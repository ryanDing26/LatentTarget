"""
Script for training VAE models (ligand and protein) separately.
This allows for modular training where VAE can be pre-trained and frozen.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

from train_all import CrossDockedDataset
from egnn.models import EGNNEncoder, EGNNDecoder
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE
from test_stability import StabilizedVAEWrapper


class VAETrainer:
    """Trainer for VAE models only."""

    def __init__(self,
                 lmdb_path: str,
                 split_file: str,
                 model_type: str = 'ligand',  # 'ligand' or 'protein'
                 device: str = 'cuda',
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 num_workers: int = 4,
                 use_amp: bool = True):

        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.use_amp = use_amp

        # Create datasets
        print(f"Loading datasets for {model_type} VAE training...")
        self.train_dataset = CrossDockedDataset(
            lmdb_path, split_file, 'train',
            max_protein_atoms=350, max_ligand_atoms=50
        )
        self.test_dataset = CrossDockedDataset(
            lmdb_path, split_file, 'test',
            max_protein_atoms=350, max_ligand_atoms=50
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

        # Feature dimensions
        self.ligand_features = 10 + 1 + 1  # elements + aromatic + charge
        self.protein_features = 10 + 1 + 20 + 1  # elements + backbone + aa_type + charge
        self.latent_dim = 32

        # Create VAE model
        self.create_model()

        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP) for faster training")

    def collate_fn(self, batch):
        """Custom collate function to handle batching."""
        from collections import defaultdict

        collated = defaultdict(list)
        for sample in batch:
            for key, value in sample.items():
                collated[key].append(value)

        # Stack numeric tensors
        result = {}
        for key, values in collated.items():
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif isinstance(values[0], dict):
                # Handle nested dicts (h features)
                result[key] = {}
                for sub_key in values[0].keys():
                    result[key][sub_key] = torch.stack([v[sub_key] for v in values])
            else:
                result[key] = torch.tensor(values)

        return result

    def create_model(self):
        """Create the VAE model."""
        print(f"Creating {self.model_type} VAE...")

        if self.model_type == 'ligand':
            in_features = self.ligand_features
            structure = 'ligand'
        else:  # protein
            in_features = self.protein_features
            structure = 'protein'

        # Create encoder
        encoder = EGNNEncoder(
            in_node_nf=in_features,
            context_node_nf=0,
            out_node_nf=self.latent_dim,
            n_dims=3,
            hidden_nf=128,
            n_layers=3,
            device=self.device,
            include_charges=True
        )

        # Create decoder
        decoder = EGNNDecoder(
            in_node_nf=self.latent_dim,
            context_node_nf=0,
            out_node_nf=in_features,
            n_dims=3,
            hidden_nf=128,
            n_layers=3,
            device=self.device,
            include_charges=True
        )

        # Create VAE
        self.vae = EnHierarchicalVAE(
            encoder=encoder,
            decoder=decoder,
            in_node_nf=in_features,
            n_dims=3,
            latent_node_nf=self.latent_dim,
            kl_weight=0.01,
            structure=structure,
            include_charges=True
        ).to(self.device)

        # Wrap with stability improvements
        self.stable_vae = StabilizedVAEWrapper(self.vae, latent_scale=0.1)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )

        print(f"VAE created successfully!")
        print(f"Parameters: {sum(p.numel() for p in self.vae.parameters())}")

    def prepare_features(self, batch):
        """Prepare features for the model."""
        if self.model_type == 'ligand':
            # Combine features for ligand
            h_cat = batch['ligand_h']['categorical']
            aromatic = batch['ligand_aromatic'].unsqueeze(-1).to(self.device)
            h_int = batch['ligand_h']['integer']

            h = {
                'categorical': torch.cat([h_cat, aromatic], dim=-1),
                'integer': h_int
            }
            x = batch['ligand_x']
            mask = batch['ligand_mask']
            edge_mask = batch['ligand_edge_mask']
        else:  # protein
            # Combine features for protein
            h_cat = batch['protein_h']['categorical']
            backbone = batch['protein_backbone'].to(self.device)
            aa_type = batch['protein_aa_type'].to(self.device)
            h_int = batch['protein_h']['integer']

            h = {
                'categorical': torch.cat([h_cat, backbone, aa_type], dim=-1),
                'integer': h_int
            }
            x = batch['protein_x']
            mask = batch['protein_mask']
            edge_mask = batch['protein_edge_mask']

        return x, h, mask, edge_mask

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.vae.train()

        total_loss = 0
        total_recon_loss = 0
        n_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
                elif isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        batch[key][sub_key] = batch[key][sub_key].to(self.device, non_blocking=True)

            # Prepare features
            x, h, mask, edge_mask = self.prepare_features(batch)

            # Mixed precision context
            with autocast(enabled=self.use_amp):
                # Forward pass
                loss = self.vae(x, h, mask, edge_mask)
                loss = loss.mean()

            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected, skipping batch")
                continue

            # Backward pass with mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    max_norm=1.0
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update statistics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / n_batches
            })

        # Scheduler step
        self.scheduler.step()

        return total_loss / max(n_batches, 1)

    def evaluate(self):
        """Evaluate on test set."""
        self.vae.eval()

        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                    elif isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            batch[key][sub_key] = batch[key][sub_key].to(self.device, non_blocking=True)

                # Prepare features
                x, h, mask, edge_mask = self.prepare_features(batch)

                with autocast(enabled=self.use_amp):
                    # Forward pass
                    loss = self.vae(x, h, mask, edge_mask)
                    loss = loss.mean()

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, num_epochs=100, save_interval=10):
        """Main training loop."""
        print(f"Starting {self.model_type} VAE training...")
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")

        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience = 0
        max_patience = 20

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)

            # Evaluate
            test_loss = self.evaluate()
            test_losses.append(test_loss)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

            # Check for improvement
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience = 0

                # Save best model
                save_path = f'best_{self.model_type}_vae.pt'
                state_dict = {
                    'epoch': epoch,
                    'vae_state': self.vae.state_dict(),
                    'stable_vae_state': self.stable_vae.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'model_type': self.model_type,
                    'latent_dim': self.latent_dim,
                }

                if self.use_amp:
                    state_dict['scaler_state'] = self.scaler.state_dict()

                torch.save(state_dict, save_path)
                print(f"Saved best {self.model_type} VAE with test loss {test_loss:.4f}")
            else:
                patience += 1

            # Periodic save
            if epoch % save_interval == 0:
                checkpoint_path = f'./checkpoints/{self.model_type}_vae_epoch_{epoch}.pt'
                state_dict = {
                    'epoch': epoch,
                    'vae_state': self.vae.state_dict(),
                    'stable_vae_state': self.stable_vae.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                }

                if self.use_amp:
                    state_dict['scaler_state'] = self.scaler.state_dict()

                torch.save(state_dict, checkpoint_path)

                # Plot losses
                self.plot_losses(train_losses, test_losses, epoch)

            # Early stopping
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"{self.model_type.capitalize()} VAE training completed!")

        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses, epoch):
        """Plot training and test losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_type.capitalize()} VAE Training Progress - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.model_type}_vae_losses_epoch_{epoch}.png')
        plt.close()


def main():
    """Main function for VAE training."""
    parser = argparse.ArgumentParser(description='Train VAE models separately')
    parser.add_argument('--model_type', type=str, default='ligand',
                       choices=['ligand', 'protein'],
                       help='Type of VAE to train (ligand or protein)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use AMP')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Paths
    lmdb_path = "./cd2020/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    split_file = "./cd2020/crossdocked_pocket10_pose_split.pt"

    # Check if files exist
    if not os.path.exists(lmdb_path):
        print(f"Error: LMDB file not found at {lmdb_path}")
        return
    if not os.path.exists(split_file):
        print(f"Error: Split file not found at {split_file}")
        return

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Create trainer
    trainer = VAETrainer(
        lmdb_path=lmdb_path,
        split_file=split_file,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        use_amp=args.use_amp and args.device != 'cpu'
    )

    # Train model
    train_losses, test_losses = trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval
    )

    print(f"\n{args.model_type.capitalize()} VAE training complete!")
    print(f"Best model saved as: best_{args.model_type}_vae.pt")


if __name__ == "__main__":
    main()
