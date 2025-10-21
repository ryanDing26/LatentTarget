"""
Script for training the denoiser model with frozen VAE weights.
This allows for modular training where pre-trained VAEs can be loaded and frozen.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

from train_all import CrossDockedDataset, WrappedDynamics, ClassifierFreeGuidanceEGNNDynamics
from egnn.models import EGNNEncoder, EGNNDecoder
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE, EnLatentDiffusion
from test_stability import StabilizedVAEWrapper, StabilizedEnLatentDiffusion


class DenoiserTrainer:
    """Trainer for denoiser model with frozen VAE."""

    def __init__(self,
                 lmdb_path: str,
                 split_file: str,
                 ligand_vae_path: str = None,
                 protein_vae_path: str = None,
                 device: str = 'cuda',
                 batch_size: int = 16,
                 learning_rate: float = 1e-4,
                 num_workers: int = 4,
                 cfg_dropout_prob: float = 0.1,
                 cfg_scale: float = 7.5,
                 use_amp: bool = True):

        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_scale = cfg_scale
        self.use_amp = use_amp

        # Create datasets
        print("Loading datasets for denoiser training...")
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

        # Create or load VAE models
        self.create_vaes(ligand_vae_path, protein_vae_path)

        # Create denoiser model
        self.create_denoiser()

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

    def create_vaes(self, ligand_vae_path=None, protein_vae_path=None):
        """Create or load VAE models."""
        print("Setting up VAE models...")

        # Ligand VAE
        if ligand_vae_path and os.path.exists(ligand_vae_path):
            print(f"Loading pre-trained ligand VAE from {ligand_vae_path}")
            checkpoint = torch.load(ligand_vae_path, map_location=self.device)

            # Create ligand VAE with same architecture
            ligand_encoder = EGNNEncoder(
                in_node_nf=self.ligand_features,
                context_node_nf=0,
                out_node_nf=self.latent_dim,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            ligand_decoder = EGNNDecoder(
                in_node_nf=self.latent_dim,
                context_node_nf=0,
                out_node_nf=self.ligand_features,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            self.ligand_vae = EnHierarchicalVAE(
                encoder=ligand_encoder,
                decoder=ligand_decoder,
                in_node_nf=self.ligand_features,
                n_dims=3,
                latent_node_nf=self.latent_dim,
                kl_weight=0.01,
                structure='ligand',
                include_charges=True
            ).to(self.device)

            self.ligand_vae.load_state_dict(checkpoint['vae_state'])
            print("Loaded ligand VAE successfully")
        else:
            print("Creating new ligand VAE (will be trained from scratch)")
            ligand_encoder = EGNNEncoder(
                in_node_nf=self.ligand_features,
                context_node_nf=0,
                out_node_nf=self.latent_dim,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            ligand_decoder = EGNNDecoder(
                in_node_nf=self.latent_dim,
                context_node_nf=0,
                out_node_nf=self.ligand_features,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            self.ligand_vae = EnHierarchicalVAE(
                encoder=ligand_encoder,
                decoder=ligand_decoder,
                in_node_nf=self.ligand_features,
                n_dims=3,
                latent_node_nf=self.latent_dim,
                kl_weight=0.01,
                structure='ligand',
                include_charges=True
            ).to(self.device)

        # Protein VAE
        if protein_vae_path and os.path.exists(protein_vae_path):
            print(f"Loading pre-trained protein VAE from {protein_vae_path}")
            checkpoint = torch.load(protein_vae_path, map_location=self.device)

            # Create protein VAE with same architecture
            protein_encoder = EGNNEncoder(
                in_node_nf=self.protein_features,
                context_node_nf=0,
                out_node_nf=self.latent_dim,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            protein_decoder = EGNNDecoder(
                in_node_nf=self.latent_dim,
                context_node_nf=0,
                out_node_nf=self.protein_features,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            self.protein_vae = EnHierarchicalVAE(
                encoder=protein_encoder,
                decoder=protein_decoder,
                in_node_nf=self.protein_features,
                n_dims=3,
                latent_node_nf=self.latent_dim,
                kl_weight=0.01,
                structure='protein',
                include_charges=True
            ).to(self.device)

            self.protein_vae.load_state_dict(checkpoint['vae_state'])
            print("Loaded protein VAE successfully")
        else:
            print("Creating new protein VAE (will be trained from scratch)")
            protein_encoder = EGNNEncoder(
                in_node_nf=self.protein_features,
                context_node_nf=0,
                out_node_nf=self.latent_dim,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            protein_decoder = EGNNDecoder(
                in_node_nf=self.latent_dim,
                context_node_nf=0,
                out_node_nf=self.protein_features,
                n_dims=3,
                hidden_nf=128,
                n_layers=3,
                device=self.device,
                include_charges=True
            )

            self.protein_vae = EnHierarchicalVAE(
                encoder=protein_encoder,
                decoder=protein_decoder,
                in_node_nf=self.protein_features,
                n_dims=3,
                latent_node_nf=self.latent_dim,
                kl_weight=0.01,
                structure='protein',
                include_charges=True
            ).to(self.device)

        # Wrap VAEs with stability layers and FREEZE them
        self.stable_ligand_vae = StabilizedVAEWrapper(self.ligand_vae, latent_scale=0.1)
        self.stable_protein_vae = StabilizedVAEWrapper(self.protein_vae, latent_scale=0.1)

        # Freeze VAE parameters
        for param in self.ligand_vae.parameters():
            param.requires_grad = False
        for param in self.protein_vae.parameters():
            param.requires_grad = False

        self.ligand_vae.eval()
        self.protein_vae.eval()

        print("VAE models frozen and set to eval mode")
        print(f"Ligand VAE params: {sum(p.numel() for p in self.ligand_vae.parameters())}")
        print(f"Protein VAE params: {sum(p.numel() for p in self.protein_vae.parameters())}")

    def create_denoiser(self):
        """Create the denoiser (dynamics + diffusion) model."""
        print("Creating denoiser with classifier-free guidance...")

        # Create CFG-enabled dynamics
        self.dynamics = ClassifierFreeGuidanceEGNNDynamics(
            ligand_feature_nf=self.latent_dim,
            protein_feature_nf=self.latent_dim + 3,  # +3 for coordinates
            n_dims=3,
            hidden_nf=256,
            device=self.device,
            n_layers=3,
            n_cross_layers=2,
            cross_distance_threshold=8.0,
            cfg_dropout_prob=self.cfg_dropout_prob
        ).to(self.device)

        wrapped_dynamics = WrappedDynamics(self.dynamics, self.stable_protein_vae)

        # Create base diffusion model
        base_diffusion = EnLatentDiffusion(
            dynamics=wrapped_dynamics,
            vae=self.stable_ligand_vae,
            in_node_nf=self.latent_dim,
            n_dims=3,
            timesteps=1000,
            noise_schedule='polynomial_2',  # More stable
            parametrization='eps',
            loss_type='l2',
            include_charges=False,
            trainable_ae=False,  # VAE is frozen
            noise_precision=1e-4
        ).to(self.device)

        # Wrap with stability improvements
        self.diffusion_model = StabilizedEnLatentDiffusion(
            base_diffusion,
            gradient_clip=1.0
        )

        # Create optimizer (only for denoiser parameters)
        self.optimizer, self.scheduler = self.diffusion_model.configure_optimizers(
            lr=self.learning_rate
        )

        print(f"Denoiser created successfully!")
        print(f"Diffusion params: {sum(p.numel() for p in self.diffusion_model.model.parameters())}")
        print(f"Trainable params: {sum(p.numel() for p in self.diffusion_model.model.parameters() if p.requires_grad)}")
        print(f"CFG dropout probability: {self.cfg_dropout_prob}")

    def prepare_features(self, batch):
        """Prepare features for the model."""
        # Combine features for ligand
        ligand_h_cat = batch['ligand_h']['categorical']
        ligand_aromatic = batch['ligand_aromatic'].unsqueeze(-1).to(self.device)
        ligand_h_int = batch['ligand_h']['integer']

        ligand_h = {
            'categorical': torch.cat([ligand_h_cat, ligand_aromatic], dim=-1),
            'integer': ligand_h_int
        }

        # Combine features for protein
        protein_h_cat = batch['protein_h']['categorical']
        protein_backbone = batch['protein_backbone'].to(self.device)
        protein_aa = batch['protein_aa_type'].to(self.device)
        protein_h_int = batch['protein_h']['integer']

        protein_h = {
            'categorical': torch.cat([protein_h_cat, protein_backbone, protein_aa], dim=-1),
            'integer': protein_h_int
        }

        return ligand_h, protein_h

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.diffusion_model.model.train()
        self.dynamics.train()

        total_loss = 0
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
            ligand_h, protein_h = self.prepare_features(batch)

            # Mixed precision context
            with autocast(enabled=self.use_amp):
                # Encode protein (VAE is frozen, so no_grad)
                with torch.no_grad():
                    z_x_prot, _, z_h_prot, _ = self.stable_protein_vae.encode(
                        batch['protein_x'], protein_h,
                        batch['protein_mask'], batch['protein_edge_mask']
                    )
                    protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)

                    # Set protein latent in dynamics
                    self.diffusion_model.model.dynamics.protein_latent = protein_latent

                # Forward pass (only denoiser is trainable)
                loss = self.diffusion_model(
                    batch['ligand_x'], ligand_h,
                    batch['ligand_mask'], batch['ligand_edge_mask']
                )
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
                    self.diffusion_model.model.parameters(),
                    max_norm=1.0
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_model.model.parameters(),
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
        self.diffusion_model.model.eval()
        self.dynamics.eval()

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
                ligand_h, protein_h = self.prepare_features(batch)

                with autocast(enabled=self.use_amp):
                    # Encode protein
                    z_x_prot, _, z_h_prot, _ = self.stable_protein_vae.encode(
                        batch['protein_x'], protein_h,
                        batch['protein_mask'], batch['protein_edge_mask']
                    )
                    protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)

                    # Set protein latent
                    self.diffusion_model.model.dynamics.protein_latent = protein_latent

                    # Forward pass
                    loss = self.diffusion_model(
                        batch['ligand_x'], ligand_h,
                        batch['ligand_mask'], batch['ligand_edge_mask']
                    )
                    loss = loss.mean()

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, num_epochs=100, save_interval=10):
        """Main training loop."""
        print("Starting denoiser training with frozen VAE...")
        print(f"CFG dropout: {self.cfg_dropout_prob}, CFG scale for inference: {self.cfg_scale}")
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")

        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience = 0
        max_patience = 10

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
                state_dict = {
                    'epoch': epoch,
                    'diffusion_state': self.diffusion_model.state_dict(),
                    'dynamics_state': self.dynamics.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'cfg_dropout_prob': self.cfg_dropout_prob,
                    'cfg_scale': self.cfg_scale,
                }

                if self.use_amp:
                    state_dict['scaler_state'] = self.scaler.state_dict()

                torch.save(state_dict, 'best_denoiser.pt')
                print(f"Saved best denoiser with test loss {test_loss:.4f}")
            else:
                patience += 1

            # Periodic save
            if epoch % save_interval == 0:
                checkpoint_path = f'./checkpoints/denoiser_epoch_{epoch}.pt'
                state_dict = {
                    'epoch': epoch,
                    'diffusion_state': self.diffusion_model.state_dict(),
                    'dynamics_state': self.dynamics.state_dict(),
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

        print("Denoiser training completed!")

        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses, epoch):
        """Plot training and test losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Denoiser Training Progress - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'denoiser_losses_epoch_{epoch}.png')
        plt.close()


def main():
    """Main function for denoiser training."""
    parser = argparse.ArgumentParser(description='Train denoiser with frozen VAE')
    parser.add_argument('--ligand_vae_path', type=str, default=None,
                       help='Path to pre-trained ligand VAE (optional)')
    parser.add_argument('--protein_vae_path', type=str, default=None,
                       help='Path to pre-trained protein VAE (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1, help='CFG dropout probability')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale for inference')
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
    trainer = DenoiserTrainer(
        lmdb_path=lmdb_path,
        split_file=split_file,
        ligand_vae_path=args.ligand_vae_path,
        protein_vae_path=args.protein_vae_path,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        cfg_dropout_prob=args.cfg_dropout_prob,
        cfg_scale=args.cfg_scale,
        use_amp=args.use_amp and args.device != 'cpu'
    )

    # Train model
    train_losses, test_losses = trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval
    )

    print("\nDenoiser training complete!")
    print("Best model saved as: best_denoiser.pt")


if __name__ == "__main__":
    main()
