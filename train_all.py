import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lmdb
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the stabilized models from your files
from test_stability import (
    create_stable_model_pipeline,
    StabilizedVAEWrapper,
    StabilizedProteinConditionedEGNNDynamics,
    StabilizedEnLatentDiffusion
)
from egnn.models import EGNNEncoder, EGNNDecoder, ProteinConditionedEGNNDynamics
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE, EnLatentDiffusion
import equivariant_diffusion.utils as diffusion_utils

class WrappedDynamics(nn.Module):
    """Wrapper for protein-conditioned dynamics to work with EnLatentDiffusion."""
    
    def __init__(self, dynamics, protein_vae=None):
        super().__init__()
        self.dynamics = dynamics
        self.protein_vae = protein_vae
        self.protein_latent = None  # Will be set before diffusion
        self._protein_mask_cache = {}  # Cache for protein masks
        
    def set_protein_latent(self, protein_latent):
        """Set the protein latent for conditioning."""
        self.protein_latent = protein_latent
        self._protein_mask_cache.clear()  # Clear cache when protein changes
        
    def _get_protein_mask(self, batch_size, n_protein_atoms, device):
        """Get or create protein mask for given batch size."""
        cache_key = (batch_size, n_protein_atoms, device)
        if cache_key not in self._protein_mask_cache:
            self._protein_mask_cache[cache_key] = torch.ones(
                batch_size, n_protein_atoms, 1, device=device
            )
        return self._protein_mask_cache[cache_key]
        
    def _forward(self, t, xh, node_mask, edge_mask, context):
        """Forward pass for denoising with protein conditioning.
        
        Args:
            t: Timestep
            xh: Ligand latent features [bs, n_ligand, 3 + latent_dim]
            node_mask: Ligand node mask
            edge_mask: Ligand edge mask
            context: Additional context (unused in this case)
        """
        if self.protein_latent is None:
            raise ValueError("Protein latent must be set before forward pass")
        
        # Get batch size from input
        bs = xh.shape[0]
        
        # Handle batch size mismatch with protein latent
        if self.protein_latent.shape[0] != bs:
            if self.protein_latent.shape[0] > bs:
                # Use subset of protein latents
                protein_latent_batch = self.protein_latent[:bs]
            else:
                # Repeat the first protein for all samples in batch
                protein_latent_batch = self.protein_latent[:1].repeat(bs, 1, 1)
        else:
            protein_latent_batch = self.protein_latent
        
        # Create protein mask matching the batch size
        n_protein_atoms = protein_latent_batch.shape[1]
        protein_mask = self._get_protein_mask(bs, n_protein_atoms, xh.device)
        
        # Call the protein-conditioned dynamics
        return self.dynamics(t, xh, protein_latent_batch, node_mask, protein_mask)
    
    def forward(self, t, xh, node_mask, edge_mask, context=None):
        """Standard forward interface."""
        return self._forward(t, xh, node_mask, edge_mask, context)

class ClassifierFreeGuidanceEGNNDynamics(ProteinConditionedEGNNDynamics):
    """Extended dynamics model with classifier-free guidance support."""
    
    def __init__(self, *args, cfg_dropout_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_dropout_prob = cfg_dropout_prob
        # Note: protein_feature_nf includes coordinates, but unconditional embedding only needs features
        self.unconditional_protein_embedding = nn.Parameter(
            torch.randn(1, 1, self.protein_feature_nf - 3) * 0.01  # -3 because we'll add dummy coords
        )
        
    def forward(self, t, z_xh, y_xh, ligand_mask=None, protein_mask=None, 
                use_conditional=True, cfg_scale=None):
        """
        Forward pass with optional classifier-free guidance.
        
        Args:
            use_conditional: If False, use unconditional generation
            cfg_scale: If provided during inference, applies guidance scaling
        """
        bs, n_ligand, _ = z_xh.shape
        
        if self.training:
            # During training, randomly drop conditioning
            if torch.rand(1).item() < self.cfg_dropout_prob:
                # Use unconditional embedding with dummy coordinates
                y_xh_uncond = self.unconditional_protein_embedding.expand(bs, 1, -1)
                # Add dummy coordinates (zeros)
                y_xh = torch.cat([
                    torch.zeros(bs, 1, 3, device=z_xh.device),
                    y_xh_uncond
                ], dim=-1)
                protein_mask = torch.ones(bs, 1, 1, device=z_xh.device)
                use_conditional = False
        
        elif cfg_scale is not None and cfg_scale != 1.0:
            # During inference with guidance
            # Get conditional prediction
            with torch.no_grad():
                vel_conditional = super().forward(t, z_xh, y_xh, ligand_mask, protein_mask)
                
                # Get unconditional prediction
                y_uncond = self.unconditional_protein_embedding.expand(bs, 1, -1)
                y_uncond = torch.cat([
                    torch.zeros(bs, 1, 3, device=z_xh.device),  # dummy positions
                    y_uncond
                ], dim=-1)
                protein_mask_uncond = torch.ones(bs, 1, 1, device=z_xh.device)
                vel_unconditional = super().forward(t, z_xh, y_uncond, ligand_mask, protein_mask_uncond)
            
            # Apply guidance: vel = vel_uncond + scale * (vel_cond - vel_uncond)
            return vel_unconditional + cfg_scale * (vel_conditional - vel_unconditional)
        
        # Standard forward pass
        return super().forward(t, z_xh, y_xh, ligand_mask, protein_mask)


class CrossDockedDataset(Dataset):
    """Dataset for loading CrossDocked protein-ligand complexes from LMDB."""
    
    def __init__(self, 
                 lmdb_path: str,
                 split_file: str,
                 split: str = 'train',
                 max_protein_atoms: int = 350,
                 max_ligand_atoms: int = 50,
                 pocket_radius: float = 10.0):
        
        self.lmdb_path = lmdb_path
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.pocket_radius = pocket_radius
        
        # Load split indices
        splits = torch.load(split_file)
        self.indices = list(splits[split])
        
        # Don't open LMDB here - do it lazily
        self.env = None  # Will be opened in __getitem__
        
        # Rest of init remains the same...
        self.ligand_size_distribution = []
        self.protein_size_distribution = []
        
        self.element_mapping = {
            1: 0, 6: 1, 7: 2, 8: 3, 9: 4,
            15: 5, 16: 6, 17: 7, 35: 8, 53: 9
        }
        self.num_elements = 10
        self.aa_mapping = {i: i for i in range(20)}
        self.num_aa_types = 20
    
    def __getitem__(self, idx):
        # Open LMDB environment lazily (once per worker process)
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=False)
        
        actual_idx = self.indices[idx]
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, value) in enumerate(cursor):
                if i == actual_idx:
                    data = pickle.loads(value)
                    break
            else:
                raise IndexError(f"Index {actual_idx} not found in LMDB")
        
        processed = self.process_complex(data)
        return processed
        
    def __len__(self):
        return len(self.indices)
    
    def process_complex(self, data: Dict) -> Dict:
        """Process a protein-ligand complex into model-ready format."""
        
        # Extract ligand data
        ligand_pos = data['ligand_pos']
        ligand_element = data['ligand_element']
        n_ligand = len(ligand_element)
        
        # Extract protein pocket
        protein_pos = data['protein_pos']
        protein_element = data['protein_element']
        protein_is_backbone = data['protein_is_backbone']
        protein_aa_type = data['protein_atom_to_aa_type']
        
        # Find pocket atoms
        ligand_center = ligand_pos.mean(dim=0)
        protein_dists = torch.norm(protein_pos - ligand_center.unsqueeze(0), dim=1)
        pocket_mask = protein_dists < self.pocket_radius
        
        # Extract pocket atoms
        pocket_pos = protein_pos[pocket_mask]
        pocket_element = protein_element[pocket_mask]
        pocket_is_backbone = protein_is_backbone[pocket_mask]
        pocket_aa_type = protein_aa_type[pocket_mask]
        n_pocket = len(pocket_element)
        
        # Center protein and ligand
        if n_pocket > 0:
            # Calculate protein COM
            protein_com = pocket_pos.mean(dim=0)
            # Center both protein and ligand relative to protein COM
            pocket_pos_centered = pocket_pos - protein_com
            ligand_pos_relative = ligand_pos - protein_com
        else:
            pocket_pos_centered = pocket_pos
            ligand_pos_relative = ligand_pos
        
        # Prepare data with centering disabled (already done above)
        ligand_data = self.prepare_molecule_data(
            ligand_pos_relative, ligand_element, None, None,
            self.max_ligand_atoms, is_ligand=True, center_coords=False
        )
        
        protein_data = self.prepare_molecule_data(
            pocket_pos_centered, pocket_element, pocket_is_backbone, pocket_aa_type,
            self.max_protein_atoms, is_ligand=False, center_coords=False
        )
        
        # Store size statistics
        self.ligand_size_distribution.append(min(n_ligand, self.max_ligand_atoms))
        self.protein_size_distribution.append(min(n_pocket, self.max_protein_atoms))
        
        # Determine aromatic atoms for ligand (simplified - ring detection)
        ligand_aromatic = self.detect_aromatic_atoms(
            data.get('ligand_bond_index', None),
            data.get('ligand_bond_type', None),
            n_ligand
        )
        
        # IMPORTANT FIX: Properly pad the aromatic tensor to max_ligand_atoms
        aromatic_padded = torch.zeros(self.max_ligand_atoms, device=ligand_aromatic.device)
        n_atoms_to_copy = min(len(ligand_aromatic), self.max_ligand_atoms)
        aromatic_padded[:n_atoms_to_copy] = ligand_aromatic[:n_atoms_to_copy]
        
        return {
            'ligand_x': ligand_data['x'],
            'ligand_h': ligand_data['h'],
            'ligand_mask': ligand_data['mask'],
            'ligand_edge_mask': ligand_data['edge_mask'],
            'ligand_aromatic': aromatic_padded,  # Now properly padded
            'protein_x': protein_data['x'],
            'protein_h': protein_data['h'],
            'protein_mask': protein_data['mask'],
            'protein_edge_mask': protein_data['edge_mask'],
            'protein_backbone': protein_data.get('backbone', None),
            'protein_aa_type': protein_data.get('aa_type', None),
            'n_ligand_atoms': min(n_ligand, self.max_ligand_atoms),
            'n_protein_atoms': min(n_pocket, self.max_protein_atoms),
        }
    
    def prepare_molecule_data(self, pos, element, is_backbone=None, aa_type=None,
                         max_atoms=50, is_ligand=True, center_coords=True):
        """Prepare molecule data with padding/truncation."""
        
        n_atoms = min(len(element), max_atoms)
        device = pos.device if torch.is_tensor(pos) else torch.device('cpu')
        
        # Initialize tensors
        x = torch.zeros(max_atoms, 3, device=device)
        h_element = torch.zeros(max_atoms, self.num_elements, device=device)
        mask = torch.zeros(max_atoms, 1, device=device)
        
        # Fill with actual data
        if n_atoms > 0:
            x[:n_atoms] = pos[:n_atoms] if torch.is_tensor(pos) else torch.tensor(pos, device=device)
            
            # Convert elements to one-hot
            for i in range(n_atoms):
                elem = element[i].item() if torch.is_tensor(element[i]) else element[i]
                elem_idx = self.element_mapping.get(elem, 1)  # Default to carbon
                h_element[i, elem_idx] = 1
            
            mask[:n_atoms] = 1
            
            # Center coordinates if needed - do it BEFORE padding
            if center_coords and not is_ligand:  # Only center proteins
                # Calculate COM only for valid atoms
                com = x[:n_atoms].mean(dim=0)
                x[:n_atoms] = x[:n_atoms] - com
        
        # Create edge mask
        edge_mask = (mask @ mask.T).view(-1, 1)
        
        result = {
            'x': x,
            'h': {'categorical': h_element, 'integer': torch.zeros(max_atoms, 1, device=device)},
            'mask': mask,
            'edge_mask': edge_mask
        }
        
        # Add protein-specific features
        if not is_ligand:
            if is_backbone is not None:
                backbone_feat = torch.zeros(max_atoms, 1, device=device)
                backbone_feat[:n_atoms] = is_backbone[:n_atoms].float().unsqueeze(1)
                result['backbone'] = backbone_feat
            
            if aa_type is not None:
                aa_feat = torch.zeros(max_atoms, self.num_aa_types, device=device)
                for i in range(n_atoms):
                    aa_idx = aa_type[i].item() if torch.is_tensor(aa_type[i]) else aa_type[i]
                    aa_idx = self.aa_mapping.get(aa_idx, 0)
                    aa_feat[i, aa_idx] = 1
                result['aa_type'] = aa_feat
        
        return result
    
    def detect_aromatic_atoms(self, bond_index, bond_type, n_atoms):
        """Simple aromatic atom detection based on bond types."""
        aromatic = torch.zeros(n_atoms)
        
        if bond_index is not None and bond_type is not None:
            # Bond type 4 typically indicates aromatic bonds
            aromatic_bonds = (bond_type == 4).nonzero(as_tuple=True)[0]
            if len(aromatic_bonds) > 0:
                aromatic_atoms = torch.unique(bond_index[:, aromatic_bonds].flatten())
                aromatic_atoms = aromatic_atoms[aromatic_atoms < n_atoms]
                aromatic[aromatic_atoms] = 1
        
        return aromatic
    
    def get_size_distribution(self):
        """Return size distributions for sampling."""
        return {
            'ligand': torch.tensor(self.ligand_size_distribution),
            'protein': torch.tensor(self.protein_size_distribution)
        }


class CrossDockedTrainer:
    """Trainer for the stabilized latent diffusion model on CrossDocked data."""
    
    def __init__(self,
                 lmdb_path: str,
                 split_file: str,
                 device: str = 'cuda',
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 num_workers: int = 4,
                 cfg_dropout_prob: float = 0.1,
                 cfg_scale: float = 7.5,
                 rank: int = 0,
                 world_size: int = 1,
                 distributed: bool = False,
                 use_amp: bool = True):
        
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.use_amp = use_amp
        self.device = f'cuda:{rank}' if distributed else device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_scale = cfg_scale  # For inference
        
        # Create datasets
        if rank == 0:
            print("Loading datasets...")
        self.train_dataset = CrossDockedDataset(
            lmdb_path, split_file, 'train',
            max_protein_atoms=350, max_ligand_atoms=50
        )
        self.test_dataset = CrossDockedDataset(
            lmdb_path, split_file, 'test',
            max_protein_atoms=350, max_ligand_atoms=50
        )
        
        # Create dataloaders with distributed sampler if needed
        if distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=True
            )
            self.test_sampler = DistributedSampler(
                self.test_dataset, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=False
            )
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=batch_size,
                sampler=self.train_sampler,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
            self.test_loader = DataLoader(
                self.test_dataset, 
                batch_size=batch_size,
                sampler=self.test_sampler,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        else:
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
        
        if rank == 0:
            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Test samples: {len(self.test_dataset)}")
            if distributed:
                print(f"Distributed training on {world_size} GPUs")
                print(f"Effective batch size: {batch_size * world_size}")
        
        # Feature dimensions
        self.ligand_features = 10 + 1 + 1  # elements + aromatic + charge
        self.protein_features = 10 + 1 + 20 + 1  # elements + backbone + aa_type + charge
        self.latent_dim = 32
        
        # Create models with CFG
        self.create_models()
        
        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
            if rank == 0:
                print("Using Automatic Mixed Precision (AMP) for faster training")
        
    def collate_fn(self, batch):
        """Custom collate function to handle batching."""
        # Stack all tensors
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
    
    def create_models(self):
        """Create the VAEs and diffusion model with CFG support."""
        if self.rank == 0:
            print("Creating models with classifier-free guidance...")
        
        # Create VAE for ligand (no centering in VAE since we handle it in dataset)
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
        
        # Create VAE for protein
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
        
        # Wrap VAEs with stability layers
        self.stable_ligand_vae = StabilizedVAEWrapper(self.ligand_vae, latent_scale=0.1)
        self.stable_protein_vae = StabilizedVAEWrapper(self.protein_vae, latent_scale=0.1)
        
        # Create CFG-enabled dynamics
        self.dynamics = ClassifierFreeGuidanceEGNNDynamics(
            ligand_feature_nf=self.latent_dim,
            protein_feature_nf=self.latent_dim + 3,  # +3 for coordinates that are concatenated
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
            trainable_ae=False,
            noise_precision=1e-4
        ).to(self.device)
        
        # Wrap with stability improvements
        self.diffusion_model = StabilizedEnLatentDiffusion(
            base_diffusion,
            gradient_clip=1.0
        )
        
        # Wrap models with DDP if using distributed training
        if self.distributed:
            self.diffusion_model = DDP(
                self.diffusion_model, 
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
            self.dynamics = DDP(
                self.dynamics,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
            # Get the underlying module for accessing attributes
            self.diffusion_module = self.diffusion_model.module
            self.dynamics_module = self.dynamics.module
        else:
            self.diffusion_module = self.diffusion_model
            self.dynamics_module = self.dynamics
        
        # Create optimizer (adjusted learning rate for multi-GPU)
        adjusted_lr = self.learning_rate * np.sqrt(self.world_size) if self.distributed else self.learning_rate
        self.optimizer, self.scheduler = self.diffusion_module.configure_optimizers(
            lr=adjusted_lr
        )
        
        if self.rank == 0:
            print(f"Models created successfully!")
            print(f"Ligand VAE params: {sum(p.numel() for p in self.ligand_vae.parameters())}")
            print(f"Protein VAE params: {sum(p.numel() for p in self.protein_vae.parameters())}")
            print(f"Diffusion params: {sum(p.numel() for p in self.diffusion_module.model.parameters())}")
            print(f"CFG dropout probability: {self.cfg_dropout_prob}")
            if self.distributed:
                print(f"Learning rate scaled from {self.learning_rate} to {adjusted_lr}")
    
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
        """Train for one epoch with optional mixed precision."""
        if self.distributed:
            self.train_sampler.set_epoch(epoch)  # Important for proper shuffling
            
        self.diffusion_model.train() if self.distributed else self.diffusion_module.model.train()
        self.dynamics.train() if self.distributed else self.dynamics_module.train()
        
        total_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=(self.rank != 0))
        
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
                # Encode protein
                with torch.no_grad():
                    z_x_prot, _, z_h_prot, _ = self.stable_protein_vae.encode(
                        batch['protein_x'], protein_h,
                        batch['protein_mask'], batch['protein_edge_mask']
                    )
                    # Don't concatenate coordinates and features here
                    # The dynamics model expects them separately
                    protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)
                    
                    # Set protein latent in the correct module
                    if self.distributed:
                        self.diffusion_model.module.model.dynamics.protein_latent = protein_latent
                    else:
                        self.diffusion_module.model.dynamics.protein_latent = protein_latent
                
                # Forward pass
                loss = self.diffusion_model(
                    batch['ligand_x'], ligand_h,
                    batch['ligand_mask'], batch['ligand_edge_mask']
                )
                loss = loss.mean()
            
            # Check for NaN
            if torch.isnan(loss):
                if self.rank == 0:
                    print(f"NaN loss detected, skipping batch")
                continue
            
            # Backward pass with mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_model.parameters() if self.distributed else self.diffusion_module.model.parameters(),
                    max_norm=1.0
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_model.parameters() if self.distributed else self.diffusion_module.model.parameters(),
                    max_norm=1.0
                )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update statistics
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            if self.rank == 0:
                progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / n_batches})
        
        # Scheduler step
        self.scheduler.step()
        
        # Gather losses from all processes if distributed
        if self.distributed:
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            n_batches_tensor = torch.tensor(n_batches).to(self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_tensor.item()
            n_batches = n_batches_tensor.item()
        
        return total_loss / max(n_batches, 1)
    
    def evaluate(self):
        """Evaluate on test set with optional mixed precision."""
        self.diffusion_model.eval() if self.distributed else self.diffusion_module.model.eval()
        self.dynamics.eval() if self.distributed else self.dynamics_module.eval()
        
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", disable=(self.rank != 0)):
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
                    if self.distributed:
                        self.diffusion_model.module.model.dynamics.protein_latent = protein_latent
                    else:
                        self.diffusion_module.model.dynamics.protein_latent = protein_latent
                    
                    # Forward pass (no CFG dropout during evaluation)
                    loss = self.diffusion_model(
                        batch['ligand_x'], ligand_h,
                        batch['ligand_mask'], batch['ligand_edge_mask']
                    )
                    loss = loss.mean()
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    n_batches += 1
        
        # Gather losses from all processes if distributed
        if self.distributed:
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            n_batches_tensor = torch.tensor(n_batches).to(self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_tensor.item()
            n_batches = n_batches_tensor.item()
        
        return total_loss / max(n_batches, 1)
    
    def sample_ligand_size(self):
        """Sample ligand size from training distribution."""
        size_dist = self.train_dataset.get_size_distribution()['ligand']
        # Add some noise for diversity
        sampled_size = np.random.choice(size_dist.numpy())
        sampled_size = int(np.clip(sampled_size + np.random.randn() * 2, 5, 50))
        return sampled_size
    
    def generate_ligands(self, protein_batch, n_samples=1, use_cfg=True):
        """Generate ligands for given protein pockets with optional CFG."""
        self.diffusion_model.model.eval()
        self.dynamics.eval()
        
        with torch.no_grad():
            # Prepare protein features
            _, protein_h = self.prepare_features(protein_batch)
            
            # Encode protein
            z_x_prot, _, z_h_prot, _ = self.stable_protein_vae.encode(
                protein_batch['protein_x'], protein_h,
                protein_batch['protein_mask'], protein_batch['protein_edge_mask']
            )
            protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)
            
            # Store protein latent for conditional generation
            self.diffusion_model.model.dynamics.protein_latent = protein_latent
            
            # Sample ligand size
            n_ligand_atoms = self.sample_ligand_size()
            
            # Create ligand mask
            ligand_mask = torch.zeros(n_samples, 50, 1).to(self.device)
            ligand_mask[:, :n_ligand_atoms, :] = 1
            
            edge_mask = (ligand_mask @ ligand_mask.transpose(1, 2)).view(n_samples, -1, 1)
            
            # Modify sampling to use CFG
            if use_cfg:
                # Temporarily set dynamics to use CFG scale during sampling
                original_forward = self.dynamics.forward
                
                def cfg_forward(t, z_xh, y_xh, ligand_mask=None, protein_mask=None):
                    return original_forward(t, z_xh, y_xh, ligand_mask, protein_mask, 
                                          cfg_scale=self.cfg_scale)
                
                self.dynamics.forward = cfg_forward
            
            # Generate ligands
            x_gen, h_gen = self.diffusion_model.model.sample(
                n_samples=n_samples,
                n_nodes=50,
                node_mask=ligand_mask,
                edge_mask=edge_mask,
                context=None
            )
            
            # Restore original forward if we modified it
            if use_cfg:
                self.dynamics.forward = original_forward
            
            return x_gen, h_gen, ligand_mask
    
    def train(self, num_epochs=100, save_interval=10):
        """Main training loop with distributed support."""
        if self.rank == 0:
            print("Starting training with classifier-free guidance...")
            print(f"CFG dropout: {self.cfg_dropout_prob}, CFG scale for inference: {self.cfg_scale}")
            if self.distributed:
                print(f"Distributed training on {self.world_size} GPUs")
                print(f"Effective batch size: {self.batch_size * self.world_size}")
            if self.use_amp:
                print("Using Automatic Mixed Precision (AMP)")
        
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience = 0
        max_patience = 5
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Evaluate
            test_loss = self.evaluate()
            test_losses.append(test_loss)
            
            if self.rank == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
                
                # Check for improvement
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience = 0
                    
                    # Save best model (only rank 0 saves)
                    state_dict = {
                        'epoch': epoch,
                        'diffusion_state': self.diffusion_module.state_dict(),
                        'dynamics_state': self.dynamics_module.state_dict() if hasattr(self, 'dynamics_module') else self.dynamics.state_dict(),
                        'ligand_vae_state': self.ligand_vae.state_dict(),
                        'protein_vae_state': self.protein_vae.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'cfg_dropout_prob': self.cfg_dropout_prob,
                        'cfg_scale': self.cfg_scale,
                    }
                    
                    if self.use_amp:
                        state_dict['scaler_state'] = self.scaler.state_dict()
                    
                    torch.save(state_dict, 'best_model.pt')
                    print(f"Saved best model with test loss {test_loss:.4f}")
                else:
                    patience += 1
                
                # Periodic save
                if epoch % save_interval == 0:
                    state_dict = {
                        'epoch': epoch,
                        'diffusion_state': self.diffusion_module.state_dict(),
                        'dynamics_state': self.dynamics_module.state_dict() if hasattr(self, 'dynamics_module') else self.dynamics.state_dict(),
                        'ligand_vae_state': self.ligand_vae.state_dict(), 
                        'protein_vae_state': self.protein_vae.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                    }
                    
                    if self.use_amp:
                        state_dict['scaler_state'] = self.scaler.state_dict()
                    
                    torch.save(state_dict, f'./checkpoints/checkpoint_epoch_{epoch}.pt')
                    
                    # Plot losses
                    self.plot_losses(train_losses, test_losses, epoch)
            
            # Broadcast patience to all processes if distributed
            if self.distributed:
                patience_tensor = torch.tensor(patience).to(self.device)
                dist.broadcast(patience_tensor, src=0)
                patience = patience_tensor.item()
            
            # Early stopping
            if patience >= max_patience:
                if self.rank == 0:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if self.rank == 0:
            print("Training completed!")
        
        return train_losses, test_losses
    
    def plot_losses(self, train_losses, test_losses, epoch):
        """Plot training and test losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'losses_epoch_{epoch}.png')
        plt.close()


def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, lmdb_path, split_file, args):
    """Training function for each distributed process."""
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    # Create trainer for this process
    trainer = CrossDockedTrainer(
        lmdb_path=lmdb_path,
        split_file=split_file,
        device=f'cuda:{rank}',
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        cfg_dropout_prob=args.cfg_dropout_prob,
        cfg_scale=args.cfg_scale,
        rank=rank,
        world_size=world_size,
        distributed=True,
        use_amp=args.use_amp
    )
    
    # Train model
    train_losses, test_losses = trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval
    )
    
    # Clean up
    cleanup()
    
    return train_losses, test_losses

def main():
    """Main training script with multi-GPU support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CrossDocked Latent Diffusion Model')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--save_interval', type=int, default=3, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers per GPU')
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1, help='CFG dropout probability')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale for inference')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use AMP')
    
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
    
    # Determine number of GPUs
    n_gpus_available = torch.cuda.device_count()
    n_gpus = min(args.num_gpus, n_gpus_available)
    
    print(f"Available GPUs: {n_gpus_available}")
    print(f"Using {n_gpus} GPU(s)")
    
    if n_gpus > 1:
        # Multi-GPU training
        print(f"Starting distributed training on {n_gpus} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total effective batch size: {args.batch_size * n_gpus}")
        print(f"Workers per GPU: {args.num_workers}")
        print(f"Total workers: {args.num_workers * n_gpus}")
        
        mp.spawn(
            train_distributed,
            args=(n_gpus, lmdb_path, split_file, args),
            nprocs=n_gpus,
            join=True
        )
        
        # After distributed training, load the best model for evaluation
        # Only main process does final evaluation
        device = 'cuda:0'
        distributed = False
        
    else:
        # Single GPU training
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        distributed = False
        
        print(f"Running on single GPU: {device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Workers: {args.num_workers}")
        
        # Create trainer with CFG
        trainer = CrossDockedTrainer(
            lmdb_path=lmdb_path,
            split_file=split_file,
            device=device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            cfg_dropout_prob=args.cfg_dropout_prob,
            cfg_scale=args.cfg_scale,
            rank=0,
            world_size=1,
            distributed=False,
            use_amp=args.use_amp and device != 'cpu'
        )
        
        # Train model
        train_losses, test_losses = trainer.train(
            num_epochs=args.num_epochs,
            save_interval=args.save_interval
        )
    
    # Final evaluation and generation (for both single and multi-GPU cases)
    if not distributed or n_gpus == 1:
        # Reload trainer for final evaluation if multi-GPU was used
        if n_gpus > 1:
            trainer = CrossDockedTrainer(
                lmdb_path=lmdb_path,
                split_file=split_file,
                device=device,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_workers=2,  # Fewer workers for evaluation
                cfg_dropout_prob=args.cfg_dropout_prob,
                cfg_scale=args.cfg_scale,
                rank=0,
                world_size=1,
                distributed=False,
                use_amp=args.use_amp
            )
            
            # Load best model
            if os.path.exists('best_model.pt'):
                checkpoint = torch.load('best_model.pt')
                trainer.diffusion_module.load_state_dict(checkpoint['diffusion_state'])
                trainer.dynamics_module.load_state_dict(checkpoint['dynamics_state'])
                print("Loaded best model for evaluation")
        
        # Final evaluation
        print("\nFinal evaluation on test set:")
        final_test_loss = trainer.evaluate()
        print(f"Final test loss: {final_test_loss:.4f}")
        
        # Generate samples with and without CFG
        print("\nGenerating sample ligands...")
        test_batch = next(iter(trainer.test_loader))
        for key in test_batch:
            if isinstance(test_batch[key], torch.Tensor):
                test_batch[key] = test_batch[key].to(trainer.device)
            elif isinstance(test_batch[key], dict):
                for sub_key in test_batch[key]:
                    test_batch[key][sub_key] = test_batch[key][sub_key].to(trainer.device)
        
        # Take first protein from batch
        protein_batch = {
            'protein_x': test_batch['protein_x'][:1],
            'protein_h': {k: v[:1] for k, v in test_batch['protein_h'].items()},
            'protein_mask': test_batch['protein_mask'][:1],
            'protein_edge_mask': test_batch['protein_edge_mask'][:1],
            'protein_backbone': test_batch['protein_backbone'][:1],
            'protein_aa_type': test_batch['protein_aa_type'][:1],
        }
        
        # Generate with CFG
        print("Generating with classifier-free guidance...")
        x_gen_cfg, h_gen_cfg, mask_cfg = trainer.generate_ligands(
            protein_batch, n_samples=5, use_cfg=True
        )
        print(f"Generated {x_gen_cfg.shape[0]} ligands with CFG")
        
        # Generate without CFG for comparison
        print("Generating without classifier-free guidance...")
        x_gen_no_cfg, h_gen_no_cfg, mask_no_cfg = trainer.generate_ligands(
            protein_batch, n_samples=5, use_cfg=False
        )
        print(f"Generated {x_gen_no_cfg.shape[0]} ligands without CFG")
        
        # Save final results
        final_results = {
            'final_test_loss': final_test_loss,
            'generated_ligands_cfg': (x_gen_cfg, h_gen_cfg, mask_cfg),
            'generated_ligands_no_cfg': (x_gen_no_cfg, h_gen_no_cfg, mask_no_cfg),
            'cfg_settings': {'dropout_prob': args.cfg_dropout_prob, 'scale': args.cfg_scale},
            'training_config': {
                'num_gpus': n_gpus,
                'batch_size_per_gpu': args.batch_size,
                'total_batch_size': args.batch_size * n_gpus if n_gpus > 1 else args.batch_size,
                'learning_rate': args.learning_rate,
                'num_workers': args.num_workers
            }
        }
        
        # Load training history if available
        if n_gpus == 1:
            final_results['train_losses'] = train_losses
            final_results['test_losses'] = test_losses
        elif os.path.exists('checkpoints/checkpoint_epoch_10.pt'):
            # Load from checkpoint if multi-GPU was used
            checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')
            if 'train_losses' in checkpoint:
                final_results['train_losses'] = checkpoint['train_losses']
                final_results['test_losses'] = checkpoint['test_losses']
        
        torch.save(final_results, 'final_results.pt')
        
        print("\nTraining complete! Results saved.")
        print(f"CFG improved generation quality with scale={args.cfg_scale}")
        print(f"Training configuration: {n_gpus} GPU(s), batch size {args.batch_size} per GPU")



if __name__ == "__main__":
    main()