import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from egnn.models import ProteinConditionedEGNNDynamics

class StabilizedProteinConditionedEGNNDynamics(ProteinConditionedEGNNDynamics):
    """Version with numerical stability improvements."""
    
    def __init__(self, *args, gradient_clip_val=1.0, latent_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_clip_val = gradient_clip_val
        self.latent_scale = latent_scale
        
        # Add layer normalization for stability
        self.input_norm = nn.LayerNorm(self.ligand_feature_nf)
        self.output_norm = nn.LayerNorm(self.ligand_feature_nf)
        
    def forward(self, t, z_xh, y_xh, ligand_mask=None, protein_mask=None):
        """Forward with stability improvements."""
        bs, n_ligand, _ = z_xh.shape
        
        # Apply input scaling to prevent explosion
        z_xh = z_xh * self.latent_scale
        y_xh = y_xh * self.latent_scale
        
        # Split coordinates and features
        z_x = z_xh[:, :, :self.n_dims]
        z_h = z_xh[:, :, self.n_dims:]
        
        y_x = y_xh[:, :, :self.n_dims]
        y_h = y_xh[:, :, self.n_dims:]
        
        # Normalize features before processing
        z_h = self.input_norm(z_h)
        
        # Call parent forward with normalized inputs
        z_xh_normalized = torch.cat([z_x, z_h], dim=-1)
        y_xh_normalized = torch.cat([y_x, y_h], dim=-1)
        
        # Get output from parent class
        vel_xh = super().forward(t, z_xh_normalized, y_xh_normalized, ligand_mask, protein_mask)
        
        # Split output
        vel_x = vel_xh[:, :, :self.n_dims]
        vel_h = vel_xh[:, :, self.n_dims:]
        
        # Normalize and clip feature velocities
        vel_h = self.output_norm(vel_h)
        vel_h = torch.clamp(vel_h, -1, 1)  # Prevent extreme values
        
        # Clip coordinate velocities
        vel_x = torch.clamp(vel_x, -1, 1)  # Reasonable molecular distances
        
        # Rescale back
        vel_xh = torch.cat([vel_x, vel_h], dim=-1) * 0.01
        
        return vel_xh


class StabilizedVAEWrapper(nn.Module):
    """Wrapper that adds stability to VAE encoding/decoding."""
    
    def __init__(self, vae, latent_scale=0.1):
        super().__init__()
        self.vae = vae
        self.latent_scale = latent_scale
        
        # Add normalization layers
        self.latent_norm = nn.LayerNorm(vae.latent_node_nf)
        
        # Copy important attributes from base VAE
        self.latent_node_nf = vae.latent_node_nf
        self.n_dims = vae.n_dims
        self.in_node_nf = vae.in_node_nf
        self.num_classes = vae.num_classes
        self.include_charges = vae.include_charges
        
    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Encode with normalization."""
        # Standard encode
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context)
        
        # Normalize and scale latents
        z_h_mu = self.latent_norm(z_h_mu) * self.latent_scale
        
        # Clamp coordinates to reasonable range
        z_x_mu = torch.clamp(z_x_mu, -10, 10)
        
        # Ensure small, stable sigma
        z_x_sigma = torch.ones_like(z_x_sigma) * 0.01
        z_h_sigma = torch.ones_like(z_h_sigma) * 0.01
        
        return z_x_mu, z_x_sigma, z_h_mu, z_h_sigma
    
    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Decode with denormalization."""
        # Split latents
        z_x = z_xh[:, :, :3]
        z_h = z_xh[:, :, 3:]
        
        # Denormalize features
        z_h = z_h / self.latent_scale
        
        # Reconstruct
        z_xh = torch.cat([z_x, z_h], dim=-1)
        return self.vae.decode(z_xh, node_mask, edge_mask, context)
    
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Delegate sampling to base VAE."""
        return self.vae.sample_normal(mu, sigma, node_mask, fix_noise)
    
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """Delegate noise sampling to base VAE."""
        return self.vae.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
    
    def compute_reconstruction_error(self, xh_rec, xh):
        """Delegate reconstruction error to base VAE."""
        return self.vae.compute_reconstruction_error(xh_rec, xh)


class StabilizedEnLatentDiffusion(nn.Module):
    """Wrapper for EnLatentDiffusion with stability improvements."""
    
    def __init__(self, latent_diffusion, gradient_clip=1.0):
        super().__init__()
        self.model = latent_diffusion
        self.gradient_clip = gradient_clip
        
        # Override gamma for gentler noise schedule
        self._modify_noise_schedule()
        
    def _modify_noise_schedule(self):
        """Adjust noise schedule to be more stable."""
        # Cap the gamma values to prevent extreme SNR
        if hasattr(self.model.gamma, 'gamma'):
            with torch.no_grad():
                self.model.gamma.gamma.data = torch.clamp(
                    self.model.gamma.gamma.data, 
                    min=-10.0, 
                    max=10.0
                )
    
    def forward(self, *args, **kwargs):
        """Forward with gradient clipping."""
        loss = self.model(*args, **kwargs)
        
        # Clip loss to prevent instability
        loss = torch.clamp(loss, max=1000.0)
        
        return loss
    
    def configure_optimizers(self, lr=1e-4):
        """Configure optimizer with appropriate settings."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=1000,
            eta_min=1e-6
        )
        
        return optimizer, scheduler
    
    def training_step(self, batch, optimizer):
        """Training step with gradient clipping."""
        loss = self.forward(**batch)
        loss = loss.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=self.gradient_clip
        )
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        return loss


def create_stable_model_pipeline(
    ligand_vae,
    protein_vae,
    latent_dim=16,
    hidden_dim=128,
    device='cuda',
    timesteps=1000
):
    """Create the full stabilized pipeline."""
    
    # Wrap VAEs with stability layers
    stable_ligand_vae = StabilizedVAEWrapper(ligand_vae, latent_scale=0.1)
    stable_protein_vae = StabilizedVAEWrapper(protein_vae, latent_scale=0.1)
    
    # Create stabilized dynamics
    dynamics = StabilizedProteinConditionedEGNNDynamics(
        ligand_feature_nf=latent_dim,
        protein_feature_nf=latent_dim,
        n_dims=3,
        hidden_nf=hidden_dim,
        device=device,
        n_layers=3,
        n_cross_layers=2,
        cross_distance_threshold=8.0,
        gradient_clip_val=1.0,
        latent_scale=0.1
    )
    
    # Wrap dynamics for latent diffusion
    class WrappedDynamics(nn.Module):
        def __init__(self, dynamics, protein_vae):
            super().__init__()
            self.dynamics = dynamics
            self.protein_vae = protein_vae
            self.protein_latent = None
            
        def _forward(self, t, xh, node_mask, edge_mask, context):
            if self.protein_latent is None:
                raise ValueError("Protein latent must be set")
            
            bs = xh.shape[0]
            if self.protein_latent.shape[0] != bs:
                if self.protein_latent.shape[0] > bs:
                    protein_latent_batch = self.protein_latent[:bs]
                else:
                    protein_latent_batch = self.protein_latent[:1].repeat(bs, 1, 1)
            else:
                protein_latent_batch = self.protein_latent
            
            protein_mask = torch.ones(bs, protein_latent_batch.shape[1], 1, device=xh.device)
            return self.dynamics(t, xh, protein_latent_batch, node_mask, protein_mask)
    
    wrapped_dynamics = WrappedDynamics(dynamics, stable_protein_vae)
    
    # Import and create latent diffusion
    from equivariant_diffusion.en_diffusion import EnLatentDiffusion
    
    # Use polynomial_2 schedule which is more stable
    base_diffusion = EnLatentDiffusion(
        dynamics=wrapped_dynamics,
        vae=stable_ligand_vae,
        in_node_nf=latent_dim,
        n_dims=3,
        timesteps=timesteps,
        noise_schedule='polynomial_2',  # More stable than cosine
        parametrization='eps',
        loss_type='l2',  # Simpler loss for initial training
        include_charges=False,
        trainable_ae=False,
        noise_precision=1e-4  # Higher precision for stability
    ).to(device)
    
    # Wrap with stability improvements
    stable_diffusion = StabilizedEnLatentDiffusion(
        base_diffusion, 
        gradient_clip=1.0
    )
    
    return stable_diffusion, stable_ligand_vae, stable_protein_vae


def test_stability_improvements():
    """Test the stabilized pipeline."""
    import matplotlib.pyplot as plt
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy VAEs for testing
    from test_models import create_vae
    ligand_vae = create_vae(25, 5, 16, 'ligand', device)
    protein_vae = create_vae(100, 5, 16, 'protein', device)
    
    # Create stabilized pipeline
    stable_model, stable_lig_vae, stable_prot_vae = create_stable_model_pipeline(
        ligand_vae, 
        protein_vae,
        device=device
    )
    
    print("Stabilized model created successfully!")
    
    # Test with dummy data
    from test_models import generate_protein_ligand_data
    data = generate_protein_ligand_data(batch_size=2, device=device)
    
    # Encode protein
    with torch.no_grad():
        z_x_prot, _, z_h_prot, _ = stable_prot_vae.encode(
            data['protein_x'], data['protein_h'],
            data['protein_mask'], data['protein_edge_mask']
        )
        protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)
        stable_model.model.dynamics.protein_latent = protein_latent
        
        print(f"Protein latent stats: mean={protein_latent.mean():.4f}, std={protein_latent.std():.4f}")
    
    # Test forward pass
    stable_model.model.train()
    loss = stable_model(
        data['ligand_x'], data['ligand_h'],
        data['ligand_mask'], data['ligand_edge_mask']
    )
    
    print(f"Stabilized loss: {loss.detach().cpu().numpy()}")
    print(f"Loss range: [{loss.min().item():.2f}, {loss.max().item():.2f}]")
    
    # Test sampling
    stable_model.model.eval()
    with torch.no_grad():
        x_gen, h_gen = stable_model.model.sample(
            n_samples=1,
            n_nodes=25,
            node_mask=data['ligand_mask'][:1],
            edge_mask=data['ligand_edge_mask'][:25*25],
            context=None
        )
        
        print(f"\nGenerated coordinates stats:")
        print(f"  Mean: {x_gen.mean():.4f}")
        print(f"  Std: {x_gen.std():.4f}")
        print(f"  Range: [{x_gen.min():.2f}, {x_gen.max():.2f}]")
    
    return stable_model


if __name__ == "__main__":
    test_stability_improvements()