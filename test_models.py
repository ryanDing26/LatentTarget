import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import models (adjust paths as needed)
from egnn.models import ProteinConditionedEGNNDynamics, EGNNEncoder, EGNNDecoder
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE, EnLatentDiffusion
import equivariant_diffusion.utils as diffusion_utils

def create_edge_mask_from_node_mask(node_mask, n_atoms):
    """Create edge mask from node mask."""
    batch_size = node_mask.shape[0]
    edge_mask_list = []
    for b in range(batch_size):
        node_mask_b = node_mask[b]
        edge_mask_b = node_mask_b @ node_mask_b.T
        edge_mask_b = edge_mask_b.view(-1, 1)
        edge_mask_list.append(edge_mask_b)
    return torch.cat(edge_mask_list, dim=0)

def generate_protein_ligand_data(batch_size=2, n_ligand_atoms=25, n_protein_atoms=100,
                                 n_atom_types=5, include_charges=True, device='cpu'):
    """Generate fake protein-ligand complex data."""
    
    # Generate protein at origin (reference frame)
    protein_x = torch.randn(batch_size, n_protein_atoms, 3, device=device) * 5.0
    protein_mask = torch.ones(batch_size, n_protein_atoms, 1, device=device)
    
    # Apply mask and center protein
    protein_x = protein_x * protein_mask
    protein_x = diffusion_utils.remove_mean_with_mask(protein_x, protein_mask)
    
    # Generate ligand positioned relative to protein
    binding_site_offset = torch.randn(batch_size, 1, 3, device=device) * 3.0
    ligand_x = torch.randn(batch_size, n_ligand_atoms, 3, device=device) * 2.0
    ligand_x = ligand_x + binding_site_offset
    ligand_mask = torch.ones(batch_size, n_ligand_atoms, 1, device=device)
    ligand_x = ligand_x * ligand_mask
    
    # Generate features
    def make_features(n_atoms, batch_size, n_atom_types, include_charges, mask, device):
        atom_types = torch.randint(0, n_atom_types, (batch_size * n_atoms,), device=device)
        h_cat = F.one_hot(atom_types, n_atom_types).float()
        h_cat = h_cat.view(batch_size, n_atoms, n_atom_types) * mask
        
        if include_charges:
            h_int = torch.randint(-1, 2, (batch_size, n_atoms, 1), device=device).float() * mask
        else:
            h_int = torch.zeros(batch_size, n_atoms, 0, device=device)
        
        return {'categorical': h_cat, 'integer': h_int}
    
    protein_h = make_features(n_protein_atoms, batch_size, n_atom_types, include_charges, protein_mask, device)
    ligand_h = make_features(n_ligand_atoms, batch_size, n_atom_types, include_charges, ligand_mask, device)
    
    # Create edge masks
    protein_edge_mask = create_edge_mask_from_node_mask(protein_mask, n_protein_atoms)
    ligand_edge_mask = create_edge_mask_from_node_mask(ligand_mask, n_ligand_atoms)
    
    return {
        'protein_x': protein_x, 'protein_h': protein_h,
        'protein_mask': protein_mask, 'protein_edge_mask': protein_edge_mask,
        'ligand_x': ligand_x, 'ligand_h': ligand_h,
        'ligand_mask': ligand_mask, 'ligand_edge_mask': ligand_edge_mask
    }


def create_vae(n_atoms, n_atom_types, latent_dim, structure, device):
    """Create a VAE for encoding/decoding molecules."""
    in_node_nf = n_atom_types + 1  # +1 for charges
    
    encoder = EGNNEncoder(
        in_node_nf=in_node_nf,
        context_node_nf=0,
        out_node_nf=latent_dim,
        n_dims=3,
        hidden_nf=64,
        n_layers=2,
        device=device,
        include_charges=True
    )
    
    decoder = EGNNDecoder(
        in_node_nf=latent_dim,
        context_node_nf=0,
        out_node_nf=in_node_nf,
        n_dims=3,
        hidden_nf=64,
        n_layers=2,
        device=device,
        include_charges=True
    )
    
    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=in_node_nf,
        n_dims=3,
        latent_node_nf=latent_dim,
        kl_weight=0.1,
        structure=structure,
        include_charges=True
    ).to(device)
    
    return vae


def test_latent_diffusion_integration():
    """Test the full EnLatentDiffusion pipeline with protein conditioning."""
    
    print("=" * 70)
    print("TESTING EnLatentDiffusion WITH PROTEIN CONDITIONING")
    print("=" * 70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Model parameters
    n_ligand_atoms = 25
    n_protein_atoms = 100
    n_atom_types = 5
    latent_dim = 16
    batch_size = 2
    
    print("-" * 70)
    print("STEP 1: Creating VAEs for Ligand and Protein")
    print("-" * 70)
    
    # Create VAEs
    ligand_vae = create_vae(n_ligand_atoms, n_atom_types, latent_dim, 'ligand', device)
    protein_vae = create_vae(n_protein_atoms, n_atom_types, latent_dim, 'protein', device)
    
    print(f"Created ligand VAE with {sum(p.numel() for p in ligand_vae.parameters())} parameters")
    print(f"Created protein VAE with {sum(p.numel() for p in protein_vae.parameters())} parameters")
    
    print("\n" + "-" * 70)
    print("STEP 2: Creating Protein-Conditioned Denoiser")
    print("-" * 70)
    
    # Create the protein-conditioned dynamics model
    dynamics = ProteinConditionedEGNNDynamics(
        ligand_feature_nf=latent_dim,
        protein_feature_nf=latent_dim,
        n_dims=3,
        hidden_nf=128,
        device=device,
        n_layers=3,
        n_cross_layers=2,
        cross_distance_threshold=8.0
    )
    
    print(f"Created denoiser with {sum(p.numel() for p in dynamics.parameters())} parameters")
    
    # Wrap dynamics for compatibility with EnLatentDiffusion
    class WrappedDynamics(nn.Module):
        def __init__(self, dynamics, protein_vae):
            super().__init__()
            self.dynamics = dynamics
            self.protein_vae = protein_vae
            self.protein_latent = None  # Will be set before diffusion
            
        def _forward(self, t, xh, node_mask, edge_mask, context):
            # xh contains ligand latent: [bs, n_ligand, 3 + latent_dim]
            # self.protein_latent contains protein latent
            
            if self.protein_latent is None:
                raise ValueError("Protein latent must be set before forward pass")
            
            # Match protein latent batch size to current batch
            bs = xh.shape[0]
            if self.protein_latent.shape[0] != bs:
                # Repeat or slice protein latent to match batch size
                if self.protein_latent.shape[0] > bs:
                    protein_latent_batch = self.protein_latent[:bs]
                else:
                    # Repeat the first protein for all samples
                    protein_latent_batch = self.protein_latent[:1].repeat(bs, 1, 1)
            else:
                protein_latent_batch = self.protein_latent
            
            # Prepare protein mask matching the batch size
            protein_mask = torch.ones(bs, protein_latent_batch.shape[1], 1, device=xh.device)
            
            # Call the protein-conditioned dynamics
            return self.dynamics(t, xh, protein_latent_batch, node_mask, protein_mask)
    
    wrapped_dynamics = WrappedDynamics(dynamics, protein_vae)
    
    print("\n" + "-" * 70)
    print("STEP 3: Creating EnLatentDiffusion Model")
    print("-" * 70)
    
    # Create the latent diffusion model
    latent_diffusion = EnLatentDiffusion(
        dynamics=wrapped_dynamics,
        vae=ligand_vae,
        in_node_nf=latent_dim,  # Features in latent space
        n_dims=3,
        timesteps=1000,  # Standard diffusion timesteps
        noise_schedule='cosine',
        parametrization='eps',
        loss_type='vlb',
        include_charges=False,  # Working in latent space
        trainable_ae=False  # Keep VAE fixed
    ).to(device)
    
    print(f"Created latent diffusion model with T={latent_diffusion.T} timesteps")
    print(f"Noise schedule: cosine")
    
    print("\n" + "-" * 70)
    print("STEP 4: Testing Forward Pass (Training Mode)")
    print("-" * 70)
    
    # Generate test data
    data = generate_protein_ligand_data(
        batch_size=batch_size,
        n_ligand_atoms=n_ligand_atoms,
        n_protein_atoms=n_protein_atoms,
        device=device
    )
    
    # Encode protein and set as conditioning
    with torch.no_grad():
        z_x_prot, _, z_h_prot, _ = protein_vae.encode(
            data['protein_x'], data['protein_h'],
            data['protein_mask'], data['protein_edge_mask']
        )
        protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)
        wrapped_dynamics.protein_latent = protein_latent
    
    print(f"Encoded protein to latent: {protein_latent.shape}")
    
    # Test forward pass (computes loss)
    latent_diffusion.train()
    loss = latent_diffusion(
        data['ligand_x'], data['ligand_h'],
        data['ligand_mask'], data['ligand_edge_mask']
    )
    
    print(f"Loss shape: {loss.shape}")
    print(f"Loss values: {loss.detach().cpu().numpy()}")
    
    print("\n" + "-" * 70)
    print("STEP 5: Testing Sampling (Generation)")
    print("-" * 70)
    
    # Test sampling
    latent_diffusion.eval()
    with torch.no_grad():
        # Generate new ligands conditioned on protein
        n_samples = 2
        x_generated, h_generated = latent_diffusion.sample(
            n_samples=n_samples,
            n_nodes=n_ligand_atoms,
            node_mask=data['ligand_mask'][:n_samples],
            edge_mask=data['ligand_edge_mask'][:n_samples * n_ligand_atoms * n_ligand_atoms],
            context=None
        )
    
    print(f"Generated ligand positions: {x_generated.shape}")
    print(f"Generated ligand features: {h_generated['categorical'].shape}")
    print(f"Position statistics: mean={x_generated.mean():.4f}, std={x_generated.std():.4f}")
    
    print("\n" + "-" * 70)
    print("STEP 6: Testing Different Noise Schedules")
    print("-" * 70)
    
    noise_schedules = ['cosine', 'polynomial_2', 'polynomial_3']
    
    for schedule in noise_schedules:
        try:
            # Create model with different schedule
            test_model = EnLatentDiffusion(
                dynamics=wrapped_dynamics,
                vae=ligand_vae,
                in_node_nf=latent_dim,
                n_dims=3,
                timesteps=50,
                noise_schedule=schedule,
                parametrization='eps',
                loss_type='vlb',
                include_charges=False,
                trainable_ae=False
            ).to(device)
            
            # Get gamma values at different timesteps
            t_values = torch.linspace(0, 1, 5, device=device).unsqueeze(1)
            gamma_values = test_model.gamma(t_values)
            
            print(f"{schedule:15} - gamma range: [{gamma_values[0].item():.3f}, {gamma_values[-1].item():.3f}]")
            
        except Exception as e:
            print(f"{schedule:15} - Error: {e}")
    
    print("\n" + "-" * 70)
    print("STEP 7: Testing Chain Sampling (Visualization)")
    print("-" * 70)
    
    # Test chain sampling for visualization
    keep_frames = 10
    with torch.no_grad():
        chain = latent_diffusion.sample_chain(
            n_samples=1,
            n_nodes=n_ligand_atoms,
            node_mask=data['ligand_mask'][:1],
            edge_mask=data['ligand_edge_mask'][:n_ligand_atoms * n_ligand_atoms],
            context=None,
            keep_frames=keep_frames
        )
    
    print(f"Chain shape: {chain.shape} (frames × atoms × features)")
    
    # The chain contains decoded molecules (not latent space)
    # Extract coordinates for visualization
    chain_coords = chain[:, :, :3].cpu().numpy()
    
    # Reshape to get individual frames
    chain_frames = chain_coords.reshape(keep_frames, n_ligand_atoms, 3)
    
    # Create visualization with better scaling
    fig = plt.figure(figsize=(15, 8))
    
    # Plot 2D projections (X-Y plane)
    for i in range(keep_frames):
        ax = plt.subplot(2, 5, i + 1)
        frame = chain_frames[i]
        
        # Calculate bounds for this frame
        x_coords = frame[:, 0]
        y_coords = frame[:, 1]
        
        # Plot atoms
        scatter = ax.scatter(x_coords, y_coords, alpha=0.6, s=50, c=np.arange(n_ligand_atoms), cmap='viridis')
        
        # Set title with timestep info
        timestep = (keep_frames - 1 - i) * (latent_diffusion.T // (keep_frames - 1))
        ax.set_title(f"t = {timestep}", fontsize=10)
        
        # Calculate bounds with some padding
        if x_coords.size > 0:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            x_range = max(x_max - x_min, 1.0)
            y_range = max(y_max - y_min, 1.0)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            ax.set_xlim(x_center - x_range * 0.7, x_center + x_range * 0.7)
            ax.set_ylim(y_center - y_range * 0.7, y_center + y_range * 0.7)
        else:
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Denoising Trajectory: From Noise (t=1000) to Generated Ligand (t=0)", fontsize=12)
    plt.tight_layout()
    plt.savefig("latent_diffusion_trajectory.png", dpi=150)
    print("Saved denoising trajectory to 'latent_diffusion_trajectory.png'")
    
    # Print statistics about the trajectory
    print("\nTrajectory statistics:")
    for i in [0, keep_frames//2, keep_frames-1]:
        frame = chain_frames[i]
        timestep = (keep_frames - 1 - i) * (latent_diffusion.T // (keep_frames - 1))
        print(f"  t={timestep:4d}: mean={frame.mean():.3f}, std={frame.std():.3f}, "
              f"range=[{frame.min():.2f}, {frame.max():.2f}]")
    
    print("\n" + "=" * 70)
    print("✓ EnLatentDiffusion INTEGRATION TEST COMPLETED!")
    print("=" * 70)
    
    return latent_diffusion, ligand_vae, protein_vae


if __name__ == "__main__":
    # Run the test
    latent_diffusion, ligand_vae, protein_vae = test_latent_diffusion_integration()
    
    print("\nKey Insights:")
    print("1. EnLatentDiffusion performs diffusion in VAE latent space")
    print("2. Protein conditioning happens through cross-attention in latent space")
    print("3. Different noise schedules affect the diffusion process")
    print("4. The full pipeline: Encode → Diffuse in latent → Decode")