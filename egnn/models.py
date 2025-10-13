import torch
import numpy as np
import torch.nn as nn
from egnn.egnn import EGNN, GCL, EquivariantBlock
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer between ligand and protein using EGNN-style message passing.
    Maintains E(n) equivariance through distance-based attention and normalized direction vectors.
    """
    def __init__(self, ligand_nf, protein_nf, hidden_nf, act_fn=nn.SiLU(), 
                 normalization_factor=100, aggregation_method='sum', distance_threshold=10.0):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.distance_threshold = distance_threshold
        
        # Attention weight computation (invariant)
        self.attention_mlp = nn.Sequential(
            nn.Linear(ligand_nf + protein_nf + 1, hidden_nf),  # +1 for distance
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid()
        )
        
        # Value transformation for features (invariant)
        self.value_mlp = nn.Sequential(
            nn.Linear(protein_nf + 1, hidden_nf),  # +1 for distance
            act_fn,
            nn.Linear(hidden_nf, ligand_nf)
        )
        
        # Coordinate influence weights (scalar output for equivariant update)
        self.coord_mlp = nn.Sequential(
            nn.Linear(ligand_nf + protein_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )
        
    def forward(self, h_ligand, x_ligand, h_protein, x_protein, 
                ligand_mask=None, protein_mask=None):
        """
        GPU-efficient batched forward pass.
        
        Args:
            h_ligand: [bs, n_ligand, ligand_nf] ligand node features
            x_ligand: [bs, n_ligand, 3] ligand coordinates
            h_protein: [bs, n_protein, protein_nf] protein node features
            x_protein: [bs, n_protein, 3] protein coordinates
            ligand_mask: [bs, n_ligand, 1] ligand node mask
            protein_mask: [bs, n_protein, 1] protein node mask
        
        Returns:
            h_cross: [bs, n_ligand, ligand_nf] cross-attention messages for features
            x_cross: [bs, n_ligand, 3] cross-attention messages for coordinates
        """
        bs, n_ligand, ligand_nf = h_ligand.shape
        _, n_protein, protein_nf = h_protein.shape
        device = h_ligand.device
        
        # Flatten batch dimensions for efficient processing
        h_lig_flat = h_ligand.view(bs * n_ligand, ligand_nf)
        x_lig_flat = x_ligand.view(bs * n_ligand, 3)
        h_prot_flat = h_protein.view(bs * n_protein, protein_nf)
        x_prot_flat = x_protein.view(bs * n_protein, 3)
        
        # Build batch-aware edge indices based on distance threshold
        edge_indices_list = []
        batch_lig_indices = []
        batch_prot_indices = []
        
        for b in range(bs):
            # Compute pairwise distances for this batch
            x_lig_b = x_ligand[b]  # [n_ligand, 3]
            x_prot_b = x_protein[b]  # [n_protein, 3]
            dist_matrix = torch.cdist(x_lig_b, x_prot_b)  # [n_ligand, n_protein]
            
            # Find pairs within threshold
            lig_idx, prot_idx = torch.where(dist_matrix < self.distance_threshold)
            
            if lig_idx.numel() > 0:
                # Add batch offsets to create global indices
                lig_idx_global = lig_idx + b * n_ligand
                prot_idx_global = prot_idx + b * n_protein
                
                batch_lig_indices.append(lig_idx_global)
                batch_prot_indices.append(prot_idx_global)
        
        # Handle case where no edges exist
        if len(batch_lig_indices) == 0:
            return torch.zeros_like(h_ligand), torch.zeros_like(x_ligand)
        
        # Concatenate all edges across batches
        all_lig_indices = torch.cat(batch_lig_indices)
        all_prot_indices = torch.cat(batch_prot_indices)
        
        # Get features and positions for all interacting pairs at once
        h_lig_pairs = h_lig_flat[all_lig_indices]  # [n_edges, ligand_nf]
        x_lig_pairs = x_lig_flat[all_lig_indices]  # [n_edges, 3]
        h_prot_pairs = h_prot_flat[all_prot_indices]  # [n_edges, protein_nf]
        x_prot_pairs = x_prot_flat[all_prot_indices]  # [n_edges, 3]
        
        # Compute relative positions and distances (vectorized)
        rel_pos = x_lig_pairs - x_prot_pairs  # [n_edges, 3]
        dist_squared = torch.sum(rel_pos ** 2, dim=1, keepdim=True)  # [n_edges, 1]
        dist = torch.sqrt(dist_squared + 1e-8)
        direction = rel_pos / (dist + 1e-8)  # [n_edges, 3]
        
        # Compute attention weights (all edges at once)
        attention_input = torch.cat([h_lig_pairs, h_prot_pairs, dist_squared], dim=1)
        attention_weights = self.attention_mlp(attention_input)  # [n_edges, 1]
        
        # Apply protein mask if provided
        if protein_mask is not None:
            prot_mask_flat = protein_mask.view(bs * n_protein, 1)
            prot_mask_pairs = prot_mask_flat[all_prot_indices]
            attention_weights = attention_weights * prot_mask_pairs
        
        # Compute value messages for features
        value_input = torch.cat([h_prot_pairs, dist_squared], dim=1)
        value_messages = self.value_mlp(value_input)  # [n_edges, ligand_nf]
        weighted_values = attention_weights * value_messages
        
        # Initialize output tensors
        h_cross_flat = torch.zeros(bs * n_ligand, ligand_nf, device=device)
        x_cross_flat = torch.zeros(bs * n_ligand, 3, device=device)
        
        # Aggregate feature messages to ligand nodes
        h_cross_flat.scatter_add_(0, all_lig_indices.unsqueeze(1).expand(-1, ligand_nf), 
                                  weighted_values)
        
        # Compute coordinate updates
        coord_input = torch.cat([h_lig_pairs, h_prot_pairs, dist_squared], dim=1)
        coord_weights = self.coord_mlp(coord_input)  # [n_edges, 1]
        coord_weights = torch.tanh(coord_weights)
        
        if protein_mask is not None:
            coord_weights = coord_weights * prot_mask_pairs
        
        coord_messages = direction * coord_weights  # [n_edges, 3]
        x_cross_flat.scatter_add_(0, all_lig_indices.unsqueeze(1).expand(-1, 3), 
                                  coord_messages)
        
        # Normalize by aggregation method
        if self.aggregation_method == 'sum':
            h_cross_flat = h_cross_flat / self.normalization_factor
            x_cross_flat = x_cross_flat / self.normalization_factor
        elif self.aggregation_method == 'mean':
            # Count number of edges per ligand node
            counts = torch.zeros(bs * n_ligand, 1, device=device)
            counts.scatter_add_(0, all_lig_indices.unsqueeze(1), 
                               torch.ones(all_lig_indices.shape[0], 1, device=device))
            h_cross_flat = h_cross_flat / (counts + 1e-8)
            x_cross_flat = x_cross_flat / (counts + 1e-8)
        
        # Reshape back to batch format
        h_cross = h_cross_flat.view(bs, n_ligand, ligand_nf)
        x_cross = x_cross_flat.view(bs, n_ligand, 3)
        
        # Apply ligand mask if provided
        if ligand_mask is not None:
            h_cross = h_cross * ligand_mask
            x_cross = x_cross * ligand_mask
        
        return h_cross, x_cross


class ProteinConditionedEGNNDynamics(nn.Module):
    """
    EGNN Dynamics network with cross-attention conditioning on protein targets.
    Used for denoising ligand latent representations conditioned on protein structure.
    """
    def __init__(self, 
                 ligand_feature_nf,      # Dimension of ligand latent features z_h
                 protein_feature_nf,      # Dimension of protein latent features y_h
                 n_dims=3,               # Spatial dimensions (3 for 3D coordinates)
                 hidden_nf=64,           
                 device='cpu',
                 act_fn=torch.nn.SiLU(), 
                 n_layers=4,             # Number of EGNN layers
                 n_cross_layers=2,       # Number of cross-attention layers
                 attention=False,        # Self-attention in EGNN layers
                 tanh=False, 
                 norm_constant=0,
                 inv_sublayers=2, 
                 sin_embedding=False, 
                 normalization_factor=100, 
                 aggregation_method='sum',
                 cross_distance_threshold=10.0):
        super().__init__()
        
        self.ligand_feature_nf = ligand_feature_nf
        self.protein_feature_nf = protein_feature_nf
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_cross_layers = n_cross_layers
        
        # Time embedding projection
        self.time_emb_dim = hidden_nf
        self.time_emb_mlp = nn.Sequential(
            nn.Linear(1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.time_emb_dim)
        )
        
        # Initial projection for ligand features (includes time embedding)
        self.ligand_input_proj = nn.Linear(ligand_feature_nf + self.time_emb_dim, hidden_nf)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                ligand_nf=hidden_nf,
                protein_nf=protein_feature_nf - n_dims if protein_feature_nf > n_dims else protein_feature_nf,  # Subtract coordinate dims
                hidden_nf=hidden_nf,
                act_fn=act_fn,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                distance_threshold=cross_distance_threshold
            ) for _ in range(n_cross_layers)
        ])
        
        # EGNN backbone for ligand self-interactions
        self.egnn = EGNN(
            in_node_nf=hidden_nf,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            out_node_nf=ligand_feature_nf  # Output original feature dimension
        )
        
        # Layer normalization for stability
        self.h_layer_norm = nn.LayerNorm(hidden_nf)
        
        self._edges_dict = {}
        self.to(device)
    
    def forward(self, t, z_xh, y_xh, ligand_mask=None, protein_mask=None):
        """
        Forward pass for denoising ligand latents conditioned on protein.
        
        Args:
            t: Diffusion timestep [bs] or scalar
            z_xh: Ligand latent [bs, n_ligand, 3 + ligand_feature_nf]
                 First 3 dims are coordinates z_x, rest are features z_h
            y_xh: Protein latent [bs, n_protein, 3 + protein_feature_nf]
                 First 3 dims are coordinates y_x, rest are features y_h
            ligand_mask: [bs, n_ligand, 1] mask for valid ligand nodes
            protein_mask: [bs, n_protein, 1] mask for valid protein nodes
        
        Returns:
            vel_xh: Predicted noise/velocity [bs, n_ligand, 3 + ligand_feature_nf]
        """
        bs, n_ligand, _ = z_xh.shape
        _, n_protein, _ = y_xh.shape
        
        # Split coordinates and features
        z_x = z_xh[:, :, :self.n_dims]  # [bs, n_ligand, 3]
        z_h = z_xh[:, :, self.n_dims:]   # [bs, n_ligand, ligand_feature_nf]
        
        y_x = y_xh[:, :, :self.n_dims]  # [bs, n_protein, 3]
        y_h = y_xh[:, :, self.n_dims:]   # [bs, n_protein, protein_feature_nf-3]
        # Note: y_h is now 32 dimensions, not 35
        
        # Generate time embedding
        if np.prod(t.size()) == 1:
            t_emb = torch.full((bs, 1), t.item(), device=self.device)
        else:
            t_emb = t.view(bs, 1)
        
        t_emb = self.time_emb_mlp(t_emb)  # [bs, hidden_nf]
        t_emb = t_emb.unsqueeze(1).expand(bs, n_ligand, -1)  # [bs, n_ligand, hidden_nf]
        
        # Concatenate time embedding with ligand features
        h_with_time = torch.cat([z_h, t_emb], dim=-1)  # [bs, n_ligand, ligand_feature_nf + hidden_nf]
        h = self.ligand_input_proj(h_with_time)  # [bs, n_ligand, hidden_nf]
        
        # Apply layer norm
        h = self.h_layer_norm(h)
        
        # Apply cross-attention layers
        x = z_x.clone()
        for cross_attn in self.cross_attention_layers:
            h_cross, x_cross = cross_attn(h, x, y_h, y_x, ligand_mask, protein_mask)
            h = h + h_cross  # Residual connection
            x = x + x_cross * 0.1  # Scaled residual for coordinates
        
        # Prepare for EGNN (flatten batch dimension)
        edges = self.get_adj_matrix(n_ligand, bs, self.device)
        edges = [e.to(self.device) for e in edges]
        
        h_flat = h.view(bs * n_ligand, -1)
        x_flat = x.view(bs * n_ligand, -1)
        
        if ligand_mask is not None:
            node_mask_flat = ligand_mask.view(bs * n_ligand, 1)
            edge_mask_flat = self.get_edge_mask(ligand_mask, n_ligand, bs)
        else:
            node_mask_flat = None
            edge_mask_flat = None
        
        # Apply EGNN for ligand self-interactions
        h_final, x_final = self.egnn(h_flat, x_flat, edges, 
                                     node_mask=node_mask_flat, 
                                     edge_mask=edge_mask_flat)
        
        # Compute velocity (coordinate change)
        vel_x = (x_final - x_flat).view(bs, n_ligand, self.n_dims)
        vel_h = h_final.view(bs, n_ligand, -1)
        
        # Apply mask to velocities BEFORE mean removal
        if ligand_mask is not None:
            vel_x = vel_x * ligand_mask
            vel_h = vel_h * ligand_mask
        
        # Remove mean from velocities for translation equivariance
        if ligand_mask is None:
            vel_x = remove_mean(vel_x)
        else:
            vel_x = remove_mean_with_mask(vel_x, ligand_mask)
        
        # Combine coordinate and feature velocities
        vel_xh = torch.cat([vel_x, vel_h], dim=-1)
        
        # Check for NaNs
        if torch.any(torch.isnan(vel_xh)):
            print('Warning: detected NaN, resetting output to zero.')
            vel_xh = torch.zeros_like(vel_xh)
        
        return vel_xh
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        """Generate fully connected adjacency matrix for batch."""
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
    
    def get_edge_mask(self, node_mask, n_nodes, batch_size):
        """Generate edge mask from node mask."""
        # Create edge mask for each batch element
        edge_mask_list = []
        for b in range(batch_size):
            # Get mask for this batch [n_nodes, 1]
            node_mask_b = node_mask[b]
            # Create edge mask [n_nodes, n_nodes] by outer product
            edge_mask_b = node_mask_b @ node_mask_b.T  # [n_nodes, n_nodes]
            edge_mask_b = edge_mask_b.view(-1, 1)  # [n_nodes*n_nodes, 1]
            edge_mask_list.append(edge_mask_b)
        
        # Concatenate all batch edge masks
        edge_mask = torch.cat(edge_mask_list, dim=0)  # [bs*n_nodes*n_nodes, 1]
        return edge_mask

class EGNNDynamics(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        # Slice off context size if conditioned on context
        h_final = h_final[:, :-self.context_node_nf] if context is not None else h_final

        # Slice off last dimension which represents time if conditioned on time (should be)
        h_final = h_final[:, :-1] if self.condition_time else h_final

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        vel = remove_mean(vel) if node_mask is None else remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
     
        return vel if h_dims == 0 else torch.cat([vel, h_final.view(bs, n_nodes, -1)], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNNEncoder(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)
        num_classes = in_node_nf - include_charges
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, out_node_nf=hidden_nf, 
            in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1))

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}

        self.out_node_nf = out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):      
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            print(f"(Encoder) We're conditioning, awesome!")
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = x_final * node_mask

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = self.final_mlp(h_final)
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1:1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])

        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one.')
            vel_std = torch.ones_like(vel_std)
        
        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one.')
            h_std = torch.ones_like(h_std)
        
        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization

        return vel_mean, vel_std, h_mean, h_std
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNNDecoder(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges

        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, out_node_nf=out_node_nf, 
            in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        h = torch.ones(bs*n_nodes, 1).to(self.device) if h_dims == 0 else xh[:, self.n_dims:].clone()

        if context is not None:
            print(f"(Decoder) We're conditioning, awesome!")
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = x_final * node_mask  # This masking operation is redundant but just in case

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        vel = remove_mean(vel) if node_mask is None else remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        
        h_final = h_final * node_mask if node_mask is not None else h_final
        
        h_final = h_final.view(bs, n_nodes, -1)

        return vel, h_final
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)