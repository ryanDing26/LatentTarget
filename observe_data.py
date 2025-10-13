import lmdb
import pickle
import torch
from equivariant_diffusion.en_diffusion import *
from egnn.models import *
from configs.datasets_config import get_dataset_info

def main():
    split_file = "./cd2020/crossdocked_pocket10_pose_split.pt"

    splits = torch.load(split_file)
    # keys: [train, val, test]
    # train as 99.99k points, val 0 (diffusion doesn't use), test 100
    # these are indices for rows in the lmdb

    train_rows = splits['train']
    test_rows = splits['test']
    train_rows = set(splits['train'])
    test_rows = set(splits['test'])
    all_rows_to_keep = train_rows.union(test_rows)
    lmdb_path = "./cd2020/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"

    extracted_data = []
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

    with env.begin() as txn:
        cursor = txn.cursor()

        for idx, (key, value) in enumerate(cursor):
            if idx in all_rows_to_keep:
                print(f"Row idx: {idx}, Key: {key[:50]}... Value type: {type(value)}")
                
                # Optionally deserialize a single sample to inspect
                sample = pickle.loads(value)
                print(sample)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # include_charges = False
    # dataset_info = get_dataset_info('cd2020')
    # in_node_nf = len(dataset_info['ligand_decoder']) + int(include_charges)
    # context_node_nf = 0
    # hidden_nf = 192
    # sin_embedding = True
    # attention = True
    # inv_sublayers = 1
    
    # tanh = True
    # n_dims = 3
    # out_node_nf = 2
    # aggregation_method = 'sum' # can also do mean
    # normalize_factors = [1, 4, 10] # also [1, 8, 1]
    # normalization_factor = 1
    # norm_constant = 1
    # encoder = EGNNEncoder(
    #     in_node_nf=in_node_nf, context_node_nf=0, out_node_nf=2,
    #     n_dims=3, device=device, hidden_nf=192,
    #     act_fn=torch.nn.SiLU(), n_layers=9,
    #     attention=True, tanh=True, norm_constant=1,
    #     inv_sublayers=1, sin_embedding=False,
    #     normalization_factor=1, aggregation_method='sum',
    #     include_charges=False
    #     )
    
    # decoder = EGNNDecoder(
    #     in_node_nf=2, context_node_nf=0, out_node_nf=in_node_nf,
    #     n_dims=3, device=device, hidden_nf=192,
    #     act_fn=torch.nn.SiLU(), n_layers=9,
    #     attention=True, tanh=True, norm_constant=1,
    #     inv_sublayers=1, sin_embedding=False,
    #     normalization_factor=1, aggregation_method='sum',
    #     include_charges=False
    #     )

    # vae = EnHierarchicalVAE(
    #     encoder=encoder,
    #     decoder=decoder,
    #     in_node_nf=in_node_nf,
    #     n_dims=3,
    #     latent_node_nf=2,
    #     kl_weight=0.01, #play with this
    #     norm_values=[1, 4, 10],
    #     include_charges=False
    # )
    # Hyperparameters
    # B = 4          # batch size
    # N = 5          # number of nodes per graph
    # in_node_nf = 5 # categorical features + charges
    # latent_node_nf = 2
    # include_charges = 0
    # encoder = EGNNEncoder(in_node_nf=in_node_nf, context_node_nf=0, out_node_nf=latent_node_nf, n_dims=3)
    # decoder = EGNNDecoder(in_node_nf=latent_node_nf, context_node_nf=0, out_node_nf=in_node_nf, n_dims=3)
    # denoiser = EGNNDynamics(in_node_nf=latent_node_nf, context_node_nf=0, n_dims=3)
    # # Coordinates (x)
    # x = torch.rand(B, N, 3)

    # # Centering operation
    # x -= x.mean(dim=1, keepdim=True)

    # # Categorical features (h['categorical'])
    # h_categorical = torch.rand(B, N, in_node_nf)

    # h = {
    #     'categorical': h_categorical,
    #     'integer': torch.zeros(B, N, 0)
    # }

    # # Node mask (all nodes exist)
    # node_mask = torch.ones(B, N, 1)

    # # Edge mask (fully connected)
    # edge_mask = torch.ones(B, N*N, 1)

    # context = None
    # # Instantiate your VAE
    # vae = EnHierarchicalVAE(
    #     encoder=encoder,
    #     decoder=decoder,
    #     in_node_nf=in_node_nf,
    #     n_dims=3,
    #     latent_node_nf=latent_node_nf,
    #     kl_weight=0.01,
    #     include_charges=include_charges
    # )
    # print("x:", x)
    # # Encode
    # z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = vae.encode(x, h, node_mask=node_mask, edge_mask=edge_mask, context=context)
    # print("z_x:", z_x_mu)
    # print("h:", h)
    # print("z_h", z_h_mu)
    # print("h['categorical'].shape:", h["categorical"].shape)
    # print("h['integer'].shape:", h["integer"].shape)
    # print("z_h.shape:", z_h_mu.shape)
    # # print("z_x_mu.shape:", z_x_mu.shape)  # [B, N, latent_node_nf]
    # # print("z_h_mu.shape:", z_h_mu.shape)  # [B, N, latent_node_nf]
    # print("z_h_sigma:", z_h_sigma, z_h_sigma.shape)
    # print("z_x_sigma:", z_x_sigma, z_x_sigma.shape)
    # # Decode
    # x_recon, h_recon = vae.decode(torch.cat([z_x_mu, z_h_mu], dim=2), node_mask=node_mask, edge_mask=edge_mask, context=context)

    # print("x_recon.shape:", x_recon.shape)            # [B, N, 3]
    # print("h_recon['categorical'].shape:", h_recon['categorical'])  # [B, N, in_node_nf - include_charges]


if __name__ == '__main__':
    main()