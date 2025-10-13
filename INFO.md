# Atom Feature List
- `protein_element (use)`: feature of protein; shaped tensor([6, 7, 8...])
- `protein_molecule_name (don't use)`: literally just 'pocket'
- `protein_pos (use)`: protein coordinates; shaped tensor([[x_1, y_1, z_1], [x_2, y_2, z_2]])
- `protein_is_backbone (maybe)`: T/F value of whether or not an atom is a backbone; used in TargetDiff as protein feature
- `protein_atom_name (don't use)`: specific atom name; shaped ['N', 'CA',...]; too high cardinality for encoding proteins, just use element number
- `protein_atom_to_aa_type (kept in TargetDiff maybe)`: associ
- `ligand_smiles (use)`: SMILES string of a ligand; str; keep to calculate avg. QED and SA later on easily
- `ligand_element (use)`: atomic number of a ligand atom; shaped tensor([6, 7, 8...])
- `ligand_pos (use)`: ligand coordinates; shaped tensor([[x_1, y_1, z_1], [x_2, y_2, z_2]])
- `ligand_bond_type `: bond type of ligands; shaped tensor([4, 1, 2, ...])
- `ligand_center_of_mass (maybe)`: CoM of ligands; shaped tensor([x, y, z]); we center based on protein coordinates for our case, probably don't need
- `ligand_atom_feature (idk)`: I have no idea; shaped tensor([[0, 1, 0, ...], [1, 1, ...]])
- `ligand_hybridization (maybe keep; used in TargetDiff)`: ligand feature; shaped like ['SP3', 'SP2'...]
- `ligand_nbh_list (maybe)`: ligand neighbor list; shaped like {0: [1, 6, 9], 1: [0, ...], ...} 
- `protein_filename (keep, might need later for Vina calc)`: filename of stored protein PDB file from TargetDiff preprocess
- `ligand_filename (keep, might need later for Vina calc)`: filename of stored ligand SDF file from TargetDiff 

## Training Process/Parameters
- `batch_size`: 4 (1 step per 4 complexes evaluated)
- `optimizer`: Adam with init lr 1e-3, betas=0.95, 0.999, clip_gradient_norm=8
-  atom type loss multiplied by 100 to balance between scale of coordinate loss
- evaluation done every 2000 steps aka 8000 training samples
- during training, the protein had gaussian noise N(0, 0.1) added to it (we won't be doing this, using VAE for protein latents add stochasticity already)
- exponentially decay lr by factor of 0.6 to a minimum 1e-6; only occurs if no validation loss improvement in 10 consecutive evals (20k steps or 80k samples)
- ideal goal: converge within 24 hours on 1 3090 GPU; training should be first done on 4x 3090 GPUs (which I have access to, need to refactor code to run on 4 GPUs though)

            