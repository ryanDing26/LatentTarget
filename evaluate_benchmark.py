"""
evaluate_model.py - Comprehensive evaluation with proper coordinate handling
Handles CoM transformations for accurate Vina docking scores
"""

import torch
import numpy as np
import os
import pickle
import lmdb
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import subprocess
from multiprocessing import Pool
import tempfile
import shutil

# Import your model classes
from train_all import CrossDockedDataset, CrossDockedTrainer
from test_stability import StabilizedVAEWrapper
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE

# For SA score
try:
    from sascorer import calculateScore
except ImportError:
    print("Installing SA scorer...")
    os.system("pip install git+https://github.com/rdkit/rdkit.git")

class CoordinateTransformer:
    """Handles coordinate transformations between centered and original frames."""
    
    @staticmethod
    def get_center_of_mass(coords, mask=None):
        """Calculate center of mass of coordinates."""
        if mask is not None:
            valid_coords = coords[mask.squeeze() > 0]
        else:
            valid_coords = coords
        return valid_coords.mean(axis=0)
    
    @staticmethod
    def center_coords(coords, center=None, mask=None):
        """Center coordinates by subtracting CoM or provided center."""
        if center is None:
            center = CoordinateTransformer.get_center_of_mass(coords, mask)
        return coords - center, center
    
    @staticmethod
    def uncenter_coords(coords, center):
        """Restore coordinates to original frame by adding center."""
        return coords + center

class LigandEvaluator:
    """Evaluates generated ligands with multiple metrics."""
    
    def __init__(self, vina_executable="vina", exhaustiveness=8):
        self.vina_executable = vina_executable
        self.exhaustiveness = exhaustiveness
        self.coord_transformer = CoordinateTransformer()
        
    def coords_to_mol(self, coords, atom_types, aromatic_mask=None, bond_index=None, bond_type=None):
        """Convert generated coordinates and features to RDKit mol."""
        mol = Chem.RWMol()
        
        # Map atom types - adjust based on your actual encoding
        # Common mapping: H, C, N, O, F, P, S, Cl, Br, I
        element_map = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9, 5: 15, 6: 16, 7: 17, 8: 35, 9: 53}
        
        # Add atoms
        atom_indices = []
        for i, atom_type in enumerate(atom_types):
            if len(atom_type.shape) > 0:  # One-hot encoded
                elem = element_map.get(atom_type.argmax().item(), 6)
            else:  # Direct encoding
                elem = element_map.get(int(atom_type), 6)
            atom = mol.AddAtom(Chem.Atom(elem))
            atom_indices.append(atom)
        
        # Add bonds if provided
        if bond_index is not None and bond_type is not None:
            for (i, j), btype in zip(bond_index.T, bond_type):
                if i < j:  # Avoid duplicate bonds
                    bond_type_map = {
                        1: Chem.BondType.SINGLE,
                        2: Chem.BondType.DOUBLE,
                        3: Chem.BondType.TRIPLE,
                        4: Chem.BondType.AROMATIC
                    }
                    mol.AddBond(int(i), int(j), bond_type_map.get(btype.item(), Chem.BondType.SINGLE))
        else:
            # Distance-based connectivity
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < 1.6:  # Typical bond length threshold
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
        
        # Set 3D coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(coords):
            conf.SetAtomPosition(i, coord.tolist())
        mol.AddConformer(conf)
        
        # Sanitize
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None
    
    def calculate_qed(self, mol):
        """Calculate QED score."""
        if mol is None:
            return 0.0
        try:
            return QED.qed(mol)
        except:
            return 0.0
    
    def calculate_sa(self, mol):
        """Calculate synthetic accessibility score."""
        if mol is None:
            return 10.0  # Worst score
        try:
            return calculateScore(mol)
        except:
            return 10.0
    
    def calculate_diversity(self, mols):
        """Calculate diversity using Tanimoto similarity of Morgan fingerprints."""
        if len(mols) < 2:
            return 0.0
        
        fps = []
        for mol in mols:
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                fps.append(fp)
        
        if len(fps) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 1.0
        return 1.0 - avg_similarity
    
    def prepare_vina_files(self, protein_pdb, ligand_mol, output_dir):
        """Prepare files for Vina docking."""
        # Save protein
        protein_file = os.path.join(output_dir, "protein.pdbqt")
        os.system(f"obabel {protein_pdb} -O {protein_file} -xr")
        
        # Save ligand
        ligand_pdb = os.path.join(output_dir, "ligand.pdb")
        ligand_pdbqt = os.path.join(output_dir, "ligand.pdbqt")
        Chem.MolToPDBFile(ligand_mol, ligand_pdb)
        os.system(f"obabel {ligand_pdb} -O {ligand_pdbqt}")
        
        return protein_file, ligand_pdbqt
    
    def calculate_vina_score(self, protein_pdb, ligand_mol, pocket_center, box_size=20):
        """
        Calculate Vina docking score.
        pocket_center should be in the original protein coordinate frame.
        """
        if ligand_mol is None:
            return 0.0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Prepare files
                protein_file, ligand_file = self.prepare_vina_files(
                    protein_pdb, ligand_mol, tmpdir
                )
                
                # Run Vina scoring
                output_file = os.path.join(tmpdir, "output.pdbqt")
                cmd = [
                    self.vina_executable,
                    "--receptor", protein_file,
                    "--ligand", ligand_file,
                    "--center_x", str(pocket_center[0]),
                    "--center_y", str(pocket_center[1]),
                    "--center_z", str(pocket_center[2]),
                    "--size_x", str(box_size),
                    "--size_y", str(box_size),
                    "--size_z", str(box_size),
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--out", output_file,
                    "--score_only"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse score from output
                for line in result.stdout.split('\n'):
                    if 'Affinity' in line:
                        score = float(line.split()[1])
                        return -score  # Return negative for better score = higher value
                
                return 0.0
                
            except Exception as e:
                print(f"Vina error: {e}")
                return 0.0


def load_model(checkpoint_path, device='cuda'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate the trainer with same configuration
    trainer = CrossDockedTrainer(
        lmdb_path="./cd2020/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
        split_file="./cd2020/crossdocked_pocket10_pose_split.pt",
        device=device,
        batch_size=1,  # For generation
        learning_rate=1e-4,
        num_workers=0,
        cfg_dropout_prob=checkpoint.get('cfg_dropout_prob', 0.1),
        cfg_scale=checkpoint.get('cfg_scale', 7.5),
        rank=0,
        world_size=1,
        distributed=False,
        use_amp=False
    )
    
    # Load model weights
    trainer.diffusion_module.load_state_dict(checkpoint['diffusion_state'])
    trainer.dynamics_module.load_state_dict(checkpoint['dynamics_state'])
    trainer.ligand_vae.load_state_dict(checkpoint['ligand_vae_state'])
    trainer.protein_vae.load_state_dict(checkpoint['protein_vae_state'])
    
    trainer.diffusion_module.eval()
    trainer.dynamics_module.eval()
    
    return trainer


def generate_ligands_for_protein(trainer, protein_data, n_ligands=100, batch_size=10):
    """
    Generate multiple ligands for a single protein.
    Returns ligands in centered coordinate frame.
    """
    generated_ligands = []
    
    # Store original protein center for later transformation
    protein_coords = protein_data['protein_x'].cpu().numpy()
    protein_mask = protein_data['protein_mask'].cpu().numpy()
    protein_center = CoordinateTransformer.get_center_of_mass(protein_coords, protein_mask)
    
    # Prepare protein features (already centered in dataset)
    ligand_h, protein_h = trainer.prepare_features(protein_data)
    
    # Encode protein once
    with torch.no_grad():
        z_x_prot, _, z_h_prot, _ = trainer.stable_protein_vae.encode(
            protein_data['protein_x'].unsqueeze(0),
            {k: v.unsqueeze(0) for k, v in protein_h.items()},
            protein_data['protein_mask'].unsqueeze(0),
            protein_data['protein_edge_mask'].unsqueeze(0)
        )
        protein_latent = torch.cat([z_x_prot, z_h_prot], dim=2)
        trainer.diffusion_module.model.dynamics.protein_latent = protein_latent
    
    # Generate in batches
    for i in range(0, n_ligands, batch_size):
        current_batch_size = min(batch_size, n_ligands - i)
        
        # Sample ligand size from distribution
        n_ligand_atoms = trainer.sample_ligand_size()
        
        # Create masks
        ligand_mask = torch.zeros(current_batch_size, 50, 1).to(trainer.device)
        ligand_mask[:, :n_ligand_atoms, :] = 1
        edge_mask = (ligand_mask @ ligand_mask.transpose(1, 2)).view(current_batch_size, -1, 1)
        
        # Generate ligands (in centered frame)
        with torch.no_grad():
            x_gen, h_gen = trainer.diffusion_module.model.sample(
                n_samples=current_batch_size,
                n_nodes=50,
                node_mask=ligand_mask,
                edge_mask=edge_mask,
                context=None
            )
        
        # Store generated ligands with protein center info
        for j in range(current_batch_size):
            generated_ligands.append({
                'coords': x_gen[j].cpu().numpy(),  # Centered coordinates
                'features': h_gen['categorical'][j].cpu().numpy(),
                'mask': ligand_mask[j].cpu().numpy(),
                'n_atoms': n_ligand_atoms,
                'protein_center': protein_center  # Store for uncentering
            })
    
    return generated_ligands


def save_protein_for_vina(protein_data, output_path):
    """
    Save protein in original coordinates for Vina docking.
    Assumes protein_data contains original PDB information.
    """
    # This would need to be implemented based on your data structure
    # You might need to store original PDB paths in your dataset
    pass


def main():
    """Main evaluation script with proper coordinate handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated ligands')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--n_ligands', type=int, default=100,
                       help='Number of ligands to generate per protein')
    parser.add_argument('--n_proteins', type=int, default=100,
                       help='Number of test proteins to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use_vina', action='store_true',
                       help='Calculate Vina docking scores (requires Vina installation)')
    parser.add_argument('--protein_pdb_dir', type=str, default='./cd2020/pdb_files',
                       help='Directory containing original protein PDB files')
    
    args = parser.parse_args()
    
    # Load model
    trainer = load_model(args.checkpoint, device=args.device)
    evaluator = LigandEvaluator()
    transformer = CoordinateTransformer()
    
    # Load test dataset
    test_dataset = CrossDockedDataset(
        lmdb_path="./cd2020/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
        split_file="./cd2020/crossdocked_pocket10_pose_split.pt",
        split='test'
    )
    
    # Limit to specified number of proteins
    n_proteins = min(args.n_proteins, len(test_dataset))
    
    print(f"Evaluating on {n_proteins} test proteins")
    print(f"Generating {args.n_ligands} ligands per protein")
    
    all_results = []
    
    # Process each protein
    for protein_idx in tqdm(range(n_proteins), desc="Processing proteins"):
        protein_data = test_dataset[protein_idx]
        
        # Store original coordinates before moving to device
        original_protein_coords = protein_data['protein_x'].clone()
        original_protein_mask = protein_data['protein_mask'].clone()
        
        # Calculate protein center and pocket center
        protein_coords = original_protein_coords.numpy()
        protein_mask = original_protein_mask.numpy()
        
        # Get protein CoM (should already be at origin if preprocessed correctly)
        protein_com = transformer.get_center_of_mass(protein_coords, protein_mask)
        
        # Calculate pocket center (using ligand position as proxy)
        if 'ligand_x' in protein_data:
            original_ligand_coords = protein_data['ligand_x'].numpy()
            ligand_mask = protein_data['ligand_mask'].numpy()
            pocket_center = transformer.get_center_of_mass(original_ligand_coords, ligand_mask)
        else:
            # Use protein center as fallback
            pocket_center = protein_com
        
        # Move to device for generation
        for key in protein_data:
            if isinstance(protein_data[key], torch.Tensor):
                protein_data[key] = protein_data[key].to(args.device)
        
        # Generate ligands (in centered frame)
        generated_ligands = generate_ligands_for_protein(
            trainer, protein_data, n_ligands=args.n_ligands
        )
        
        # Process generated ligands
        mols = []
        mols_for_vina = []
        
        for lig in generated_ligands:
            # Extract valid atoms
            n_atoms = lig['n_atoms']
            coords_centered = lig['coords'][:n_atoms]
            features = lig['features'][:n_atoms]
            
            # Create molecule in centered frame (for QED, SA, diversity)
            mol = evaluator.coords_to_mol(coords_centered, features)
            mols.append(mol)
            
            # If using Vina, create molecule in original frame
            if args.use_vina and mol is not None:
                # Translate ligand back to original protein frame
                coords_original = transformer.uncenter_coords(coords_centered, pocket_center)
                mol_vina = evaluator.coords_to_mol(coords_original, features)
                mols_for_vina.append(mol_vina)
        
        # Calculate metrics
        valid_mols = [m for m in mols if m is not None]
        validity = len(valid_mols) / len(mols) if mols else 0
        
        qed_scores = [evaluator.calculate_qed(mol) for mol in valid_mols]
        sa_scores = [evaluator.calculate_sa(mol) for mol in valid_mols]
        diversity = evaluator.calculate_diversity(valid_mols)
        
        result = {
            'protein_idx': protein_idx,
            'protein_name': protein_data.get('name', f'protein_{protein_idx}'),
            'n_generated': len(mols),
            'validity': validity,
            'qed_mean': np.mean(qed_scores) if qed_scores else 0,
            'qed_std': np.std(qed_scores) if qed_scores else 0,
            'sa_mean': np.mean(sa_scores) if sa_scores else 0,
            'sa_std': np.std(sa_scores) if sa_scores else 0,
            'diversity': diversity
        }
        
        # Calculate Vina scores if requested
        if args.use_vina and mols_for_vina:
            # Get protein PDB path (you'll need to implement this based on your data structure)
            protein_pdb = os.path.join(args.protein_pdb_dir, f"{protein_data.get('name', '')}.pdb")
            
            if os.path.exists(protein_pdb):
                vina_scores = []
                for mol in mols_for_vina[:10]:  # Limit Vina to 10 molecules (it's slow)
                    if mol is not None:
                        score = evaluator.calculate_vina_score(
                            protein_pdb, mol, pocket_center
                        )
                        vina_scores.append(score)
                
                result['vina_mean'] = np.mean(vina_scores) if vina_scores else 0
                result['vina_std'] = np.std(vina_scores) if vina_scores else 0
                result['vina_best'] = np.max(vina_scores) if vina_scores else 0
            else:
                print(f"Warning: Protein PDB not found: {protein_pdb}")
                result['vina_mean'] = 0
                result['vina_std'] = 0
                result['vina_best'] = 0
        
        all_results.append(result)
        
        # Periodic saving
        if (protein_idx + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df.to_csv(args.output, index=False)
            print(f"Saved intermediate results for {protein_idx + 1} proteins")
            print(f"  Latest - Validity: {result['validity']:.3f}, "
                  f"QED: {result['qed_mean']:.3f}, "
                  f"SA: {result['sa_mean']:.3f}, "
                  f"Diversity: {result['diversity']:.3f}")
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total proteins evaluated: {len(df)}")
    print(f"Average Validity: {df['validity'].mean():.3f} ± {df['validity'].std():.3f}")
    print(f"Average QED: {df['qed_mean'].mean():.3f} ± {df['qed_mean'].std():.3f}")
    print(f"Average SA: {df['sa_mean'].mean():.3f} ± {df['sa_mean'].std():.3f}")
    print(f"Average Diversity: {df['diversity'].mean():.3f} ± {df['diversity'].std():.3f}")
    
    if args.use_vina and 'vina_mean' in df.columns:
        print(f"Average Vina Score: {df['vina_mean'].mean():.3f} ± {df['vina_mean'].std():.3f}")
        print(f"Best Vina Scores: {df['vina_best'].mean():.3f} ± {df['vina_best'].std():.3f}")
    
    print(f"\nResults saved to {args.output}")
    
    # Additional analysis
    print("\n" + "="*50)
    print("ADDITIONAL STATISTICS")
    print("="*50)
    
    # Success rate (valid molecules with good properties)
    good_molecules = df[(df['validity'] > 0.5) & (df['qed_mean'] > 0.5)]
    print(f"Proteins with >50% validity and QED>0.5: {len(good_molecules)}/{len(df)} "
          f"({100*len(good_molecules)/len(df):.1f}%)")
    
    # Distribution of validity
    print(f"\nValidity distribution:")
    print(f"  >90% valid: {(df['validity'] > 0.9).sum()} proteins")
    print(f"  >75% valid: {(df['validity'] > 0.75).sum()} proteins")
    print(f"  >50% valid: {(df['validity'] > 0.5).sum()} proteins")
    print(f"  <25% valid: {(df['validity'] < 0.25).sum()} proteins")


if __name__ == "__main__":
    main()