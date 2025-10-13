"""
Test script to verify your environment and imports before running the main training.
"""

import torch
import os
import sys

def test_environment():
    """Test the environment setup."""
    print("=" * 70)
    print("ENVIRONMENT TEST")
    print("=" * 70)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Memory: {memory:.1f} GB")
    
    # Check for required files
    print("\n" + "=" * 70)
    print("FILE CHECK")
    print("=" * 70)
    
    lmdb_path = "./cd2020/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    split_file = "./cd2020/crossdocked_pocket10_pose_split.pt"
    
    print(f"LMDB exists: {os.path.exists(lmdb_path)}")
    if os.path.exists(lmdb_path):
        size_mb = os.path.getsize(lmdb_path) / 1e6
        print(f"  Size: {size_mb:.1f} MB")
    
    print(f"Split file exists: {os.path.exists(split_file)}")
    if os.path.exists(split_file):
        size_kb = os.path.getsize(split_file) / 1e3
        print(f"  Size: {size_kb:.1f} KB")
    
    # Check imports
    print("\n" + "=" * 70)
    print("IMPORT CHECK")
    print("=" * 70)
    
    required_imports = [
        "torch",
        "numpy",
        "lmdb",
        "pickle",
        "tqdm",
        "matplotlib"
    ]
    
    for module_name in required_imports:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
    
    # Check for the stabilized_models import
    try:
        from test_stability import create_stable_model_pipeline
        print("✓ stabilized_models")
    except ImportError as e:
        print(f"✗ stabilized_models: {e}")
        print("  Make sure stabilized_models.py is in the same directory")
    
    # Check for egnn imports
    try:
        from egnn.models import EGNNEncoder, EGNNDecoder
        print("✓ egnn.models")
    except ImportError as e:
        print(f"✗ egnn.models: {e}")
        print("  Make sure the egnn package is available")
    
    # Check for equivariant_diffusion imports
    try:
        from equivariant_diffusion.en_diffusion import EnHierarchicalVAE, EnLatentDiffusion
        print("✓ equivariant_diffusion")
    except ImportError as e:
        print(f"✗ equivariant_diffusion: {e}")
        print("  Make sure the equivariant_diffusion package is available")
    
    print("\n" + "=" * 70)
    print("Setup test complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_environment()