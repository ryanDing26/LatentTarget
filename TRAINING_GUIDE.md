# Training Guide for Latent Diffusion Model

## Changes Made

### 1. Fixed Training/Validation Loss Disparity

**Problem**: Training loss was showing ~0.2 while validation loss was showing ~30, making it impossible to properly monitor model performance.

**Root Cause**: The loss calculation in `equivariant_diffusion/en_diffusion.py` had inconsistent normalization between training and evaluation modes:
- During training with L2 loss: loss was normalized by dividing by `(n_dims + in_node_nf) * n_nodes`
- During evaluation: loss was NOT normalized AND was upweighted by the number of timesteps (1001)

**Fixes Applied**:
1. **Consistent Error Normalization** (`compute_error()` at line 450-459):
   - Removed the `self.training` check
   - Now always normalizes by `(n_dims + in_node_nf) * n_nodes` when using L2 loss

2. **Consistent SNR Weighting** (line 616-623):
   - Removed the `self.training` check
   - Now always uses unit weights for L2 loss (both train and eval)

3. **Consistent Constants** (line 629-631):
   - Removed the `self.training` check
   - Now always zeros out constants for L2 loss

4. **Consistent Upweighting** (line 674-679):
   - Removed the `self.training` check
   - Now never upweights when using L2 loss

5. **Consistent t0_always** (line 702-713 and 1205-1215):
   - For L2 loss, always uses `t0_always=False` for both training and evaluation
   - This ensures the same loss computation path

6. **VAE Reconstruction Error** (line 929-933):
   - Also fixed to always normalize consistently

**Result**: Training and validation losses are now on the same scale and directly comparable!

---

## 2. Modular Training Architecture

### New Training Scripts

Three training scripts are now available for flexible, modular training:

#### A. `train_vae.py` - Train VAE Models Separately

Train ligand or protein VAE independently:

```bash
# Train ligand VAE
python train_vae.py --model_type ligand --batch_size 32 --num_epochs 100

# Train protein VAE
python train_vae.py --model_type protein --batch_size 32 --num_epochs 100
```

**Arguments**:
- `--model_type`: 'ligand' or 'protein'
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Maximum number of epochs (default: 100)
- `--save_interval`: Save checkpoint every N epochs (default: 10)
- `--num_workers`: Number of data loading workers (default: 4)
- `--use_amp`: Use automatic mixed precision (default: True)
- `--device`: Device to use (default: 'cuda')

**Outputs**:
- `best_ligand_vae.pt` or `best_protein_vae.pt` - Best model checkpoint
- `checkpoints/{model_type}_vae_epoch_{N}.pt` - Periodic checkpoints
- `{model_type}_vae_losses_epoch_{N}.png` - Loss plots

#### B. `train_denoiser.py` - Train Denoiser with Frozen VAE

Train the denoiser (diffusion model) with pre-trained, frozen VAE weights:

```bash
# Train denoiser with pre-trained VAEs
python train_denoiser.py \
    --ligand_vae_path best_ligand_vae.pt \
    --protein_vae_path best_protein_vae.pt \
    --batch_size 16 \
    --num_epochs 100

# Train denoiser without pre-trained VAEs (joint training)
python train_denoiser.py --batch_size 16 --num_epochs 100
```

**Arguments**:
- `--ligand_vae_path`: Path to pre-trained ligand VAE (optional)
- `--protein_vae_path`: Path to pre-trained protein VAE (optional)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Maximum number of epochs (default: 100)
- `--save_interval`: Save checkpoint every N epochs (default: 10)
- `--num_workers`: Number of data loading workers (default: 4)
- `--cfg_dropout_prob`: Classifier-free guidance dropout probability (default: 0.1)
- `--cfg_scale`: CFG scale for inference (default: 7.5)
- `--use_amp`: Use automatic mixed precision (default: True)
- `--device`: Device to use (default: 'cuda')

**Outputs**:
- `best_denoiser.pt` - Best model checkpoint
- `checkpoints/denoiser_epoch_{N}.pt` - Periodic checkpoints
- `denoiser_losses_epoch_{N}.png` - Loss plots

**Key Features**:
- VAE weights are automatically frozen when loaded
- Only denoiser parameters are trained
- Reduces memory usage and training time
- Allows experimentation with different VAE/denoiser combinations

#### C. `train_all.py` - Joint Training (Original)

The original script for joint training of VAE and denoiser:

```bash
# Single GPU
python train_all.py --num_gpus 1 --batch_size 16

# Multi-GPU
python train_all.py --num_gpus 4 --batch_size 16
```

**Note**: This script has been updated with the loss fixes but retains the joint training approach.

---

## Recommended Training Pipeline

### Option 1: Two-Stage Training (Recommended for Modularity)

**Stage 1: Pre-train VAEs**
```bash
# Train ligand VAE
python train_vae.py --model_type ligand --batch_size 32 --num_epochs 100

# Train protein VAE
python train_vae.py --model_type protein --batch_size 32 --num_epochs 100
```

**Stage 2: Train Denoiser with Frozen VAEs**
```bash
python train_denoiser.py \
    --ligand_vae_path best_ligand_vae.pt \
    --protein_vae_path best_protein_vae.pt \
    --batch_size 16 \
    --num_epochs 100
```

**Advantages**:
- Modular: Can swap VAEs or denoisers independently
- Easier debugging: Issues can be isolated to specific components
- Faster iteration: Can retrain only the component that needs improvement
- Better for ablation studies

### Option 2: Joint Training

```bash
python train_all.py --num_gpus 1 --batch_size 16 --num_epochs 100
```

**Advantages**:
- Simpler: Single command
- End-to-end optimization
- May find better joint optimum

---

## Monitoring Training

All scripts now output comparable losses:

```
Epoch 1: Train Loss = 0.1234, Test Loss = 0.1456
Epoch 2: Train Loss = 0.1123, Test Loss = 0.1345
...
```

**What to Expect**:
- Both losses should be on the same scale (e.g., both ~0.1 or both ~1.0)
- Test loss should track train loss closely
- If test loss >> train loss, you likely have overfitting
- If test loss ≈ train loss, training is healthy

---

## Checkpoint Format

All saved checkpoints contain:
- `epoch`: Epoch number
- `vae_state` or `diffusion_state`: Model state dict
- `optimizer_state`: Optimizer state dict
- `train_loss` and `test_loss`: Loss values at checkpoint
- `scaler_state`: AMP scaler state (if using AMP)

**Loading Checkpoints**:
```python
checkpoint = torch.load('best_ligand_vae.pt')
model.load_state_dict(checkpoint['vae_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch']
```

---

## Memory and Performance Tips

1. **Batch Size**:
   - VAE training: Can use larger batches (32-64)
   - Denoiser training: Use smaller batches (8-16)

2. **Mixed Precision**:
   - Always enabled by default
   - Reduces memory usage by ~50%
   - Speeds up training by ~2x on modern GPUs

3. **Number of Workers**:
   - Set to 4-8 for optimal data loading
   - Too many workers can cause memory issues

4. **Multi-GPU Training**:
   - Only available in `train_all.py`
   - Automatically scales learning rate
   - Use for faster training on large datasets

---

## Files Modified

1. `equivariant_diffusion/en_diffusion.py`:
   - Fixed loss normalization inconsistencies
   - Made L2 loss calculation consistent between train/eval

2. Created new files:
   - `train_vae.py` - VAE-only training script
   - `train_denoiser.py` - Denoiser-only training script
   - `TRAINING_GUIDE.md` - This guide

---

## Troubleshooting

### Loss is NaN
- Reduce learning rate
- Check for gradient explosion (already clipped at norm=1.0)
- Verify input data is normalized

### Train/Test Loss Mismatch
- Should be fixed with the updates
- If still present, check if using correct model checkpoint

### Out of Memory
- Reduce batch size
- Reduce number of workers
- Enable gradient checkpointing (not implemented yet)

### VAE Not Loading
- Check file path is correct
- Verify checkpoint was saved correctly
- Ensure model architecture matches

---

## Citation

If you use this code, please cite the original latent diffusion work and the EGNN architecture.
