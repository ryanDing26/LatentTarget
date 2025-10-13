#!/bin/bash
#SBATCH --job-name=test_single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # number of CPU cores
#SBATCH --mem=32G               # RAM
#SBATCH --gpus=1                # <-- request only 1 GPU
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=job_%j.log

# this trains LatentTarget on 1 GPU

srun python train_model.py

# for interactive mode:
# export CUDA_VISIBLE_DEVICES=0   # only GPU 0 is visible to your program
# python main_cd2020.py
