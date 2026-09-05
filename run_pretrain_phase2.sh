#!/bin/bash
#SBATCH --job-name=pretrain_full106k_p2
#SBATCH --partition=dgx_all
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv

# Resumes from last.ckpt (full trainer state) — NOT best.ckpt.
# --epochs is CUMULATIVE total (see pretrain_hms.py --help).
python src/pretrain_hms.py \
    --name pretrain_hms \
    --epochs 100 \
    --devices 2 \
    --batch_size 64 \
    --resume_ckpt checkpoints/pretrain_hms/last.ckpt
