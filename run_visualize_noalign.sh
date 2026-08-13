#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=viz_noalign
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/visualize_noalign_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/visualize_noalign_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final

python src/visualize_representations.py \
    --baseline_ckpt checkpoint/baseline_5200/best.ckpt \
    --graphcond_ckpt checkpoint/abl_GC_noalign_5202/best.ckpt \
    --data_root /home/dsamantaai/krishna/data \
    --test_csv full106k_test \
    --signal_cache data/signal_cache \
    --max_samples 100000 \
    --out figures/noalign_vs_baseline
