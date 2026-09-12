#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --exclude=dgx3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_pearson
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/smoke_pearson_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/smoke_pearson_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint
echo "Job started : $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# Smoke test: 1 GPU, tiny subset, few epochs -- validates the full
# graph+align+pearson code path runs correctly before committing a
# multi-day 2-GPU job to it.
srun python src/finetune_graphcond.py \
    --name smoke_pearson \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv smoke_pearson_train \
    --val_csv smoke_pearson_val \
    --test_csv smoke_pearson_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --connectivity_measure pearson \
    --pearson_cache data/pearson \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 8 \
    --epochs 3 \
    --patience 3 \
    --lambda_align 0.1 \
    --use_graph 1 \
    --devices 1 \
    --wandb_project CGEEGDM \
    --wandb_group SmokeTest
echo "Job finished: $(date)"
