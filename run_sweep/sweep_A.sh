#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --exclude=dgx3
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=sweep_A
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/sweep_A_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/sweep_A_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint

srun python src/finetune_graphcond.py \
    --name sweep_A \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv sweep30_train \
    --val_csv sweep30_val \
    --test_csv sweep30_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 30 \
    --lambda_align 0.1 \
    --graph_lr 5e-6 \
    --graph_max_lr 2e-4 \
    --graph_weight_decay 0.05 \
    --use_graph 1 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group Sweep

echo "Job finished: $(date)"
