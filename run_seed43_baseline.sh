#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=seed43_baseline
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/seed43_baseline_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/seed43_baseline_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint

echo "Job started : $(date)"

srun python src/finetune_graphcond.py \
    --name seed43_baseline \
    --seed 43 \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --test_csv full106k_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 30 \
    --lambda_align 0.0 \
    --use_graph 0 \
    --gcn_dropout 0.2 \
    --augment_icoh 1 \
    --icoh_noise_std 0.05 \
    --icoh_edge_dropout_p 0.1 \
    --lambda_graph_l2 0.01 \
    --devices 2 \
    --patience 5 \
    --wandb_project CGEEGDM \
    --wandb_group Seed43Baseline

echo "Job finished: $(date)"
