#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=abl_GC_noalign
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/abl_GC_noalign_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/abl_GC_noalign_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint/ablation/GraphCond_noalign

echo "Job started: $(date)"
srun python src/finetune_graphcond.py \
    --name abl_GC_noalign \
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
    --use_graph 1 \
    --devices 2 \
    --patience 5 \
    --wandb_project CGEEGDM \
    --wandb_group GraphCond_NoAlign 
echo "Job finished: $(date)"
