#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gc_v2_resume_p2
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_v2_resume_p2_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_v2_resume_p2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint

echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Resume from existing Phase 1 checkpoint (gc_v2_p2_5249), run only Phase 2
# (classifier-head-only finetune) which crashed last time due to missing methods
srun python src/finetune_graphcond_v2.py \
    --name gc_v2_resume_p2 \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --test_csv full106k_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 30 \
    --lambda_align 0.1 \
    --use_graph 1 \
    --resume_phase1_ckpt checkpoint/gc_v2_phase2_5249/best.ckpt \
    --phase2_epochs 10 \
    --phase2_lr 1e-4 \
    --phase2_weight_decay 0.01 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group GraphCond_Phase2Resume

echo "Job finished: $(date)"
