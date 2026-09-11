#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --exclude=dgx3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=graphcond_noalign
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_noalign_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_noalign_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint
echo "Job started : $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# Graph conditioning ON, alignment loss OFF (lambda_align=0.0 -- confirmed
# in src/finetune_graphcond.py this zeroes effective_lambda_align
# regardless of align_start_epoch/ramp logic)
srun python src/finetune_graphcond.py \
    --name graphcond_noalign \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv finetune_train \
    --val_csv finetune_val \
    --test_csv finetune_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 50 \
    --lambda_align 0.0 \
    --use_graph 1 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group NoAlign
echo "Job finished: $(date)"
