#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --exclude=dgx3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=gc_quickcheck
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_quickcheck_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_quickcheck_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint

echo "Job started : $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

srun python src/finetune_graphcond.py \
    --name gc_quickcheck \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv finetune_train \
    --val_csv finetune_val \
    --test_csv finetune_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 30 \
    --lambda_align 0.1 \
    --use_graph 1 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group QuickCheck

echo "Job finished: $(date)"
