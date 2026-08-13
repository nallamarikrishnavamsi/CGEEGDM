#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=pretrain_smoke
#SBATCH --output=logs/pretrain_smoke_%j.log
#SBATCH --error=logs/pretrain_smoke_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoints

echo "Job started : $(date)"
srun python src/pretrain_hms.py \
    --name pretrain_smoke \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv smoke_train \
    --val_csv smoke_val \
    --signal_cache data/signal_cache \
    --batch_size 32 \
    --epochs 5 \
    --devices 2
echo "Job finished: $(date)"
