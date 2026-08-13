#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pretrain_hms
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/pretrain_hms_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/pretrain_hms_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoints

echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

export NCCL_DEBUG=INFO

export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_ASYNC_ERROR_HANDLING=1

srun python src/pretrain_hms.py \
    --name pretrain_hms \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --signal_cache data/signal_cache \
    --batch_size 32 \
    --epochs 100 \
    --devices 2

echo "Job finished: $(date)"
