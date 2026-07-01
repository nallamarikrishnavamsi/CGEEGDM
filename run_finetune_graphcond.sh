#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=graphcond_v2
#SBATCH --output=logs/graphcond_%j.log
#SBATCH --error=logs/graphcond_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
cd ~/krishna/files/Connectivity-Guided-EEGDM-GraphCond
mkdir -p logs checkpoint

echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python src/finetune_graphcond.py \
    --name graphcond_v2 \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full_train \
    --val_csv pretrain_val \
    --test_csv finetune_test \
    --icoh_cache data/icoh_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs_p1 10 \
    --epochs_p2 20 \
    --lambda_align 0.1 \
    --align_type cosine

echo "Job finished: $(date)"
