#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=graphcond_small
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_small_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_small_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint

echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Small data test — finetune splits
python src/finetune_graphcond.py \
    --name graphcond_small \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv finetune_train \
    --val_csv finetune_val \
    --test_csv finetune_test \
    --icoh_cache /home/dsamantaai/krishna/files/CGEEGDM_Final/data/icoh_cache \
    --backbone_ckpt /home/dsamantaai/krishna/files/CGEEGDM_Final/checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs_p1 10 \
    --epochs_p2 20 \
    --lambda_align 0.1 \
    --devices 2

echo "Job finished: $(date)"
