#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=gc_p10e50
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_p10e50_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/gc_p10e50_%j.err

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

# patience=10, epochs=50, warmup_epochs=0 (graph module full strength from epoch 1),
# align_start_epoch=0 (alignment loss also active from epoch 1)
srun python src/finetune_graphcond.py \
    --name gc_p10e50 \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --test_csv full106k_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 50 \
    --patience 10 \
    --warmup_epochs 0 \
    --align_start_epoch 0 \
    --lambda_align 0.1 \
    --use_graph 1 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group Patience10Epoch50

echo "Job finished: $(date)"
