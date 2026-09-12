#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --exclude=dgx3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=graphcond_noalign_pc
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_noalign_pc_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_noalign_pc_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint
echo "Job started : $(date)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# Graph conditioning ON, alignment loss OFF, connectivity measure = Pearson
srun python src/finetune_graphcond.py \
    --name graphcond_noalign_pc \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --test_csv full106k_test \
    --icoh_cache data/icoh_cache \
    --signal_cache data/signal_cache \
    --connectivity_measure pearson \
    --pearson_cache data/pearson \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 50 \
    --lambda_align 0.0 \
    --use_graph 1 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group PearsonNoAlign
echo "Job finished: $(date)"
