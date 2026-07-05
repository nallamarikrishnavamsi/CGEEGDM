#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ft_B0_matched
#SBATCH --output=/home/dsamantaai/krishna/files/Connectivity-Guided-EEGDM-GraphCond/logs/ft_B0_matched_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/Connectivity-Guided-EEGDM-GraphCond/logs/ft_B0_matched_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs checkpoint/finetune/B0_matched

echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python src/finetune_baseline_B0.py \
    +rng_seeding.seed=42 \
    +rng_seeding.workers=True \
    finetune=hms_B0_matched

echo "Job finished: $(date)"
