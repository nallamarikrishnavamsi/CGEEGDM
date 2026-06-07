#!/bin/bash
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=eegdm_ablation
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate eeg310

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd /home/mtech1/25CS60R51/EEGDM_connectivity
mkdir -p logs checkpoint/ablation

echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python src/ablation_hms.py

echo "Job finished: $(date)"
