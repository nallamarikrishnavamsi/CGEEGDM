#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=signal_cache
#SBATCH --output=logs/signal_cache_%j.log
#SBATCH --error=logs/signal_cache_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv

cd ~/krishna/files/CGEEGDM_Final

mkdir -p logs
mkdir -p data/signal_cache

python src/precompute_signal_cache.py
