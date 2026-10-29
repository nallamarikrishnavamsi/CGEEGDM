#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=precompute_icoh
#SBATCH --output=logs/precompute_icoh_%j.log
#SBATCH --error=logs/precompute_icoh_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export OMP_NUM_THREADS=16
cd ~/krishna/files/CGEEGDM_Final
mkdir -p logs data/icoh_cache

echo "Job started : $(date)"
echo "Node        : $(hostname)"

python3 src/precompute_icoh.py

echo "Job finished: $(date)"
