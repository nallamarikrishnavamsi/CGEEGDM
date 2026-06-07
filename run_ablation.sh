#!/bin/bash
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=eegdm_abl
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err
#SBATCH --nodelist=gnode6

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate eeg310

cd ~/EEGDM_connectivity
mkdir -p logs

echo "Job started: $(date)"
echo "Config: $1"

python src/ablation_hms.py --config $1

echo "Job finished: $(date)"
