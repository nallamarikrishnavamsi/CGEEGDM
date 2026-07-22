#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_metrics
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_metrics_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_metrics_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final
python src/eval_full_metrics.py --ckpt checkpoint/seed44_combined_5434/best.ckpt
