#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=per_class_eval
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/per_class_eval_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/per_class_eval_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final

python src/evaluate_per_class.py \
    --ckpts \
        baseline:checkpoint/baseline_5200/best.ckpt \
        graphcond:checkpoint/graphcond_full106k_5201/best.ckpt \
        gc_combined:checkpoint/gc_combined_5308/best.ckpt \
    --data_root /home/dsamantaai/krishna/data \
    --test_csv full106k_test \
    --signal_cache data/signal_cache \
    --out figures/per_class_breakdown

echo "Job finished: $(date)"
