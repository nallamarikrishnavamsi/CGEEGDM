#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_rich_intra_seed42
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_rich_intra_seed42_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_rich_intra_seed42_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final
python src/eval_rich_ckpt.py \
    --ckpt checkpoint/baseline_finetune_intra_seed42_35824/best.ckpt \
    --model_type global \
    --data_root /home/dsamantaai/krishna/data \
    --test_csv intra106k_test \
    --signal_cache data/signal_cache \
    --connectivity_measure icoh \
    --out figures/per_class_intra_seed42/baseline_rich \
    --title_prefix "Intra-patient baseline: "
python src/eval_rich_ckpt.py \
    --ckpt checkpoint/graphcond_channelwise_icoh_noalign_intra_seed42_35825/best.ckpt \
    --model_type channelwise \
    --data_root /home/dsamantaai/krishna/data \
    --test_csv intra106k_test \
    --signal_cache data/signal_cache \
    --connectivity_measure icoh \
    --out figures/per_class_intra_seed42/cgeegdm_rich \
    --title_prefix "Intra-patient CGEEGDM: "
