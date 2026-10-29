#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_cw_newfilm_orig_seed42
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_cw_newfilm_orig_seed42_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/eval_cw_newfilm_orig_seed42_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final
python src/evaluate_per_class_v3.py \
    --ckpts cw_newfilm:checkpoint/graphcond_channelwise_icoh_noalign_orig_seed42_34721/best.ckpt:channelwise \
    --data_root /home/dsamantaai/krishna/data \
    --test_csv full106k_test \
    --signal_cache data/signal_cache \
    --connectivity_measure icoh \
    --out figures/per_class_orig_seed42/cw_newfilm
