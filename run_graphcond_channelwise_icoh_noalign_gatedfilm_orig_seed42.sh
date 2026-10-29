#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --exclude=dgx3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --job-name=graphcond_channelwise_icoh_noalign_gatedfilm_orig_seed42
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_channelwise_icoh_noalign_gatedfilm_orig_seed42_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/graphcond_channelwise_icoh_noalign_gatedfilm_orig_seed42_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
cd /home/dsamantaai/krishna/files/CGEEGDM_Final

# Channelwise iCOH, no alignment, OLD gated FiLM (warm-up removed):
#   T' = T + g * (alpha*T + beta - T), alpha/beta/g = MLP([T_c ; G_c])
srun python src/finetune_graphcond_channelwise.py \
    --name graphcond_channelwise_icoh_noalign_gatedfilm_orig_seed42 \
    --data_root /home/dsamantaai/krishna/data \
    --train_csv full106k_train \
    --val_csv full106k_val \
    --test_csv full106k_test \
    --signal_cache data/signal_cache \
    --connectivity_measure icoh \
    --backbone_ckpt checkpoints/backbone.ckpt \
    --batch_size 32 \
    --epochs 50 \
    --lambda_align 0.0 \
    --use_graph 1 \
    --augment_icoh 1 \
    --film gated \
    --fix_extractor_fold 0 \
    --seed 42 \
    --devices 2 \
    --wandb_project CGEEGDM \
    --wandb_group ChannelwiseIcohNoAlignGatedFilmOrigSeed42
echo "Job finished: $(date)"
