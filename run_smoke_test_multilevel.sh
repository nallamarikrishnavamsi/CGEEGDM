#!/bin/bash
#SBATCH --partition=dgx_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=smoke_multilevel
#SBATCH --output=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/smoke_multilevel_%j.log
#SBATCH --error=/home/dsamantaai/krishna/files/CGEEGDM_Final/logs/smoke_multilevel_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegenv
cd ~/krishna/files/CGEEGDM_Final

python3 -c "
import sys; sys.path.insert(0, '.')
import torch

device = 'cuda'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Free memory before: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB')

from model.diffusion_model import Wavenet
from model.classifier import Classifier
from model.graph_conditioned_classifier import GraphConditionedClassifier

backbone = Wavenet(
    in_channels=1, d_model=128, d_state=128, n_layer=20,
    n_ssm=None, kernel_init='diag-lin', kernel_mode='diag',
    bidirectional=True, d_cond=512, d_cond_embed=128,
    local_cond_ch=0, n_class=19, have_null_class=False,
    self_gated=False,
)
cls_kwargs = dict(
    start=0, end=None, diffusion_t=1,
    query=['gate'], reduce=['std'], rescale=False,
    L=2000, window_size=200, window_step=200,
    pool_merge='share', multi_query_merge='seq',
    d_embed=None, init_weight=False, embed_query=False,
    d_query_embed=None, have_ch_pos_embed=False,
    cat_ch_pos_embed=True, ch_pos_emb_sym='mirror',
    ch_order=['Fp1','F3','C3','P3','F7','T3','T5','O1',
              'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2'],
    clst_dim='TP', clst_pos_embed_dim='', n_clst=16,
    stack_struct='scf', num_heads=8, ff=4, dropout=0,
    have_crossnorm=False, across_pool_stack_struct='',
    n_ap_clst=0, ap_clst_dim='T',
    classifier_use_ap_clst=False, classifier_have_pos_embed=True,
    classifier_pos_embed_dim='TPN',
    classifier_stack_struct='sfsfsfsfsfsfsfsf',
    classifier_final_act='pool', n_class=6,
)
base_cls = Classifier(model=backbone, **cls_kwargs)

model = GraphConditionedClassifier(
    classifier = base_cls,
    graph_dim  = 256,
    token_dim  = 128,
    num_nodes  = 19,
    gcn_hidden = 128,
    gcn_layers = 2,
    gcn_dropout = 0.1,
    use_graph  = True,
)
model = model.to(device)
model.train()

B = 32
signal   = torch.randn(B, 19, 2000, device=device)
icoh_vec = torch.rand(B, 171, device=device)

print(f'Running forward pass with batch_size={B}...')
import time
t0 = time.time()
logits, z_token, z_graph = model((signal, None), icoh_vec, return_alignment=True, warmup_alpha=1.0)
print(f'Forward OK in {time.time()-t0:.1f}s. logits shape: {logits.shape}')
print(f'Peak memory after forward: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

t0 = time.time()
loss = logits.sum()
loss.backward()
print(f'Backward OK in {time.time()-t0:.1f}s.')
print(f'Peak memory after backward: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
print(f'Free memory after: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB')

print('SMOKE TEST PASSED — no OOM at batch_size=32 on GPU')
"

echo "Job finished: \$(date)"
