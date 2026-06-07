#!/bin/bash
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eegdm_pretrain
#SBATCH --output=/home/mtech1/25CS60R51/EEGDM_connectivity/logs/pretrain_%j.log
#SBATCH --error=/home/mtech1/25CS60R51/EEGDM_connectivity/logs/pretrain_%j.err
#SBATCH --exclude=gnode5

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate eeg310

cd /home/mtech1/25CS60R51/EEGDM_connectivity

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python << 'PYEOF'
import sys, os, torch
import lightning as pl
from torch.utils.data import DataLoader
sys.path.insert(0, '/home/mtech1/25CS60R51/EEGDM_connectivity')
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.diffusion_model_pl import PLDiffusionModel
from model.connectivity_encoder import ConnectivityEncoder
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

pl.seed_everything(42)

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']

train_ds = ConnectivityHMSDataset(
    root='/home/mtech1/25CS60R51/EEGDM/data/hms',
    split='train_small',
    icoh_cache_dir='/home/mtech1/25CS60R51/EEGDM_connectivity/data/icoh_cache',
    window_sec=10, fs=200
)
val_ds = ConnectivityHMSDataset(
    root='/home/mtech1/25CS60R51/EEGDM/data/hms',
    split='val_small',
    icoh_cache_dir='/home/mtech1/25CS60R51/EEGDM_connectivity/data/icoh_cache',
    window_sec=10, fs=200
)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

os.makedirs('./gen/hms_icoh', exist_ok=True)
os.makedirs('./checkpoint/pretrain/hms_icoh', exist_ok=True)

conn_encoder = ConnectivityEncoder(in_dim=171, hidden_dim=256, out_dim=256)
pl_model = PLDiffusionModel(
    model_kwargs=dict(
        in_channels=1, d_model=128, d_state=128, n_layer=20,
        n_ssm=None, kernel_init="diag-lin", kernel_mode="diag",
        bidirectional=True, d_cond=512, d_cond_embed=128,
        local_cond_ch=0, n_class=19, have_null_class=False,
        self_gated=False, use_icoh=True, icoh_dim=256
    ),
    ema_kwargs=dict(beta=0.999, update_after_step=100, update_every=10),
    noise_sch_kwargs=dict(
        num_train_timesteps=50, beta_start=0.0001, beta_end=0.05,
        beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction"
    ),
    opt_kwargs=dict(lr=1e-4, weight_decay=0),
    gen_kwargs=dict(
        n_sample=4, shape=[1,2000], save_intermediate=False,
        root="./gen/", save_dir="hms_icoh", rescale=1e-4, sfreq=200,
        ch_names=HMS_CHANNELS, inv_data_trans=None
    ),
    use_icoh=True,
    conn_encoder=conn_encoder
)
print(f"Total params: {sum(p.numel() for p in pl_model.parameters())/1e6:.2f}M")

logger = WandbLogger(project='eegdm_connectivity', name='hms_icoh_pretrain_small',
                     save_dir='./logs/')
ckpt_cb = ModelCheckpoint(
    save_top_k=1, every_n_epochs=10,
    dirpath='./checkpoint/pretrain/hms_icoh/',
    filename='backbone_icoh'
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    strategy='auto',
    callbacks=[ckpt_cb],
    logger=logger,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,
    check_val_every_n_epoch=10,
    num_sanity_val_steps=0,
    enable_progress_bar=True
)

print("Starting pretraining...")
trainer.fit(pl_model, train_loader, val_loader)
print("Pretraining done!")
PYEOF

echo "Job finished: $(date)"
