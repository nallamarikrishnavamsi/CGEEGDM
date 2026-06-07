import os
import sys
import math
import torch
import lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.diffusion_model_pl import PLDiffusionModel
from model.connectivity_encoder import ConnectivityEncoder

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    pl.seed_everything(config.rng_seeding.seed, workers=config.rng_seeding.workers)

    data_cfg = instantiate(config.pretrain.data)

    # Datasets
    train_ds = ConnectivityHMSDataset(
        root=data_cfg['root'],
        split=data_cfg['train_dir'],
        icoh_cache_dir=data_cfg['icoh_cache_dir'],
        window_sec=data_cfg['window_sec'],
        fs=data_cfg['fs']
    )
    val_ds = ConnectivityHMSDataset(
        root=data_cfg['root'],
        split=data_cfg['val_dir'],
        icoh_cache_dir=data_cfg['icoh_cache_dir'],
        window_sec=data_cfg['window_sec'],
        fs=data_cfg['fs']
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    # Connectivity encoder
    conn_encoder = ConnectivityEncoder(in_dim=171, hidden_dim=256, out_dim=256)

    # Diffusion model with iCOH conditioning
    model_cfg = OmegaConf.to_container(config.pretrain.model, resolve=True)
    pl_model = PLDiffusionModel(
        model_kwargs=model_cfg['model_kwargs'],
        ema_kwargs=model_cfg['ema_kwargs'],
        noise_sch_kwargs=model_cfg['noise_sch_kwargs'],
        opt_kwargs=model_cfg['opt_kwargs'],
        gen_kwargs=model_cfg['gen_kwargs'],
        use_icoh=model_cfg['use_icoh'],
        conn_encoder=conn_encoder
    )

    print(f"Diffusion model params: {sum(p.numel() for p in pl_model.model.parameters())/1e6:.2f}M")
    print(f"Connectivity encoder params: {sum(p.numel() for p in conn_encoder.parameters())/1e6:.2f}M")

    # Trainer
    os.makedirs("logs/", exist_ok=True)
    os.makedirs("checkpoint/pretrain/hms_icoh/", exist_ok=True)

    trainer = instantiate(config.pretrain.trainer)

    print("Starting pretraining...")
    trainer.fit(pl_model, train_loader, val_loader)
    print("Pretraining done!")

if __name__ == '__main__':
    main()
