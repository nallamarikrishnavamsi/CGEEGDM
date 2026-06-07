import os, sys, math
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.classifier_connectivity import PLClassifierConnectivity

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    pl.seed_everything(config.rng_seeding.seed, workers=config.rng_seeding.workers)
    torch.set_float32_matmul_precision('medium')

    data_cfg = OmegaConf.to_container(
        hydra.utils.instantiate(config.finetune.data), resolve=True
    )

    train_ds = ConnectivityHMSDataset(
        root=data_cfg['root'], split=data_cfg['train_dir'],
        icoh_cache_dir=data_cfg['icoh_cache_dir'],
        window_sec=data_cfg['window_sec'], fs=data_cfg['fs']
    )
    val_ds = ConnectivityHMSDataset(
        root=data_cfg['root'], split=data_cfg['val_dir'],
        icoh_cache_dir=data_cfg['icoh_cache_dir'],
        window_sec=data_cfg['window_sec'], fs=data_cfg['fs']
    )
    test_ds = ConnectivityHMSDataset(
        root=data_cfg['root'], split=data_cfg['test_dir'],
        icoh_cache_dir=data_cfg['icoh_cache_dir'],
        window_sec=data_cfg['window_sec'], fs=data_cfg['fs']
    )
    print(f"Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

    # Dynamic total_steps
    steps_per_epoch = math.ceil(len(train_ds) / data_cfg['batch_size'])
    total_steps     = steps_per_epoch * config.finetune.trainer.max_epochs
    print(f"Steps/epoch:{steps_per_epoch}  Total:{total_steps}")

    train_loader = DataLoader(train_ds, batch_size=data_cfg['batch_size'],
                              shuffle=True,  num_workers=data_cfg['num_workers'],
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=data_cfg['batch_size'],
                              shuffle=False, num_workers=data_cfg['num_workers'],
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=data_cfg['batch_size'],
                              shuffle=False, num_workers=2)

    model_cfg = OmegaConf.to_container(config.finetune.model, resolve=True)
    model_cfg['sch_kwargs']['total_steps'] = total_steps

    # Phase 1: frozen backbone
    model = PLClassifierConnectivity(
        pretrain_checkpoint = model_cfg['pretrain_checkpoint'],
        model_kwargs        = model_cfg['model_kwargs'],
        ema_kwargs          = model_cfg['ema_kwargs'],
        opt_kwargs          = model_cfg['opt_kwargs'],
        sch_kwargs          = model_cfg['sch_kwargs'],
        n_class             = model_cfg['n_class'],
        lambda_supcon       = model_cfg['lambda_supcon'],
        proj_dim            = model_cfg['proj_dim'],
        freeze_backbone     = True,
        use_kl              = model_cfg['use_kl'],
        use_supcon          = model_cfg['use_supcon'],
    )

    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoint/finetune/hms_connectivity', exist_ok=True)

    callbacks_p1 = [
        pl.callbacks.ModelCheckpoint(
            monitor='val/kappa', mode='max', save_top_k=1,
            dirpath='checkpoint/finetune/hms_connectivity',
            filename='phase1_best', save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val/kappa', mode='max', patience=5
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]

    trainer_p1 = pl.Trainer(
        max_epochs              = model_cfg.get('phase1_epochs', 10),
        accelerator             = 'gpu', devices=1,
        precision               = '16-mixed',
        callbacks               = callbacks_p1,
        log_every_n_steps       = 10,
        num_sanity_val_steps    = 0,
        default_root_dir        = 'logs'
    )
    print("Phase 1: training heads only...")
    trainer_p1.fit(model, train_loader, val_loader)

    # Phase 2: unfreeze all
    model.unfreeze_backbone()
    steps_p2    = steps_per_epoch * model_cfg.get('phase2_epochs', 20)
    model.hparams['sch_kwargs']['total_steps'] = steps_p2

    callbacks_p2 = [
        pl.callbacks.ModelCheckpoint(
            monitor='val/kappa', mode='max', save_top_k=1,
            dirpath='checkpoint/finetune/hms_connectivity',
            filename='phase2_best', save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val/kappa', mode='max', patience=8
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]

    trainer_p2 = pl.Trainer(
        max_epochs              = model_cfg.get('phase2_epochs', 20),
        accelerator             = 'gpu', devices=1,
        precision               = '16-mixed',
        callbacks               = callbacks_p2,
        log_every_n_steps       = 10,
        num_sanity_val_steps    = 0,
        default_root_dir        = 'logs'
    )
    print("Phase 2: full fine-tuning...")
    trainer_p2.fit(model, train_loader, val_loader)

    best_ckpt = trainer_p2.checkpoint_callbacks[0].best_model_path
    print(f"Best checkpoint: {best_ckpt}")
    best = PLClassifierConnectivity.load_from_checkpoint(best_ckpt, weights_only=False)
    trainer_p2.test(best, test_loader)

if __name__ == '__main__':
    main()
