"""
Pretrain a fresh diffusion backbone (Wavenet) from scratch on HMS,
using the same 19-channel-labeled, per-channel diffusion objective as
original EEGDM's TUEV pretraining (src/pretrain.py), adapted to argparse
and HMS's data (correctly offset-windowed signal_cache).

Usage:
    python src/pretrain_hms.py --epochs 50 --devices 2
"""
import os, sys, argparse
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.diffusion_model_pl import PLDiffusionModel
from dataloader.HMSPretrainDataset import HMSPretrainDataset, HMS_CHANNELS


class PrintEpochCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train/mse_loss', float('nan'))
        print(f"[Epoch {trainer.current_epoch+1}/{trainer.max_epochs}] train/mse_loss={loss:.4f}", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            return
        val_loss = trainer.callback_metrics.get('val/mse_loss', float('nan'))
        print(f"[Epoch {trainer.current_epoch}] val/mse_loss={val_loss:.4f}", flush=True)


def main(args):
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    train_ds = HMSPretrainDataset(args.data_root, args.train_csv, args.signal_cache)
    val_ds   = HMSPretrainDataset(args.data_root, args.val_csv, args.signal_cache)
    print(f"Train (per-channel rows): {len(train_ds)}  Val: {len(val_ds)}")

    _g = torch.Generator()
    _g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True,
                               generator=_g, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True,
                               drop_last=True)

    # Fresh 19-channel backbone (HMS has 19 channels vs original TUEV's 22)
    model_kwargs = {
        'in_channels': 1, 'd_model': 128, 'd_state': 128,
        'n_layer': 20, 'n_ssm': None, 'kernel_init': 'diag-lin',
        'kernel_mode': 'diag', 'bidirectional': True,
        'd_cond': 512, 'd_cond_embed': 128, 'local_cond_ch': 0,
        'n_class': 19, 'have_null_class': False, 'self_gated': False,
    }
    ema_kwargs       = {'beta': 0.999, 'update_after_step': 100, 'update_every': 10}
    noise_sch_kwargs = {'num_train_timesteps': 50, 'beta_start': 0.0001,
                         'beta_end': 0.05, 'beta_schedule': 'squaredcos_cap_v2',
                         'prediction_type': 'v_prediction'}
    # backbone lr/wd match verified original EEGDM base_*.yaml values
    opt_kwargs = {'lr': 1e-4, 'weight_decay': 0.0}
    gen_kwargs = {'root': './gen/', 'save_dir': 'pretrain_hms',
                  'n_sample': 1, 'shape': [1, 2000],
                  'save_intermediate': False, 'rescale': 1e-4, 'sfreq': 200,
                  'ch_names': HMS_CHANNELS}

    model = PLDiffusionModel(
        model_kwargs=model_kwargs, ema_kwargs=ema_kwargs,
        noise_sch_kwargs=noise_sch_kwargs, opt_kwargs=opt_kwargs,
        gen_kwargs=gen_kwargs,
    )

    import os as _os
    job_id = _os.environ.get("SLURM_JOB_ID", "local")
    ckpt_dir = f"checkpoints/{args.name}_{job_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=args.epochs,
        accelerator='gpu', devices=args.devices,
        strategy='ddp' if args.devices > 1 else 'auto',
        precision='32-true', log_every_n_steps=10, num_sanity_val_steps=0,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
        default_root_dir=f'logs/{args.name}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/mse_loss', mode='min', save_top_k=1,
                                          dirpath=ckpt_dir, filename='best', save_last=True),
            PrintEpochCallback(),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {trainer.checkpoint_callbacks[0].best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='pretrain_hms')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--train_csv', type=str, default='full106k_train')
    parser.add_argument('--val_csv', type=str, default='full106k_val')
    parser.add_argument('--signal_cache', type=str, default='data/signal_cache')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
