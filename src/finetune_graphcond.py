"""
Finetune Graph-Conditioned EEGDM on HMS.

Loads the ORIGINAL official EEGDM backbone.ckpt (pretrained on TUEV,
22 channels) and removes only the dataset-specific label_embed.weight
before loading, per transfer-learning philosophy of EEGDM.
"""
import os, sys, math, argparse
import torch
from ema_pytorch import EMA
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion_model_pl import PLDiffusionModel
from model.classifier import Classifier
from model.graph_conditioned_classifier import GraphConditionedClassifier
from model.alignment import cosine_alignment_loss
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached


def load_original_backbone(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']

    incompatible_keys = [
        'model.label_embed.weight',
        'ema.online_model.label_embed.weight',
        'ema.ema_model.label_embed.weight',
    ]
    for k in incompatible_keys:
        if k in state_dict:
            del state_dict[k]
            print(f"Removed incompatible key: {k}")

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
    opt_kwargs  = {'lr': 1e-4, 'weight_decay': 0}
    gen_kwargs  = {'root': './gen/', 'save_dir': 'graphcond',
                   'n_sample': 1, 'shape': [1, 2000],
                   'save_intermediate': False, 'rescale': 1e-4, 'sfreq': 200}

    pretrain_model = PLDiffusionModel(
        model_kwargs=model_kwargs, ema_kwargs=ema_kwargs,
        noise_sch_kwargs=noise_sch_kwargs, opt_kwargs=opt_kwargs,
        gen_kwargs=gen_kwargs,
    )
    pretrain_model.load_state_dict(state_dict, strict=False)
    backbone = pretrain_model.ema.ema_model
    return backbone


class PLGraphConditionedClassifier(pl.LightningModule):
    def __init__(self, classifier_model_kwargs, opt_kwargs, sch_kwargs,
                 n_class=6, lambda_align=0.1,
                 use_graph=True,
                 backbone_ckpt='checkpoints/backbone.ckpt'):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_align = lambda_align

        backbone = load_original_backbone(backbone_ckpt)

        base_classifier = Classifier(model=backbone, **classifier_model_kwargs)
        self.model = GraphConditionedClassifier(
            classifier = base_classifier,
            graph_dim  = 256,
            token_dim  = 128,  # d_model of the diffusion backbone (latent token feature dim)
            num_nodes  = 19,
            gcn_hidden = 128,
            gcn_layers = 3,
            use_graph  = use_graph,
        )

        # EMA on classifier — matches original EEGDM
        self.ema = EMA(
            self.model,
            beta=0.999,
            update_after_step=100,
            update_every=10,
            ignore_startswith_names={"classifier.extractor", "classifier.reducer"},
        )
        self.should_update_ema = True

        # No freezing — train end-to-end like original EEGDM

        self.val_metrics = MetricCollection({
            'kappa': MulticlassCohenKappa(num_classes=n_class, validate_args=False),
            'bacc' : MulticlassAccuracy(num_classes=n_class, average='macro', validate_args=False),
            'wf1'  : MulticlassF1Score(num_classes=n_class, average='weighted', validate_args=False),
        }, prefix='val/')
        self.test_metrics = self.val_metrics.clone(prefix='test/')

    def _shared_step(self, batch, use_ema=False):
        signal, soft_label, icoh_vec = batch
        model = self.ema if use_ema else self.model
        logits, z_token, z_graph = model((signal, None), icoh_vec, return_alignment=True)

        task_loss = F.kl_div(
            F.log_softmax(logits, dim=-1), soft_label, reduction='batchmean'
        )

        if z_token is None:
            align_loss = torch.tensor(
                0.0,
                device=task_loss.device,
                dtype=task_loss.dtype,
            )
        else:
            align_loss = cosine_alignment_loss(z_token, z_graph)

        loss = task_loss + self.lambda_align * align_loss
        return loss, task_loss, align_loss, logits, soft_label

    def training_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/task_loss', task_loss)
        self.log('train/align_loss', align_loss)
        if self.should_update_ema:
            self.ema.update()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch, use_ema=True)
        preds = logits.argmax(dim=-1)
        hard  = soft_label.argmax(dim=-1)
        self.val_metrics.update(preds, hard)
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch, use_ema=True)
        preds = logits.argmax(dim=-1)
        hard  = soft_label.argmax(dim=-1)
        self.test_metrics.update(preds, hard)
        self.log('test/loss', loss)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()



    def configure_optimizers(self):
        # Single AdamW on all parameters — matches original EEGDM exactly
        base_lr      = self.hparams['opt_kwargs'].get('lr', 1e-4)
        weight_decay = self.hparams['opt_kwargs'].get('weight_decay', 0.05)
        betas        = self.hparams['opt_kwargs'].get('betas', (0.9, 0.98))
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=base_lr, weight_decay=weight_decay, betas=betas
        )
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams['sch_kwargs']['max_lr'],
            total_steps=self.hparams['sch_kwargs']['total_steps'],
            pct_start=self.hparams['sch_kwargs'].get('pct_start', 0.1),
        )
        return [opt], [{'scheduler': sch, 'interval': 'step'}]

CLASSIFIER_MODEL_KWARGS = dict(
    start=0, end=None, diffusion_t=1,
    query=["gate"], reduce=["std"], rescale=False,
    L=2000, window_size=200, window_step=200,
    pool_merge="share", multi_query_merge="seq",
    d_embed=None, init_weight=False, embed_query=False,
    d_query_embed=None, have_ch_pos_embed=False,
    cat_ch_pos_embed=True, ch_pos_emb_sym="mirror",
    ch_order=["Fp1","F3","C3","P3","F7","T3","T5","O1",
              "Fz","Cz","Pz","Fp2","F4","C4","P4","F8","T4","T6","O2"],
    clst_dim="TP", clst_pos_embed_dim="", n_clst=16,
    stack_struct="scf", num_heads=8, ff=4, dropout=0,
    have_crossnorm=False, across_pool_stack_struct="",
    n_ap_clst=0, ap_clst_dim="T",
    classifier_use_ap_clst=False, classifier_have_pos_embed=True,
    classifier_pos_embed_dim="TPN",
    classifier_stack_struct="sfsfsfsfsfsfsfsf",
    classifier_final_act="pool", n_class=6,
    use_cond=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
)


def main(args):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    train_ds = ConnectivityHMSDatasetCached(args.data_root, args.train_csv, args.signal_cache, window_sec=10)
    val_ds   = ConnectivityHMSDatasetCached(args.data_root, args.val_csv, args.signal_cache, window_sec=10)
    test_ds  = ConnectivityHMSDatasetCached(args.data_root, args.test_csv, args.signal_cache, window_sec=10)
    print(f"Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = PLGraphConditionedClassifier(
        classifier_model_kwargs = CLASSIFIER_MODEL_KWARGS,
        opt_kwargs   = dict(lr=1e-4, weight_decay=0.05, betas=[0.9, 0.98]),
        sch_kwargs   = dict(max_lr=5e-4, total_steps=steps_per_epoch*args.epochs, pct_start=0.1),
        n_class      = 6,
        lambda_align = args.lambda_align,
        use_graph    = args.use_graph,
        
        backbone_ckpt = args.backbone_ckpt,
    )

    ckpt_dir = f"checkpoint/{args.name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Single end-to-end training — matches original EEGDM philosophy
    total_steps = steps_per_epoch * args.epochs
    model.hparams['sch_kwargs']['total_steps'] = total_steps


    wandb_logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.name,
        save_dir="logs/wandb",
    )

    trainer = pl.Trainer(
        logger=wandb_logger,

        max_epochs=args.epochs,
        accelerator='gpu', devices=args.devices,
        strategy='ddp' if args.devices > 1 else 'auto',
        precision='32-true', log_every_n_steps=10, num_sanity_val_steps=0,
        gradient_clip_val=3,
        default_root_dir=f'logs/{args.name}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/kappa', mode='max', save_top_k=1,
                                         dirpath=ckpt_dir, filename='best', save_last=True),
            pl.callbacks.EarlyStopping(monitor='val/kappa', mode='max', patience=10),
        ]
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callbacks[0].best_model_path
    best = PLGraphConditionedClassifier.load_from_checkpoint(best_ckpt, weights_only=False)
    results = trainer.test(best, test_loader)
    print(results)

    import wandb
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='graphcond')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--train_csv', type=str, default='finetune_train')
    parser.add_argument('--val_csv',   type=str, default='finetune_val')
    parser.add_argument('--test_csv',  type=str, default='finetune_test')
    parser.add_argument('--icoh_cache', type=str, default='data/icoh_cache')
    parser.add_argument('--signal_cache', type=str,
                        default='data/signal_cache')
    parser.add_argument('--backbone_ckpt', type=str, default='checkpoints/backbone.ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--lambda_align', type=float, default=0.1)
    parser.add_argument('--use_graph', type=int, default=1)

    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--wandb_project', type=str, default='CGEEGDM')
    parser.add_argument('--wandb_group', type=str, default='GraphCond')
    args = parser.parse_args()
    args.use_graph = bool(args.use_graph)
    main(args)
