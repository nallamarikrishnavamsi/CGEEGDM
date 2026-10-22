"""
finetune_graphcond_channelwise.py

Channel-wise variant of src/finetune_graphcond.py (ORIGINAL FILE UNCHANGED).

Identical to finetune_graphcond.py in every respect EXCEPT the graph
conditioning mechanism: imports GraphConditionedClassifierChannelwise
(channel c's EEG tokens conditioned by graph node c's own embedding)
instead of GraphConditionedClassifier (one global graph embedding
conditioning every channel identically).

Everything else -- backbone loading, label_embed slicing, CLASSIFIER_MODEL_KWARGS,
dataset loading (icoh/pearson connectivity switch), optimizer param groups,
training loop, all CLI args -- is copied verbatim from finetune_graphcond.py.

--name defaults to 'graphcond_channelwise' (not 'graphcond') so checkpoint
dirs and wandb runs never collide with the original entry point's runs.
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
from model.graph_conditioned_classifier_channelwise import GraphConditionedClassifierChannelwise
from model.alignment import cosine_alignment_loss
from src.util import EarlyStoppingWithWarmup
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from model.extractor_foldfix import LatentActivityExtractorFoldFix


USED_INDICES_22_TO_20 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]


def load_original_backbone(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']

    label_embed_keys = [
        'model.label_embed.weight',
        'ema.online_model.label_embed.weight',
        'ema.ema_model.label_embed.weight',
    ]
    for k in label_embed_keys:
        if k in state_dict:
            original = state_dict[k]
            assert original.shape[0] == 22, f"{k}: expected 22 rows, got {original.shape[0]}"
            state_dict[k] = original[USED_INDICES_22_TO_20]
            print(f"Sliced {k}: 22 -> {state_dict[k].shape[0]} rows (kept pretrained values for available channels)")

    model_kwargs = {
        'in_channels': 1, 'd_model': 128, 'd_state': 128,
        'n_layer': 20, 'n_ssm': None, 'kernel_init': 'diag-lin',
        'kernel_mode': 'diag', 'bidirectional': True,
        'd_cond': 512, 'd_cond_embed': 128, 'local_cond_ch': 0,
        'n_class': 20, 'have_null_class': False, 'self_gated': False,
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


class PrintEpochCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get('train/loss', float('nan'))
        print(f"[Epoch {epoch+1}/{trainer.max_epochs}] train/loss={train_loss:.4f}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss  = metrics.get('val/loss', float('nan'))
        val_kappa = metrics.get('val/kappa', float('nan'))
        val_bacc  = metrics.get('val/bacc', float('nan'))
        print(f"[Epoch {epoch}] val/loss={val_loss:.4f} val/kappa={val_kappa:.4f} val/bacc={val_bacc:.4f}", flush=True)


class PLGraphConditionedClassifierChannelwise(pl.LightningModule):
    def __init__(self, classifier_model_kwargs, opt_kwargs, sch_kwargs,
                 n_class=6, lambda_align=0.1,
                 use_graph=True,
                 backbone_ckpt='checkpoints/backbone.ckpt',
                 align_start_epoch=5,
                 align_ramp_epochs=5,
                 gcn_layers=3,
                 graph_dim=128,
                 gcn_dropout=0.1,
                 augment_icoh=False,
                 random_tokens=False,
                 film='standard',
                 icoh_edge_dropout_p=0.1,
                 fix_extractor_fold=False):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_align = lambda_align

        backbone = load_original_backbone(backbone_ckpt)

        base_classifier = Classifier(model=backbone, **classifier_model_kwargs)
        self.model = GraphConditionedClassifierChannelwise(
            classifier = base_classifier,
            graph_dim  = graph_dim,
            token_dim  = 128,
            num_nodes  = 20,
            gcn_hidden = 128,
            gcn_layers = gcn_layers,
            gcn_dropout = gcn_dropout,
            use_graph  = use_graph,
            augment_icoh = augment_icoh,
            random_tokens = random_tokens,
            film = film,
            icoh_edge_dropout_p = icoh_edge_dropout_p,
        )

        if fix_extractor_fold:   # must happen before EMA deep-copies self.model
            LatentActivityExtractorFoldFix.from_existing(self.model.classifier.extractor)
        self.ema = EMA(
            self.model,
            beta=0.999,
            update_after_step=100,
            update_every=10,
            ignore_startswith_names={"classifier.extractor", "classifier.reducer"},
        )
        self.should_update_ema = True

        self.val_metrics = MetricCollection({
            'kappa': MulticlassCohenKappa(num_classes=n_class, validate_args=False),
            'bacc' : MulticlassAccuracy(num_classes=n_class, average='macro', validate_args=False),
            'wf1'  : MulticlassF1Score(num_classes=n_class, average='weighted', validate_args=False),
        }, prefix='val/')
        self.test_metrics = self.val_metrics.clone(prefix='test/')

    def _shared_step(self, batch, use_ema=False):
        signal, soft_label, icoh_vec = batch
        model = self.ema if use_ema else self.model
        logits, z_token, z_graph = model(
            (signal, None), icoh_vec, return_alignment=True
        )

        task_loss = F.kl_div(
            F.log_softmax(logits, dim=-1), soft_label, reduction='batchmean'
        )

        align_start = self.hparams.get('align_start_epoch', 5)
        if z_token is None or self.current_epoch < align_start:
            align_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
            effective_lambda_align = 0.0
        else:
            align_loss = cosine_alignment_loss(z_token, z_graph)
            align_ramp_epochs = self.hparams.get('align_ramp_epochs', 5)
            epochs_since_start = self.current_epoch - align_start
            ramp_frac = min(1.0, (epochs_since_start + 1) / max(1, align_ramp_epochs))
            effective_lambda_align = self.lambda_align * ramp_frac

        loss = task_loss + effective_lambda_align * align_loss
        self.log('train/lambda_align_effective', effective_lambda_align,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss, task_loss, align_loss, logits, soft_label

    def training_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/task_loss', task_loss)
        self.log('train/align_loss', align_loss)
        return loss

    def optimizer_step(self, *args, **kwargs):
        # EMA is updated once per OPTIMIZER step, as in EEGDM
        # model/classifier_pl.py:244-248 -- not once per batch. With
        # accumulate_grad_batches=4 a per-batch update would run 4x per
        # weight change and shrink the effective EMA window ~4x.
        super().optimizer_step(*args, **kwargs)
        if self.should_update_ema:
            self.ema.update()

    def validation_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch, use_ema=True)
        preds = logits.argmax(dim=-1)
        hard  = soft_label.argmax(dim=-1)
        self.val_metrics.update(preds, hard)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, task_loss, align_loss, logits, soft_label = self._shared_step(batch, use_ema=True)
        preds = logits.argmax(dim=-1)
        hard  = soft_label.argmax(dim=-1)
        self.test_metrics.update(preds, hard)
        self.log('test/loss', loss)
        self.log('test/task_loss', task_loss)
        self.log('test/align_loss', align_loss)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        # Paper: "Latent fusion tokens and bias terms were exempted from weight
        # decay."  EEGDM config: no_wd=["decoder.cls_token","decoder.ap_cls_token"]
        # plus bias_1dim_no_wd=True.  Reproduced here: every 1-D parameter
        # (biases, LayerNorm scales) and every *cls_token gets weight_decay=0.
        backbone_lr     = self.hparams['opt_kwargs'].get('lr', 1e-5)
        backbone_max_lr = self.hparams['sch_kwargs'].get('backbone_max_lr', 5e-4)
        backbone_wd     = self.hparams['opt_kwargs'].get('weight_decay', 0.05)
        graph_lr        = self.hparams['opt_kwargs'].get('graph_lr', 1e-5)
        graph_max_lr    = self.hparams['sch_kwargs'].get('graph_max_lr', 5e-4)
        graph_wd        = self.hparams['opt_kwargs'].get('graph_weight_decay', 0.05)
        betas = self.hparams['opt_kwargs'].get('betas', (0.9, 0.98))

        GRAPH_PREFIXES = ('graph_encoder', 'graph_modulator', 'alignment_head')
        buckets = {'bb_decay': [], 'bb_nodecay': [], 'gr_decay': [], 'gr_nodecay': []}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_graph   = n.startswith(GRAPH_PREFIXES)
            is_nodecay = (p.ndim <= 1) or ('cls_token' in n)
            buckets[('gr_' if is_graph else 'bb_') +
                    ('nodecay' if is_nodecay else 'decay')].append(p)

        spec = [('bb_decay',   backbone_lr, backbone_wd, backbone_max_lr),
                ('bb_nodecay', backbone_lr, 0.0,         backbone_max_lr),
                ('gr_decay',   graph_lr,    graph_wd,    graph_max_lr),
                ('gr_nodecay', graph_lr,    0.0,         graph_max_lr)]
        param_groups, max_lr_list = [], []
        for key, lr, wd, mlr in spec:
            if buckets[key]:
                param_groups.append({'params': buckets[key], 'lr': lr, 'weight_decay': wd})
                max_lr_list.append(mlr)
        print("[optim] " + "  ".join(f"{k}={len(v)}" for k, v in buckets.items()), flush=True)

        opt = torch.optim.AdamW(param_groups, betas=betas)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr_list,
            total_steps=self.hparams['sch_kwargs']['total_steps'],
            pct_start=self.hparams['sch_kwargs'].get('pct_start', 0.2),
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
    ch_order=["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6",
              "T6-O2", "T3-C3", "C3-CZ", "C4-CZ", "T4-C4", "FP1-F3", "F3-C3",
              "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"],
    clst_dim="TP", clst_pos_embed_dim="", n_clst=16,
    stack_struct="scf", num_heads=8, ff=4, dropout=0,
    have_crossnorm=False, across_pool_stack_struct="",
    n_ap_clst=0, ap_clst_dim="T",
    classifier_use_ap_clst=False, classifier_have_pos_embed=True,
    classifier_pos_embed_dim="TPN",
    classifier_stack_struct="sfsfsfsfsfsfsfsf",
    classifier_final_act="pool", n_class=6,
    use_cond=list(range(20)),
)


def main(args):
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_ds = ConnectivityHMSDatasetCached(args.data_root, args.train_csv, args.signal_cache, window_sec=10,
                                             connectivity_measure=args.connectivity_measure, pearson_cache_dir=args.pearson_cache, apply_mu_law=bool(args.mu_law))
    val_ds   = ConnectivityHMSDatasetCached(args.data_root, args.val_csv, args.signal_cache, window_sec=10,
                                             connectivity_measure=args.connectivity_measure, pearson_cache_dir=args.pearson_cache, apply_mu_law=bool(args.mu_law))
    test_ds  = ConnectivityHMSDatasetCached(args.data_root, args.test_csv, args.signal_cache, window_sec=10,
                                             connectivity_measure=args.connectivity_measure, pearson_cache_dir=args.pearson_cache, apply_mu_law=bool(args.mu_law))
    print(f"Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.devices * args.accumulate_grad_batches))

    _g = torch.Generator()
    _g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              generator=_g)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = PLGraphConditionedClassifierChannelwise(
        classifier_model_kwargs = CLASSIFIER_MODEL_KWARGS,
        opt_kwargs   = dict(
            lr=1e-5, weight_decay=0.05,
            graph_lr=args.graph_lr, graph_weight_decay=args.graph_weight_decay,
            betas=[0.9, 0.98],
        ),
        sch_kwargs   = dict(
            backbone_max_lr=5e-4,
            graph_max_lr=args.graph_max_lr,
            total_steps=steps_per_epoch*args.epochs,
            pct_start=0.1,
        ),
        n_class      = 6,
        lambda_align = args.lambda_align,
        use_graph    = args.use_graph,
        backbone_ckpt = args.backbone_ckpt,
        align_start_epoch = args.align_start_epoch,
        align_ramp_epochs = args.align_ramp_epochs,
        gcn_layers         = args.gcn_layers,
        graph_dim          = args.graph_dim,
        gcn_dropout        = args.gcn_dropout,
        augment_icoh       = bool(args.augment_icoh),
        random_tokens      = bool(args.random_tokens),
        film               = args.film,
        icoh_edge_dropout_p = args.icoh_edge_dropout_p,
        fix_extractor_fold = bool(args.fix_extractor_fold),
    )

    import os as _os
    _job_id = _os.environ.get("SLURM_JOB_ID", "local")
    ckpt_dir = f"checkpoint/{args.name}_{_job_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    total_steps = steps_per_epoch * args.epochs
    model.hparams['sch_kwargs']['total_steps'] = total_steps

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.name,
        save_dir="logs/wandb",
    )

    trainer = pl.Trainer(
        deterministic=True,
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator='gpu', devices=args.devices,
        strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto',
        precision='32-true', log_every_n_steps=10, num_sanity_val_steps=0,
        gradient_clip_val=3,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=False,
        default_root_dir=f'logs/{args.name}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/kappa', mode='max', save_top_k=1,
                                         dirpath=ckpt_dir, filename='best', save_last=True),
            EarlyStoppingWithWarmup(monitor="val/kappa", mode="max", patience=args.patience, warmup=args.es_warmup),
            PrintEpochCallback(),
        ]
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callbacks[0].best_model_path
    best = PLGraphConditionedClassifierChannelwise.load_from_checkpoint(best_ckpt, weights_only=False)
    results = trainer.test(best, test_loader)
    print(results)

    import wandb
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='graphcond_channelwise')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--train_csv', type=str, default='finetune_train')
    parser.add_argument('--val_csv',   type=str, default='finetune_val')
    parser.add_argument('--test_csv',  type=str, default='finetune_test')
    parser.add_argument('--connectivity_measure', type=str, default='icoh', choices=['icoh', 'pearson'],
                        help='Which connectivity statistic to use as the graph edge feature (ablation switch).')
    parser.add_argument('--pearson_cache', type=str, default='data/pearson')
    parser.add_argument('--mu_law', type=int, default=0,
                        help='Keep 0. Mu-law is already applied at cache time by precompute_signal_cache.py (crop -> /100 -> staged_mu_law), reproducing EEGDM div_100_staged_mu_law. Setting 1 would compand a second time.')
    parser.add_argument('--signal_cache', type=str,
                        default='data/signal_cache')
    parser.add_argument('--backbone_ckpt', type=str, default='checkpoints/backbone.ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--lambda_align', type=float, default=0.1)
    parser.add_argument('--align_start_epoch', type=int, default=5)
    parser.add_argument('--align_ramp_epochs', type=int, default=5)
    parser.add_argument('--graph_lr', type=float, default=1e-5)
    parser.add_argument('--graph_max_lr', type=float, default=5e-4)
    parser.add_argument('--graph_weight_decay', type=float, default=0.05)
    parser.add_argument('--use_graph', type=int, default=1)

    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--wandb_project', type=str, default='CGEEGDM')
    parser.add_argument('--wandb_group', type=str, default='GraphCondChannelwise')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)  # 32 x 2 GPUs x 4 = 256 global (paper)
    parser.add_argument('--es_warmup', type=int, default=20)  # paper: early stop only after epoch 20
    parser.add_argument('--gcn_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--graph_dim', type=int, default=128)
    parser.add_argument('--gcn_dropout', type=float, default=0.1)
    parser.add_argument('--augment_icoh', type=int, default=1)  # DropEdge on (training only)
    parser.add_argument('--random_tokens', type=int, default=0)  # 1 = edge-weights-only ablation
    parser.add_argument('--film', type=str, default='standard', choices=['standard', 'gated'])
    parser.add_argument('--icoh_edge_dropout_p', type=float, default=0.1)
    parser.add_argument('--fix_extractor_fold', type=int, default=0)
    args = parser.parse_args()
    args.use_graph = bool(args.use_graph)
    main(args)
