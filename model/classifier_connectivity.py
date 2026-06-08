import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from ema_pytorch import EMA
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score
)
from .diffusion_model_pl import PLDiffusionModel
from .connectivity_encoder import ConnectivityEncoder
from .util import setup_optimizer


def supcon_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss.
    embeddings: [B, D] L2-normalized
    labels:     [B]    hard class labels
    """
    B = embeddings.shape[0]
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]
    # Mask: same class = positive, diagonal = self (exclude)
    labels = labels.unsqueeze(0)
    pos_mask  = (labels == labels.T).float()
    self_mask = torch.eye(B, device=embeddings.device)
    pos_mask  = pos_mask * (1 - self_mask)
    # No positives → skip
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    # Log-sum-exp over all non-self
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return loss.mean()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PLClassifierConnectivity(pl.LightningModule):
    def __init__(
        self,
        pretrain_checkpoint,      # path to connectivity pretrained backbone
        model_kwargs,             # classifier model kwargs (same as EEGDM)
        ema_kwargs,
        opt_kwargs,
        sch_kwargs,
        n_class=6,
        lambda_supcon=0.1,
        proj_dim=128,
        freeze_backbone=True,     # Phase 1: freeze, Phase 2: unfreeze
        use_kl=True,
        use_supcon=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_supcon  = lambda_supcon
        self.use_kl         = use_kl
        self.use_supcon     = use_supcon

        # Load pretrained connectivity backbone
        pretrain_model = PLDiffusionModel.load_from_checkpoint(
            pretrain_checkpoint, map_location='cpu'
        )
        backbone = pretrain_model.ema.ema_model  # Wavenet with iCOH

        # Connectivity encoder (also pretrained)
        self.conn_encoder = pretrain_model.conn_encoder \
            if hasattr(pretrain_model, 'conn_encoder') \
            else ConnectivityEncoder(in_dim=171, hidden_dim=256, out_dim=256)

        # Import Classifier from EEGDM
        from .classifier import Classifier
        self.model = Classifier(model=backbone, **model_kwargs)

        if ema_kwargs is not None:
            self.ema = EMA(
                self.model,
                ignore_startswith_names={"extractor", "reducer"},
                **ema_kwargs
            )
        else:
            self.ema = self.model

        self.noise_sch = pretrain_model.noise_sch

        # Projection head for SupCon
        lft_dim = model_kwargs.get('d_embed', None)
        if lft_dim is None:
            # classifier_final_act='pool' outputs [B, n_class] directly
            lft_dim = n_class
            self.cls_head = nn.Linear(n_class, n_class)  # trainable even when backbone frozen
        else:
            self.cls_head = nn.Linear(lft_dim, n_class)
        self.proj_head = ProjectionHead(in_dim=lft_dim, out_dim=proj_dim)

        # Freeze backbone Phase 1
        if freeze_backbone:
            for p in self.model.extractor.parameters():
                p.requires_grad = False
            if self.conn_encoder is not None:
                for p in self.conn_encoder.parameters():
                    p.requires_grad = False

        # Metrics
        self.val_metrics = MetricCollection({
            'kappa': MulticlassCohenKappa(num_classes=n_class, validate_args=False),
            'bacc' : MulticlassAccuracy(num_classes=n_class, average='macro', validate_args=False),
            'wf1'  : MulticlassF1Score(num_classes=n_class, average='weighted', validate_args=False),
        }, prefix='val/')
        self.test_metrics = self.val_metrics.clone(prefix='test/')

    def _get_embedding(self, signal, icoh_vec):
        # icoh_vec → connectivity embedding
        icoh_emb = self.conn_encoder(icoh_vec) if self.conn_encoder is not None else None
        # Set icoh_embed on extractor so calc_cond can access it with fold support
        self.model.extractor._icoh_embed = icoh_emb
        emb = self.model((signal, None))
        self.model.extractor._icoh_embed = None  # cleanup
        return emb  # [B, D]

    def _shared_step(self, batch):
        signal, soft_label, icoh_vec = batch
        # signal:     [B, 1, T]
        # ch_label:   [B, 1]
        # soft_label: [B, 6]
        # icoh_vec:   [B, 171]
        emb    = self._get_embedding(signal, icoh_vec)
        logits = self.cls_head(emb)                       # [B, 6]
        z_proj = self.proj_head(emb)                      # [B, 128]

        # KL divergence loss (soft labels)
        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            soft_label, reduction='batchmean'
        ) if self.use_kl else F.cross_entropy(logits, soft_label.argmax(dim=-1))

        # SupCon loss (hard labels from argmax of soft)
        hard_label = soft_label.argmax(dim=-1)
        sc_loss = supcon_loss(z_proj, hard_label) \
            if self.use_supcon else torch.tensor(0.0, device=logits.device)

        loss = kl_loss + self.lambda_supcon * sc_loss
        return loss, kl_loss, sc_loss, logits, soft_label

    def training_step(self, batch, batch_idx):
        loss, kl, sc, logits, soft_label = self._shared_step(batch)
        self.log('train/loss',  loss, prog_bar=True)
        self.log('train/kl',    kl)
        self.log('train/supcon',sc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl, sc, logits, soft_label = self._shared_step(batch)
        hard = soft_label.argmax(dim=-1)
        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, hard)
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, kl, sc, logits, soft_label = self._shared_step(batch)
        hard  = soft_label.argmax(dim=-1)
        preds = logits.argmax(dim=-1)
        self.test_metrics.update(preds, hard)
        self.log('test/loss', loss)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.test_metrics.reset()

    def unfreeze_backbone(self):
        for p in self.model.extractor.parameters():
            p.requires_grad = True
        if self.conn_encoder is not None:
            for p in self.conn_encoder.parameters():
                p.requires_grad = True
        print("Backbone unfrozen for Phase 2")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.hparams['opt_kwargs']
        )
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams['sch_kwargs']['max_lr'],
            total_steps=self.hparams['sch_kwargs']['total_steps'],
            pct_start=self.hparams['sch_kwargs'].get('pct_start', 0.1)
        )
        return [opt], [{'scheduler': sch, 'interval': 'step'}]
