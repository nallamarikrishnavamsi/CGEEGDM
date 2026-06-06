import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from model.diffusion_model import Wavenet
from diffusers import DDPMScheduler
from ema_pytorch import EMA
from model.util import setup_optimizer
import os
from einops import rearrange
import mne
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class PLDiffusionModel(pl.LightningModule):
    def __init__(self, model_kwargs, ema_kwargs, noise_sch_kwargs, opt_kwargs, gen_kwargs, target_dist=None, kde_kwargs=None):
        if target_dist is not None: assert kde_kwargs is not None
        super().__init__()
        self.save_hyperparameters()

        self.model = Wavenet(**model_kwargs)
        self.ema = EMA(self.model, **ema_kwargs)
        
    def loss(self, pred, target, mask):
        flipped_mask = mask - 1
        return (F.mse_loss(pred, target, reduction="none") * flipped_mask).sum() / flipped_mask.sum()

    def configure_optimizers(self):
        return setup_optimizer(self.model, **self.hparams["opt_kwargs"])

    def training_step(self, batch_input, batch_idx):
        batch = batch_input[0]
        label = batch_input[1].view(-1, 1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        noisy_signal, mask, times = self.forward_sample(batch)
        target = batch

        pred = self.model(noisy_signal, times, label, local_cond)

        loss = self.loss(pred, target, mask)
        self.log("train/mse_loss", loss, on_epoch=True, on_step=False)
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()
    
    @torch.no_grad()
    def validation_step(self, batch_input, batch_idx):
        # validate via MSE loss: does it make sense for diffusion model?
        
        batch = batch_input[0]
        label = batch_input[1].view(-1, 1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        noisy_signal, mask, times = self.forward_sample(batch)
        target = batch

        pred = self.ema(noisy_signal, times, label, local_cond)

        loss = self.loss(pred, target, mask)
        
        self.log("val/mse_loss", loss, on_epoch=True, on_step=False)

    def forward_sample(self, batch, times=None, noiseless=False):
        bs = batch.shape[0]
        # HACK assume 5 sec segment
        mask = torch.randint(0, 2, (bs, 1, 5), device=batch.device).repeat_interleave(batch.shape[-1] // 5, dim=-1)
        noisy_signal = batch * mask
        noise = mask
        times = torch.zeros(bs, 1, device=batch.device)
        # print(batch.shape, mask.shape, times.shape)
        return noisy_signal, noise, times


import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
from dataloader.TUEVDataset import TUEVDataset
import os
from omegaconf import DictConfig
from hydra.utils import instantiate

def entry(config: DictConfig):
    pl.seed_everything(**config["rng_seeding"])

    trainer = instantiate(config["trainer"])

    model = PLDiffusionModel(
        model_kwargs=config["model"]["model_kwargs"],
        ema_kwargs=config["model"]["ema_kwargs"],
        noise_sch_kwargs=config["model"]["noise_sch_kwargs"],
        opt_kwargs=config["model"]["opt_kwargs"],
        gen_kwargs=instantiate(config["model"]["gen_kwargs"]),
        target_dist=None,
        kde_kwargs=None
    )

    data_config = instantiate(config["data"])
    train_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["train_dir"]),
            schema=data_config["schema"],
            stft_kwargs=data_config["stft_kwargs"]
        ), 
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )
    
    val_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["val_dir"]),
            schema=data_config["schema"],
            stft_kwargs=data_config["stft_kwargs"]
        ), 
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )
    
    # test_loader = torch.utils.data.DataLoader(
    #     TUEVDataset(
    #         os.path.join(data_config["root"], data_config["test_dir"]),
    #         schema=data_config["schema"]
    #     ), 
    #     batch_size=data_config["batch_size"],
    #     num_workers=data_config["num_workers"],
    # )
    
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)
