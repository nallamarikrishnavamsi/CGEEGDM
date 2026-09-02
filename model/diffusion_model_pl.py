import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from .diffusion_model import Wavenet
from diffusers import DDPMScheduler
from ema_pytorch import EMA
from .util import setup_optimizer
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
        self.noise_sch = DDPMScheduler(**noise_sch_kwargs)

        self.gen_save_dir = os.path.join(gen_kwargs["root"], gen_kwargs["save_dir"])
        self.pred_target = noise_sch_kwargs["prediction_type"]
        if not os.path.exists(self.gen_save_dir):
            os.makedirs(self.gen_save_dir)

    def configure_optimizers(self):
        return setup_optimizer(self.model, **self.hparams["opt_kwargs"])
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-4, epochs=epoch, steps_per_epoch=, pct_start=0.1)
        # return {"optimizer": optimizer, "scheduler": scheduler}

    def forward_sample(self, batch, times=None, noiseless=False):
        bs = batch.shape[0]
        noise = torch.randn_like(batch) if not noiseless else torch.zeros_like(batch)
        if times is None:
            times = torch.randint(0, self.hparams["noise_sch_kwargs"]["num_train_timesteps"], (bs, 1), device=batch.device)
        noisy_signal = self.noise_sch.add_noise(batch, noise, times)
        return noisy_signal, noise, times

    def get_pred_target(self, batch, noise, times):
        if self.pred_target == "sample":
            return batch
        elif self.pred_target == "epsilon":
            return noise
        elif self.pred_target == "v_prediction":
            return self.noise_sch.get_velocity(batch, noise, times)

    def training_step(self, batch_input, batch_idx):
        # batch_input = self.transfer_batch_to_device(batch_input, self.device, batch_idx)

        batch = batch_input[0]
        label = batch_input[1].view(-1, 1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        noisy_signal, noise, times = self.forward_sample(batch)
        target = self.get_pred_target(batch, noise, times)

        pred = self.model(noisy_signal, times, label, local_cond)

        loss = F.mse_loss(pred, target)

        self.log(
            "train/mse_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
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

        noisy_signal, noise, times = self.forward_sample(batch)
        target = self.get_pred_target(batch, noise, times)

        pred = self.ema(noisy_signal, times, label, local_cond)


        loss = F.mse_loss(pred, target)
        
        self.log(
            "val/mse_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    # def test_step(self, batch_input, batch_idx):
    #     # validate via MSE loss: does it make sense for diffusion model?
        
    #     batch = batch_input[0]
    #     label = batch_input[1].view(-1, 1)

    #     noisy_signal, noise, times = self.forward_sample(batch)
    #     target = self.get_pred_target(batch, noise, times)

    #     pred = self.model(noisy_signal, times, label)

    #     loss = F.mse_loss(pred, target)
        
    #     self.log("test/mse_loss", loss, on_epoch=True, on_step=False)


    @torch.no_grad()
    @rank_zero_only
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # NOTE: original reverse-diffusion sampling/KLD logic intentionally
        # removed for HMS pretraining wall-clock budget. See
        # GROUND_TRUTH_EXCEPTIONS.md, section 2. Does not affect checkpoint
        # selection (monitors val/mse_loss).
        return
