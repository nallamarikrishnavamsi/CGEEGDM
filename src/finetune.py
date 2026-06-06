import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
from model.classifier_pl import PLClassifier as PLClassifier_v2
from dataloader.TUEVDataset import TUEVDataset as TUEVDataset
import os
from omegaconf import DictConfig
from hydra.utils import instantiate
import pickle
import random
import string
from tqdm import tqdm
import numpy as np

def entry(config: DictConfig):
    pl.seed_everything(**config["rng_seeding"])

    trainer = instantiate(config["trainer"])
    data_is_cached = config.get("data_is_cached", False)
    lmax = config["model"].get("set_lmax", None)
    if data_is_cached:
        metadata_inferred = {
            "diffusion_model_checkpoint": config["model"]["diffusion_model_checkpoint"],
            "diffusion_t": config["model"]["model_kwargs"]["diffusion_t"],
            "fwd_with_noise": config["model"]["fwd_with_noise"],
            "use_cond": config["model"]["model_kwargs"].get("use_cond", None),
            "query": config["model"]["model_kwargs"]["query"],
            "reduce": config["model"]["model_kwargs"]["reduce"],
            "rescale": config["model"]["model_kwargs"]["rescale"],
            "L": config["model"]["model_kwargs"]["L"],
            "window_size": config["model"]["model_kwargs"]["window_size"],
            "window_step": config["model"]["model_kwargs"]["window_step"],
            "pool_merge": config["model"]["model_kwargs"]["pool_merge"],
            "multi_query_merge": config["model"]["model_kwargs"]["multi_query_merge"],
        }

        if lmax is not None: metadata_inferred["lmax"] = lmax
        
        with open(os.path.join(config["data"]["root"], "metadata.pkl"), "rb") as m:
            metadata = pickle.load(m)
        assert metadata.keys() == metadata_inferred.keys()
        for k in metadata_inferred.keys(): assert metadata[k] == metadata_inferred[k]

    pl_cls = [None, None, PLClassifier_v2][config.get("pl_cls_version", 1)]
    model = pl_cls(
        diffusion_model_checkpoint=config["model"]["diffusion_model_checkpoint"],
        model_kwargs=config["model"]["model_kwargs"],
        ema_kwargs=config["model"]["ema_kwargs"],
        opt_kwargs=config["model"]["opt_kwargs"],
        sch_kwargs=config["model"]["sch_kwargs"],
        criterion_kwargs=config["model"]["criterion_kwargs"],
        fwd_with_noise=config["model"]["fwd_with_noise"],
        data_is_cached=data_is_cached,
        run_test_together=config["model"]["run_test_together"],
        test_data_is_cached=config["model"].get("test_data_is_cached", False),
        cls_version=config["model"]["cls_version"],
        lrd_kwargs=config["model"]["lrd_kwargs"],
        is_binary=config["model"].get("is_binary", False),
        is_multibinarylabel=config["model"].get("is_multibinarylabel", False),
        is_multilabel=config["model"].get("is_multilabel", False)
    )

    if lmax is not None:
        for l in model.model.extractor.model.layers: l.layer.L = lmax

    data_config = instantiate(config["data"])


    if data_is_cached:
        train_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["train_dir"]),
                schema=data_config["schema"],
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )

        val_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["val_dir"]),
                schema=data_config["schema"],
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )

    else:
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
    
    # Reduce train/val on the fly
    scarce = data_config.get("scarce", None)
    if scarce is not None:
        scarce_pct = scarce["pct"]
        scarce_mode = scarce["mode"]
        scarce_seed = scarce["seed"]
        n_class = 2 if config["model"].get("is_binary", False) else config["model"]["model_kwargs"]["n_class"]
        train_loader.dataset.setup_scarce(scarce_seed, scarce_mode, scarce_pct, n_class)
        val_loader.dataset.setup_scarce(scarce_seed, scarce_mode, scarce_pct, n_class)
        
    test_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["test_dir"]),
            schema=data_config.get("test_schema", data_config["schema"]),
            stft_kwargs=data_config["stft_kwargs"]
        ), 
        # batch_size=data_config["batch_size"],
        # num_workers=data_config["num_workers"],
        batch_size=16,
        num_workers=2,
    )
    
    if config["model"]["run_test_together"]:
        trainer.fit(model, train_loader, [val_loader, test_loader])
    else:
        trainer.fit(model, train_loader, val_loader)
        best_model = pl_cls.load_from_checkpoint(
            trainer.checkpoint_callbacks[0].best_model_path
        )
        # pl.Trainer(devices=config["trainer"]["devices"][:1]).test(best_model, test_loader)
        trainer.test(best_model, test_loader)