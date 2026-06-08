import os, sys, math
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.classifier_connectivity import PLClassifierConnectivity

PRETRAIN_CKPT = "./checkpoint/pretrain/hms_icoh/backbone_icoh.ckpt"
ORIG_CKPT = "./checkpoint/pretrain/hms_baseline/backbone_hms_baseline.ckpt"  # original EEGDM backbone
ICOH_CACHE    = "./data/icoh_cache"
DATA_ROOT     = "./data/hms"
BATCH_SIZE    = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20

# Ablation configs
ABLATIONS = {
    "B0_eegdm"        : dict(pretrain=ORIG_CKPT,    use_kl=False, use_supcon=False, use_icoh=False, shuffle_icoh=False, random_icoh=False),
    "B1_random_cond"  : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=False, use_icoh=True,  shuffle_icoh=False, random_icoh=True),
    "A1_shuffled_icoh": dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=False, use_icoh=True,  shuffle_icoh=True,  random_icoh=False),
    "A2_real_icoh"    : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=False, use_icoh=True,  shuffle_icoh=False, random_icoh=False),
    "A3_icoh_dropout" : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=False, use_icoh=True,  shuffle_icoh=False, random_icoh=False, ch_dropout=0.2),
    "A4_icoh_kl"      : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=False, use_icoh=True,  shuffle_icoh=False, random_icoh=False),
    "A5_icoh_supcon"  : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=True,  use_icoh=True,  shuffle_icoh=False, random_icoh=False),
    "A5b_kl_supcon"   : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=True,  use_icoh=True,  shuffle_icoh=False, random_icoh=False),
    "A6_full"         : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=True,  use_icoh=True,  shuffle_icoh=False, random_icoh=False, ch_dropout=0.2),
    "A7_frozen_enc"   : dict(pretrain=PRETRAIN_CKPT, use_kl=True,  use_supcon=True,  use_icoh=True,  shuffle_icoh=False, random_icoh=False, freeze_conn_enc=True),
}

BASE_MODEL_KWARGS = dict(
    start=0, end=None, diffusion_t=1,
    use_cond=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
    query=["gate"], reduce=["std"], rescale=False,
    L=2000, window_size=200, window_step=200,
    pool_merge="share", multi_query_merge="seq",
    d_embed=None, init_weight=False, embed_query=False,
    d_query_embed=None, have_ch_pos_embed=False,
    cat_ch_pos_embed=True, ch_pos_emb_sym="mirror",
    ch_order=["Fp1","F3","C3","P3","F7","T3","T5","O1",
              "Fz","Cz","Pz","Fp2","F4","C4","P4","F8","T4","T6","O2"],
    clst_dim="TP", clst_pos_embed_dim="", n_clst=16,
    stack_struct="scf", num_heads=8, ff=4, dropout=0.1,
    have_crossnorm=False, across_pool_stack_struct="",
    n_ap_clst=0, ap_clst_dim="T",
    classifier_use_ap_clst=False, classifier_have_pos_embed=True,
    classifier_pos_embed_dim="TPN",
    classifier_stack_struct="sfsfsfsfsfsfsfsf",
    classifier_final_act="pool", n_class=6
)

def run_one(name, cfg, train_loader, val_loader, test_loader, steps_per_epoch):
    print(f"\n{'='*60}\nAblation: {name}\n{'='*60}")
    ckpt_dir = f"checkpoint/ablation/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    total_steps_p1 = steps_per_epoch * PHASE1_EPOCHS
    total_steps_p2 = steps_per_epoch * PHASE2_EPOCHS

    model = PLClassifierConnectivity(
        pretrain_checkpoint = cfg['pretrain'],
        model_kwargs        = BASE_MODEL_KWARGS,
        ema_kwargs          = dict(beta=0.999, update_after_step=100, update_every=10),
        opt_kwargs          = dict(lr=1e-4, weight_decay=0.05, betas=[0.9, 0.98]),
        sch_kwargs          = dict(pct_start=0.1, max_lr=5e-4, total_steps=total_steps_p1),
        n_class             = 6,
        lambda_supcon       = 0.1,
        proj_dim            = 128,
        freeze_backbone     = True,
        use_kl              = cfg['use_kl'],
        use_supcon          = cfg['use_supcon'],
    )

    # Handle special conditions
    if cfg.get('random_icoh', False):
        model.random_icoh = True
    if cfg.get('shuffle_icoh', False):
        model.shuffle_icoh = True
    if cfg.get('freeze_conn_enc', False):
        for p in model.conn_encoder.parameters():
            p.requires_grad = False

    def make_trainer(max_epochs, ckpt_filename, patience):
        return pl.Trainer(
            max_epochs=max_epochs, accelerator='gpu', devices=1,
            precision='32-true', log_every_n_steps=10,
            num_sanity_val_steps=0, default_root_dir=f'logs/ablation/{name}',
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor='val/kappa', mode='max', save_top_k=1,
                    dirpath=ckpt_dir, filename=ckpt_filename
                ),
                pl.callbacks.EarlyStopping(
                    monitor='val/kappa', mode='max', patience=patience
                )
            ]
        )

    # Phase 1
    t1 = make_trainer(PHASE1_EPOCHS, 'phase1_best', patience=5)
    t1.fit(model, train_loader, val_loader)

    # Phase 2
    model.unfreeze_backbone()
    model.hparams['sch_kwargs']['total_steps'] = total_steps_p2
    t2 = make_trainer(PHASE2_EPOCHS, 'phase2_best', patience=8)
    t2.fit(model, train_loader, val_loader)

    best_ckpt = t2.checkpoint_callbacks[0].best_model_path
    best = PLClassifierConnectivity.load_from_checkpoint(best_ckpt, weights_only=False)
    results = t2.test(best, test_loader)
    return results[0] if results else {}

def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    os.makedirs('logs/ablation', exist_ok=True)

    train_ds = ConnectivityHMSDataset(DATA_ROOT, 'train_split', ICOH_CACHE, window_sec=10)
    val_ds   = ConnectivityHMSDataset(DATA_ROOT, 'val_split',   ICOH_CACHE, window_sec=10)
    test_ds  = ConnectivityHMSDataset(DATA_ROOT, 'test_split',  ICOH_CACHE, window_sec=10)
    print(f"Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_results = {}
    for name, cfg in ABLATIONS.items():
        try:
            r = run_one(name, cfg, train_loader, val_loader, test_loader, steps_per_epoch)
            all_results[name] = r
        except Exception as e:
            import traceback
            print(f"FAILED {name}: {e}")
            traceback.print_exc()
            all_results[name] = {}

    print(f"\n{'='*60}\nABLATION SUMMARY\n{'='*60}")
    df = pd.DataFrame(all_results).T
    print(df.round(4).to_string())
    df.to_csv('logs/ablation_results.csv')
    print("Saved: logs/ablation_results.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Run single config by name')
    args = parser.parse_args()

    if args.config:
        pl.seed_everything(42)
        torch.set_float32_matmul_precision("medium")
        os.makedirs("logs/ablation", exist_ok=True)
        train_ds = ConnectivityHMSDataset(DATA_ROOT, "train_split", ICOH_CACHE, window_sec=10)
        val_ds   = ConnectivityHMSDataset(DATA_ROOT, "val_split",   ICOH_CACHE, window_sec=10)
        test_ds  = ConnectivityHMSDataset(DATA_ROOT, "test_split",  ICOH_CACHE, window_sec=10)
        steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        cfg = ABLATIONS[args.config]
        r   = run_one(args.config, cfg, train_loader, val_loader, test_loader, steps_per_epoch)
        print(f"Result for {args.config}: {r}")
        pd.DataFrame([r], index=[args.config]).to_csv(f"logs/ablation_{args.config}.csv")
        print(f"Saved: logs/ablation_{args.config}.csv")
    else:
        main()
