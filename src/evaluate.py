import os, sys, torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.classifier_connectivity import PLClassifierConnectivity

ICOH_CACHE = "./data/icoh_cache"
DATA_ROOT  = "./data/hms"
BATCH_SIZE = 32

def evaluate(ckpt_path, split="test_split"):
    print(f"Evaluating: {ckpt_path}")
    model = PLClassifierConnectivity.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()

    ds     = ConnectivityHMSDataset(DATA_ROOT, split, ICOH_CACHE, window_sec=10)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    results = trainer.test(model, loader)
    return results[0] if results else {}

def evaluate_all_ablations(ablation_dir="checkpoint/ablation"):
    results = {}
    for name in sorted(os.listdir(ablation_dir)):
        ckpt_dir = os.path.join(ablation_dir, name)
        # Find best checkpoint
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and 'phase2' in f]
        if not ckpts:
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not ckpts:
            print(f"No checkpoint found for {name}")
            continue
        ckpt_path = os.path.join(ckpt_dir, ckpts[0])
        try:
            r = evaluate(ckpt_path)
            results[name] = r
            print(f"{name}: kappa={r.get('test/kappa', 'N/A'):.4f}  bacc={r.get('test/bacc', 'N/A'):.4f}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
            results[name] = {}

    df = pd.DataFrame(results).T
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    print(df.round(4).to_string())
    df.to_csv('logs/eval_results.csv')
    print("Saved: logs/eval_results.csv")
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--split', type=str, default='test_split')
    args = parser.parse_args()

    if args.all:
        evaluate_all_ablations()
    elif args.ckpt:
        r = evaluate(args.ckpt, args.split)
        print(r)
    else:
        print("Use --ckpt <path> or --all")
