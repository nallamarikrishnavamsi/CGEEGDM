"""Is the cosine alignment objective collapsed? Compares a λ_align=0.1 checkpoint
against a λ_align=0 one. Collapse => z vectors from DIFFERENT samples are ~identical."""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from src.finetune_graphcond import PLGraphConditionedClassifier

p = argparse.ArgumentParser()
p.add_argument('--ckpts', nargs='+', required=True)      # name:path
p.add_argument('--n', type=int, default=32)
p.add_argument('--csv', default='full106k_val')
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = ConnectivityHMSDatasetCached('/home/dsamantaai/krishna/data', args.csv,
                                  'data/signal_cache', window_sec=10,
                                  connectivity_measure='icoh', pearson_cache_dir='data/pearson')
sig, _, ico = next(iter(DataLoader(ds, batch_size=args.n, shuffle=False, num_workers=0)))

for spec in args.ckpts:
    name, path = spec.split(':', 1)
    m = PLGraphConditionedClassifier.load_from_checkpoint(path, weights_only=False).eval().to(dev)
    with torch.no_grad():
        _, zt, zg = m.ema((sig.to(dev), None), ico.to(dev), return_alignment=True)
    def offdiag_cos(z):
        C = z @ z.T
        n = z.size(0)
        return C[~torch.eye(n, dtype=torch.bool, device=z.device)].mean().item()
    paired = (zt * zg).sum(-1)
    print(f"\n{name}")
    print(f"  cos(z_token_i, z_graph_i)      mean {paired.mean():.6f}  min {paired.min():.6f}")
    print(f"  cos(z_token_i, z_token_j) i!=j mean {offdiag_cos(zt):.6f}   <- ~1.0 means COLLAPSED")
    print(f"  cos(z_graph_i, z_graph_j) i!=j mean {offdiag_cos(zg):.6f}   <- ~1.0 means COLLAPSED")
    print(f"  per-dim std across samples: z_token {zt.std(0).mean():.6f}  z_graph {zg.std(0).mean():.6f}")
    del m
