"""Graph-shuffle control: does the model's output depend on WHICH graph it gets?
Compares real graphs vs graphs permuted across the batch, on the same checkpoint."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from src.finetune_graphcond import PLGraphConditionedClassifier
from src.finetune_graphcond_channelwise import PLGraphConditionedClassifierChannelwise

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True); p.add_argument('--kind', default='global')
p.add_argument('--csv', default='full106k_test'); p.add_argument('--bs', type=int, default=32)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
Cls = PLGraphConditionedClassifierChannelwise if a.kind == 'channelwise' else PLGraphConditionedClassifier
m = Cls.load_from_checkpoint(a.ckpt, weights_only=False).eval().to(dev)
ds = ConnectivityHMSDatasetCached('/home/dsamantaai/krishna/data', a.csv, 'data/signal_cache',
                                  window_sec=10, connectivity_measure='icoh', pearson_cache_dir='data/pearson')
g = torch.Generator().manual_seed(0)
P = {k: ([], []) for k in ('real', 'shuffled', 'zeroed')}
with torch.no_grad():
    for sig, soft, ico in DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=2):
        sig, ico = sig.to(dev), ico.to(dev)
        variants = {'real': ico,
                    'shuffled': ico[torch.randperm(ico.size(0), generator=g).to(dev)],
                    'zeroed': torch.zeros_like(ico)}
        for k, v in variants.items():
            lg, _, _ = m.ema((sig, None), v, return_alignment=True)
            P[k][0].append(lg.argmax(-1).cpu().numpy()); P[k][1].append(soft.argmax(-1).numpy())
for k, (pr, la) in P.items():
    pr, la = np.concatenate(pr), np.concatenate(la)
    print(f"{k:9s} kappa={cohen_kappa_score(la, pr):.4f}  "
          f"macro_f1={f1_score(la, pr, average='macro', zero_division=0):.4f}")
print("\nIf real ~= shuffled ~= zeroed, the graph branch carries no information.")
