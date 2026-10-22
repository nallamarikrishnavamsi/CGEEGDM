"""Does the graph branch carry sample-specific information, and does FiLM do anything?"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from model.graph_utils import vector_to_adjacency
from src.finetune_graphcond import PLGraphConditionedClassifier
from src.finetune_graphcond_channelwise import PLGraphConditionedClassifierChannelwise

p = argparse.ArgumentParser()
p.add_argument('--ckpts', nargs='+', required=True)   # name:path:global|channelwise
p.add_argument('--n', type=int, default=16)
p.add_argument('--csv', default='full106k_val')
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = ConnectivityHMSDatasetCached('/home/dsamantaai/krishna/data', args.csv, 'data/signal_cache',
                                  window_sec=10, connectivity_measure='icoh',
                                  pearson_cache_dir='data/pearson')
sig, _, ico = next(iter(DataLoader(ds, batch_size=args.n, shuffle=False, num_workers=0)))

def offdiag(z):                      # mean cosine between DIFFERENT samples
    z = torch.nn.functional.normalize(z, dim=-1)
    C = z @ z.T; n = z.size(0)
    return C[~torch.eye(n, dtype=torch.bool, device=z.device)].mean().item()

for spec in args.ckpts:
    name, path, kind = spec.split(':')
    Cls = PLGraphConditionedClassifierChannelwise if kind == 'channelwise' else PLGraphConditionedClassifier
    m = Cls.load_from_checkpoint(path, weights_only=False).eval().to(dev)
    net = m.ema.ema_model
    with torch.no_grad():
        la  = net.classifier.extractor((sig.to(dev), None), rate=1)
        tok = net.classifier.reducer(la)
        adj = vector_to_adjacency(ico.to(dev), augment=False)
        nf  = tok.mean(dim=(1, 3)).reshape(tok.size(0), 20, -1)
        out = net.graph_encoder(adj, node_features=nf)
        g   = out[0] if isinstance(out, tuple) else out
        film = net.graph_modulator(tok, g)

    print(f"\n=== {name} ({kind}) ===")
    print(f"  adjacency: per-entry std across samples {adj.std(0).mean():.4f} (mean {adj.mean():.4f})")
    if g.dim() == 2:
        print(f"  GLOBAL graph_emb [B,{g.size(-1)}]  -> FiLM conditioning")
        print(f"    cos(g_i, g_j) i!=j      {offdiag(g):.6f}   <- ~1.0 = same signal for every sample")
        print(f"    per-dim std across samples {g.std(0).mean():.6f}  (unit-norm scale ~{1/g.size(-1)**0.5:.4f})")
    else:
        print(f"  CHANNEL-WISE node_emb {tuple(g.shape)} -> FiLM conditioning")
        across = sum(offdiag(g[:, c, :]) for c in range(g.size(1))) / g.size(1)
        within = sum(offdiag(g[b]) for b in range(g.size(0))) / g.size(0)
        print(f"    cos across SAMPLES, same node   {across:.6f}   <- ~1.0 = no per-sample info")
        print(f"    cos across NODES, same sample   {within:.6f}   <- ~1.0 = all 20 channels identical")
        print(f"    per-dim std across samples {g.std(0).mean():.6f}")
    rel = (film - tok).norm() / tok.norm()
    print(f"  FiLM effect: ||film-tok|| / ||tok|| = {rel:.6f}   <- ~0 = conditioning does nothing")
    del m
