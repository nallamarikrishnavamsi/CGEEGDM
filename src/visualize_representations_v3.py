"""
UMAP visualization comparing Baseline (no graph) vs GraphCond (with graph)
latent representations, extracted directly from trained checkpoints.

Usage:
    python src/visualize_representations.py \
        --baseline_ckpt checkpoint/baseline/best.ckpt \
        --graphcond_ckpt checkpoint/graphcond_full106k/best.ckpt \
        --data_root /home/dsamantaai/krishna/data \
        --test_csv full106k_test \
        --signal_cache data/signal_cache \
        --max_samples 1000 \
        --out logs/umap
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.graph_conditioned_classifier import GraphConditionedClassifier
from model.classifier import Classifier
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from src.finetune_graphcond import load_original_backbone, CLASSIFIER_MODEL_KWARGS, PLGraphConditionedClassifier
from src.finetune_graphcond_channelwise import PLGraphConditionedClassifierChannelwise

LABEL_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
COLORS      = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


@torch.no_grad()
def extract_embeddings(ckpt_path, data_root, test_csv, signal_cache, max_samples, device='cuda',
                        connectivity_measure='icoh', pearson_cache='data/pearson', model_type='global'):
    print(f"Loading checkpoint: {ckpt_path}")
    ModelCls = PLGraphConditionedClassifierChannelwise if model_type == 'channelwise' else PLGraphConditionedClassifier
    model = ModelCls.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to(device)

    test_ds = ConnectivityHMSDatasetCached(data_root, test_csv, signal_cache, window_sec=10,
                                            connectivity_measure=connectivity_measure,
                                            pearson_cache_dir=pearson_cache)
    # Seeded random subset (same samples for every model), not the first rows of the CSV
    idx = np.sort(np.random.default_rng(0).choice(len(test_ds), size=min(max_samples, len(test_ds)), replace=False))
    test_ds = torch.utils.data.Subset(test_ds, idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    embeddings, labels = [], []
    n_seen = 0
    for batch in loader:
        signal, soft_label, icoh_vec = batch
        signal   = signal.to(device)
        icoh_vec = icoh_vec.to(device)

        gc_model = model.ema.ema_model  # use EMA weights, matching validation_step/test_step convention

        # Replicate forward() up through FiLM (Step 1 + Step 2), stop before decode.
        # Must match the real forward() exactly: real per-channel node_feats derived
        # from tokens, NOT the untrained fallback node_embedding (that branch only
        # triggers when node_features=None, which never happens in real training).
        latent_activity = gc_model.classifier.extractor((signal, None), rate=1)
        tokens = gc_model.classifier.reducer(latent_activity)
        tokens_pre_film = tokens

        if gc_model.use_graph:
            from model.graph_utils import vector_to_adjacency
            import torch.nn.functional as TF
            adj = vector_to_adjacency(icoh_vec, augment=False)
            node_feats = tokens_pre_film.mean(dim=(1, 3)).reshape(
                tokens_pre_film.size(0), gc_model.graph_encoder.num_nodes, -1)
            if model_type == 'channelwise':
                # GraphEncoderChannelwise.forward() returns a tuple:
                # (node_embeddings [B,N,H], raw (no L2);
                #  pooled_for_align [B,H], alignment branch only -- unused here)
                node_embeddings, _pooled_for_align = gc_model.graph_encoder(adj, node_features=node_feats)
                tokens = gc_model.graph_modulator(tokens, node_embeddings)
            else:
                graph_emb = gc_model.graph_encoder(adj, node_features=node_feats)
                tokens = gc_model.graph_modulator(tokens, graph_emb)

        # Penultimate representation: decoder output that feeds the classification head
        # (the part every model actually trains; pre-decoder tokens are the frozen backbone)
        rep = gc_model.classifier.decoder(tokens)[gc_model.classifier.use_rep_idx]
        B = rep.size(0)
        pooled = rep.reshape(B, -1)

        embeddings.append(pooled.cpu().numpy())
        labels.append(soft_label.argmax(dim=-1).cpu().numpy())

        n_seen += B
        if n_seen >= max_samples:
            break

    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    return embeddings, labels


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def compute_cluster_metrics(embeddings, labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil = silhouette_score(embeddings, labels)
    dbi = davies_bouldin_score(embeddings, labels)
    return sil, dbi


def plot_umap(coords, labels, title, ax):
    for cls_idx in range(6):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[cls_idx], label=LABEL_NAMES[cls_idx],
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    specs = []
    for spec in args.ckpts:
        parts = spec.split(':')
        if len(parts) == 3:
            name, ckpt_path, model_type = parts
        elif len(parts) == 2:
            name, ckpt_path = parts
            model_type = 'global'
        else:
            raise ValueError(f"Invalid --ckpts spec: {spec!r} (expected name:path or name:path:type)")
        specs.append((name, ckpt_path, model_type))

    results = {}
    for name, ckpt_path, model_type in specs:
        print(f"=== Extracting {name} embeddings ({model_type}) ===")
        emb, lab = extract_embeddings(
            ckpt_path, args.data_root, args.test_csv,
            args.signal_cache, args.max_samples, device,
            connectivity_measure=args.connectivity_measure, pearson_cache=args.pearson_cache,
            model_type=model_type)
        coords = compute_umap(emb)
        sil, dbi = compute_cluster_metrics(emb, lab)
        results[name] = dict(emb=emb, lab=lab, coords=coords, sil=sil, dbi=dbi)

    print(f"\nSamples used for clustering metrics:")
    for name, r in results.items():
        print(f"  {name:<20}: {len(r['emb'])} samples")
    print(f"\n{'Model':<20}{'Silhouette':>12}{'Davies-Bouldin':>18}")
    for name, r in results.items():
        print(f"{name:<20}{r['sil']:>12.4f}{r['dbi']:>18.4f}")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        plot_umap(r['coords'], r['lab'],
                  f"{name} (n={len(r['emb'])}, Sil={r['sil']:.3f}, DBI={r['dbi']:.3f})", ax)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(args.out, 'umap_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Save raw metrics
    with open(os.path.join(args.out, 'cluster_metrics.txt'), 'w') as f:
        for name, r in results.items():
            f.write(f"{name}: silhouette={r['sil']:.4f}, davies_bouldin={r['dbi']:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='name:path or name:path:type (type=global|channelwise, default global)')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--test_csv', type=str, default='full106k_test')
    parser.add_argument('--signal_cache', type=str, default='data/signal_cache')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--connectivity_measure', type=str, default='icoh', choices=['icoh', 'pearson'])
    parser.add_argument('--pearson_cache', type=str, default='data/pearson')
    parser.add_argument('--out', type=str, default='logs/umap')
    args = parser.parse_args()
    main(args)
