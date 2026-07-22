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

LABEL_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
COLORS      = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


@torch.no_grad()
def extract_embeddings(ckpt_path, data_root, test_csv, signal_cache, max_samples, device='cuda'):
    print(f"Loading checkpoint: {ckpt_path}")
    model = PLGraphConditionedClassifier.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to(device)

    test_ds = ConnectivityHMSDatasetCached(data_root, test_csv, signal_cache, window_sec=10)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    embeddings, labels = [], []
    n_seen = 0
    for batch in loader:
        signal, soft_label, icoh_vec = batch
        signal   = signal.to(device)
        icoh_vec = icoh_vec.to(device)

        gc_model = model.model  # GraphConditionedClassifier

        # Replicate forward() up through FiLM (Step 1 + Step 2), stop before decode
        latent_activity = gc_model.classifier.extractor((signal, None), rate=1)
        tokens = gc_model.classifier.reducer(latent_activity)

        if gc_model.use_graph:
            from model.graph_utils import vector_to_adjacency
            adj = vector_to_adjacency(icoh_vec)
            graph_emb = gc_model.graph_encoder(adj)
            tokens = gc_model.graph_modulator(tokens, graph_emb)

        # Pool tokens to a single vector per sample: mean over all dims except batch and H
        B, H = tokens.size(0), tokens.size(-1)
        pooled = tokens.reshape(B, -1, H).mean(dim=1)  # [B, H]

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

    print("=== Extracting Baseline embeddings ===")
    emb_base, lab_base = extract_embeddings(
        args.baseline_ckpt, args.data_root, args.test_csv,
        args.signal_cache, args.max_samples, device)

    print("=== Extracting GraphCond embeddings ===")
    emb_gc, lab_gc = extract_embeddings(
        args.graphcond_ckpt, args.data_root, args.test_csv,
        args.signal_cache, args.max_samples, device)

    print("Computing UMAP projections...")
    coords_base = compute_umap(emb_base)
    coords_gc   = compute_umap(emb_gc)

    print("Computing cluster quality metrics...")
    sil_base, dbi_base = compute_cluster_metrics(emb_base, lab_base)
    sil_gc, dbi_gc     = compute_cluster_metrics(emb_gc, lab_gc)

    print(f"\nSamples used for clustering metrics:")
    print(f"  Baseline : {len(emb_base)} samples")
    print(f"  GraphCond: {len(emb_gc)} samples")
    print(f"\n{'Model':<15}{'Silhouette':>12}{'Davies-Bouldin':>18}")
    print(f"{'Baseline':<15}{sil_base:>12.4f}{dbi_base:>18.4f}")
    print(f"{'GraphCond':<15}{sil_gc:>12.4f}{dbi_gc:>18.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_umap(coords_base, lab_base,
              f"Baseline (n={len(emb_base)}, Sil={sil_base:.3f}, DBI={dbi_base:.3f})", axes[0])
    plot_umap(coords_gc, lab_gc,
              f"GraphCond (n={len(emb_gc)}, Sil={sil_gc:.3f}, DBI={dbi_gc:.3f})", axes[1])
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(args.out, 'umap_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Save raw metrics
    with open(os.path.join(args.out, 'cluster_metrics.txt'), 'w') as f:
        f.write(f"Baseline: silhouette={sil_base:.4f}, davies_bouldin={dbi_base:.4f}\n")
        f.write(f"GraphCond: silhouette={sil_gc:.4f}, davies_bouldin={dbi_gc:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', type=str, default='checkpoint/baseline/best.ckpt')
    parser.add_argument('--graphcond_ckpt', type=str, default='checkpoint/graphcond_full106k/best.ckpt')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--test_csv', type=str, default='full106k_test')
    parser.add_argument('--signal_cache', type=str, default='data/signal_cache')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--out', type=str, default='logs/umap')
    args = parser.parse_args()
    main(args)
