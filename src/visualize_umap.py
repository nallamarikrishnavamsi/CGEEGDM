"""
UMAP visualization comparing B0 vs A2 representations.
Usage:
    python src/visualize_umap.py \
        --b0_cache data/cached_hms/B0 \
        --a2_cache data/cached_hms/A2 \
        --split test \
        --out logs/umap
"""
import os, sys, argparse, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
COLORS      = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628']

def load_cache(cache_dir, split, max_samples=500):
    files = sorted([
        os.path.join(cache_dir, split, f)
        for f in os.listdir(os.path.join(cache_dir, split))
        if f.endswith('.pkl')
    ])[:max_samples]

    embeddings, labels = [], []
    for fp in files:
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        # Flatten cached token to 1D
        emb = d['__cache_data__'].flatten()
        lab = np.argmax(d['__cache_label__'])
        embeddings.append(emb)
        labels.append(lab)

    return np.array(embeddings, dtype=np.float32), np.array(labels)


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except ImportError:
        os.system("pip install umap-learn --quiet")
        import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


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
    ax.legend(loc='upper right', fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)


def compute_cluster_metrics(coords, labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil = silhouette_score(coords, labels)
    dbi = davies_bouldin_score(coords, labels)
    return sil, dbi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b0_cache', default='data/cached_hms/B0')
    parser.add_argument('--a2_cache', default='data/cached_hms/A2')
    parser.add_argument('--split',    default='test')
    parser.add_argument('--out',      default='logs/umap')
    parser.add_argument('--max_samples', default=500, type=int)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading B0 embeddings...")
    b0_emb, b0_lab = load_cache(args.b0_cache, args.split, args.max_samples)
    print(f"  B0: {b0_emb.shape}")

    print("Loading A2 embeddings...")
    a2_emb, a2_lab = load_cache(args.a2_cache, args.split, args.max_samples)
    print(f"  A2: {a2_emb.shape}")

    print("Computing UMAP for B0...")
    b0_coords = compute_umap(b0_emb)

    print("Computing UMAP for A2...")
    a2_coords = compute_umap(a2_emb)

    # Cluster metrics
    b0_sil, b0_dbi = compute_cluster_metrics(b0_coords, b0_lab)
    a2_sil, a2_dbi = compute_cluster_metrics(a2_coords, a2_lab)

    print(f"\nCluster Quality Metrics:")
    print(f"{'Model':<20} {'Silhouette↑':<15} {'Davies-Bouldin↓'}")
    print(f"{'B0 (no iCOH)':<20} {b0_sil:<15.4f} {b0_dbi:.4f}")
    print(f"{'A2 (real iCOH)':<20} {a2_sil:<15.4f} {a2_dbi:.4f}")

    # Save metrics
    with open(os.path.join(args.out, 'cluster_metrics.txt'), 'w') as f:
        f.write(f"Model,Silhouette,Davies-Bouldin\n")
        f.write(f"B0,{b0_sil:.4f},{b0_dbi:.4f}\n")
        f.write(f"A2,{a2_sil:.4f},{a2_dbi:.4f}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('UMAP: EEGDM Representations (HMS Test Set)', fontsize=14)

    plot_umap(b0_coords, b0_lab,
              f'B0 — Original EEGDM\nSilhouette={b0_sil:.3f}  DBI={b0_dbi:.3f}',
              axes[0])
    plot_umap(a2_coords, a2_lab,
              f'A2 — Connectivity-Guided EEGDM\nSilhouette={a2_sil:.3f}  DBI={a2_dbi:.3f}',
              axes[1])

    plt.tight_layout()
    out_path = os.path.join(args.out, 'umap_b0_vs_a2.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
