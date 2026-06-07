import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDataset
from model.classifier_connectivity import PLClassifierConnectivity

ICOH_CACHE  = "./data/icoh_cache"
DATA_ROOT   = "./data/hms"
BATCH_SIZE  = 32
SAVE_DIR    = "logs/viz"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

def load_model_and_data(ckpt_path, split='test_split', n_samples=500):
    model = PLClassifierConnectivity.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval().cuda()

    ds     = ConnectivityHMSDataset(DATA_ROOT, split, ICOH_CACHE, window_sec=10)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return model, loader

def plot_ablation_comparison(csv_path='logs/ablation_results.csv'):
    """Bar chart comparing ablation configurations."""
    df = pd.read_csv(csv_path, index_col=0)
    metrics = ['test/kappa', 'test/bacc', 'test/wf1']
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1: axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))

    for ax, metric in zip(axes, metrics):
        vals = df[metric].fillna(0)
        bars = ax.barh(df.index, vals, color=colors)
        ax.set_xlabel(metric.replace('test/', '').upper())
        ax.set_title(metric)
        ax.axvline(vals.max(), color='red', linestyle='--', alpha=0.5, label=f'Best: {vals.max():.4f}')
        for bar, val in zip(bars, vals):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
        ax.legend(fontsize=8)

    plt.suptitle('Ablation Study: Connectivity-Conditioned EEGDM', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'ablation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_icoh_matrix(eeg_id, icoh_cache_dir='./data/icoh_cache'):
    """Visualize iCOH connectivity matrix for one EEG."""
    cache = torch.load(f"{icoh_cache_dir}/{eeg_id}.pt", weights_only=True)
    A = cache['icoh_matrix'].numpy()

    HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                    'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(A, cmap='RdYlBu_r', vmin=0, vmax=A.max())
    plt.colorbar(im, ax=ax, label='iCOH')
    ax.set_xticks(range(19)); ax.set_xticklabels(HMS_CHANNELS, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(19)); ax.set_yticklabels(HMS_CHANNELS, fontsize=8)
    ax.set_title(f'Imaginary Coherence Matrix — EEG {eeg_id}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f'icoh_matrix_{eeg_id}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_tsne(ckpt_path, split='test_split', n_samples=1000):
    """t-SNE of learned representations colored by class."""
    model, loader = load_model_and_data(ckpt_path, split)

    embeddings, labels = [], []
    with torch.no_grad():
        for batch in loader:
            signal, ch_label, soft_label, icoh_vec = [x.cuda() for x in batch]
            # Get embedding from model
            emb = model.get_embedding(signal, ch_label, icoh_vec)
            embeddings.append(emb.cpu())
            labels.append(soft_label.argmax(dim=-1).cpu())
            if sum(len(e) for e in embeddings) >= n_samples:
                break

    emb = torch.cat(embeddings)[:n_samples].numpy()
    lbl = torch.cat(labels)[:n_samples].numpy()

    print(f"Running t-SNE on {len(emb)} samples...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab10(np.linspace(0, 1, 6))
    for i, name in enumerate(CLASS_NAMES):
        mask = lbl == i
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name, alpha=0.6, s=15)
    ax.legend(fontsize=10)
    ax.set_title('t-SNE of Learned Representations', fontsize=12)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'tsne_representations.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_confusion_matrix(ckpt_path, split='test_split'):
    """Confusion matrix on test set."""
    model, loader = load_model_and_data(ckpt_path, split)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            signal, ch_label, soft_label, icoh_vec = [x.cuda() for x in batch]
            logits = model(signal, ch_label, icoh_vec)
            preds  = logits.argmax(dim=-1).cpu()
            labels = soft_label.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_csv', type=str, default=None)
    parser.add_argument('--icoh_matrix',  type=int, default=None)
    parser.add_argument('--tsne',         type=str, default=None, help='ckpt path')
    parser.add_argument('--confusion',    type=str, default=None, help='ckpt path')
    parser.add_argument('--all',          type=str, default=None, help='ckpt path for tsne+confusion')
    args = parser.parse_args()

    if args.ablation_csv:
        plot_ablation_comparison(args.ablation_csv)
    if args.icoh_matrix:
        plot_icoh_matrix(args.icoh_matrix)
    if args.tsne:
        plot_tsne(args.tsne)
    if args.confusion:
        plot_confusion_matrix(args.confusion)
    if args.all:
        plot_tsne(args.all)
        plot_confusion_matrix(args.all)
        if os.path.exists('logs/ablation_results.csv'):
            plot_ablation_comparison('logs/ablation_results.csv')
