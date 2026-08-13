"""
Per-class evaluation: F1 and Cohen's Kappa broken down by HMS class
(Seizure, LPD, GPD, LRDA, GRDA, Other), comparing multiple checkpoints
on the same test set.

Usage:
    python src/evaluate_per_class.py \
        --ckpts baseline:checkpoint/baseline_5200/best.ckpt \
                gc_combined:checkpoint/gc_combined_5308/best.ckpt \
        --data_root /home/dsamantaai/krishna/data \
        --test_csv full106k_test \
        --signal_cache data/signal_cache \
        --out figures/per_class_breakdown
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from src.finetune_graphcond import PLGraphConditionedClassifier

LABEL_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']


@torch.no_grad()
def get_predictions(ckpt_path, data_root, test_csv, signal_cache, device='cuda'):
    print(f"Loading checkpoint: {ckpt_path}")
    model = PLGraphConditionedClassifier.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to(device)

    test_ds = ConnectivityHMSDatasetCached(data_root, test_csv, signal_cache, window_sec=10)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    for batch in loader:
        signal, soft_label, icoh_vec = batch
        signal   = signal.to(device)
        icoh_vec = icoh_vec.to(device)

        # Use EMA weights, matching validation_step/test_step convention
        logits, _, _ = model.ema(
            (signal, None), icoh_vec, return_alignment=True, warmup_alpha=1.0
        )
        preds = logits.argmax(dim=-1).cpu().numpy()
        hard  = soft_label.argmax(dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(hard)

    return np.concatenate(all_preds), np.concatenate(all_labels)


def per_class_report(name, preds, labels):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    overall_kappa = cohen_kappa_score(labels, preds)
    overall_f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    print(f"Overall: kappa={overall_kappa:.4f}  "
          f"macro_f1={overall_f1_macro:.4f}  weighted_f1={overall_f1_weighted:.4f}")

    f1_per_class = f1_score(labels, preds, average=None,
                            labels=list(range(6)), zero_division=0)

    print(f"\n{'Class':<12}{'F1':>8}{'Support':>10}")
    class_kappas = {}
    for i, cls_name in enumerate(LABEL_NAMES):
        support = int((labels == i).sum())
        print(f"{cls_name:<12}{f1_per_class[i]:>8.4f}{support:>10}")

        # One-vs-rest kappa per class: binarize labels/preds for this class
        bin_labels = (labels == i).astype(int)
        bin_preds  = (preds  == i).astype(int)
        if bin_labels.sum() > 0:
            class_kappas[cls_name] = cohen_kappa_score(bin_labels, bin_preds)
        else:
            class_kappas[cls_name] = float('nan')

    print(f"\n{'Class':<12}{'One-vs-rest Kappa':>20}")
    for cls_name in LABEL_NAMES:
        print(f"{cls_name:<12}{class_kappas[cls_name]:>20.4f}")

    return {
        'overall_kappa': overall_kappa,
        'overall_f1_macro': overall_f1_macro,
        'overall_f1_weighted': overall_f1_weighted,
        'f1_per_class': dict(zip(LABEL_NAMES, f1_per_class.tolist())),
        'kappa_per_class': class_kappas,
    }


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for spec in args.ckpts:
        name, ckpt_path = spec.split(':', 1)
        preds, labels = get_predictions(
            ckpt_path, args.data_root, args.test_csv, args.signal_cache, device)
        results[name] = per_class_report(name, preds, labels)

    # Comparison table: F1 per class, side by side
    names = list(results.keys())
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE — F1 per class")
    print(f"{'='*80}")
    header = f"{'Class':<12}" + "".join(f"{n:>15}" for n in names)
    print(header)
    for cls_name in LABEL_NAMES:
        row = f"{cls_name:<12}"
        for n in names:
            row += f"{results[n]['f1_per_class'][cls_name]:>15.4f}"
        print(row)

    print(f"\n{'Overall kappa':<12}" + "".join(f"{results[n]['overall_kappa']:>15.4f}" for n in names))
    print(f"{'Macro F1':<12}" + "".join(f"{results[n]['overall_f1_macro']:>15.4f}" for n in names))

    # Save to file
    out_path = os.path.join(args.out, 'per_class_results.txt')
    with open(out_path, 'w') as f:
        f.write(header + "\n")
        for cls_name in LABEL_NAMES:
            row = f"{cls_name:<12}"
            for n in names:
                row += f"{results[n]['f1_per_class'][cls_name]:>15.4f}"
            f.write(row + "\n")
        f.write(f"\n{'Overall kappa':<12}" +
                "".join(f"{results[n]['overall_kappa']:>15.4f}" for n in names) + "\n")
        f.write(f"{'Macro F1':<12}" +
                "".join(f"{results[n]['overall_f1_macro']:>15.4f}" for n in names) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='List of name:path pairs, e.g. baseline:checkpoint/baseline_5200/best.ckpt')
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--test_csv', type=str, default='full106k_test')
    parser.add_argument('--signal_cache', type=str, default='data/signal_cache')
    parser.add_argument('--out', type=str, default='figures/per_class_breakdown')
    args = parser.parse_args()
    main(args)
