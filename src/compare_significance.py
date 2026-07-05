"""
Statistical significance test comparing Baseline vs GraphCond test predictions.
Uses paired bootstrap resampling on per-sample correctness to get a
confidence interval on the kappa difference, plus McNemar's test on
right/wrong disagreement counts.

Usage:
    python src/compare_significance.py \
        --baseline_ckpt checkpoint/baseline/best.ckpt \
        --graphcond_ckpt checkpoint/graphcond_full106k/best.ckpt \
        --data_root /home/dsamantaai/krishna/data \
        --test_csv full106k_test \
        --signal_cache data/signal_cache
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from src.finetune_graphcond import PLGraphConditionedClassifier


@torch.no_grad()
def get_predictions(ckpt_path, data_root, test_csv, signal_cache, device='cuda'):
    model = PLGraphConditionedClassifier.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval().to(device)

    test_ds = ConnectivityHMSDatasetCached(data_root, test_csv, signal_cache, window_sec=10)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    for signal, soft_label, icoh_vec in loader:
        signal, icoh_vec = signal.to(device), icoh_vec.to(device)
        logits = model.model((signal, None), icoh_vec)
        preds = logits.argmax(dim=-1).cpu().numpy()
        hard  = soft_label.argmax(dim=-1).numpy()
        all_preds.append(preds)
        all_labels.append(hard)

    return np.concatenate(all_preds), np.concatenate(all_labels)


def bootstrap_kappa_diff(preds_a, preds_b, labels, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        k_a = cohen_kappa_score(labels[idx], preds_a[idx])
        k_b = cohen_kappa_score(labels[idx], preds_b[idx])
        diffs.append(k_b - k_a)
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs > 0).mean())
    return diffs.mean(), ci_low, ci_high, p_value


def mcnemar_test(preds_a, preds_b, labels):
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b01 = np.sum(correct_a & ~correct_b)  # a right, b wrong
    b10 = np.sum(~correct_a & correct_b)  # a wrong, b right
    if b01 + b10 == 0:
        return 0.0, 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    from scipy.stats import chi2
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p


def main(args):
    print("Extracting Baseline predictions...")
    preds_base, labels = get_predictions(args.baseline_ckpt, args.data_root, args.test_csv, args.signal_cache)
    print("Extracting GraphCond predictions...")
    preds_gc, _ = get_predictions(args.graphcond_ckpt, args.data_root, args.test_csv, args.signal_cache)

    kappa_base = cohen_kappa_score(labels, preds_base)
    kappa_gc   = cohen_kappa_score(labels, preds_gc)

    print(f"\nBaseline  kappa: {kappa_base:.4f}")
    print(f"GraphCond kappa: {kappa_gc:.4f}")
    print(f"Difference     : {kappa_gc - kappa_base:+.4f}")

    mean_diff, ci_low, ci_high, p_boot = bootstrap_kappa_diff(preds_base, preds_gc, labels)
    print(f"\nBootstrap (n=2000): mean diff={mean_diff:+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}], p={p_boot:.4f}")

    stat, p_mcnemar = mcnemar_test(preds_base, preds_gc, labels)
    print(f"McNemar's test    : chi2={stat:.4f}, p={p_mcnemar:.4f}")

    sig = "SIGNIFICANT" if (ci_low > 0 or ci_high < 0) else "not significant"
    print(f"\nResult: difference is {sig} at 95% confidence")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', default='checkpoint/baseline/best.ckpt')
    parser.add_argument('--graphcond_ckpt', default='checkpoint/graphcond_full106k/best.ckpt')
    parser.add_argument('--data_root', default='/home/dsamantaai/krishna/data')
    parser.add_argument('--test_csv', default='full106k_test')
    parser.add_argument('--signal_cache', default='data/signal_cache')
    args = parser.parse_args()
    main(args)
