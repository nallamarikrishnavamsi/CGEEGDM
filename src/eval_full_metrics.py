"""
Full evaluation: KL-divergence, macro AUROC, macro F1, plus existing
kappa/bacc/wf1, on a given checkpoint against the test set.
Usage:
    python src/eval_full_metrics.py --ckpt checkpoint/seed43_combined_5433/best.ckpt
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAUROC, MulticlassF1Score, MulticlassCohenKappa, MulticlassAccuracy
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.finetune_graphcond import PLGraphConditionedClassifier
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached

def evaluate(ckpt_path, data_root, test_csv, signal_cache, batch_size=32, device='cuda'):
    model = PLGraphConditionedClassifier.load_from_checkpoint(ckpt_path, weights_only=False)
    model.eval()
    model.to(device)
    test_ds = ConnectivityHMSDatasetCached(data_root, test_csv, signal_cache, window_sec=10)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} samples")
    all_probs, all_soft_labels, all_hard_labels = [], [], []
    with torch.no_grad():
        for signal, soft_label, icoh_vec in test_loader:
            signal, soft_label, icoh_vec = signal.to(device), soft_label.to(device), icoh_vec.to(device)
            logits = model.ema((signal, None), icoh_vec, return_alignment=False)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_soft_labels.append(soft_label.cpu())
            all_hard_labels.append(soft_label.argmax(dim=-1).cpu())
    probs       = torch.cat(all_probs, dim=0)
    soft_labels = torch.cat(all_soft_labels, dim=0)
    hard_labels = torch.cat(all_hard_labels, dim=0)
    n_class = probs.shape[1]
    kl = F.kl_div(
        torch.log(probs.clamp(min=1e-8)), soft_labels, reduction='none'
    ).sum(dim=-1).mean().item()
    auroc = MulticlassAUROC(num_classes=n_class, average='macro')(probs, hard_labels).item()
    f1    = MulticlassF1Score(num_classes=n_class, average='macro')(probs, hard_labels).item()
    kappa = MulticlassCohenKappa(num_classes=n_class)(probs, hard_labels).item()
    bacc  = MulticlassAccuracy(num_classes=n_class, average='macro')(probs, hard_labels).item()
    print(f"\n=== Results: {ckpt_path} ===")
    print(f"N samples        : {len(test_ds)}")
    print(f"KL divergence    : {kl:.4f}")
    print(f"Macro AUROC      : {auroc:.4f}")
    print(f"Macro F1         : {f1:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print(f"Balanced Acc     : {bacc:.4f}")
    return dict(kl=kl, auroc=auroc, f1=f1, kappa=kappa, bacc=bacc, n=len(test_ds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/home/dsamantaai/krishna/data')
    parser.add_argument('--test_csv', type=str, default='full106k_test')
    parser.add_argument('--signal_cache', type=str, default='data/signal_cache')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate(args.ckpt, args.data_root, args.test_csv, args.signal_cache,
             batch_size=args.batch_size, device=device)
