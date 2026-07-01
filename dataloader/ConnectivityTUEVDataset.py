import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
LABEL_COLS = ['seizure_vote','lpd_vote','gpd_vote',
              'lrda_vote','grda_vote','other_vote']

class ConnectivityHMSDataset(Dataset):
    """
    HMS dataset — returns all 19 channels together.
    Returns tuple:
      [0] signal    : [19, T]   float32  all channels
      [1] soft_label: [6]       float32  vote distribution
      [2] icoh_vec  : [171]     float32  precomputed iCOH upper triangle
    """
    def __init__(self, root, split, icoh_cache_dir, window_sec=10, fs=200):
        self.root           = root
        self.icoh_cache_dir = icoh_cache_dir
        self.window         = window_sec * fs
        self.fs             = fs
        self.channels       = HMS_CHANNELS

        csv_path = os.path.join(root, f"{split}.csv")
        self.df  = pd.read_csv(csv_path).reset_index(drop=True)
        self.eeg_dir = os.path.join(root, "train_eegs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        eeg_id = int(row.eeg_id)

        eeg_raw = pd.read_parquet(os.path.join(self.eeg_dir, f"{eeg_id}.parquet"))
        segs = []
        for ch in self.channels:
            sig_full = eeg_raw[ch].values.astype(np.float32) if ch in eeg_raw.columns else np.zeros(self.window, dtype=np.float32)
            sig_full = np.nan_to_num(sig_full, nan=0.0, posinf=0.0, neginf=0.0)
            mid  = len(sig_full) // 2
            half = self.window // 2
            seg  = sig_full[max(0,mid-half):min(len(sig_full),mid+half)]
            if len(seg) < self.window:
                seg = np.concatenate([seg, np.zeros(self.window-len(seg), dtype=np.float32)])
            segs.append(seg[:self.window] / 100.0)
        signal = torch.tensor(np.stack(segs), dtype=torch.float32)  # [19, T]

        cache_path = os.path.join(self.icoh_cache_dir, f"{eeg_id}.pt")
        icoh_vec = torch.load(cache_path, weights_only=True)['icoh_vector'] if os.path.exists(cache_path) else torch.zeros(171, dtype=torch.float32)

        votes = row[LABEL_COLS].values.astype(np.float32)
        total = votes.sum()
        label = votes / total if total > 0 else np.ones(6, dtype=np.float32) / 6

        return signal, torch.tensor(label, dtype=torch.float32), icoh_vec
