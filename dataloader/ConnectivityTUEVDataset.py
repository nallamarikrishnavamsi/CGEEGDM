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
    HMS dataset that returns:
      [0] signal      : [1, T]        float32  (single bipolar channel, EEGDM style)
      [1] ch_label    : [1]           long     (channel index, 0-18)
      [2] soft_label  : [6]           float32  (vote distribution)
      [3] icoh_vector : [171]         float32  (precomputed iCOH upper triangle)
    """
    def __init__(self, root, split, icoh_cache_dir, window_sec=50, fs=200):
        self.root           = root
        self.icoh_cache_dir = icoh_cache_dir
        self.window         = window_sec * fs
        self.fs             = fs
        self.channels       = HMS_CHANNELS

        csv_path = os.path.join(root, f"{split}.csv")
        self.df  = pd.read_csv(csv_path).reset_index(drop=True)
        self.eeg_dir = os.path.join(root, "train_eegs")

        # Expand: one row per channel per EEG
        rows = []
        for _, row in self.df.iterrows():
            for ch_idx, ch in enumerate(self.channels):
                rows.append({
                    'eeg_id'   : int(row.eeg_id),
                    'ch_idx'   : ch_idx,
                    'ch_name'  : ch,
                    **{c: row[c] for c in LABEL_COLS}
                })
        self.samples = pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row    = self.samples.iloc[idx]
        eeg_id = int(row.eeg_id)
        ch_idx = int(row.ch_idx)
        ch     = row.ch_name

        # Load EEG
        eeg_path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
        eeg_raw  = pd.read_parquet(eeg_path)
        sig_full = eeg_raw[ch].values.astype(np.float32) if ch in eeg_raw.columns else np.zeros(self.window, dtype=np.float32)
        sig_full = np.nan_to_num(sig_full, nan=0.0, posinf=0.0, neginf=0.0)

        # Center crop
        mid  = len(sig_full) // 2
        half = self.window // 2
        start = max(0, mid - half)
        end   = min(len(sig_full), mid + half)
        seg   = sig_full[start:end]
        if len(seg) < self.window:
            seg = np.concatenate([seg, np.zeros(self.window - len(seg), dtype=np.float32)])
        seg = seg[:self.window] / 100.0  # normalize like EEGDM

        signal = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # [1, T]

        # Load iCOH cache
        cache_path = os.path.join(self.icoh_cache_dir, f"{eeg_id}.pt")
        if os.path.exists(cache_path):
            cache     = torch.load(cache_path, weights_only=True)
            icoh_vec  = cache['icoh_vector']   # [171]
        else:
            icoh_vec  = torch.zeros(171, dtype=torch.float32)

        # Soft label
        votes = np.array([row[c] for c in LABEL_COLS], dtype=np.float32)
        total = votes.sum()
        label = votes / total if total > 0 else np.ones(6, dtype=np.float32) / 6
        soft_label = torch.tensor(label, dtype=torch.float32)

        ch_label = torch.tensor(ch_idx, dtype=torch.long).unsqueeze(0)  # [1]

        return signal, ch_label, soft_label, icoh_vec
