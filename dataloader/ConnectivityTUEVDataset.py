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
    Returns per EEG row:
      [0] signal   : [19, 2000]  float32  all channels like HMSDataset
      [1] label    : [6]         float32  soft vote distribution
      [2] icoh_vec : [171]       float32  precomputed iCOH upper triangle
    """
    def __init__(self, root, split, icoh_cache_dir, window_sec=10, fs=200):
        self.root           = root
        self.icoh_cache_dir = icoh_cache_dir
        self.window         = window_sec * fs
        self.channels       = HMS_CHANNELS
        self.eeg_dir        = os.path.join(root, "train_eegs")
        self.df             = pd.read_csv(os.path.join(root, f"{split}.csv")).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        eeg_id = int(row.eeg_id)

        # Load all 19 channels — same as HMSDataset
        eeg_raw = pd.read_parquet(os.path.join(self.eeg_dir, f"{eeg_id}.parquet"))
        eeg = pd.DataFrame(index=eeg_raw.index)
        for ch in self.channels:
            eeg[ch] = eeg_raw[ch] if ch in eeg_raw.columns else 0.0

        seg = eeg.values.astype(np.float32)
        mid  = len(seg) // 2
        half = self.window // 2
        seg  = seg[max(0, mid-half):min(len(seg), mid+half)]
        if seg.shape[0] < self.window:
            seg = np.concatenate([seg, np.zeros((self.window-seg.shape[0], seg.shape[1]), dtype=np.float32)])
        seg = (seg[:self.window] / 100.0).T  # [19, 2000]
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        signal = torch.tensor(seg, dtype=torch.float32)  # [19, 2000]

        # Load iCOH cache
        cache_path = os.path.join(self.icoh_cache_dir, f"{eeg_id}.pt")
        if os.path.exists(cache_path):
            icoh_vec = torch.load(cache_path, weights_only=True)['icoh_vector']
        else:
            icoh_vec = torch.zeros(171, dtype=torch.float32)

        # Soft label
        votes = row[LABEL_COLS].values.astype(np.float32)
        total = votes.sum()
        label = votes / total if total > 0 else np.ones(6, dtype=np.float32) / 6
        soft_label = torch.tensor(label, dtype=torch.float32)

        return signal, soft_label, icoh_vec
