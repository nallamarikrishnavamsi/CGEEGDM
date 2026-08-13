"""
HMS pretraining dataset for diffusion backbone pretraining.

Matches original EEGDM's TUEVDataset output format: one channel per
sample (not all 19 at once, unlike ConnectivityHMSDatasetCached used
for finetuning). Reads from the existing signal_cache (19-channel,
correctly offset-windowed tensors) and expands each (eeg_id, offset)
row into 19 separate per-channel training examples.

Returns:
    [0] signal : FloatTensor [1, T]  single channel, T=2000 (10s @ 200Hz)
    [1] label  : LongTensor  []      channel index (0-18), matches
                 original EEGDM's per-channel diffusion conditioning
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']


class HMSPretrainDataset(Dataset):
    def __init__(self, data_root, split_csv, signal_cache_dir):
        self.signal_cache_dir = signal_cache_dir
        self.df = pd.read_csv(os.path.join(data_root, f"{split_csv}.csv")).reset_index(drop=True)
        self.n_channels = len(HMS_CHANNELS)

    def __len__(self):
        return len(self.df) * self.n_channels

    def __getitem__(self, index):
        row_idx = index // self.n_channels
        ch_idx  = index % self.n_channels

        row    = self.df.iloc[row_idx]
        eeg_id = int(row.eeg_id)
        offset = float(row.eeg_label_offset_seconds)
        key    = f"{eeg_id}_{int(offset)}"

        cache_path = os.path.join(self.signal_cache_dir, f"{key}.pt")
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=True)
            signal_19ch = cache['signal']  # [19, 2000]
        else:
            signal_19ch = torch.zeros(self.n_channels, 2000, dtype=torch.float32)

        signal = signal_19ch[ch_idx].unsqueeze(0)  # [1, 2000]
        label  = torch.tensor(ch_idx, dtype=torch.long)

        return signal, label
