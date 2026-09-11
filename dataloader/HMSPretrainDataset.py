"""
HMS pretraining dataset for diffusion backbone pretraining.

Matches original EEGDM's TUEVDataset output format: one channel per
sample (not all 20 at once, unlike ConnectivityHMSDatasetCached used
for finetuning). Reads from the existing signal_cache (20-channel
bipolar TCP, correctly offset-windowed tensors -- see src/bipolar.py)
and expands each (eeg_id, offset) row into 20 separate per-channel
training examples.

20 of EEGDM's original 22 bipolar derivations are constructible from
HMS (A1-T3, A2-T4 excluded -- HMS has no A1/A2 mastoid electrodes).

Returns:
    [0] signal : FloatTensor [1, T]  single channel, T=2000 (10s @ 200Hz)
    [1] label  : LongTensor  []      channel index (0-19), matches
                 original EEGDM's per-channel diffusion conditioning
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

# 20 of EEGDM's 22 bipolar derivations, in original relative order
# (A1-T3, A2-T4 dropped -- see src/bipolar.py). Must match the channel
# order build_bipolar() produces in precompute_signal_cache.py.
BIPOLAR_CHANNELS_20 = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "C4-CZ", "T4-C4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


class HMSPretrainDataset(Dataset):
    def __init__(self, data_root, split_csv, signal_cache_dir):
        self.signal_cache_dir = signal_cache_dir
        self.df = pd.read_csv(os.path.join(data_root, f"{split_csv}.csv")).reset_index(drop=True)
        self.n_channels = len(BIPOLAR_CHANNELS_20)

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
            signal_20ch = cache['signal']  # [20, 2000]
        else:
            signal_20ch = torch.zeros(self.n_channels, 2000, dtype=torch.float32)

        signal = signal_20ch[ch_idx].unsqueeze(0)  # [1, 2000]
        label  = torch.tensor(ch_idx, dtype=torch.long)

        return signal, label
