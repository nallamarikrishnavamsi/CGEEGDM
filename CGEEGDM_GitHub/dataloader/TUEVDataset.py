import torch
import os
from glob import glob
import pickle
from dataclasses import dataclass
from typing import Callable, Sequence, NamedTuple
import numpy as np
from tqdm import tqdm

class TUEVDataField(NamedTuple):
    name: str
    dtype: torch.dtype
    trans: Callable | None = None

class TUEVDataset(torch.utils.data.Dataset):
    def __init__(self, root, schema: Sequence[TUEVDataField]=[("signal", torch.float), ("label", torch.long)], stft_kwargs=None, return_index=False):
        self.root = root
        self.files = sorted(glob(str(os.path.join(root, "*.pkl"))))
        self.fields = [TUEVDataField(*f) for f in schema]
        self.stft_kwargs = stft_kwargs
        self.return_index = return_index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            p = pickle.load(f)
        
        out = []
        for f in self.fields:
            data = p[f.name]
            if f.trans is not None: data = f.trans(data)
            out.append(torch.tensor(data, dtype=f.dtype))

        if self.stft_kwargs:
            out.append(torch.stft(out[0], return_complex=True, **self.stft_kwargs).abs())

        if self.return_index:
            out.append(torch.tensor([index], dtype=torch.long))

        return out
    
    def setup_scarce(self, seed, mode, pct, n_class=None, label_field_idx=-1):
        assert mode in ["per_class", "simple"]

        chooser = np.random.default_rng(seed) # seed data selection independent of model initialization
        
        if mode == "per_class":
            assert n_class is not None
            class_count = np.zeros(n_class)
            class_idx = [[] for _ in class_count]

            for f_idx, f in tqdm(enumerate(self.files), desc=f"Counting data of each class...", total=len(self.files)):
                with open(f, "rb") as ff:
                    p = pickle.load(ff)
                
                label = p[self.fields[label_field_idx].name]
                if self.fields[label_field_idx].trans is not None: label = f.trans(label)
                
                class_count[label] += 1
                class_idx[label.item()].append(f_idx) 
            
            reduced_class_count = np.ceil(class_count * pct / 100).astype(int)
            reduced_class_idx = []
            for c_idx, rcc in zip(class_idx, reduced_class_count):
                reduced_class_idx.append(chooser.choice(c_idx, replace=False, size=rcc))

            reduced_class_idx = np.concatenate(reduced_class_idx)
            print(f"Size {len(reduced_class_idx)} ({reduced_class_count.tolist()})")
            self.files = sorted([self.files[i] for i in reduced_class_idx])

        else: # simple
            size=np.ceil(len(self.files) * pct / 100).astype(int)
            print(f"Size {len(self.files)} -> {size}")
            self.files = sorted(chooser.choice(self.files, replace=False, size=size))
