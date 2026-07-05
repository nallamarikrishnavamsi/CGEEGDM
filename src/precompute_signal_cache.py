"""
Offline preprocessing for HMS pretraining — matches original EEGDM pattern
where data is converted to fast-loading cache files BEFORE training starts.

This replaces repeated parquet reads (19x per EEG per epoch) with a single
read per EEG, caching all 19 channels + iCOH together.

Run once before pretraining:
    python src/precompute_signal_cache.py

Output: data/signal_cache/{eeg_id}.pt
    {
      'signal': FloatTensor [19, 2000],   # all channels, centre-cropped, normalised
      'icoh_vec': FloatTensor [171],      # from existing icoh_cache
    }
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']

EEG_DIR        = '/home/dsamantaai/krishna/data/train_eegs'
ICOH_CACHE_DIR = './data/icoh_cache'
SIGNAL_CACHE_DIR = './data/signal_cache'
WINDOW_SEC = 10
FS = 200
WINDOW = WINDOW_SEC * FS  # 2000


def process_one(eeg_id):
    cache_path = os.path.join(SIGNAL_CACHE_DIR, f"{eeg_id}.pt")
    if os.path.exists(cache_path):
        return 'skipped'
    try:
        eeg_raw = pd.read_parquet(os.path.join(EEG_DIR, f"{eeg_id}.parquet"))
        sig = np.zeros((len(HMS_CHANNELS), len(eeg_raw)), dtype=np.float32)
        for i, ch in enumerate(HMS_CHANNELS):
            if ch in eeg_raw.columns:
                sig[i] = eeg_raw[ch].values.astype(np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        T = sig.shape[1]
        mid = T // 2
        half = WINDOW // 2
        seg = sig[:, max(0, mid-half):min(T, mid+half)]
        if seg.shape[1] < WINDOW:
            pad = WINDOW - seg.shape[1]
            seg = np.concatenate([seg, np.zeros((len(HMS_CHANNELS), pad), dtype=np.float32)], axis=1)
        seg = seg[:, :WINDOW] / 100.0

        icoh_path = os.path.join(ICOH_CACHE_DIR, f"{eeg_id}.pt")
        if os.path.exists(icoh_path):
            icoh_vec = torch.load(icoh_path, weights_only=True)['icoh_vector']
        else:
            icoh_vec = torch.zeros(171, dtype=torch.float32)

        torch.save({
            'signal': torch.tensor(seg, dtype=torch.float32),
            'icoh_vec': icoh_vec,
        }, cache_path)
        return 'done'
    except Exception as e:
        return f'failed:{eeg_id}:{e}'


if __name__ == '__main__':
    os.makedirs(SIGNAL_CACHE_DIR, exist_ok=True)

    csv_files = [
        '/home/dsamantaai/krishna/data/full106k_train.csv',
        '/home/dsamantaai/krishna/data/full106k_val.csv',
        '/home/dsamantaai/krishna/data/full106k_test.csv',
    ]
    all_ids = set()
    for csv in csv_files:
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            all_ids.update(df['eeg_id'].astype(int).tolist())
        else:
            print(f"WARNING: {csv} not found")
    all_ids = sorted(all_ids)
    print(f"Total EEG IDs to process: {len(all_ids)}")

    cached = set(int(f.replace('.pt', '')) for f in os.listdir(SIGNAL_CACHE_DIR) if f.endswith('.pt'))
    missing = [i for i in all_ids if i not in cached]
    print(f"Already cached: {len(cached)}  Missing: {len(missing)}")

    n_workers = min(cpu_count(), 16)
    print(f"Using {n_workers} workers")
    start = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_one, missing), total=len(missing)))

    done    = results.count('done')
    skipped = results.count('skipped')
    failed  = [r for r in results if r.startswith('failed')]
    print(f"\nDone:{done}  Skipped:{skipped}  Failed:{len(failed)}")
    if failed:
        print("First 5 failures:", failed[:5])
    print(f"Total time: {(time.time()-start)/60:.1f} min")
