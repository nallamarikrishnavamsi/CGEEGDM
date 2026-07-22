import os, sys, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.icoh import compute_icoh, icoh_upper_triangle

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']

EEG_DIR   = '/home/dsamantaai/krishna/data/train_eegs'
CACHE_DIR = 'data/icoh_cache'
FS        = 200
WINDOW_SEC = 10
WINDOW = WINDOW_SEC * FS  # 2000

# Cache eeg_raw reads within a worker across consecutive (eeg_id, offset) pairs
# sharing the same eeg_id (common — many offsets per file)
_cache = {'eeg_id': None, 'sig': None}

def process_one(args):
    eeg_id, offset = args
    key = f"{eeg_id}_{int(offset)}"
    cache_path = os.path.join(CACHE_DIR, f"{key}.pt")
    if os.path.exists(cache_path):
        return 'skipped'
    try:
        if _cache['eeg_id'] != eeg_id:
            eeg_raw = pd.read_parquet(os.path.join(EEG_DIR, f"{eeg_id}.parquet"))
            sig = np.zeros((len(HMS_CHANNELS), len(eeg_raw)), dtype=np.float32)
            for i, ch in enumerate(HMS_CHANNELS):
                if ch in eeg_raw.columns:
                    sig[i] = eeg_raw[ch].values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

            from scipy.signal import butter, filtfilt, iirnotch
            nyq = FS / 2
            b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
            sig  = filtfilt(b, a, sig, axis=-1)
            b, a = iirnotch(50.0/nyq, 30)
            sig  = filtfilt(b, a, sig, axis=-1)

            _cache['eeg_id'] = eeg_id
            _cache['sig'] = sig
        else:
            sig = _cache['sig']

        # HMS convention: eeg_label_offset_seconds marks start of a 50s labeled window.
        # Take the middle 10s of that 50s window: [offset+20, offset+30]
        total_len  = sig.shape[-1]
        start_sample = int((offset + 20) * FS)
        end_sample   = start_sample + WINDOW
        start_sample = max(0, min(start_sample, total_len - WINDOW))
        end_sample   = start_sample + WINDOW

        seg = sig[:, start_sample:end_sample]
        if seg.shape[1] < WINDOW:
            pad = WINDOW - seg.shape[1]
            seg = np.concatenate([seg, np.zeros((len(HMS_CHANNELS), pad), dtype=np.float32)], axis=1)

        A = compute_icoh(seg, fs=FS)
        v = icoh_upper_triangle(A)

        torch.save({
            'icoh_matrix' : torch.tensor(A),
            'icoh_vector' : torch.tensor(v),
        }, cache_path)
        return 'done'
    except Exception as e:
        return f'failed:{eeg_id}_{offset}:{e}'


if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    csv_files = [
        '/home/dsamantaai/krishna/data/full106k_train.csv',
        '/home/dsamantaai/krishna/data/full106k_val.csv',
        '/home/dsamantaai/krishna/data/full106k_test.csv',
    ]
    all_pairs = set()
    for csv in csv_files:
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            for eid, off in zip(df['eeg_id'].astype(int), df['eeg_label_offset_seconds']):
                all_pairs.add((eid, off))
    # Sort by eeg_id so consecutive same-eeg_id pairs benefit from in-worker sig cache
    all_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))
    print(f"Total (eeg_id, offset) pairs: {len(all_pairs)}")

    # NOTE: multiprocessing Pool workers don't share _cache across processes,
    # so within-worker caching only helps when chunksize groups same-eeg_id pairs together.
    n_workers = min(cpu_count(), 16)
    print(f"Using {n_workers} workers")
    start = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_one, all_pairs, chunksize=8), total=len(all_pairs)))

    done    = results.count('done')
    skipped = results.count('skipped')
    failed  = sum(1 for r in results if r.startswith('failed'))
    print(f"\nDone:{done}  Skipped:{skipped}  Failed:{failed}")
    if failed:
        for r in results:
            if r.startswith('failed'):
                print(r)
    print(f"Total time: {(time.time()-start)/60:.1f} min")
