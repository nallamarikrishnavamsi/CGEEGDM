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

EEG_DIR   = '/home/mtech1/25CS60R51/EEGDM/data/hms/train_eegs'
CACHE_DIR = 'data/icoh_cache'
FS        = 200
WIN_SEC   = 10  # match EEG input window

def process_one(eeg_id):
    cache_path = os.path.join(CACHE_DIR, f"{eeg_id}.pt")
    if os.path.exists(cache_path):
        return 'skipped'
    try:
        eeg_raw = pd.read_parquet(os.path.join(EEG_DIR, f"{eeg_id}.parquet"))
        sig = np.zeros((len(HMS_CHANNELS), len(eeg_raw)), dtype=np.float32)
        for i, ch in enumerate(HMS_CHANNELS):
            if ch in eeg_raw.columns:
                sig[i] = eeg_raw[ch].values.astype(np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        # Bandpass + notch before iCOH
        from scipy.signal import butter, filtfilt, iirnotch
        nyq = FS / 2
        b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
        sig  = filtfilt(b, a, sig, axis=-1)
        b, a = iirnotch(50.0/nyq, 30)
        sig  = filtfilt(b, a, sig, axis=-1)
        # Use full signal for richer connectivity estimate
        seg = sig
        # Compute iCOH
        A = compute_icoh(seg, fs=FS)          # [19, 19]
        v = icoh_upper_triangle(A)            # [171]
        torch.save({
            'icoh_matrix'  : torch.tensor(A),  # [19, 19]
            'icoh_vector'  : torch.tensor(v),  # [171]
        }, cache_path)
        return 'done'
    except Exception as e:
        return f'failed:{e}'

if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    csv_files = [
        'data/hms/train_split.csv',
        'data/hms/val_split.csv',
        'data/hms/test_split.csv'
    ]
    all_ids = set()
    for csv in csv_files:
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            all_ids.update(df['eeg_id'].astype(int).tolist())
    all_ids = sorted(all_ids)
    print(f"Total EEG files : {len(all_ids)}")
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers")
    start = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_one, all_ids), total=len(all_ids)))
    done    = results.count('done')
    skipped = results.count('skipped')
    failed  = sum(1 for r in results if r.startswith('failed'))
    print(f"\nDone:{done}  Skipped:{skipped}  Failed:{failed}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
