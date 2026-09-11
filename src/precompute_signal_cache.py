"""
Offline preprocessing for HMS pretraining — matches original EEGDM pattern
where data is converted to fast-loading cache files BEFORE training starts.

Keys by (eeg_id, eeg_label_offset_seconds) since a single eeg_id can have
multiple distinct labeled 50s windows, each needing its own 10s crop.

Output: data/signal_cache/{eeg_id}_{offset}.pt
    {
      'signal': FloatTensor [20, 2000],   # 20 of EEGDM's 22 bipolar channels;
                                            # A1-T3, A2-T4 unavailable (HMS has no A1/A2)
      'icoh_vec': FloatTensor [190],       # 20*19/2 upper-triangle entries
    }
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bipolar import build_bipolar

HMS_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']

EEG_DIR        = '/home/dsamantaai/krishna/data/train_eegs'
ICOH_CACHE_DIR = './data/icoh_cache'
SIGNAL_CACHE_DIR = './data/signal_cache'
WINDOW_SEC = 10
FS = 200
WINDOW = WINDOW_SEC * FS  # 2000

_cache = {'eeg_id': None, 'sig': None}

def process_one(args):
    eeg_id, offset = args
    key = f"{eeg_id}_{int(offset)}"
    cache_path = os.path.join(SIGNAL_CACHE_DIR, f"{key}.pt")
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

            # Bandpass + notch filter — matches precompute_icoh.py exactly,
            # using MNE with original EEGDM's exact config values
            # (EEGDM_original/conf/preprocessing/pretrain.yaml: bandpass 0.1-75Hz,
            # notch 50Hz, method='fir' default), so both the classification
            # signal and the iCOH graph branch see identically filtered EEG.
            import mne
            mne.set_log_level("CRITICAL")
            info = mne.create_info(ch_names=HMS_CHANNELS, sfreq=FS, ch_types="eeg")
            raw = mne.io.RawArray(sig, info, verbose="CRITICAL")
            raw.filter(l_freq=0.1, h_freq=75)
            raw.notch_filter(50)
            sig = raw.get_data().astype(np.float32)  # [19, T] filtered monopolar

            # Bipolar TCP conversion (EEGDM's original montage). Only 20 of
            # 22 derivations are constructible from HMS (no A1/A2 mastoid
            # electrodes) -- see src/bipolar.py. No fabrication.
            sig, used_names, used_indices, skipped = build_bipolar(sig, HMS_CHANNELS)
            # sig is now [20, T] bipolar

            _cache['eeg_id'] = eeg_id
            _cache['sig'] = sig
        else:
            sig = _cache['sig']

        T = sig.shape[1]
        start_sample = int((offset + 20) * FS)
        end_sample   = start_sample + WINDOW
        start_sample = max(0, min(start_sample, T - WINDOW))
        end_sample   = start_sample + WINDOW

        seg = sig[:, start_sample:end_sample]
        if seg.shape[1] < WINDOW:
            pad = WINDOW - seg.shape[1]
            seg = np.concatenate([seg, np.zeros((seg.shape[0], pad), dtype=np.float32)], axis=1)
        seg = seg[:, :WINDOW] / 100.0

        icoh_path = os.path.join(ICOH_CACHE_DIR, f"{key}.pt")
        if os.path.exists(icoh_path):
            icoh_vec = torch.load(icoh_path, weights_only=True)['icoh_vector']
        else:
            icoh_vec = torch.zeros(190, dtype=torch.float32)

        torch.save({
            'signal': torch.tensor(seg, dtype=torch.float32),
            'icoh_vec': icoh_vec,
        }, cache_path)
        return 'done'
    except Exception as e:
        return f'failed:{eeg_id}_{offset}:{e}'


if __name__ == '__main__':
    os.makedirs(SIGNAL_CACHE_DIR, exist_ok=True)
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
    all_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))
    print(f"Total (eeg_id, offset) pairs: {len(all_pairs)}")

    n_workers = min(cpu_count(), 16)
    print(f"Using {n_workers} workers")
    start = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_one, all_pairs, chunksize=8), total=len(all_pairs)))

    done    = results.count('done')
    skipped = results.count('skipped')
    failed  = sum(1 for r in results if r.startswith('failed'))
    print(f"\nDone:{done}  Skipped:{skipped}  Failed:{failed}")
    print(f"Total time: {(time.time()-start)/60:.1f} min")
