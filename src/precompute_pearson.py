"""
Pearson correlation connectivity for HMS, computed on the SAME 20-channel
bipolar signal windows already produced by precompute_signal_cache.py
(data/signal_cache/{eeg_id}_{offset}.pt -> 'signal': [20, 2000]).

This does NOT reprocess raw EEG or refilter anything -- it reuses the
already-filtered, already-windowed, already-scaled bipolar signal cache,
so Pearson connectivity and iCOH connectivity are guaranteed to be
computed from identical underlying signal, differing only in which
statistic (correlation vs imaginary coherence) is applied.

For each 20-channel window:
    pearson_matrix[i, j] = Pearson correlation coefficient between
                            channel i and channel j over the 2000 samples
    pearson_vector        = upper-triangle (i<j) of pearson_matrix,
                            flattened -> 190 values, SAME ordering
                            convention as icoh_vector (row-major upper
                            triangle, k = 0..189) so the two are directly
                            comparable / swappable as graph edge features.

Output: data/pearson_cache/{eeg_id}_{offset}.pt
    {
      'pearson_matrix': FloatTensor [20, 20],
      'pearson_vector': FloatTensor [190],
    }
"""
import os, sys, time
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

SIGNAL_CACHE_DIR  = './data/signal_cache'
PEARSON_CACHE_DIR = './data/pearson'
N_CHANNELS = 20

UPPER_TRI_PAIRS = [(i, j) for i in range(N_CHANNELS) for j in range(i + 1, N_CHANNELS)]
assert len(UPPER_TRI_PAIRS) == 190


def pearson_matrix_from_signal(sig):
    with np.errstate(invalid='ignore', divide='ignore'):
        mat = np.corrcoef(sig)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 1.0)
    return mat.astype(np.float32)


def process_one(fname):
    key = fname[:-3]
    out_path = os.path.join(PEARSON_CACHE_DIR, fname)
    if os.path.exists(out_path):
        return 'skipped'
    try:
        d = torch.load(os.path.join(SIGNAL_CACHE_DIR, fname), weights_only=True)
        sig = d['signal'].numpy()
        if sig.shape[0] != N_CHANNELS:
            return f'failed:{key}:unexpected channel count {sig.shape[0]}'

        mat = pearson_matrix_from_signal(sig)
        vec = np.array([mat[i, j] for i, j in UPPER_TRI_PAIRS], dtype=np.float32)

        torch.save({
            'pearson_matrix': torch.tensor(mat, dtype=torch.float32),
            'pearson_vector': torch.tensor(vec, dtype=torch.float32),
        }, out_path)
        return 'done'
    except Exception as e:
        return f'failed:{key}:{e}'


if __name__ == '__main__':
    os.makedirs(PEARSON_CACHE_DIR, exist_ok=True)

    all_files = sorted(f for f in os.listdir(SIGNAL_CACHE_DIR) if f.endswith('.pt'))
    print(f"Found {len(all_files)} cached signal windows in {SIGNAL_CACHE_DIR}")

    n_workers = min(cpu_count(), 16)
    print(f"Using {n_workers} workers")
    start = time.time()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_one, all_files, chunksize=32), total=len(all_files)))

    done    = results.count('done')
    skipped = results.count('skipped')
    failed  = [r for r in results if r.startswith('failed')]
    print(f"\nDone:{done}  Skipped:{skipped}  Failed:{len(failed)}")
    if failed:
        for r in failed[:20]:
            print(r)
    print(f"Total time: {(time.time()-start)/60:.1f} min")
