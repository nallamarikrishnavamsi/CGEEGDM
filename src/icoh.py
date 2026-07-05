import numpy as np
from scipy.signal import csd, welch


def compute_icoh(signal, fs=200):
    """
    Compute Imaginary Coherence matrix — correct per-frequency formula
    (Nolte et al. 2004).

    signal: [C, T] numpy float32
    returns: [C, C] float32

    icoh(f) = Im(Cxy(f)) / sqrt(Pxx(f) * Pyy(f))   <- per frequency bin
    icoh    = mean_f( |icoh(f)| )                   <- then average
    """
    C = signal.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    nperseg = min(signal.shape[1], 256)

    # Per-frequency PSD for every channel: [C, F]
    pxx_f = np.array([
        welch(signal[i], fs=fs, nperseg=nperseg)[1]
        for i in range(C)
    ])  # [C, F]

    for i in range(C):
        for j in range(i + 1, C):
            _, Cxy_f = csd(signal[i], signal[j], fs=fs, nperseg=nperseg)  # [F] complex

            denom = np.sqrt(pxx_f[i] * pxx_f[j]) + 1e-8       # [F]
            icoh_f = np.imag(Cxy_f) / denom                    # [F], per-frequency ratio
            icoh = np.mean(np.abs(icoh_f))                     # scalar, average AFTER ratio

            A[i, j] = A[j, i] = float(icoh)

    # Normalize matrix to [0, 1] for stable conditioning input
    max_val = A.max()
    if max_val > 1e-8:
        A = A / max_val

    return A


def icoh_upper_triangle(A):
    """
    Extract upper triangle from symmetric iCOH matrix.
    A: [C, C]
    returns: [C*(C-1)/2] float32  (171 for C=19)
    """
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx].astype(np.float32)
