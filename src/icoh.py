import numpy as np
from scipy.signal import csd, welch

def compute_icoh(signal, fs=200):
    """
    Compute Imaginary Coherence matrix.
    signal: [C, T] numpy float32
    returns: [C, C] float32
    """
    C = signal.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    nperseg = min(signal.shape[1], 256)
    pxx = np.array([
        welch(signal[i], fs=fs, nperseg=nperseg)[1].mean()
        for i in range(C)
    ])
    for i in range(C):
        for j in range(i+1, C):
            _, Cxy = csd(signal[i], signal[j], fs=fs, nperseg=nperseg)
            icoh   = np.abs(np.mean(np.imag(Cxy))) / (np.sqrt(pxx[i]*pxx[j]) + 1e-8)
            A[i,j] = A[j,i] = float(icoh)
    return A

def icoh_upper_triangle(A):
    """
    Extract upper triangle from symmetric iCOH matrix.
    A: [C, C]
    returns: [C*(C-1)/2] float32
    """
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx].astype(np.float32)
