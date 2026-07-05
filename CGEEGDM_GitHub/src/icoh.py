import numpy as np
from scipy.signal import csd, welch, butter, filtfilt, iirnotch


def bandpass_filter(signal, lo=0.5, hi=40.0, fs=200):
    """Bandpass 0.5-40Hz + 50Hz notch filter."""
    nyq = fs / 2
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    signal = filtfilt(b, a, signal, axis=-1)
    b, a = iirnotch(50.0/nyq, 30)
    signal = filtfilt(b, a, signal, axis=-1)
    return signal


def compute_icoh(signal, fs=200, apply_filter=True):
    """
    Compute Imaginary Coherence matrix.
    signal: [C, T] numpy float32
    returns: [C, C] float32
    Formula: iCOH(i,j) = mean(|Im(Cxy)|) / sqrt(Pxx * Pyy)
    """
    if apply_filter:
        signal = bandpass_filter(signal, fs=fs)

    C = signal.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    nperseg = min(signal.shape[1], 256)

    pxx = np.array([
        welch(signal[i], fs=fs, nperseg=nperseg)[1].mean()
        for i in range(C)
    ])

    for i in range(C):
        for j in range(i + 1, C):
            _, Cxy = csd(signal[i], signal[j], fs=fs, nperseg=nperseg)
            icoh = np.mean(np.abs(np.imag(Cxy))) / (np.sqrt(pxx[i] * pxx[j]) + 1e-8)
            A[i, j] = A[j, i] = float(icoh)

    # Normalize to [0, 1]
    max_val = A.max()
    if max_val > 1e-8:
        A = A / max_val

    return A


def icoh_upper_triangle(A):
    """
    Extract upper triangle from symmetric iCOH matrix.
    A: [C, C] -> [C*(C-1)/2] = [171] for C=19
    """
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx].astype(np.float32)
