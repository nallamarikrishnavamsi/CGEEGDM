"""
Imaginary part of coherency (Nolte et al., 2004), Welch estimate.

  Welch : Hann window, nperseg=400 (2 s at 200 Hz -> 0.5 Hz bins, as in
          Nentwich et al., 2020), 50% overlap -> K = 9 segments per 10 s.
  IC_xy(f) = Im(S_xy(f)) / sqrt(S_xx(f) * S_yy(f))       per frequency bin
  A_xy     = mean over 0.5..75 Hz of |IC_xy(f)|          150 bins, inclusive

Input must be the LINEAR signal (mu-law removed). fmin=0.5 excludes the
DC bin (Im(S_xy) is identically 0 there). 75 Hz = preprocessing low-pass.
"""
import numpy as np
from scipy.signal import csd

NPERSEG = 400
FMIN = 0.5
FMAX = 75.0


def compute_icoh(signal, fs=200, nperseg=NPERSEG, fmin=FMIN, fmax=FMAX):
    signal = np.asarray(signal, dtype=np.float64)
    C, T = signal.shape
    nperseg = min(T, nperseg)

    # full cross-spectral matrix S[i, j, f]; its diagonal is the Welch PSD
    f, S = csd(signal[:, None, :], signal[None, :, :], fs=fs, nperseg=nperseg)

    keep = (f >= fmin) & (f <= fmax)
    S = S[..., keep]

    P = np.real(np.diagonal(S, axis1=0, axis2=1)).T          # [C, F]
    denom = np.sqrt(P[:, None, :] * P[None, :, :])           # [C, C, F]
    ic = np.divide(np.imag(S), denom, out=np.zeros_like(denom), where=denom > 0)

    A = np.abs(ic).mean(-1).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def icoh_upper_triangle(A):
    """Row-major upper triangle (i < j): 190 values for C = 20."""
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx].astype(np.float32)
