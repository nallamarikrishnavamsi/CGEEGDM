"""
Band-limited iCOH. src/icoh.py is NOT modified.

Identical to compute_icoh() except for one thing: the average over frequency
is restricted to the bins that survive preprocessing.

compute_icoh() averages |icoh(f)| over ALL 129 Welch bins (0-100 Hz), but the
signal is band-passed 0.1-75 Hz and notched at 50 Hz. Measured on 50 real
bipolar segments, the 33 out-of-passband bins carry |icoh| = 0.163 each versus
0.197 inside the passband -- numerical noise from a ratio of two
filter-attenuated quantities, contributing 22.1% of every edge weight.

At fs=200, nperseg=256 the bin spacing is 0.78125 Hz and the defaults keep
bins 1-96 minus bins 62-66 = 91 of 129 bins.
"""
import numpy as np
from scipy.signal import csd, welch


def band_mask(f, fmin=0.1, fmax=75.0, notch=50.0, notch_width=2.0):
    m = (f >= fmin) & (f <= fmax)
    if notch is not None:
        m &= ~((f >= notch - notch_width) & (f <= notch + notch_width))
    return m


def compute_icoh_band(signal, fs=200, fmin=0.1, fmax=75.0,
                      notch=50.0, notch_width=2.0):
    """
    Imaginary Coherence (Nolte et al. 2004), averaged over the passband only.

    signal: [C, T] numpy float32 bipolar
    returns: [C, C] float32

    icoh(f) = Im(Cxy(f)) / sqrt(Pxx(f) * Pyy(f))   <- per frequency bin
    icoh    = mean_{f in band}( |icoh(f)| )         <- average over kept bins
    """
    C = signal.shape[0]
    A = np.zeros((C, C), dtype=np.float32)
    nperseg = min(signal.shape[1], 256)

    f, _ = welch(signal[0], fs=fs, nperseg=nperseg)
    sel = band_mask(f, fmin, fmax, notch, notch_width)
    assert sel.sum() > 0, "empty frequency band"

    pxx_f = np.array([
        welch(signal[i], fs=fs, nperseg=nperseg)[1]
        for i in range(C)
    ])  # [C, F]

    for i in range(C):
        for j in range(i + 1, C):
            _, Cxy_f = csd(signal[i], signal[j], fs=fs, nperseg=nperseg)
            denom = np.sqrt(pxx_f[i] * pxx_f[j]) + 1e-8
            icoh_f = np.imag(Cxy_f) / denom
            icoh = np.mean(np.abs(icoh_f[sel]))
            A[i, j] = A[j, i] = float(icoh)

    # No post-hoc normalization, same reasoning as src/icoh.py: |icoh_f| is
    # already bounded in [0, 1], so its mean is too.
    return A


def icoh_upper_triangle(A):
    """Identical to src/icoh.py's. A: [C, C] -> [C*(C-1)/2] float32 (190 for C=20)."""
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx].astype(np.float32)
