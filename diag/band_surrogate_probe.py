import sys, glob, numpy as np, torch
sys.path.insert(0, 'src')

FS, NPERSEG, NOVERLAP, NWIN = 200, 256, 128, 300
BANDS = [("delta", 1, 5), ("theta", 5, 8), ("alpha", 8, 14),
         ("beta", 15, 30), ("gamma", 30, 49)]

def cross_spectra(sig):
    """Welch cross-spectral matrix. Re and Im come from one transform,
    so the Re/Im contrast is exact. Scaling cancels in coherency."""
    C, L = sig.shape
    win, step = np.hanning(NPERSEG), NPERSEG - NOVERLAP
    segs = []
    for s in range(0, L - NPERSEG + 1, step):
        seg = sig[:, s:s+NPERSEG]
        seg = seg - seg.mean(axis=1, keepdims=True)      # detrend='constant'
        segs.append(np.fft.rfft(seg * win, axis=1))
    F = np.stack(segs, 0)                                 # [K, C, Fbins]
    S = np.einsum('kcf,kdf->cdf', F, np.conj(F)) / F.shape[0]
    return np.fft.rfftfreq(NPERSEG, 1.0/FS), S, F.shape[0]

def coh_parts(S):
    P = np.real(np.einsum('ccf->cf', S))                  # [C, Fbins] auto-spectra
    den = np.sqrt(P[:, None, :] * P[None, :, :])
    den = np.where(den <= 0, np.inf, den)
    return np.abs(S.imag)/den, np.abs(S.real)/den         # icoh_bins, rcoh_bins

def phase_randomize(x, rng):
    """Preserve each channel's power spectrum exactly, destroy all
    cross-channel phase relations -> true coherency is zero."""
    F = np.fft.rfft(x, axis=1); mag = np.abs(F)
    ph = rng.uniform(-np.pi, np.pi, size=F.shape)
    ph[:, 0] = 0.0
    if x.shape[1] % 2 == 0: ph[:, -1] = 0.0
    return np.fft.irfft(mag*np.exp(1j*ph), n=x.shape[1], axis=1)

files = sorted(glob.glob('data/signal_cache/*.pt'))
rng   = np.random.default_rng(0)
pick  = rng.choice(len(files), size=min(NWIN, len(files)), replace=False)
iu    = np.triu_indices(20, 1)

acc = {}
K_used = None
for n, k in enumerate(pick):
    d   = torch.load(files[k], weights_only=True)
    sig = d['signal'].numpy().astype(np.float64)
    for cond, x in (('real', sig), ('surr', phase_randomize(sig, rng))):
        f, S, K = cross_spectra(x); K_used = K
        ib, rb = coh_parts(S)
        for bname, lo, hi in BANDS:
            m = (f >= lo) & (f <= hi)
            acc.setdefault((bname, cond, 'icoh'), []).append(ib[:, :, m].mean(-1)[iu])
            acc.setdefault((bname, cond, 'rcoh'), []).append(rb[:, :, m].mean(-1)[iu])
    if n == 0:
        print(f"segments K = {K}   bins per band:")
        for bname, lo, hi in BANDS:
            print(f"  {bname:6} {lo:2}-{hi:2} Hz : {int(((f>=lo)&(f<=hi)).sum()):3d} bins")
        print(f"\npredicted zero-coupling bias = sqrt(1/(2*{K})) * sqrt(2/pi) "
              f"= {np.sqrt(1/(2*K))*np.sqrt(2/np.pi):.4f}\n")

print(f"{'band':7} {'measure':8} {'real':>8} {'surr':>8} {'real-surr':>10} "
      f"{'% signal':>9}")
for bname, _, _ in BANDS:
    for meas in ('icoh', 'rcoh'):
        R = np.array(acc[(bname,'real',meas)]); S_ = np.array(acc[(bname,'surr',meas)])
        d_ = R.mean() - S_.mean()
        print(f"{bname:7} {meas:8} {R.mean():8.4f} {S_.mean():8.4f} "
              f"{d_:+10.4f} {d_/R.mean()*100:8.1f}%")

print("\ncontrast  rcoh - icoh  (volume-conduction component):")
for bname, _, _ in BANDS:
    cr = np.array(acc[(bname,'real','rcoh')]).mean() - np.array(acc[(bname,'real','icoh')]).mean()
    cs = np.array(acc[(bname,'surr','rcoh')]).mean() - np.array(acc[(bname,'surr','icoh')]).mean()
    print(f"  {bname:7} real {cr:+.4f}   surrogate {cs:+.4f}  (surrogate should be ~0)")
