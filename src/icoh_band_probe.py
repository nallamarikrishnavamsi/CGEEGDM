"""Compare iCOH variants on real bipolar segments: frequency band, and per-sample
standardisation. Read-only - touches no cache. Mirrors src/precompute_icoh.py."""
import sys, os, numpy as np, pandas as pd, mne
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.signal import csd, welch
from src.bipolar import build_bipolar
mne.set_log_level("CRITICAL")

HMS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
       'Fp2','F4','C4','P4','F8','T4','T6','O2']
FS, WIN, N, C = 200, 2000, 50, 20

def icoh_spectrum(sig):
    """-> R [190, F] of |icoh(f)| per channel pair, and the frequency axis."""
    nper = min(sig.shape[1], 256)
    f, _ = welch(sig[0], fs=FS, nperseg=nper)
    pxx = np.array([welch(sig[i], fs=FS, nperseg=nper)[1] for i in range(C)])
    R = []
    for i in range(C):
        for j in range(i + 1, C):
            _, cxy = csd(sig[i], sig[j], fs=FS, nperseg=nper)
            R.append(np.abs(np.imag(cxy) / (np.sqrt(pxx[i] * pxx[j]) + 1e-8)))
    return np.array(R), f

df = pd.read_csv('/home/dsamantaai/krishna/data/full106k_val.csv').head(N)
specs, freqs = [], None
for eid, off in zip(df.eeg_id.astype(int), df.eeg_label_offset_seconds):
    raw = pd.read_parquet(f'/home/dsamantaai/krishna/data/train_eegs/{eid}.parquet')
    s = np.nan_to_num(np.stack([raw[c].values if c in raw else np.zeros(len(raw))
                                for c in HMS])).astype(np.float32)
    r = mne.io.RawArray(s, mne.create_info(HMS, FS, "eeg"), verbose="CRITICAL")
    r.filter(0.1, 75); r.notch_filter(50)
    b, *_ = build_bipolar(r.get_data().astype(np.float32), HMS)
    st = max(0, min(int((off + 20) * FS), b.shape[1] - WIN))
    R, freqs = icoh_spectrum(b[:, st:st + WIN])
    specs.append(R)
specs = np.stack(specs)                       # [N, 190, F]
np.savez('data/icoh_probe_spectra.npz', specs=specs, freqs=freqs)
print(f"{len(specs)} bipolar segments, {specs.shape[2]} freq bins "
      f"({freqs[1]-freqs[0]:.4f} Hz spacing, {freqs[0]:.2f}-{freqs[-1]:.2f} Hz)\n")

def report(tag, V, nb=None):
    band = f"{nb}/{specs.shape[2]} bins" if nb else ""
    print(f"{tag:26s} {band:13s} mean {V.mean():.4f} | within-sample std {V.std(1).mean():.4f} "
          f"| across-sample std {V.std(0).mean():.4f} "
          f"| max/min {np.median(V.max(1)/np.maximum(V.min(1),1e-9)):.2f} "
          f"| self-weight {1/(1+19*V.mean()):.3f}")

bands = [("current (0-100Hz)", None, None), ("passband 0.1-75Hz", 0.1, 75),
         ("analysis 0.5-45Hz", 0.5, 45), ("alpha-beta 8-30Hz", 8, 30)]
base = None
for tag, lo, hi in bands:
    m = np.ones_like(freqs, bool)
    if lo is not None: m &= freqs >= lo
    if hi is not None: m &= freqs <= hi
    V = specs[:, :, m].mean(2)                # [N, 190]
    report(tag, V, int(m.sum()))
    if tag.startswith("analysis"): base = V

z = (base - base.mean(1, keepdims=True)) / (base.std(1, keepdims=True) + 1e-8)
report("  +z-score -> minmax[0,1]", (z - z.min(1, keepdims=True)) /
       (z.max(1, keepdims=True) - z.min(1, keepdims=True) + 1e-8))

print()
m = (freqs >= 0.5) & (freqs <= 45)
V = specs[:, :, m].mean(2)
report("0.5-45 raw (reference)", V)
report("  +per-sample mean rescale", V * (V.mean() / V.mean(1, keepdims=True)))
zz = (V - V.mean(1, keepdims=True)) / (V.std(1, keepdims=True) + 1e-8)
report("  +z -> mean .188 std .04", np.clip(0.188 + 0.04 * zz, 0.0, 1.0))
