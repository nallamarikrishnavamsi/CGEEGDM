"""
Read-only probe: does the extractor keep EEG channel c in output slot c?
Mock backbone whose latent activity = (input sample value, channel-embedding id),
so every output position traces back to its input (batch, channel, time).
Then a real random-init 2-layer Wavenet check. CPU, ~1 min, no data/checkpoint.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from model.classifier import LatentActivityExtractor
from model.extractor_foldfix import LatentActivityExtractorFoldFix
from model.diffusion_model import Wavenet


class MockLayer(nn.Module):
    def forward(self, x, cond, lc, query=None, skip_skip=True, rate=1):
        return x, None, [torch.cat([x, cond.expand(-1, x.shape[1], -1)], dim=-1) for _ in query]


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(), MockLayer()])
        self.in_layer = nn.Identity(); self.n_class = 20; self.have_null_class = False
    def calc_cond(self, t, cond):
        return cond.double().unsqueeze(-1)


def trace(ext, L, B=2, C=20):
    b = torch.arange(B).view(B, 1, 1).double(); c = torch.arange(C).view(1, C, 1).double()
    t = torch.arange(L).view(1, 1, L).double()
    la = ext((b * 1e7 + c * 1e4 + t,))
    val, emb = la[..., 0], la[..., 1]
    db = torch.div(val, 1e7, rounding_mode="floor"); r = val - db * 1e7
    dc = torch.div(r, 1e4, rounding_mode="floor"); dt = r - dc * 1e4
    bad = [(db != b.view(B,1,1,1,1)).sum().item(), (dc != c.view(1,C,1,1,1)).sum().item(),
           (dt != t.view(1,1,1,1,L)).sum().item(), (emb != c.view(1,C,1,1,1)).sum().item()]
    return val.numel(), bad, dc, dt


for name, cls, L in [("ORIGINAL, L=1000 (fold 1)", LatentActivityExtractor, 1000),
                     ("ORIGINAL, L=2000 (fold 2)", LatentActivityExtractor, 2000),
                     ("FOLD-FIX, L=2000 (fold 2)", LatentActivityExtractorFoldFix, 2000)]:
    n, bad, dc, dt = trace(cls(MockBackbone(), query=["gate"], use_cond=list(range(20))), L)
    print(f"{name}: positions {n} | wrong batch {bad[0]} | wrong channel {bad[1]} | wrong time {bad[2]} | "
          f"wrong embedding {bad[3]} -> {'ALIGNED' if sum(bad) == 0 else 'SCRAMBLED'}")
    if sum(bad):
        for m in (0, 1, 2, 19):
            print(f"   slot {m:2d}: t=0 <- ch {int(dc[0,m,0,0,0])} t {int(dt[0,m,0,0,0])} | "
                  f"t={L//2} <- ch {int(dc[0,m,0,0,L//2])} t {int(dt[0,m,0,0,L//2])}")

torch.manual_seed(0)
wn = Wavenet(in_channels=1, d_model=128, d_state=128, n_layer=2, kernel_init='diag-lin', kernel_mode='diag',
             bidirectional=True, d_cond=512, d_cond_embed=128, local_cond_ch=0, n_class=20).eval()
sig = torch.randn(1, 20, 2000)
og = LatentActivityExtractor(wn, query=["gate"], use_cond=list(range(20)))
fx = LatentActivityExtractorFoldFix(wn, query=["gate"], use_cond=list(range(20)))
ref = torch.cat([og((sig[..., :1000],)), og((sig[..., 1000:],))], dim=4)
e_fix, e_og = (fx((sig,)) - ref).abs().max().item(), (og((sig,)) - ref).abs().max().item()
print(f"Real Wavenet: max|FOLD-FIX - per-half reference| = {e_fix:.2e} | max|ORIGINAL - per-half reference| = {e_og:.2e}")
print("PASS" if e_fix < 1e-5 and e_og > 1e-3 else "UNEXPECTED")
