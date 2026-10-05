"""
Opt-in fix for LatentActivityExtractor (model/classifier.py, identical to
upstream EEGDM, NOT modified). Only used with --fix_extractor_fold 1.

Bug (L > 1000 samples -> fold f > 1; HMS 10 s = 2000 -> f = 2):
the input is flattened channel-then-fold, index (b*C + c)*f + k, but the
channel-embedding repeat and the output un-flatten assume fold-then-channel,
index (b*f + k)*C + c. Output slot m then holds input channel floor(m/2)
(first 5 s) and 10 + floor(m/2) (second 5 s), with embedding m.
Fix: keep channel-then-fold ordering for cond and for the un-flatten.
For f = 1 the result is identical to the original.
"""
import torch
from einops import rearrange
from .classifier import LatentActivityExtractor


class LatentActivityExtractorFoldFix(LatentActivityExtractor):

    @classmethod
    def from_existing(cls, ext):
        ext.__class__ = cls      # same backbone, same buffers, only forward() changes
        return ext

    @torch.no_grad()
    def forward(self, input, is_caching=None, rate=1):
        query, start, end = self.query, self.start, self.end
        x = input[0]
        local_cond = input[1] if len(input) > 1 else None
        B = x.shape[0]
        assert x.shape[1] == self.C, f"input has {x.shape[1]} channels, extractor expects {self.C}"

        x = self.init_rearr(x)
        if x.dim() < 3: x = x.unsqueeze(1)
        fold = 1
        if x.shape[2] > 1000:
            assert rate == 1
            fold = x.shape[2] // 1000
        x = rearrange(x, "BC c (f L) -> (BC f) c L", f=fold)
        if local_cond is not None:
            local_cond = self.init_rearr(local_cond)
            if local_cond.dim() < 4: local_cond = local_cond.unsqueeze(1)
            local_cond = rearrange(local_cond, "BC ... (f L) -> (BC f) ... L", f=fold)

        diffusion_steps = self.diffusion_steps.repeat_interleave(fold, dim=0).repeat(B, 1)
        cond = self.cond.repeat_interleave(fold, dim=0).repeat(B, 1)
        cond = self.model.calc_cond(diffusion_steps, cond)

        x = rearrange(x, "N H L -> N L H")
        x = self.model.in_layer(x)
        latent_activities = []
        for i, l in enumerate(self.model.layers):
            if i == end:
                break
            x, _, query_result = l(x, cond, local_cond, query=query, skip_skip=True, rate=rate)
            if i >= start:
                latent_activities.append(torch.stack(query_result, dim=1))
        latent_activities = torch.stack(latent_activities, dim=1)          # [(B C f), n, q, L, H]
        return rearrange(latent_activities, "(B C f) n q L H -> B C n q (f L) H", B=B, C=self.C, f=fold)
