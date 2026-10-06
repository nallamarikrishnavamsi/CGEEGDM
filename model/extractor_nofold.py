"""
Opt-in no-fold extractor variant (ablation). Unlike LatentActivityExtractor
(original, buggy for L>1000) and LatentActivityExtractorFoldFix (correct
channel ordering, but still splits L>1000 into independent 1000-sample
folds), this variant NEVER splits: it feeds the full input length directly
to the backbone's SSM layers in one pass.

Architecturally safe to do: the SSM kernel (model/s4standalone.py) is
generated dynamically to match whatever length is requested at runtime
(l_max is optional, "most Kernel instantiations will return kernels of
arbitrary length as passed into .forward()"), and neither diffusion_model.py
nor the checkpoint contain any length-1000-sized learned parameter
(verified: no l_max/pos_embed/positional references in diffusion_model.py,
no pos/embed-named checkpoint tensors besides label_embed/t_embed, neither
of which is sequence-position-dependent).

However, the backbone was only ever PRETRAINED with kernels generated at
length 1000 (gen_kwargs.shape=[1,1000]), so feeding length 2000 directly
exercises the SSM kernel at a length it was never trained with -- this is
the empirical question this ablation exists to answer, compared against
the fold-based (run_finetune_graphcond.sh etc.) results.

For input length <= 1000, behavior is identical to the original/FoldFix
extractor (fold is always 1 in that case regardless).
"""
import torch
from .classifier import LatentActivityExtractor


class LatentActivityExtractorNoFold(LatentActivityExtractor):

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
        # NEVER fold/split -- feed the full length directly, whatever it is.
        fold = 1

        if local_cond is not None:
            local_cond = self.init_rearr(local_cond)
            if local_cond.dim() < 4: local_cond = local_cond.unsqueeze(1)

        diffusion_steps = self.diffusion_steps.repeat(B, 1)
        cond = self.cond.repeat(B, 1)
        cond = self.model.calc_cond(diffusion_steps, cond)

        x = x.transpose(1, 2)  # "N H L -> N L H" (fold=1, so N == B*C)
        x = self.model.in_layer(x)
        latent_activities = []
        for i, l in enumerate(self.model.layers):
            if i == end:
                break
            x, _, query_result = l(x, cond, local_cond, query=query, skip_skip=True, rate=rate)
            if i >= start:
                latent_activities.append(torch.stack(query_result, dim=1))
        latent_activities = torch.stack(latent_activities, dim=1)          # [(B C), n, q, L, H]
        B_C, n, q, L, H = latent_activities.shape
        return latent_activities.reshape(B, self.C, n, q, L, H)
