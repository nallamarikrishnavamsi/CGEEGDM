import torch
import torch.nn as nn


class GraphLatentModulation(nn.Module):
    """
    Gated, joint-conditioned FiLM.

    T' = T + warmup_alpha * gate(T,G) * ( alpha(T,G)*T + beta(T,G) - T )

    - alpha/beta depend on concat(token, graph) -> joint conditioning.
    - gate(T,G) is a learned sigmoid in [0,1] that lets the model decide,
      per-sample and per-feature, how much graph conditioning to apply.
      Some samples may benefit from near-zero graph influence; the gate
      lets the model express that instead of always applying a fixed-size
      correction.
    - Identity at init: alpha=1, beta=0 (zero-init final layer), gate
      starts near 0.5 (neutral) via zero-init gate layer + zero bias
      (sigmoid(0)=0.5). warmup_alpha (externally scheduled, 0->1) is
      what damps the initial effective contribution now.
    - Dropout inside the MLPs regularizes the new graph branch so it
      cannot simply memorize per-sample connectivity patterns.
    """
    def __init__(self, token_dim, graph_dim=128, hidden_dim=256,
                 dropout=0.2):
        super().__init__()
        self.token_dim = token_dim
        in_dim = token_dim + graph_dim

        self.alpha_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

        # Identity init for alpha/beta
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.ones_(self.alpha_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

        # Neutral init for gate: zero weight + zero bias -> sigmoid(0) = 0.5
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, tokens, graph, warmup_alpha=1.0):
        B = tokens.size(0)
        H = tokens.size(-1)
        mid_dims = tokens.shape[1:-1]

        g_shape = [B] + [1] * len(mid_dims) + [graph.size(-1)]
        g_broadcast = graph.view(*g_shape).expand(B, *mid_dims, graph.size(-1))

        joint = torch.cat([tokens, g_broadcast], dim=-1)
        alpha = self.alpha_net(joint)
        beta  = self.beta_net(joint)
        gate  = torch.sigmoid(self.gate_net(joint))   # [0,1], per-sample/per-feature

        film_out = alpha * tokens + beta
        # warmup_alpha (0->1, set externally) suppresses the graph branch at
        # the start of training and gradually ramps it in over the first few
        # epochs. gate additionally lets the model down-weight the correction
        # on a per-sample basis.
        return tokens + warmup_alpha * gate * (film_out - tokens)
