import torch
import torch.nn as nn


class GraphLatentModulation(nn.Module):
    """
    Gated, joint-conditioned FiLM.

    T' = T + gate(T,G) * ( alpha(T,G)*T + beta(T,G) - T )

    - alpha/beta depend on concat(token, graph) -> joint conditioning.
    - gate(T,G) is a learned sigmoid in [0,1] that lets the model decide,
      per-sample and per-feature, how much graph conditioning to apply.
      Some samples may benefit from near-zero graph influence; the gate
      lets the model express that instead of always applying a fixed-size
      correction.
    - Identity at init: alpha=1, beta=0 (zero-init final layer), gate
      starts near 0.5 (neutral) via zero-init gate layer + zero bias
      (sigmoid(0)=0.5).
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

    def forward(self, tokens, graph):
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
        # gate lets the model down-weight the correction per sample/feature.
        return tokens + gate * (film_out - tokens)
"""
graph_latent_modulation_channelwise.py

Channel-wise variant of model/graph_latent_modulation.py (ORIGINAL FILE UNCHANGED).

Original GraphLatentModulation.forward() broadcasts ONE global graph vector
[B, graph_dim] identically across every dim of tokens except batch and H,
including the channel dim -- so every EEG channel receives the same
conditioning.

This variant instead takes per-channel graph embeddings [B, C, graph_dim] and
broadcasts each channel c's own embedding only across the NON-channel middle
dims of tokens, so tokens[..., c, :] is conditioned only by graph[:, c, :].

Traced from the ACTUAL current tensor shape produced by
model/classifier.py's LatentActivityReducer + model/graph_conditioned_classifier.py
(confirmed on the live server, not assumed):

    tokens : [B, n_layer, 1, pool, 1, C, H]   (7-D; channel dim is index -2)

The FiLM math itself (alpha/beta/gate nets, identity init, gated residual) is copied UNCHANGED from the original -- per instructions, only
the conditioning INPUT changes from global to per-channel, never the formula.
"""
import torch
import torch.nn as nn


class GraphLatentModulationChannelwise(nn.Module):
    """
    Gated, joint-conditioned FiLM -- channel-wise variant.

    T'_c = T_c + gate(T_c,G_c) * ( alpha(T_c,G_c)*T_c + beta(T_c,G_c) - T_c )

    Identical formula to GraphLatentModulation; G is now per-channel (G_c)
    instead of one global G shared across all channels.
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

        # Identity init for alpha/beta -- unchanged from original
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.ones_(self.alpha_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

        # Neutral init for gate -- unchanged from original
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, tokens, graph_nodes):
        """
        tokens      : [B, ..., C, H]  -- channel dim is index -2, H is index -1.
                      Concretely [B, n_layer, 1, pool, 1, C, H] in the current
                      pipeline, but this code only assumes "channel is -2,
                      H is -1" -- it does not hardcode the number or names
                      of the middle dims, so it is robust to n_layer/pool/
                      query-count changes elsewhere in the pipeline.
        graph_nodes : [B, C, graph_dim]  -- per-channel graph embeddings.
        """
        B = tokens.size(0)
        C_tokens = tokens.shape[-2]
        H = tokens.shape[-1]

        C_graph = graph_nodes.shape[1]
        assert C_tokens == C_graph, (
            f"Channel mismatch: tokens has C={C_tokens} (dim -2) but "
            f"graph_nodes has C={C_graph} (dim 1). Channel ordering/count "
            f"must match between the EEG latent tokens and the graph nodes."
        )

        mid_dims = tokens.shape[1:-2]  # dims strictly between batch and (C, H)
        Dg = graph_nodes.size(-1)

        # Reshape graph_nodes [B, C, Dg] -> [B, 1, 1, ..., 1, C, Dg] (one
        # singleton per mid_dim) so each channel's OWN embedding broadcasts
        # only across the non-channel middle dims -- never across channels.
        g_shape = [B] + [1] * len(mid_dims) + [C_tokens, Dg]
        g_broadcast = graph_nodes.view(*g_shape).expand(B, *mid_dims, C_tokens, Dg)

        joint = torch.cat([tokens, g_broadcast], dim=-1)  # [..., C, H+Dg]
        alpha = self.alpha_net(joint)
        beta  = self.beta_net(joint)
        gate  = torch.sigmoid(self.gate_net(joint))

        film_out = alpha * tokens + beta
        return tokens + gate * (film_out - tokens)
