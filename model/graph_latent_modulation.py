import torch
import torch.nn as nn

class GraphLatentModulation(nn.Module):
    """
    Adaptive Graph-conditioned Latent Modulation.

    T' = alpha(T, G) * T + beta(T, G)

    Unlike standard FiLM where gamma/beta depend only on the graph
    embedding G, here alpha and beta are produced from the CONCATENATION
    of the latent token T and graph embedding G, so the modulation can
    depend on both neural activity and functional connectivity jointly.

    tokens : [B, ..., H]   latent EEG tokens
    graph  : [B, G]        graph embedding (broadcast across token dims)
    """
    def __init__(self, token_dim, graph_dim=256, hidden_dim=256):
        super().__init__()
        self.token_dim = token_dim
        in_dim = token_dim + graph_dim

        self.alpha_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # Zero-init final layers: identity transform at start of training
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.ones_(self.alpha_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, tokens, graph):
        # tokens: [B, ..., H], graph: [B, G]
        B = tokens.size(0)
        H = tokens.size(-1)
        mid_dims = tokens.shape[1:-1]

        # Broadcast graph embedding to match tokens' middle dimensions
        g_shape = [B] + [1] * len(mid_dims) + [graph.size(-1)]
        g_broadcast = graph.view(*g_shape).expand(B, *mid_dims, graph.size(-1))

        # Concatenate token and graph features for joint conditioning
        joint = torch.cat([tokens, g_broadcast], dim=-1)  # [B, ..., H+G]

        alpha = self.alpha_net(joint)  # [B, ..., H]
        beta  = self.beta_net(joint)   # [B, ..., H]

        return alpha * tokens + beta
