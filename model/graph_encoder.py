import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_normalize(adj):
    """
    Kipf & Welling symmetric normalization: D^-1/2 A D^-1/2
    adj: [B, N, N]
    Clamps degree to be non-negative before pow(-0.5) to avoid NaN
    from negative-weight edges (real iCOH is always in [0,1], but this
    guards against any upstream numerical edge case).
    """
    deg = adj.sum(-1).clamp(min=0)          # [B, N]
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt) | torch.isnan(d_inv_sqrt)] = 0.0
    D = torch.diag_embed(d_inv_sqrt)        # [B, N, N]
    return D @ adj @ D


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        # adj_norm is already symmetrically normalized upstream
        x = torch.bmm(adj_norm, x)
        x = self.linear(x)
        return F.relu(x)


class LearnedReadout(nn.Module):
    """
    Attention-free learned pooling (weighted sum over nodes).
    Replaces plain mean-pooling with a per-node importance score,
    letting the model emphasize informative electrodes rather than
    treating all 19 channels equally.

    score = mlp(node_features)          [B, N, 1]
    weight = softmax(score, dim=nodes)  [B, N, 1]
    readout = sum(weight * node_features, dim=nodes)  [B, hidden_dim]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: [B, N, hidden_dim]
        score  = self.score_mlp(x)              # [B, N, 1]
        weight = torch.softmax(score, dim=1)    # softmax over nodes
        readout = (weight * x).sum(dim=1)        # [B, hidden_dim]
        return readout, weight.squeeze(-1)       # also return weights for interpretability


class GraphEncoder(nn.Module):
    """
    Multi-layer GCN over the iCOH-derived adjacency matrix.

    Node features are EEG-derived (per-channel pooled latent tokens from
    the pretrained EEGDM backbone) rather than a single fixed learnable
    embedding shared across all samples. Falls back to a learnable node
    embedding when node_features is None (e.g. baseline compatibility).

    Uses symmetric normalization (standard GCN), residual connections
    between layers to mitigate over-smoothing, and learned attention-free
    readout (weighted sum) instead of plain mean pooling, so the model
    can emphasize informative electrodes.
    """
    def __init__(self, num_nodes=19, node_feat_dim=128, hidden_dim=128,
                 out_dim=256, layers=3, dropout=0.1, edge_dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_dropout = edge_dropout

        # Project EEG-derived per-channel features down to hidden_dim
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Fallback learnable embedding (used if node_features not provided)
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        self.layers  = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.readout = LearnedReadout(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._last_node_weights = None  # for interpretability/debugging

    def forward(self, adj, node_features=None):
        """
        adj           : [B, N, N] iCOH-derived adjacency
        node_features : [B, N, node_feat_dim] EEG-derived per-channel
                        features (optional). If None, uses the fixed
                        learnable node embedding (old behavior).
        """
        B = adj.size(0)

        # Edge dropout: randomly zero out off-diagonal edges during training
        # to regularize the graph branch (self-loops on the diagonal are
        # always preserved since they carry the node's own identity).
        if self.training and self.edge_dropout > 0:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=torch.bool).unsqueeze(0)
            drop_mask = (torch.rand_like(adj) < self.edge_dropout) & (~eye)
            adj = adj.masked_fill(drop_mask, 0.0)

        adj_norm = symmetric_normalize(adj)

        if node_features is not None:
            x = self.node_proj(node_features)          # [B, N, hidden_dim]
        else:
            x = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            x = x + self.dropout(layer(x, adj_norm))   # residual per layer

        readout, node_weights = self.readout(x)          # [B, hidden_dim], [B, N]
        self._last_node_weights = node_weights.detach()   # store for inspection
        graph_emb = self.proj(readout)
        graph_emb = F.normalize(graph_emb, dim=-1)        # stabilize FiLM inputs
        return graph_emb
