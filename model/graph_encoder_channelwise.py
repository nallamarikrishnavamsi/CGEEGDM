"""
graph_encoder_channelwise.py

Channel-wise variant of model/graph_encoder.py (ORIGINAL FILE UNCHANGED).

Original GraphEncoder.forward returns ONE pooled/readout embedding [B, hidden_dim]
that conditions every EEG channel identically. This variant instead returns the
per-node (per-channel) embeddings [B, N, hidden_dim] BEFORE readout, so channel c's
EEG representation can be conditioned by channel c's own graph embedding.

Everything else -- GraphConv, symmetric_normalize, the GCN layer stack, residual
connections, dropout -- is copied unchanged from graph_encoder.py. Only the return
value of forward() differs: node embeddings instead of the pooled readout.

The existing LearnedReadout is KEPT (not deleted) and still used, but only to
produce a SEPARATE pooled embedding for the alignment branch (see
GROUND_TRUTH: original graph_conditioned_classifier.py's alignment_head expects
a single global [B, graph_dim] vector; we preserve that by pooling explicitly
here, rather than changing AlignmentHead itself).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_normalize(adj):
    """Identical to graph_encoder.py's symmetric_normalize. Kipf & Welling
    D^-1/2 A D^-1/2, with degree clamped >= 0 before pow(-0.5)."""
    deg = adj.abs().sum(-1).clamp(min=1e-6)  # [B, N]
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt) | torch.isnan(d_inv_sqrt)] = 0.0
    D = torch.diag_embed(d_inv_sqrt)        # [B, N, N]
    return D @ adj @ D


class GraphConv(nn.Module):
    """Identical to graph_encoder.py's GraphConv."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        x = torch.bmm(adj_norm, x)
        x = self.linear(x)
        return F.relu(x)


class LearnedReadout(nn.Module):
    """Identical to graph_encoder.py's LearnedReadout. Kept and used ONLY
    for the separate alignment-branch pooling in GraphEncoderChannelwise."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        score  = self.score_mlp(x)              # [B, N, 1]
        weight = torch.softmax(score, dim=1)
        readout = (weight * x).sum(dim=1)        # [B, hidden_dim]
        return readout, weight.squeeze(-1)


class GraphEncoderChannelwise(nn.Module):
    """
    Multi-layer GCN over the iCOH-derived adjacency matrix -- same GCN body
    as GraphEncoder, but forward() returns PER-NODE embeddings instead of a
    pooled global embedding.

    forward() returns a tuple:
        node_embeddings : [B, N, hidden_dim]  -- for channel-wise FiLM
        pooled_for_align: [B, hidden_dim]     -- for AlignmentHead only,
                                                  via the SAME LearnedReadout
                                                  used in the original file
                                                  (kept, not reimplemented)

    Args identical to GraphEncoder (num_nodes, node_feat_dim, hidden_dim,
    out_dim, layers, dropout). hidden_dim must equal out_dim, same
    constraint as the original (no projection is applied to node features).
    """
    def __init__(self, num_nodes=20, node_feat_dim=128, hidden_dim=128,
                 out_dim=128, layers=3, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        assert hidden_dim == out_dim, (
            "GraphEncoderChannelwise returns node embeddings directly (no "
            f"projection); hidden_dim ({hidden_dim}) must equal out_dim ({out_dim})."
        )

        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))

        self.layers  = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.readout = LearnedReadout(hidden_dim)  # kept, alignment-branch only
        self._last_node_weights = None

    def forward(self, adj, node_features=None):
        """
        adj           : [B, N, N] iCOH-derived adjacency
        node_features : [B, N, node_feat_dim] EEG-derived per-channel
                        features (optional; falls back to learnable
                        embedding if None, matching original behavior).

        Returns:
            node_embeddings  : [B, N, hidden_dim] -- raw (no L2)
                                (dim=-1), for FiLM conditioning.
            pooled_for_align : [B, hidden_dim] -- raw, via the
                                SAME LearnedReadout mechanism as the
                                original file, for AlignmentHead ONLY.
                                Explicitly NOT used for FiLM.
        """
        B = adj.size(0)

        # No internal edge dropout, same reasoning as original: vector_to_adjacency()
        # already applies augmentation upstream.
        adj_norm = symmetric_normalize(adj)

        if node_features is not None:
            x = node_features
        else:
            x = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        # Residual GCN (Kipf & Welling 2017, App. B, Eq. 14): H' = sigma(A H W) + H.
        # Dropout on the layer INPUT of the first and last layer only (App. B).
        L = len(self.layers)
        for l, layer in enumerate(self.layers):
            h = self.dropout(x) if l in (0, L - 1) else x
            x = x + layer(h, adj_norm)

        node_embeddings = x  # [B, N, hidden_dim], passed to FiLM as-is (Perez et al. 2018)

        # Separate, explicit pooling -- ONLY for the alignment branch.
        # Uses the same LearnedReadout as the original file, applied here
        # to the un-normalized node features (matching original's own
        # order: pool first, then normalize).
        pooled, node_weights = self.readout(x)
        self._last_node_weights = node_weights.detach()
        pooled_for_align = pooled  # [B, hidden_dim]; AlignmentHead normalizes its own projections

        return node_embeddings, pooled_for_align
