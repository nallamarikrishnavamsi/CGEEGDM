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
    deg = adj.sum(-1).clamp(min=0)          # [B, N]
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


class GraphEncoderChannelwiseEdgeDrop(nn.Module):
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
                 out_dim=128, layers=3, dropout=0.1, edge_dropout=0.0):
        """
        edge_dropout : intrinsic edge dropout applied to adj BEFORE
            symmetric_normalize, during training only. Default 0.0 (off).
            NOTE: original graph_encoder.py deliberately removed an
            equivalent mechanism, reasoning that vector_to_adjacency's
            augment_icoh path already perturbs the same adjacency and a
            second dropout here would double-perturb it. This parameter
            reintroduces it anyway for the channel-wise variant, as an
            explicit experimental choice -- set augment_icoh=0 on the
            caller if you want this mechanism in isolation rather than
            stacked with the other one.
        """
        super().__init__()
        self.edge_dropout = edge_dropout
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
            node_embeddings  : [B, N, hidden_dim] -- L2-normalized per node
                                (dim=-1), for FiLM conditioning.
            pooled_for_align : [B, hidden_dim] -- L2-normalized, via the
                                SAME LearnedReadout mechanism as the
                                original file, for AlignmentHead ONLY.
                                Explicitly NOT used for FiLM.
        """
        B = adj.size(0)

        # Intrinsic edge dropout (channel-wise variant only -- see __init__
        # docstring for why the original file removed this).
        if self.training and self.edge_dropout > 0:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=torch.bool).unsqueeze(0)
            drop_mask = (torch.rand_like(adj) < self.edge_dropout) & (~eye)
            adj = adj.masked_fill(drop_mask, 0.0)

        adj_norm = symmetric_normalize(adj)

        if node_features is not None:
            x = node_features
        else:
            x = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            x = x + self.dropout(layer(x, adj_norm))   # residual per layer, same as original

        # NEW: return per-node embeddings directly, L2-normalized per node
        # for the same FiLM-input-stability reason the original normalizes
        # its single pooled readout.
        node_embeddings = F.normalize(x, dim=-1)  # [B, N, hidden_dim]

        # Separate, explicit pooling -- ONLY for the alignment branch.
        # Uses the same LearnedReadout as the original file, applied here
        # to the un-normalized node features (matching original's own
        # order: pool first, then normalize).
        pooled, node_weights = self.readout(x)
        self._last_node_weights = node_weights.detach()
        pooled_for_align = F.normalize(pooled, dim=-1)  # [B, hidden_dim]

        return node_embeddings, pooled_for_align
