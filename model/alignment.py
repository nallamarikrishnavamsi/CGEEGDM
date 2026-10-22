"""
Graph-Latent Representation Alignment.
Encourages latent EEG representation and graph (functional connectivity)
representation to agree, since both encode the same brain activity.
Used only during training; adds no inference cost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """Projects pooled latent tokens and graph embedding into shared space."""
    def __init__(self, token_dim=128, graph_dim=128, proj_dim=128):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, tokens, graph_emb):
        B, H = tokens.size(0), tokens.size(-1)
        pooled = tokens.reshape(B, -1, H).mean(dim=1)
        z_token = F.normalize(self.token_proj(pooled), dim=-1)
        z_graph = F.normalize(self.graph_proj(graph_emb), dim=-1)
        return z_token, z_graph


def cosine_alignment_loss(z_token, z_graph):
    return (1.0 - (z_token * z_graph).sum(dim=-1)).mean()
