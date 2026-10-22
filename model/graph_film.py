"""
FiLM (Perez et al., 2018):  T' = gamma * T + beta,  gamma = 1 + d_gamma.

d_gamma and beta each come from a single linear layer (the FiLM generator),
as in the original. No gate, no warm-up, no dropout in the generator.
Kept from the previous version: the generator input is concat(token, graph),
so gamma/beta are computed per token from the token and the graph embedding.
Identity at init: generator weights and biases are zero -> gamma = 1, beta = 0.
"""
import torch
import torch.nn as nn


class GraphFiLM(nn.Module):
    def __init__(self, token_dim, graph_dim=128):
        super().__init__()
        in_dim = token_dim + graph_dim
        self.d_gamma = nn.Linear(in_dim, token_dim)
        self.beta = nn.Linear(in_dim, token_dim)
        for m in (self.d_gamma, self.beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens, graph):
        B = tokens.size(0)
        mid = tokens.shape[1:-1]
        g = graph.view(B, *([1] * len(mid)), graph.size(-1)).expand(B, *mid, graph.size(-1))
        joint = torch.cat([tokens, g], dim=-1)
        gamma = 1.0 + self.d_gamma(joint)
        beta = self.beta(joint)
        return gamma * tokens + beta


class GraphFiLMChannelwise(nn.Module):
    """
    Channel-wise FiLM: T'_c = gamma_c * T_c + beta_c,  gamma_c = 1 + d_gamma.
    Same generator as GraphFiLM (one linear layer each, input concat(T_c, G_c)),
    but G_c is graph node c's own embedding instead of one global vector.
    tokens: [B, ..., C, H]   graph_nodes: [B, C, Dg]
    Identity at init: generator weights and biases are zero.
    """
    def __init__(self, token_dim, graph_dim=128):
        super().__init__()
        in_dim = token_dim + graph_dim
        self.d_gamma = nn.Linear(in_dim, token_dim)
        self.beta = nn.Linear(in_dim, token_dim)
        for m in (self.d_gamma, self.beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens, graph_nodes):
        B, C = tokens.size(0), tokens.shape[-2]
        assert graph_nodes.shape[1] == C, 'graph node c must be EEG channel c'
        mid = tokens.shape[1:-2]
        Dg = graph_nodes.size(-1)
        g = graph_nodes.view(B, *([1] * len(mid)), C, Dg).expand(B, *mid, C, Dg)
        joint = torch.cat([tokens, g], dim=-1)
        return (1.0 + self.d_gamma(joint)) * tokens + self.beta(joint)
