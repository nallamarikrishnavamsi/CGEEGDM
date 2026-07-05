import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):

        deg = adj.sum(-1, keepdim=True) + 1e-6
        adj = adj / deg

        x = torch.bmm(adj, x)
        x = self.linear(x)

        return F.relu(x)


class GraphEncoder(nn.Module):

    def __init__(
        self,
        num_nodes=19,
        hidden_dim=128,
        out_dim=256,
        layers=3,
    ):
        super().__init__()

        self.node_embedding = nn.Parameter(
            torch.randn(num_nodes, hidden_dim)
        )

        self.layers = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim)
            for _ in range(layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, adj):

        B = adj.size(0)

        x = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            x = layer(x, adj)

        x = x.transpose(1, 2)

        x = self.pool(x).squeeze(-1)

        return self.proj(x)
