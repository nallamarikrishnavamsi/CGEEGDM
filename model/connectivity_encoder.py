import torch
import torch.nn as nn

class ConnectivityEncoder(nn.Module):
    """
    MLP encoder for iCOH upper triangle vector.
    Input : [B, 171]  (19*18/2 = 171 features)
    Output: [B, 256]  connectivity embedding c
    """
    def __init__(self, in_dim=171, hidden_dim=256, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: [B, 171]
        return self.encoder(x)  # [B, 256]
