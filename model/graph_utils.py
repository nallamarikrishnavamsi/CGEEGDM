import torch

NUM_NODES = 19

def vector_to_adjacency(vec, augment=False, noise_std=0.05, edge_dropout_p=0.1):
    """
    Convert iCOH vector [B,171] -> adjacency [B,19,19]

    Args:
        augment        : if True (training only), applies noise + edge dropout
                         to combat memorization of per-sample iCOH fingerprints.
        noise_std      : std of Gaussian noise added to edge weights.
        edge_dropout_p : probability of zeroing an edge.
    """
    B = vec.shape[0]
    device = vec.device

    adj = torch.zeros(
        B,
        NUM_NODES,
        NUM_NODES,
        device=device,
        dtype=vec.dtype,
    )

    idx = 0
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            adj[:, i, j] = vec[:, idx]
            adj[:, j, i] = vec[:, idx]
            idx += 1

    if augment:
        noise = torch.randn_like(adj) * noise_std
        noise = (noise + noise.transpose(1, 2)) / 2  # keep symmetric
        adj = adj + noise
        mask = (torch.rand(B, NUM_NODES, NUM_NODES, device=device) > edge_dropout_p).float()
        mask = torch.triu(mask, diagonal=1)
        mask = mask + mask.transpose(1, 2) + torch.eye(NUM_NODES, device=device).unsqueeze(0)
        adj = adj * mask
        adj = adj.clamp(min=0.0)  # iCOH is non-negative

    eye = torch.eye(NUM_NODES, device=device).unsqueeze(0)
    adj = adj + eye

    return adj
