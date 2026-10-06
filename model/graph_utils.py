import torch

NUM_NODES = 20  # 20 of EEGDM's 22 bipolar TCP channels (A1-T3/A2-T4 unavailable, see src/bipolar.py)

def vector_to_adjacency(vec, augment=False, noise_std=0.05, edge_dropout_p=0.1,
                        allow_negative=None):
    """
    Convert iCOH vector [B,190] -> adjacency [B,20,20]

    Args:
        augment        : if True (training only), applies noise + edge dropout
                         to combat memorization of per-sample iCOH fingerprints.
        noise_std      : std of Gaussian noise added to edge weights.
        edge_dropout_p : probability of zeroing an edge.
        allow_negative : clamp bounds for the augmentation step.
            None  -- inferred from the data (default). iCOH is non-negative
                     by construction; Pearson lies in [-1, 1].
            False -- clamp to [0, inf). Correct for iCOH: noise can push a
                     small edge below zero, and clamping restores the
                     measure's own domain.
            True  -- clamp to [-1, 1]. Required for Pearson: clamping
                     negatives to zero discards 33.9% of the edges
                     (measured on data/pearson) and silently converts the
                     measure into a non-negative one.
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
        if allow_negative is None:
            allow_negative = bool((vec < 0).any())
        if allow_negative:
            adj = adj.clamp(min=-1.0, max=1.0)   # signed: Pearson
        else:
            adj = adj.clamp(min=0.0)             # non-negative: iCOH

    eye = torch.eye(NUM_NODES, device=device).unsqueeze(0)
    adj = adj + eye

    return adj
