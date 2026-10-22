import torch

NUM_NODES = 20  # 20 of EEGDM's 22 bipolar TCP channels (A1-T3/A2-T4 unavailable, see src/bipolar.py)

def vector_to_adjacency(vec, augment=False, edge_dropout_p=0.1):
    """
    Convert iCOH vector [B,190] -> adjacency [B,20,20] with self-loops (A + I).

    augment (training only): DropEdge (Rong et al., 2020). Each undirected
    edge is removed with probability edge_dropout_p (symmetric mask over the
    upper triangle, mirrored). Self-loops are added afterwards and are never
    dropped; GraphEncoder re-normalizes D^-1/2 (A+I) D^-1/2 after dropping.
    """
    B = vec.shape[0]
    device = vec.device

    adj = torch.zeros(B, NUM_NODES, NUM_NODES, device=device, dtype=vec.dtype)
    idx = 0
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            adj[:, i, j] = vec[:, idx]
            adj[:, j, i] = vec[:, idx]
            idx += 1

    if augment:
        mask = (torch.rand(B, NUM_NODES, NUM_NODES, device=device) > edge_dropout_p).to(adj.dtype)
        mask = torch.triu(mask, diagonal=1)
        adj = adj * (mask + mask.transpose(1, 2))

    adj = adj + torch.eye(NUM_NODES, device=device, dtype=adj.dtype).unsqueeze(0)
    return adj
