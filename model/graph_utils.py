import torch

NUM_NODES = 19

def vector_to_adjacency(vec):
    """
    Convert iCOH vector [B,171] -> adjacency [B,19,19]
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

    eye = torch.eye(NUM_NODES, device=device).unsqueeze(0)
    adj = adj + eye

    return adj
