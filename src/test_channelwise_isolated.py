"""
Isolated shape + channel-isolation test for the channel-wise graph
modules. Pure synthetic tensors, no real data/backbone.

CHANNEL-ISOLATION TEST DESIGN NOTE:
GraphLatentModulationChannelwise's real init makes alpha=1, beta=0,
gate=sigmoid(0)=0.5 EXACTLY for every input, at construction time. A
freshly constructed instance's output is therefore completely
insensitive to EVERY input including the graph embedding -- a naive
"perturb one channel and check which outputs numerically change" test
would trivially report "nothing changed anywhere" against the real init.
That is correct behavior for the real module, not a bug.

This test instead uses an autograd-based check: give a TEST-ONLY fresh
instance non-identity weights purely so it responds to inputs at all,
then verify d(output at channel c)/d(graph_nodes at channel c' != c) is
EXACTLY zero via backprop. This holds structurally (nn.Linear only mixes
the last dim) regardless of weights, trained or not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from model.graph_encoder_channelwise import GraphEncoderChannelwise
from model.graph_latent_modulation_channelwise import GraphLatentModulationChannelwise

torch.manual_seed(0)

B, C, H, Dg = 2, 20, 128, 128
n_layer, pool = 3, 5

print("=== 1. GraphEncoderChannelwise shape test ===")
adj = torch.rand(B, C, C)
adj = (adj + adj.transpose(1, 2)) / 2
adj = adj + torch.eye(C).unsqueeze(0)
node_feats = torch.randn(B, C, H)

enc = GraphEncoderChannelwise(num_nodes=C, hidden_dim=H, out_dim=H, layers=3, dropout=0.0).eval()
with torch.no_grad():
    node_out, pooled_for_align = enc(adj, node_features=node_feats)  # forward() returns a tuple
print("adj shape             :", tuple(adj.shape), "(expect", (B, C, C), ")")
print("node_feats shape       :", tuple(node_feats.shape), "(expect", (B, C, H), ")")
print("node_out shape         :", tuple(node_out.shape), "(expect", (B, C, H), ", NOT pooled)")
assert node_out.shape == (B, C, H), "GraphEncoderChannelwise output shape is wrong"

pooled = pooled_for_align  # no separate .pool() method exists anymore; forward() already returns it
print("pooled (.pool()) shape :", tuple(pooled.shape), "(expect", (B, H), ")")
assert pooled.shape == (B, H)
print("PASS\n")


print("=== 2. GraphLatentModulationChannelwise: shape test ===")
tokens = torch.randn(B, n_layer, 1, pool, 1, C, H)
graph_nodes = torch.randn(B, C, Dg)

film = GraphLatentModulationChannelwise(token_dim=H, graph_dim=Dg)
with torch.no_grad():
    out = film(tokens, graph_nodes, warmup_alpha=1.0)
print("tokens shape      :", tuple(tokens.shape))
print("graph_nodes shape :", tuple(graph_nodes.shape))
print("FiLM output shape :", tuple(out.shape), "(expect same as tokens)")
assert out.shape == tokens.shape
print("PASS\n")


print("=== 3. GraphLatentModulationChannelwise: GRADIENT-BASED channel isolation proof ===")
print("(see this file's module docstring for why this is the correct check)\n")

film2 = GraphLatentModulationChannelwise(token_dim=H, graph_dim=Dg)
for net in (film2.alpha_net, film2.beta_net, film2.gate_net):
    nn.init.normal_(net[-1].weight, std=0.05)
    nn.init.normal_(net[-1].bias, std=0.05)

tokens2 = torch.randn(B, n_layer, 1, pool, 1, C, H)
graph_nodes2 = torch.randn(B, C, Dg, requires_grad=True)

out2 = film2(tokens2, graph_nodes2, warmup_alpha=1.0)

target_channel = 5
loss = out2[..., target_channel, :].sum()
loss.backward()

grad = graph_nodes2.grad
grad_target = grad[:, target_channel, :].abs().sum().item()
grad_elsewhere = grad.clone()
grad_elsewhere[:, target_channel, :] = 0
max_other_grad = grad_elsewhere.abs().max().item()

print(f"|gradient| w.r.t. target channel {target_channel}'s graph embedding : {grad_target:.6f} (expect > 0)")
print(f"max |gradient| w.r.t. ALL OTHER channels' graph embeddings         : {max_other_grad:.10f} (expect exactly 0)")

assert grad_target > 0, "Target channel gradient is zero -- test setup issue."
assert max_other_grad == 0.0, (
    f"FAIL: output at channel {target_channel} has NONZERO gradient with "
    "respect to other channels' graph embeddings -- channel mixing detected!"
)
print("\nPASS: output at channel c depends ONLY on channel c's own graph embedding.")
print("\n=== ALL ISOLATED TESTS PASSED ===")
