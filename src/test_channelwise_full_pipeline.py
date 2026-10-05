"""
Full-pipeline shape-flow test: real backbone, real cached data, forward
pass only -- no training, no optimizer step.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader

from model.classifier import Classifier
from model.graph_conditioned_classifier_channelwise import GraphConditionedClassifierChannelwise
from dataloader.ConnectivityTUEVDataset import ConnectivityHMSDatasetCached
from finetune_graphcond_channelwise import load_original_backbone, CLASSIFIER_MODEL_KWARGS

DATA_ROOT     = '/home/dsamantaai/krishna/data'
SIGNAL_CACHE  = 'data/signal_cache'
BACKBONE_CKPT = 'checkpoints/backbone.ckpt'

print("Loading one real validation batch...")
val_ds = ConnectivityHMSDatasetCached(
    DATA_ROOT, 'finetune_val', SIGNAL_CACHE, window_sec=10,
    connectivity_measure='icoh', pearson_cache_dir='data/pearson',
)
loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
signal, soft_label, icoh_vec = next(iter(loader))

print("EEG input (signal) shape :", tuple(signal.shape))
print("iCOH vector shape        :", tuple(icoh_vec.shape), "(expect [B, 190] for 20 channels)")

print("\nLoading backbone + building channel-wise model...")
backbone = load_original_backbone(BACKBONE_CKPT)
base_classifier = Classifier(model=backbone, **CLASSIFIER_MODEL_KWARGS)
model = GraphConditionedClassifierChannelwise(
    classifier=base_classifier, graph_dim=128, token_dim=128, num_nodes=20,
    gcn_hidden=128, gcn_layers=3, gcn_dropout=0.1, use_graph=True,
).eval()

print("\nRunning forward pass (no_grad, eval mode -- no training)...")
with torch.no_grad():
    latent_activity = model.classifier.extractor((signal, None), rate=1)
    tokens = model.classifier.reducer(latent_activity)
    print("EEG latent representation shape :", tuple(latent_activity.shape))
    print("Reduced EEG token shape         :", tuple(tokens.shape),
          "(expect [..., C=20, H=128] with C at dim -2)")

    from model.graph_utils import vector_to_adjacency
    adj = vector_to_adjacency(icoh_vec, augment=False)
    print("Adjacency shape                 :", tuple(adj.shape), "(expect [B, 20, 20])")

    node_feats = tokens.mean(dim=(1, 3)).reshape(tokens.size(0), 20, -1)
    print("Node features (encoder input)   :", tuple(node_feats.shape), "(expect [B, 20, 128])")

    # Current GraphEncoderChannelwise.forward() returns a tuple:
    # (node_embeddings, pooled_for_align) -- unpack both.
    graph_nodes, pooled_for_align = model.graph_encoder(adj, node_features=node_feats)
    print("Graph node embedding shape      :", tuple(graph_nodes.shape),
          "(expect [B, 20, 128], NOT pooled)")
    print("Pooled-for-alignment shape      :", tuple(pooled_for_align.shape),
          "(expect [B, 128], alignment branch only)")

    film_out = model.graph_modulator(tokens, graph_nodes, warmup_alpha=1.0)
    print("GraphCond FiLM input shapes     : tokens", tuple(tokens.shape),
          " graph", tuple(graph_nodes.shape))
    print("GraphCond FiLM output shape     :", tuple(film_out.shape), "(expect same as tokens)")
    assert film_out.shape == tokens.shape

    rep = model.classifier.decoder(film_out)[model.classifier.use_rep_idx]
    print("Decoder (LFT-equivalent) output shape:", tuple(rep.shape))
    cls = model.classifier.classifier(rep)
    print("Classifier output shape         :", tuple(cls.shape), "(expect [B, 6])")

    print("\n--- Explicit channel correspondence check ---")
    print(f"graph_nodes[:, 0, :] conditions tokens[..., 0, :]  (EEG channel 0)")
    print(f"graph_nodes[:, 19, :] conditions tokens[..., 19, :] (EEG channel 19)")
    same_count = graph_nodes.shape[1] == tokens.shape[-2]
    print(f"graph_nodes.shape[1] == tokens.shape[-2] == {same_count} -> channel counts match")
    assert same_count, "channel count mismatch between graph nodes and tokens"

    print("\nFull forward with alignment:")
    cls2, z_token, z_graph = model((signal, None), icoh_vec, return_alignment=True, warmup_alpha=1.0)
    print("  cls shape     :", tuple(cls2.shape))
    print("  z_token shape :", tuple(z_token.shape), "(pooled, for alignment loss only)")
    print("  z_graph shape :", tuple(z_graph.shape), "(pooled, for alignment loss only)")

print("\n=== FULL-PIPELINE SHAPE TEST PASSED ===")
