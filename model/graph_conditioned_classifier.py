"""
GraphConditionedClassifier
Wraps the original EEGDM Classifier without modifying it.
Injects graph-conditioned FiLM modulation between reducer and decoder.

Pipeline:
    input
     ↓
    extractor  (unchanged)
     ↓
    reducer    (unchanged)
     ↓
    FiLM(tokens, graph_embedding)   ← NEW
     ↓
    decoder    (unchanged)
     ↓
    classifier (unchanged)
"""
import torch
import torch.nn as nn
from .graph_encoder import GraphEncoder
from .graph_utils import vector_to_adjacency
from .graph_latent_modulation import GraphLatentModulation
from .alignment import AlignmentHead


class GraphConditionedClassifier(nn.Module):
    """
    Wraps original EEGDM Classifier.
    Adds graph-conditioned FiLM between reducer and decoder.

    Args:
        classifier   : original EEGDM Classifier instance
        graph_dim    : output dim of GraphEncoder (default 256)
        token_dim    : last dim H of latent tokens (default 128 = d_model)
        num_nodes    : number of EEG channels (default 19)
        gcn_hidden   : hidden dim of GCN layers (default 128)
        gcn_layers   : number of GCN layers (default 3)
    """
    def __init__(
        self,
        classifier,
        graph_dim  = 256,
        token_dim  = 128,
        num_nodes  = 19,
        gcn_hidden = 128,
        gcn_layers = 3,
        use_graph  = True,
    ):
        super().__init__()
        self.classifier = classifier
        self.use_graph  = use_graph
        self.graph_encoder   = GraphEncoder(
            num_nodes  = num_nodes,
            hidden_dim = gcn_hidden,
            out_dim    = graph_dim,
            layers     = gcn_layers,
        ) if use_graph else None
        self.graph_modulator = GraphLatentModulation(
            token_dim = token_dim,
            graph_dim = graph_dim,
        ) if use_graph else None
        self.alignment_head = AlignmentHead(
            token_dim = token_dim,
            graph_dim = graph_dim,
            proj_dim  = 128,
        ) if use_graph else None

    def forward(self, input, icoh_vec, data_is_cached=False, rate=1, return_alignment=False):
        """
        Args:
            input         : EEG signal or cached tokens
            icoh_vec      : [B, 171] iCOH upper triangle vector
            data_is_cached: if True, input is already cached tokens
            rate          : SSM rate (passed to extractor)
        Returns:
            cls           : [B, n_class] logits
        """
        # ── Step 1: Extract latent tokens (unchanged from EEGDM) ──────────
        if not data_is_cached:
            latent_activity = self.classifier.extractor(input, rate=rate)
            tokens = self.classifier.reducer(latent_activity)
        else:
            tokens = input[
                :,
                self.classifier.extractor.start:self.classifier.extractor.end,
                :, :, :, :, :
            ]

        # ── Step 2: Graph-conditioned FiLM on latent tokens ───────────────
        if self.use_graph:
            adj       = vector_to_adjacency(icoh_vec)
            graph_emb = self.graph_encoder(adj)
            tokens    = self.graph_modulator(tokens, graph_emb)
        else:
            graph_emb = None

        # ── Step 3: Decode and classify (unchanged from EEGDM) ────────────
        rep = self.classifier.decoder(tokens)[self.classifier.use_rep_idx]
        cls = self.classifier.classifier(rep)

        if return_alignment:
            if self.use_graph:
                z_token, z_graph = self.alignment_head(tokens, graph_emb)
            else:
                z_token, z_graph = None, None
            return cls, z_token, z_graph

        return cls

    def get_new_params(self):
        """Return only the new graph conditioning parameters."""
        if not self.use_graph:
            return []
        return list(self.graph_encoder.parameters()) + \
               list(self.graph_modulator.parameters()) + \
               list(self.alignment_head.parameters())

    def get_backbone_params(self):
        """Return original EEGDM classifier parameters."""
        return list(self.classifier.parameters())
