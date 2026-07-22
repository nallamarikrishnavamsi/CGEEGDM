"""
GraphConditionedClassifier
Wraps the original EEGDM Classifier without modifying it.
Injects graph-conditioned FiLM modulation between reducer and decoder —
the only point where gradients can flow, since extractor/reducer are
both @torch.no_grad() in original EEGDM.

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
    Adds graph-conditioned FiLM at two points: pre-reducer and pre-decoder.

    Args:
        classifier   : original EEGDM Classifier instance
        graph_dim    : output dim of GraphEncoder (default 256)
        token_dim    : last dim H of latent tokens (default 128 = d_model)
        num_nodes    : number of EEG channels (default 19)
        gcn_hidden   : hidden dim of GCN layers (default 128)
        gcn_layers   : number of GCN layers (default 3)
        gcn_dropout  : dropout inside GCN layers (default 0.1)
        use_graph    : if False, acts as pure baseline (no graph conditioning)
        augment_icoh : if True (training only), applies noise + edge dropout
                       to the iCOH adjacency to combat per-sample memorization
        icoh_noise_std       : std of Gaussian noise for iCOH augmentation
        icoh_edge_dropout_p  : edge dropout probability for iCOH augmentation
    """
    def __init__(
        self,
        classifier,
        graph_dim  = 256,
        token_dim  = 128,
        num_nodes  = 19,
        gcn_hidden = 128,
        gcn_layers = 3,
        gcn_dropout = 0.1,
        use_graph  = True,
        augment_icoh = False,
        icoh_noise_std = 0.05,
        icoh_edge_dropout_p = 0.1,
    ):
        super().__init__()
        self.classifier = classifier
        self.use_graph  = use_graph
        self.augment_icoh = augment_icoh
        self.icoh_noise_std = icoh_noise_std
        self.icoh_edge_dropout_p = icoh_edge_dropout_p
        self.graph_encoder   = GraphEncoder(
            num_nodes  = num_nodes,
            hidden_dim = gcn_hidden,
            out_dim    = graph_dim,
            layers     = gcn_layers,
            dropout    = gcn_dropout,
        ) if use_graph else None
        # FiLM: applied to reducer output (tokens), pre-decoder — the only
        # point in the pipeline where conditioning can actually receive
        # gradients, since extractor/reducer are both @torch.no_grad() in
        # original EEGDM. An earlier "FiLM #1 before reducer" attempt was
        # removed: gradients cannot cross a no_grad() boundary, so that
        # branch was permanently untrained dead weight.
        self.graph_modulator = GraphLatentModulation(
            token_dim = token_dim,
            graph_dim = graph_dim,
        ) if use_graph else None
        self.alignment_head = AlignmentHead(
            token_dim = token_dim,
            graph_dim = graph_dim,
            proj_dim  = 128,
        ) if use_graph else None

    def forward(self, input, icoh_vec, data_is_cached=False, rate=1, return_alignment=False, warmup_alpha=1.0):
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
        tokens_pre_film = tokens  # capture before FiLM: alignment must not be self-referential

        # ── Step 2: Graph-conditioned FiLM on latent tokens ───────────────
        if self.use_graph:
            adj = vector_to_adjacency(
                icoh_vec,
                augment=self.augment_icoh and self.training,
                noise_std=self.icoh_noise_std,
                edge_dropout_p=self.icoh_edge_dropout_p,
            )  # [B, 19, 19]
            # tokens_pre_film: [B, n_layer, 1, pool, 1, C=19, H] -> mean over
            # (n_layer, pool) dims (1, 3), explicitly NOT touching batch dim 0.
            node_feats = tokens_pre_film.mean(dim=(1, 3))  # [B, 1, 1, C, H]
            node_feats = node_feats.reshape(tokens_pre_film.size(0), self.graph_encoder.num_nodes, -1)  # [B, C, H]
            graph_emb = self.graph_encoder(adj, node_features=node_feats)
            tokens    = self.graph_modulator(tokens, graph_emb, warmup_alpha=warmup_alpha)
        else:
            graph_emb = None

        # ── Step 4: Decode and classify (unchanged from EEGDM) ────────────
        rep = self.classifier.decoder(tokens)[self.classifier.use_rep_idx]
        cls = self.classifier.classifier(rep)

        if return_alignment:
            if self.use_graph:
                z_token, z_graph = self.alignment_head(tokens_pre_film, graph_emb)
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
