"""
graph_conditioned_classifier_channelwise.py

Channel-wise variant of model/graph_conditioned_classifier.py (ORIGINAL FILE UNCHANGED).

Wires together GraphEncoderChannelwise + GraphLatentModulationChannelwise so that
each EEG channel c's latent tokens are conditioned by graph node c's own
embedding, rather than one global graph embedding conditioning every channel.

AlignmentHead (model/alignment.py) is imported UNCHANGED -- it still expects a
single global [B, graph_dim] vector. Per instructions, this is preserved by
using GraphEncoderChannelwise's separate pooled_for_align output (produced via
the same LearnedReadout mechanism as the original file) ONLY for alignment,
explicitly kept out of the FiLM path.
"""
import torch
import torch.nn as nn
from .graph_encoder_channelwise_edgedrop import GraphEncoderChannelwiseEdgeDrop
from .graph_utils import vector_to_adjacency
from .graph_latent_modulation_channelwise import GraphLatentModulationChannelwise
from .alignment import AlignmentHead


class GraphConditionedClassifierChannelwiseEdgeDrop(nn.Module):
    """
    Wraps original EEGDM Classifier. Adds CHANNEL-WISE graph-conditioned
    FiLM: graph node c's embedding conditions EEG channel c's latent tokens.
    """
    def __init__(
        self,
        classifier,
        graph_dim  = 128,
        token_dim  = 128,
        num_nodes  = 20,
        gcn_hidden = 128,
        gcn_layers = 3,
        gcn_dropout = 0.1,
        use_graph  = True,
        augment_icoh = False,
        icoh_noise_std = 0.05,
        icoh_edge_dropout_p = 0.1,
        cw_edge_dropout = 0.0,
    ):
        super().__init__()
        self.classifier = classifier
        self.use_graph  = use_graph
        self.augment_icoh = augment_icoh
        self.icoh_noise_std = icoh_noise_std
        self.icoh_edge_dropout_p = icoh_edge_dropout_p
        self.cw_edge_dropout = cw_edge_dropout

        self.graph_encoder = GraphEncoderChannelwiseEdgeDrop(
            num_nodes  = num_nodes,
            hidden_dim = gcn_hidden,
            out_dim    = graph_dim,
            layers     = gcn_layers,
            dropout    = gcn_dropout,
            edge_dropout = cw_edge_dropout,
        ) if use_graph else None

        self.graph_modulator = GraphLatentModulationChannelwise(
            token_dim = token_dim,
            graph_dim = graph_dim,
        ) if use_graph else None

        self.alignment_head = AlignmentHead(
            token_dim = token_dim,
            graph_dim = graph_dim,
            proj_dim  = 128,
        ) if use_graph else None

    def forward(self, input, icoh_vec, data_is_cached=False, rate=1, return_alignment=False, warmup_alpha=1.0):
        if not data_is_cached:
            latent_activity = self.classifier.extractor(input, rate=rate)
            tokens = self.classifier.reducer(latent_activity)
        else:
            tokens = input[
                :,
                self.classifier.extractor.start:self.classifier.extractor.end,
                :, :, :, :, :
            ]
        tokens_pre_film = tokens

        if self.use_graph:
            adj = vector_to_adjacency(
                icoh_vec,
                augment=self.augment_icoh and self.training,
                noise_std=self.icoh_noise_std,
                edge_dropout_p=self.icoh_edge_dropout_p,
            )  # [B, 20, 20]

            node_feats = tokens_pre_film.mean(dim=(1, 3))  # [B, 1, 1, C, H]
            node_feats = node_feats.reshape(tokens_pre_film.size(0), self.graph_encoder.num_nodes, -1)  # [B, C, H]

            node_embeddings, pooled_for_align = self.graph_encoder(adj, node_features=node_feats)

            assert node_embeddings.shape[1] == tokens.shape[-2], (
                f"graph node count ({node_embeddings.shape[1]}) must match "
                f"EEG channel count ({tokens.shape[-2]}) and use the SAME "
                f"channel ordering -- see src/bipolar.py for the ordering "
                f"used throughout this pipeline."
            )

            tokens = self.graph_modulator(tokens, node_embeddings, warmup_alpha=warmup_alpha)
        else:
            pooled_for_align = None

        rep = self.classifier.decoder(tokens)[self.classifier.use_rep_idx]
        cls = self.classifier.classifier(rep)

        if return_alignment:
            if self.use_graph:
                z_token, z_graph = self.alignment_head(tokens_pre_film, pooled_for_align)
            else:
                z_token, z_graph = None, None
            return cls, z_token, z_graph

        return cls

    def get_new_params(self):
        if not self.use_graph:
            return []
        return list(self.graph_encoder.parameters()) + \
               list(self.graph_modulator.parameters()) + \
               list(self.alignment_head.parameters())

    def get_backbone_params(self):
        return list(self.classifier.parameters())
