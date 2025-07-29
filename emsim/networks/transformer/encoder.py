import copy
from typing import Optional, Union

import MinkowskiEngine as ME
import numpy as np
import torch
from pytorch_sparse_utils.batching import (
    BatchTopK,
    batch_topk,
    seq_lengths_to_batch_offsets,
)
from pytorch_sparse_utils.conversion import minkowski_to_torch_sparse
from pytorch_sparse_utils.indexing.indexing import batch_sparse_index
from pytorch_sparse_utils.indexing.scatter import (
    scatter_to_sparse_tensor,
)
from pytorch_sparse_utils.indexing.script_funcs import (
    flatten_nd_indices,
    linearize_sparse_and_index_tensors,
)
from torch import Tensor, nn

from emsim.config.transformer import RoPEConfig, TransformerEncoderConfig
from emsim.networks.positional_encoding.rope import prep_multilevel_positions
from emsim.networks.transformer.blocks import (
    FFNBlock,
    MultilevelSelfAttentionBlockWithRoPE,
    SelfAttentionBlock,
    SparseMSDeformableAttentionBlock,
    SparseNeighborhoodAttentionBlock,
)
from emsim.networks.transformer.blocks.background_transformer import (
    BackgroundEmbeddingTransformerLayer,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_feature_levels: int = 4,
        use_msdeform_attn: bool = True,
        n_deformable_points: int = 4,
        use_neighborhood_attn: bool = True,
        neighborhood_sizes: Optional[list[int]] = None,
        use_rope: bool = True,
        rope_config: Optional[RoPEConfig] = None,
        dropout: float = 0.1,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        max_tokens_sa: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.max_tokens_sa = max_tokens_sa
        self.use_msdeform_attn = use_msdeform_attn
        self.use_neighborhood_attn = use_neighborhood_attn
        self.use_rope = use_rope

        if not use_rope:
            self.self_attn = SelfAttentionBlock(
                d_model, n_heads, dropout, attn_proj_bias, norm_first
            )
        else:
            assert rope_config is not None
            self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
                d_model,
                n_heads,
                rope_config.spatial_dimension,
                dropout,
                attn_proj_bias,
                norm_first,
                rope_spatial_base_theta=rope_config.spatial_base_theta,
                rope_level_base_theta=rope_config.level_base_theta,
                rope_share_heads=rope_config.share_heads,
                rope_freq_group_pattern=rope_config.freq_group_pattern,
                rope_enforce_freq_groups_equal=rope_config.enforce_freq_groups_equal,
            )
        if use_msdeform_attn:
            assert n_deformable_points is not None
            self.msdeform_attn = SparseMSDeformableAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                n_deformable_points,
                dropout,
                bias=attn_proj_bias,
                norm_first=norm_first,
            )
        else:
            self.msdeform_attn = None
        if use_neighborhood_attn:
            assert rope_config is not None
            assert neighborhood_sizes is not None
            self.neighborhood_attn = SparseNeighborhoodAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                neighborhood_sizes=neighborhood_sizes,
                position_dim=rope_config.spatial_dimension,
                dropout=dropout,
                bias=attn_proj_bias,
                norm_first=norm_first,
                rope_spatial_base_theta=rope_config.spatial_base_theta,
                rope_level_base_theta=rope_config.level_base_theta,
                rope_share_heads=rope_config.share_heads,
                rope_freq_group_pattern=rope_config.freq_group_pattern,
                rope_enforce_freq_groups_equal=rope_config.enforce_freq_groups_equal,
            )
        else:
            self.neighborhood_attn = None
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,  # shape: [n_queries]
        query_batch_offsets: Tensor,  # [bsz+1]
        token_predicted_salience_score: Tensor,  # [n_queries]
        query_spatial_indices: Tensor,  # [spatial_dim+2, n_queries]
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        token_electron_scores: Tensor,
        background_embedding: Optional[Tensor] = None,  # [bsz, n_levels, embed_dim]
    ):
        token_scores = token_electron_scores + token_predicted_salience_score

        # Find the max_tokens_sa topk for self attention
        topk_indices, topk_offsets, _ = batch_topk(
            token_scores, query_batch_offsets, self.max_tokens_sa
        )

        queries_sa = queries[topk_indices]
        spatial_indices_sa = query_spatial_indices[:, topk_indices]

        if not self.use_rope:
            raise NotImplementedError("Non-RoPE SA not implemented")
        else:
            self_attn_out = self.self_attn(
                x=queries_sa,
                spatial_positions=spatial_indices_sa[1:-1].T,
                level_indices=spatial_indices_sa[-1],
                level_spatial_shapes=level_spatial_shapes,
                batch_offsets=topk_offsets,
            )
        queries = queries.index_copy(0, topk_indices, self_attn_out)

        if self.msdeform_attn is not None:
            query_level_spatial_positions = query_spatial_indices[1:-1].T + 0.5
            queries = self.msdeform_attn(
                queries,
                query_level_spatial_positions,
                query_batch_offsets,
                stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
                background_embedding=background_embedding,
                query_level_indices=query_spatial_indices[-1],
            )

        if self.neighborhood_attn is not None:
            query_positions = prep_multilevel_positions(
                query_spatial_indices[1:-1].T,
                query_spatial_indices[0],
                query_spatial_indices[-1],
                level_spatial_shapes,
            )
            queries = self.neighborhood_attn(
                query=queries,
                query_spatial_positions=query_positions[:, :-1],
                query_batch_offsets=query_batch_offsets,
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
                background_embedding=background_embedding,
                query_level_indices=query_spatial_indices[-1],
            )
        queries = self.ffn(queries)

        return queries

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if self.msdeform_attn is not None:
            self.msdeform_attn.reset_parameters()
        if self.neighborhood_attn is not None:
            self.neighborhood_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        config: TransformerEncoderConfig,
        score_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.n_layers)]
        )
        self.n_layers = config.n_layers
        self.d_model = encoder_layer.d_model
        self.use_rope = config.use_rope
        self.max_tokens_sa = config.max_tokens_sa
        self.max_tokens_non_sa = config.max_tokens_non_sa
        self.use_background_embedding = config.use_background_embedding

        if self.use_background_embedding:
            self.background_transformer_layers = nn.ModuleList(
                [
                    BackgroundEmbeddingTransformerLayer(
                        config.background_transformer.embed_dim,
                        config.background_transformer.n_heads,
                        config.background_transformer.dim_feedforward,
                        config.background_transformer.dropout,
                        config.background_transformer.attn_proj_bias,
                        config.background_transformer.norm_first,
                        config.background_transformer.rope_base_theta,
                        config.background_transformer.rope_share_heads,
                    )
                    for _ in range(self.n_layers)
                ]
            )
        else:
            self.register_module("background_transformer_layers", None)

        self.register_buffer(
            "layer_filter_ratio", torch.as_tensor(config.layer_filter_ratio)
        )

        self.reset_parameters()

        self.enhance_score_predictor = score_predictor

    def forward(
        self,
        sparse_feature_maps: list[ME.SparseTensor],
        level_spatial_shapes: Tensor,
        token_salience_scores: list[Tensor],  # foreground scores
        token_spatial_indices: list[Tensor],
        token_level_indices: list[Tensor],
        background_embedding: Optional[Tensor] = None,
        position_encoding: Optional[list[ME.SparseTensor]] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        # stack the MinkowskiEngine tensors over the batch dimension for faster indexing
        stacked_feature_maps = self.stack_sparse_tensors(
            sparse_feature_maps, level_spatial_shapes[-1]
        )
        if position_encoding is not None:
            assert not self.use_rope
            stacked_pos_encodings = self.stack_sparse_tensors(
                position_encoding, level_spatial_shapes[-1]
            )
        else:
            stacked_pos_encodings = None

        spatial_dim = level_spatial_shapes.size(-1)
        n_batch_tokens = torch.tensor(
            [scores.size(0) for scores in token_salience_scores],
            device=sparse_feature_maps[0].device,
        )
        batch_offsets = seq_lengths_to_batch_offsets(n_batch_tokens)

        for layer_idx, layer in enumerate(self.layers):
            # Get the top tokens for this layer according to ratio

            # Per-batch number of tokens for this layer, capped by the predetermined
            # cap (max_tokens_non_sa) and tokens actually available (n_batch_tokens)
            batch_k = (
                (n_batch_tokens * self.layer_filter_ratio[layer_idx])
                .long()
                .clamp_max_(self.max_tokens_non_sa)
                .clamp_max_(n_batch_tokens)
            )

            batch_topk_out: BatchTopK = batch_topk(
                torch.cat([score.detach() for score in token_salience_scores]),
                batch_offsets,
                batch_k,
                return_values=True,
            )

            # Fill in the spatial indices from the token seq indices
            batch_topk_spatial_indices = token_spatial_indices[0].new_empty(
                int(batch_topk_out.offsets[-1]), spatial_dim + 2
            )
            for b in range(len(token_salience_scores)):
                slice_start = int(batch_topk_out.offsets[b])
                slice_end = int(batch_topk_out.offsets[b + 1])
                batch_start = int(batch_offsets[b])
                batch_topk_indices = (
                    batch_topk_out.indices[slice_start:slice_end] - batch_start
                )
                batch_topk_spatial_indices[slice_start:slice_end, 0] = b
                batch_topk_spatial_indices[slice_start:slice_end, 1:-1] = (
                    token_spatial_indices[b].index_select(0, batch_topk_indices)
                )
                batch_topk_spatial_indices[slice_start:slice_end, -1] = (
                    token_level_indices[b].index_select(0, batch_topk_indices)
                )

            # Extract the embeddings at each spatial index
            query_for_layer = batch_sparse_index(
                stacked_feature_maps, batch_topk_spatial_indices, True
            )[0]
            if position_encoding is not None:
                # additive position embedding instead of RoPE
                pos_encoding_for_layer = batch_sparse_index(
                    stacked_pos_encodings, batch_topk_spatial_indices, True
                )[0]
            else:
                pos_encoding_for_layer = None

            if self.enhance_score_predictor is not None:
                # Compute classification score for each embedding
                electron_score = self.enhance_score_predictor(query_for_layer)
                electron_score = electron_score.squeeze(-1)
            else:
                electron_score = None
            query_for_layer = layer(
                query_for_layer,
                batch_topk_out.offsets,
                batch_topk_out.values,
                batch_topk_spatial_indices.T.contiguous(),
                stacked_feature_maps,
                level_spatial_shapes,
                electron_score,  # score_tgt
                background_embedding,
            )

            stacked_feature_maps = scatter_to_sparse_tensor(
                stacked_feature_maps, batch_topk_spatial_indices, query_for_layer
            )

            # Update background embedding
            if self.background_transformer_layers is not None:
                background_embedding = self.background_transformer_layers[layer_idx](
                    background_embedding, stacked_feature_maps
                )

        return stacked_feature_maps, background_embedding

    @staticmethod
    def stack_sparse_tensors(
        tensor_list: Union[list[ME.SparseTensor], list[Tensor]],
        full_scale_spatial_shape: Union[Tensor, list[int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Stacks a list of sparse tensors along a new _level_ dimension.

        Every input tensor is expected to have shape
        ``[batch, *spatial_shape, feature_dim]`` and identical sparse / dense
        dimensionalities.  The function converts ``MinkowskiEngine`` tensors to
        regular PyTorch COO format and concatenates their indices / values while
        inserting a new _level_ coordinate, returning a single sparse COO tensor
        whose shape is  ``[max_batch, *max_spatial, #levels, max_feature]``.

        Args:
            tensor_list (list[ME.SparseTensor | torch.Tensor]): Input sparse
                tensors to be stacked.
            full_scale_spatial_shape (Tensor | list[int]): Spatial extents
                (``len == tensor.ndim - 2``) used when converting Minkowski
                tensors.
            device (torch.device, optional): Device for the resulting tensor.
                Defaults to the device of the first tensor (or of the
                ``full_scale_spatial_shape`` argument when the list is empty).

        Returns:
            torch.Tensor: A sparse COO tensor that contains all input tensors
                stacked along the new level dimension.

        Notes:
            - If the input tensors are already coalesced, this function bypasses an
                explicit coalescing of the output tensor by sorting the concatenated
                sparse indices without the accompanying deduplication step.
        """

        # Determine device
        if device is None:
            if tensor_list:
                device = tensor_list[0].device
            else:  # empty input: use device of spatial shape tensor
                device = (
                    full_scale_spatial_shape.device
                    if isinstance(full_scale_spatial_shape, torch.Tensor)
                    else torch.device("cpu")
                )

        # Convert spatial shape to tensor if needed
        full_scale_spatial_shape = torch.as_tensor(
            full_scale_spatial_shape, device=device
        )

        # Convert possible Minkowski tensors to Pytorch sparse
        converted = [
            (
                minkowski_to_torch_sparse(t, full_scale_spatial_shape)
                if isinstance(t, ME.SparseTensor)
                else t
            )
            for t in tensor_list
        ]

        # Early exit for empty tensor list
        if len(converted) == 0:
            empty_sz = [0] + full_scale_spatial_shape.tolist() + [0, 0]
            return torch.sparse_coo_tensor(
                torch.empty(
                    full_scale_spatial_shape.numel() + 2,
                    0,
                    dtype=torch.long,
                    device=device,
                ),
                torch.empty(0, device=device),
                empty_sz,
                is_coalesced=True,
            )

        # Determine shape information and run shape checks
        sparse_dims = [t.sparse_dim() for t in converted]
        dense_dims = [t.dense_dim() for t in converted]
        if len(set(sparse_dims)) != 1:
            raise ValueError(f"Inconsistent sparse_dim values: {sparse_dims}")
        if len(set(dense_dims)) != 1:
            raise ValueError(f"Inconsistent dense_dim values:  {dense_dims}")

        sparse_dim = sparse_dims[0]  # includes batch + spatial dims
        dense_dim = dense_dims[0]  # usually 1  (feature)
        level_dim = sparse_dim  # we append the level as last sparse dim

        spatial_rank_expected = sparse_dim - 1  # batch excluded, feature excluded
        if full_scale_spatial_shape.numel() != spatial_rank_expected:
            raise ValueError(
                "full_scale_spatial_shape must have length "
                f"{spatial_rank_expected}, got {full_scale_spatial_shape.numel()}"
            )

        # Collect information for constructing output tensor
        total_nnz = 0
        max_sizes = list(converted[0].shape)  # copy to mutate in-place
        all_coalesced = True

        for t in converted:
            total_nnz += t._nnz()
            all_coalesced &= t.is_coalesced()
            for i, dim_size in enumerate(t.shape):
                max_sizes[i] = max(dim_size, max_sizes[i])

        final_shape = torch.Size(max_sizes[:-1] + [len(converted), max_sizes[-1]])

        if total_nnz == 0:  # All tensors are empty
            return torch.sparse_coo_tensor(
                torch.empty(level_dim + 1, 0, dtype=torch.long, device=device),
                torch.empty(0, device=device),
                final_shape,
                is_coalesced=True,
            )

        # Allocate output index and value tensors
        indices_out = torch.empty(
            level_dim + 1, total_nnz, dtype=torch.long, device=device
        )

        sample_values = converted[0].values()
        if dense_dim == 0:
            values_out = torch.empty(
                total_nnz, dtype=sample_values.dtype, device=device
            )
        else:
            values_out = torch.empty(
                (total_nnz,) + sample_values.shape[1:],
                dtype=sample_values.dtype,
                device=device,
            )

        # Fill output tensors
        cursor = 0
        for lvl, t in enumerate(converted):
            nnz = t._nnz()
            if nnz == 0:
                continue
            slicer = slice(cursor, cursor + nnz)
            values_out[slicer] = t.values()
            indices_out[:sparse_dim, slicer] = t.indices()
            indices_out[level_dim, slicer] = lvl
            cursor += nnz

        # If the input tensors are all coalesced, we can "pre-coalesce" the output tensor
        # with a flattening+sort since we know deduplication is not needed
        if all_coalesced:
            sparse_sizes = torch.tensor(
                final_shape[:-1],
                device=device,
                dtype=torch.long,
            )
            flat_indices, _ = flatten_nd_indices(indices_out, sparse_sizes)
            perm = torch.argsort(flat_indices.squeeze(0))
            indices_out = indices_out[:, perm]
            values_out = values_out[perm]

        # Construct final output tensor
        return torch.sparse_coo_tensor(
            indices_out,
            values_out,
            final_shape,
            is_coalesced=all_coalesced,
        ).coalesce()  # coalesce is no-op if already coalesced

    @staticmethod
    def get_background_indices(stacked_feature_maps, foreground_indices):
        (
            linear_sparse_indices,
            index_tensor_linearized,
        ) = linearize_sparse_and_index_tensors(stacked_feature_maps, foreground_indices)
        background_token_indices = ~torch.isin(
            linear_sparse_indices, index_tensor_linearized
        )
        background_indices = stacked_feature_maps.indices()[
            :, background_token_indices
        ].transpose(0, 1)
        return background_indices

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
