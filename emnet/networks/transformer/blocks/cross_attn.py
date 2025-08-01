from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nd_rotary_encodings import (
    RoPEEncodingND,
    prep_multilevel_positions,
)
from pytorch_sparse_utils.batching import (
    batch_offsets_from_sparse_tensor_indices,
    concatenated_to_padded,
    padded_to_concatenated,
)


class MultilevelCrossAttentionBlockWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_theta: float = 10.0,
        rope_dtype: torch.dtype = torch.double,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(d_model)
        self.q_in_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_in_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        position_dim = 2
        raise NotImplementedError("Not updated yet")
        self.pos_encoding = RoPEEncodingND(
            position_dim + 1,
            d_model,
            n_heads,
            [0] * position_dim + [1],
            [rope_theta] * position_dim + [rope_theta / 100],
            dtype=rope_dtype,
        )

        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        query_normalized_xy_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert query.ndim == 2  # (stacked sequences x d_model)
        assert query_normalized_xy_positions.shape == (query.shape[0], 2)
        value = stacked_feature_maps.values()
        value_bijl_positions = stacked_feature_maps.indices().T
        assert value.ndim == 2  # (stacked sequences x d_model)
        assert value_bijl_positions.shape[-1] == 4  # (batch, i, j, level)
        value_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            stacked_feature_maps.indices()
        )

        residual = query
        if self.norm_first:
            query = self.norm(query)
        q = self.q_in_proj(query)
        k, v = self.kv_in_proj(value).chunk(2, dim=-1)

        finest_image_size, finest_image_level = level_spatial_shapes.max(0)
        finest_image_level = finest_image_level.unique()
        assert finest_image_level.numel() == 1

        query_prepped_positions = torch.cat(
            [
                query_normalized_xy_positions.flip(-1) * finest_image_size,
                finest_image_level.expand(query.shape[0], 1).to(
                    query_normalized_xy_positions
                ),
            ],
            -1,
        )
        key_prepped_positions = prep_multilevel_positions(
            value_bijl_positions[:, 1:3],
            level_spatial_shapes,
            value_bijl_positions[:, -1],
            level_spatial_shapes,
        )
        key_prepped_positions = key_prepped_positions[:, 1:]

        q, query_pad_mask = concatenated_to_padded(q, query_batch_offsets)
        k, key_pad_mask = concatenated_to_padded(k, value_batch_offsets)
        v, value_pad_mask = concatenated_to_padded(v, value_batch_offsets)
        assert torch.equal(key_pad_mask, value_pad_mask)
        src_seq_len = k.shape[1]
        # pad value of 1.0 instead of 0.0 to suppress a false warning
        query_prepped_positions, _ = concatenated_to_padded(
            query_prepped_positions,
            query_batch_offsets,
            query_prepped_positions.new_ones([]),
        )
        key_prepped_positions, _ = concatenated_to_padded(
            key_prepped_positions,
            value_batch_offsets,
            key_prepped_positions.new_ones([]),
        )

        q, k = self.pos_encoding(q, query_prepped_positions, k, key_prepped_positions)

        # (batch x seq_len x n_heads x head_dim) -> (batch x n_heads x seq_len x head_dim)
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 3
        bsz, tgt_seq_len, n_heads, head_dim = q.shape
        q: Tensor = q.transpose(1, 2).to(v).contiguous()
        k: Tensor = k.transpose(1, 2).to(v).contiguous()
        v: Tensor = (
            v.view(bsz, src_seq_len, n_heads, head_dim).transpose(1, 2).contiguous()
        )

        if key_pad_mask.any():
            assert key_pad_mask.shape == (bsz, src_seq_len)
            key_pad_mask = (
                key_pad_mask.logical_not()
                .view(bsz, 1, 1, src_seq_len)
                .expand(-1, -1, tgt_seq_len, -1)
            )
        else:
            key_pad_mask = None
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            if attn_mask.ndim == 3:
                assert attn_mask.shape[0] in (bsz, bsz * n_heads)
                if attn_mask.shape[0] == bsz:
                    attn_mask = attn_mask.view(bsz, 1, tgt_seq_len, src_seq_len)
                else:
                    attn_mask = attn_mask.view(bsz, n_heads, tgt_seq_len, src_seq_len)
            else:
                assert attn_mask.ndim == 4

            # switch from true being should not be attended to, to true being
            # should be attended to
            attn_mask = attn_mask.logical_not()
            if key_pad_mask is not None:
                attn_mask = torch.logical_and(attn_mask, key_pad_mask)
        else:
            attn_mask = key_pad_mask
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_rate if self.training else 0.0,
        )

        # (batch x n_heads x tgt_seq_len x head_dim) ->
        # (batch x tgt_seq_len x n_heads x head_dim)
        x = x.transpose(1, 2)

        x = x.reshape(bsz, tgt_seq_len, self.d_model)
        x, batch_offsets_2 = padded_to_concatenated(x, query_pad_mask)
        assert torch.equal(query_batch_offsets, batch_offsets_2)

        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.q_in_proj.reset_parameters()
        self.kv_in_proj.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()
