from typing import Optional

import torch
from pytorch_sparse_utils.batching import (
    concatenated_to_padded,
    padded_to_concatenated,
)
from torch import Tensor, nn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout, bias, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward_batched(
        self,
        query: Tensor,
        query_pos_embedding: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        if pad_mask is not None:
            assert pad_mask.ndim == 2  # batch, seq_len
        residual = query
        query_with_pos_embed = query + query_pos_embedding
        if self.norm_first:
            query_with_pos_embed = self.norm(query_with_pos_embed)
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    query_with_pos_embed,
                    query,
                    key_padding_mask=pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
        else:
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    query_with_pos_embed,
                    query,
                    key_padding_mask=pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
            query = self.norm(query)
        return query

    def forward(
        self,
        query: Tensor,
        query_pos_embedding: Tensor,
        batch_offsets: Tensor,
        attn_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
    ):
        if query.ndim == 3:
            assert query_pos_embedding.ndim == 3
            assert pad_mask is not None
            return self.forward_batched(
                query, query_pos_embedding, pad_mask=pad_mask, attn_mask=attn_mask
            )
        assert batch_offsets is not None
        residual = query
        query_with_pos_embed = query + query_pos_embedding
        if self.norm_first:
            query_with_pos_embed = self.norm(query_with_pos_embed)
        query, pad_mask = concatenated_to_padded(query, batch_offsets)
        query_with_pos_embed, pad_mask_2 = concatenated_to_padded(
            query_with_pos_embed, batch_offsets
        )
        assert isinstance(pad_mask, Tensor)
        assert torch.equal(pad_mask, pad_mask_2)
        query = self.dropout(
            self.attn(
                query_with_pos_embed,
                query_with_pos_embed,
                query,
                key_padding_mask=pad_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )[0]
        )
        query, batch_offsets_2 = padded_to_concatenated(query, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)
        query = query + residual
        if not self.norm_first:
            query = self.norm(query)
        return query

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.norm.reset_parameters()
