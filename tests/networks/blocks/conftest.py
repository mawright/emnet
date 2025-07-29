from typing import Optional, Union

import torch
import numpy as np
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import unflatten_nd_indices


def simple_sparse_input_tensors(
    device: Union[str, torch.device],
    n_queries: int = 4,
    position_dim: int = 2,
    embed_dim: int = 64,
    random_seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Generate test data for SparseNeighborhoodAttentionBlock."""
    if isinstance(device, str):
        device = torch.device(device, 0)

    if random_seed is not None:
        if device.type == "cuda":
            rng_state = torch.cuda.get_rng_state(device)
        else:
            rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    else:
        rng_state = None

    # Query embeddings
    query = torch.randn(n_queries, embed_dim, device=device)

    # Query spatial positions
    query_spatial_positions = torch.rand(n_queries, position_dim, device=device) * 16.0

    # Batch information (assume single batch for simplicity)
    query_batch_offsets = torch.tensor([0, n_queries], device=device)

    # Feature maps at different levels with decreasing resolution
    level_spatial_shapes = torch.tensor([[16, 16], [8, 8], [4, 4]], device=device)

    # Create feature maps for all levels
    nonzero_indices_list = []
    for level, (h, w) in enumerate(level_spatial_shapes):
        hw_indices = torch.stack(
            torch.meshgrid(
                torch.arange(int(h), device=device),
                torch.arange(int(w), device=device),
                indexing="ij",
            ),
            dim=-1,
        ).flatten(0, 1)
        n_level_indices = hw_indices.size(0)
        level_indices = torch.cat(
            [
                torch.tensor([[0]], device=device).expand(n_level_indices, 1),
                hw_indices,
                torch.tensor([[level]], device=device).expand(n_level_indices, 1),
            ],
            dim=-1,
        )
        # 80% sparsity
        nonzero_indices = level_indices[
            torch.randperm(n_level_indices, device=device)[: int(n_level_indices * 0.2)]
        ]
        nonzero_indices_list.append(nonzero_indices)

    nonzero_indices = torch.cat(nonzero_indices_list)
    sparse_size = (1, 16, 16, 3, embed_dim)

    stacked_feature_maps = torch.sparse_coo_tensor(
        nonzero_indices.T,
        torch.randn(nonzero_indices.size(0), embed_dim, device=device),
        sparse_size,
        device=device,
    ).coalesce()

    if rng_state is not None:
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng_state, device)
        else:
            torch.set_rng_state(rng_state)

    return {
        "query": query,
        "query_spatial_positions": query_spatial_positions,
        "query_batch_offsets": query_batch_offsets,
        "stacked_feature_maps": stacked_feature_maps,
        "level_spatial_shapes": level_spatial_shapes,
    }


def random_multilevel_sparse_tensor_indices(
    level_spatial_shapes: Union[np.ndarray, Tensor],
    sparsity: float,
    batch_size: int,
    max_nnz: int = 1000,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    assert 0.0 <= sparsity <= 1.0
    if device is None:
        device = torch.get_default_device()
    device = torch.device(device)

    nonzero_indices = []
    for level, level_shape in enumerate(level_spatial_shapes):
        # add batch dimension to level shape
        extended_level_shape = level_shape.new_full(
            (level_shape.size(0) + 1,), fill_value=batch_size
        )
        extended_level_shape[1:] = level_shape
        n_indices_in_level = int(torch.prod(extended_level_shape).item())

        n_sampled_indices = int(n_indices_in_level * (1 - sparsity))
        n_sampled_indices = min(max(n_sampled_indices, 1), n_indices_in_level, max_nnz)

        # sample indices in flattened/1D form
        if n_indices_in_level <= max_nnz:
            # sample from available indices
            sampled_flat_indices = torch.randperm(n_indices_in_level, device=device)[
                :n_sampled_indices
            ]
        else:
            # avoid creating the whole randperm tensor
            # may have duplicates but that should be okay since there are so many nnzs
            sampled_flat_indices = (
                torch.rand(n_sampled_indices, device=device) * n_indices_in_level
            ).long()

        # "unflatten" 1D indices to true N-D indices
        # shape: dim x nnz, last entry of axis 0 is level index
        sampled_indices = sampled_flat_indices.new_full(
            (extended_level_shape.size(0) + 1, n_sampled_indices), fill_value=level
        )
        sampled_indices[:-1] = unflatten_nd_indices(
            sampled_flat_indices.unsqueeze(0), extended_level_shape
        )

        assert torch.all(sampled_indices[:-1].T < extended_level_shape)

        nonzero_indices.append(sampled_indices)

    return torch.cat(nonzero_indices, dim=1)
