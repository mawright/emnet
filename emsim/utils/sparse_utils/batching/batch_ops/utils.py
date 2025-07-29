import torch
from torch import Tensor


def _lex_compare(row1: Tensor, row2: Tensor, descending: bool = False) -> bool:
    """Compares two 1D tensors lexicographically."""
    diff_positions = (row1 != row2).nonzero()

    if diff_positions.numel() == 0:
        return True

    first_diff_pos = int(diff_positions[0].item())

    if descending:
        return bool((row1[first_diff_pos] >= row2[first_diff_pos]).item())
    else:
        return bool((row1[first_diff_pos] <= row2[first_diff_pos]).item())


def is_lexsorted(tensor: Tensor, descending: bool = False) -> Tensor:
    """
    Lexicographic sorting check for batches of 2D tensors.

    Args:
        tensor: 3D tensor of shape (batch_size, sort_dim, vector_dim)
        descending: Whether to check for descending lexsort

    Returns:
        Tensor: Boolean tensor of shape (batch_size,) indicating sorted status
    """
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("Expected 2D or 3D tensor")
    batch_size, sort_len, vector_len = tensor.shape
    device = tensor.device

    if sort_len <= 1 or vector_len <= 1 or tensor.numel() == 0:
        return torch.ones(batch_size, dtype=torch.bool, device=tensor.device)

    batch_size, n_rows, _ = tensor.shape

    if n_rows <= 1:
        return tensor.new_ones(batch_size, dtype=torch.bool)

    tensor = tensor.cpu()
    results = tensor.new_empty(batch_size, dtype=torch.bool)

    for b in range(batch_size):
        batch_sorted = True
        for r in range(n_rows - 1):
            curr_row = tensor[b, r]
            next_row = tensor[b, r + 1]
            if not _lex_compare(curr_row, next_row, descending=descending):
                batch_sorted = False
                break

        results[b] = batch_sorted

    return results.to(device)
