from typing import Union
import numpy as np
import sparse
import torch
import torch.nn.functional as F
from scipy.ndimage import grey_dilation


class NSigmaSparsifyTransform:
    def __init__(
        self,
        background_threshold_n_sigma: Union[int, float] = 4,
        window_size: int = 7,
        channels_last: bool = True,
        max_pixels_to_keep: int = int(1e7),
        reduction: str = "median",
    ):
        self.background_threshold_n_sigma = background_threshold_n_sigma
        self.window_size = window_size
        self.channels_last = channels_last
        self.max_pixels_to_keep = max_pixels_to_keep
        self.reduction = reduction

    def __call__(self, batch):
        image = batch["image"]
        if isinstance(image, np.ndarray):
            sparsified = numpy_sigma_energy_threshold_sparsify(
                image,
                background_threshold_n_sigma=self.background_threshold_n_sigma,
                window_size=self.window_size,
                max_pixels_to_keep=self.max_pixels_to_keep,
                reduction=self.reduction,
            )
        elif isinstance(image, torch.Tensor):
            sparsified = torch_sigma_energy_threshold_sparsify(
                image,
                background_threshold_n_sigma=self.background_threshold_n_sigma,
                window_size=self.window_size,
                reduction=self.reduction,
            )
        else:
            raise ValueError

        if self.channels_last:
            if isinstance(sparsified, sparse.COO):
                sparsified = (
                    sparsified.transpose([0, 2, 3, 1])
                    if sparsified.ndim == 4
                    else sparsified.transpose([1, 2, 0])
                )
            elif isinstance(sparsified, torch.Tensor):
                sparsified = (
                    sparsified.permute(0, 2, 3, 1)
                    if sparsified.ndim == 4
                    else sparsified.permute(1, 2, 0)
                ).coalesce()

        batch["image_sparsified"] = sparsified
        return batch


def torch_sigma_energy_threshold_sparsify(
    image: torch.Tensor,
    background_threshold_n_sigma: Union[int, float] = 4,
    window_size: int = 7,
    reduction: str = "median",
):
    if window_size % 2 != 1:
        raise ValueError(f"Expected an odd `window_size`, got {window_size=}")
    if image.ndim > 3:
        reduce_dims = (-1, -2, -3)
    else:
        reduce_dims = (-1, -2)

    if reduction == "median":
        avg = torch.median(
            image,
            reduce_dims,  # pyright: ignore[reportCallIssue, reportArgumentType]
            keepdim=True,
        )
    elif reduction == "mean":
        avg = torch.mean(
            image,
            reduce_dims,  # pyright: ignore[reportCallIssue, reportArgumentType]
            keepdim=True
        )
    else:
        raise ValueError(
            f"Unexpected reduction {reduction}, expected 'mean' or 'median'."
        )
    std = torch.std(image, reduce_dims, keepdim=True)
    thresholded = image > avg + background_threshold_n_sigma * std

    kernel = torch.ones((1, 1, window_size, window_size), device=thresholded.device)

    conved = F.conv2d(thresholded.float(), kernel, padding="same")
    indices = conved.nonzero(as_tuple=True)
    values = image[indices]
    out = torch.sparse_coo_tensor(torch.stack(indices), values, size=image.shape)
    return out.coalesce()


def numpy_sigma_energy_threshold_sparsify(
    image: np.ndarray,
    background_threshold_n_sigma: Union[int, float] = 4,
    window_size: int = 7,
    max_pixels_to_keep: int = int(1e7),
    reduction: str = "median",
):
    if window_size % 2 != 1:
        raise ValueError(f"Expected an odd `window_size`, got {window_size=}")
    if image.ndim == 4:
        reduce_dims = (-1, -2, -3)
        kernel_size = (1, 1, window_size, window_size)
    elif image.ndim == 3:
        reduce_dims = (-1, -2)
        kernel_size = (1, window_size, window_size)
    else:
        raise ValueError

    if reduction == "median":
        avg = np.median(image, reduce_dims, keepdims=True)
    elif reduction == "mean":
        avg = np.mean(image, reduce_dims, keepdims=True)
    else:
        raise ValueError(
            f"Unexpected reduction {reduction}, expected 'mean' or 'median'."
        )
    std = np.std(image, reduce_dims, keepdims=True)

    def _get_indices(n_sigma):
        thresholded = image > avg + n_sigma * std
        conved = grey_dilation(thresholded, kernel_size)
        indices = conved.nonzero()
        return indices

    # if we end up with more than 1M pixels after sparsification, raise the threshold
    below_limit = False
    n_sigma = background_threshold_n_sigma
    while not below_limit:
        indices = _get_indices(n_sigma)
        n_indices = indices[0].size
        if n_indices <= max_pixels_to_keep:
            below_limit = True
        else:
            n_sigma += 0.1

    values = image[indices]
    out = sparse.COO(indices, values, shape=image.shape)
    return out
