# Based on https://github.com/xiuqhou/Salience-DETR/blob/main/models/detectors/salience_detr.py
from typing import Union

import MinkowskiEngine as ME
import torch
from pytorch_sparse_utils.conversion import minkowski_to_torch_sparse
from pytorch_sparse_utils.indexing.indexing import (
    union_sparse_indices,
)
from pytorch_sparse_utils.shape_ops import sparse_squeeze
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss


def pixel_coord_grid(height: int, width: int, stride: Tensor, device: torch.device):
    coord_y, coord_x = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device)
        * stride[0],
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device)
        * stride[1],
        indexing="ij",
    )
    return coord_y, coord_x


class ElectronSalienceCriterion(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predicted_foreground_masks: list[Tensor],
        peak_normalized_images: list[Tensor],
    ) -> Tensor:
        assert len(predicted_foreground_masks) == len(peak_normalized_images)

        predicted_pixels = []
        true_pixels = []

        for predicted_mask, true_mask in zip(
            predicted_foreground_masks, peak_normalized_images
        ):
            assert predicted_mask.shape == true_mask.shape
            predicted_unioned, true_unioned = union_sparse_indices(
                predicted_mask, true_mask
            )
            assert torch.equal(predicted_unioned.indices(), true_unioned.indices())
            predicted_pixels.append(predicted_unioned.values())
            true_pixels.append(true_unioned.values())

        predicted_pixels = torch.cat(predicted_pixels)
        true_pixels = torch.cat(true_pixels)

        num_pos = (true_pixels > 0.5).sum().clamp_min(1)
        loss = sigmoid_focal_loss(
            predicted_pixels, true_pixels, alpha=self.alpha, gamma=self.gamma
        )
        loss = loss.sum() / num_pos
        return loss

    @staticmethod
    def prep_inputs(
        score_dict: dict[str, Union[Tensor, list[ME.SparseTensor]]],
        target_dict: dict[str, Tensor],
    ):
        predicted_foreground_masks = [
            sparse_squeeze(
                minkowski_to_torch_sparse(
                    mask,
                    full_scale_spatial_shape=target_dict["image_size_pixels_rc"].max(0)[
                        0
                    ],
                ),
                -1,
            )
            for mask in score_dict["score_feature_maps"]
        ]

        downsample_scales = [1, 2, 4, 8, 16, 32, 64]
        downsample_scales = downsample_scales[: len(predicted_foreground_masks)]
        downsample_scales.reverse()
        peak_normalized_images = [
            target_dict[f"peak_normalized_noiseless_image_1/{ds}"]
            for ds in downsample_scales
        ]
        return predicted_foreground_masks, peak_normalized_images
