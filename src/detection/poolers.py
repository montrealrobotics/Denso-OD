# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat

__all__ = ["ROIPooler"]

def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes]):
            A list of N Boxes, where N is the number of images in the batch.

    Returns:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index on which a canonically-sized box
                should be placed. The default is defined as level 4 in the FPN paper.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False)
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = ROIAlign( output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True)
        elif pooler_type == "ROIPool":
            self.level_poolers = RoIPool(output_size, spatial_scale=scale)
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

    def forward(self, x, box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps with scales matching those used to
                construct this module.
            box_lists (list[Boxes]):
                A list of N Boxes , where N is the number of images in the batch.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        output = self.level_poolers(x, pooler_fmt_boxes)
        
        return output

