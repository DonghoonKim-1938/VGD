from typing import List, Tuple
import math
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):

    def __init__(
            self,
            strides: List[int],  # [8, 16, 32, 64, 128], resolution ratio of features from high to low resolution.
            sizes: List[float],  # [32, 64, 128, 256, 512], object size in terms of side length in IMAGE PIXELS.
            scales: List[float],  # [1, 2^(1/3), 2^(2/3)] size scaling
            aspect_ratios: List[float],  # [1, 0.5, 2.0] anchor aspect ratios h/w
    ) -> None:
        super().__init__()

        if len(strides) != len(sizes):
            raise ValueError(f"AnchorGenerator strides {strides} and sizes {sizes} length mismatch.")

        self.strides = strides
        self.sizes = sizes
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        self.num_anchors = len(scales) * len(aspect_ratios)  # anchors per level

        self.anchors = []  # do not necessarily save as buffers
        for size in sizes:
            # generate relative anchors for each coordinate, for each level.
            self.anchors.append(self._generate_anchor_per_size(size, scales, aspect_ratios))

    @staticmethod
    def _generate_anchor_per_size(size: float, scales: List[float], aspect_ratios: List[float]) -> torch.Tensor:
        """Generate (relative) anchor boxes.
        :return:            (len(scales) x len(aspect_ratios), 4) xyxy format.
        """
        anchors = []
        for scale in scales:
            area = float(size ** 2) * scale
            for ratio in aspect_ratios:
                # area = h * w = ratio * w * w
                w = math.sqrt(area / ratio)
                h = w * ratio
                x1, y1, x2, y2 = -w / 2, -h / 2, w / 2, h / 2
                anchors.append([x1, y1, x2, y2])
        anchors = torch.tensor(anchors, dtype=torch.float32)
        return anchors

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Anchor generator forward
        :param features:      (batch_size, ch_l, h_l, w_l) for each level
        :return:              (batch_size, SUM(h_l * w_l * num_anchors), 4)
        """
        # TODO