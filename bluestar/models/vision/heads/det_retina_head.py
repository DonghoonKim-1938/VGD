from typing import List, Tuple
import math
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from .base_head import VisionHeadBase, register_vision_head

__all__ = ["DetectionRetinaHead"]


@register_vision_head("det_retina")
class DetectionRetinaHead(VisionHeadBase):
    """Object detection head from RetinaNet.
    Includes (1) classification head (2) bounding box regression head.
    Heads are shared for all feature levels.
    """

    def __init__(
            self,
            feature_ch: int,  # input feature channel.
            conv_chs: List[int],  # channels for stacked conv-bn-act.
            num_classes: int,  # object classes
            num_anchors: int,  # anchors per pixel (location)
            *, act_layer=nn.ReLU,
            use_norm: bool = False,
            prior_prob: float = 0.01,  # initial object-less (foreground) probability.
    ) -> None:
        super().__init__()
        self.feature_ch = feature_ch
        self.conv_chs = conv_chs
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        cls_net = []
        bbox_net = []

        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)

        channels = [feature_ch] + conv_chs
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            if use_norm:
                cls_net.append(ConvBN2d(in_ch, out_ch, **conv_kwargs))
                bbox_net.append(ConvBN2d(in_ch, out_ch, **conv_kwargs))
            else:
                cls_net.append(nn.Conv2d(in_ch, out_ch, **conv_kwargs))
                bbox_net.append(nn.Conv2d(in_ch, out_ch, **conv_kwargs))
            cls_net.append(act_layer())
            bbox_net.append(act_layer())

        # final output
        cls_net.append(nn.Conv2d(channels[-1], num_anchors * num_classes, **conv_kwargs))
        bbox_net.append(nn.Conv2d(channels[-1], num_anchors * 4, **conv_kwargs))

        # initialize bias of cls_net.
        prior_bias = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(cls_net[-1].bias, prior_bias)

        self.cls_net = nn.Sequential(*cls_net)
        self.bbox_net = nn.Sequential(*bbox_net)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """RetinaNet head forward.
        :param features:        (batch_size, feature_ch, h_l, w_l) for each level
        :return:      logits    (batch_size, num_anchors * num_classes, h_l, w_l) for each level
                      deltas    (batch_size, num_anchors * 4, h_l, w_l) for each level
        """
        logits = []
        deltas = []
        for feat in features:
            logits.append(self.cls_net(feat))
            deltas.append(self.bbox_net(feat))
        return logits, deltas

    def merge_pred(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Merge multi-level outputs to one single Tensor.
        :param outputs:     (batch_size, num_anchors * X, h_l, w_l) for each level
        :return:            (batch_size, SUM(h_l * w_l * num_anchors), X)
        """
        c = outputs[0].shape[1]
        assert c % self.num_anchors == 0
        dim = c // self.num_anchors  # 4 or num_classes

        pred = []
        for out in outputs:
            b, _, h, w = out.shape
            # (b, c, h, w) -> (b, a, d, h, w) -> (b, h, w, a, d) -> (b, hwa, d)
            out = out.view(b, self.num_anchors, dim, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, dim)
            pred.append(out)
        pred = torch.cat(pred, dim=1)
        return pred
