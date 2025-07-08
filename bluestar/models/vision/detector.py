from typing import List, Tuple
import torch

from bluestar.models.base_model import ModelBase
from bluestar.models.vision.backbones import build_vision_backbone
from bluestar.models.vision.necks import build_vision_neck
from bluestar.models.vision.special import ExtraVisionBackboneLevel
from bluestar.models.vision.heads import build_vision_head


class VisionDetector(ModelBase):

    def __init__(
            self,
            backbone_name: str,
            neck_name: str,
            head_name: str,
            **model_kwargs,
    ) -> None:
        """
        **model_kwargs
            "backbone": {...}
            "extra": {...}
            "feature_indices": List[int], indicates which features to selectively use.
            "neck": {...}
            "head": {...}
        """
        super().__init__()

        # -------- configuration ------------------------ #
        backbone_kwargs = model_kwargs["backbone"] if ("backbone" in model_kwargs) else {}
        neck_kwargs = model_kwargs["neck"] if ("neck" in model_kwargs) else {}
        head_kwargs = model_kwargs["head"] if ("head" in model_kwargs) else {}

        # -------- backbone ------------------------ #
        self.backbone = build_vision_backbone(name=backbone_name, **backbone_kwargs)
        features_ch = self.backbone.features_ch

        if "extra" in model_kwargs:
            self.extra = ExtraVisionBackboneLevel(**model_kwargs["extra"])
            features_ch += [self.extra.output_ch] * self.extra.num_levels
        else:
            self.extra = None

        # select which features to send to neck
        # default is to use every feature from the third
        self.feature_indices = model_kwargs.get("feature_indices", list(range(2, len(features_ch))))
        features_ch = features_ch[self.feature_indices]

        # -------- neck ------------------------ #
        neck_kwargs["features_ch"] = features_ch
        self.neck = build_vision_neck(name=neck_name, **neck_kwargs)

        # -------- head ------------------------ #
        # TODO align num_anchors
        self.head = build_vision_head(name=head_name, **head_kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vision object detection forward
        :param x:   (batch_size, 3, h, w)
        :return:      logits    (batch_size, SUM(h_l * w_l * num_anchors), num_classes)
                      deltas    (batch_size, SUM(h_l * w_l * num_anchors), 4)
                      anchors   (SUM(h_l * w_l * num_anchors), 4)
        """
        features = self.backbone.forward_features(x)
        if self.extra is not None:
            extra_features = self.extra(features[-1])
            features.extend(extra_features)

        features = features[self.feature_indices]  # select

        features = self.neck(features)
        logits, deltas = self.head(features)
        logits = self.head.merge_pred(logits)
        deltas = self.head.merge_pred(deltas)
        return logits, deltas, anchors  # TODO 
