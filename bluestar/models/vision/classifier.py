import torch

from bluestar.models.base_model import ModelBase
from bluestar.models.vision.backbones import build_vision_backbone
from bluestar.models.vision.heads import build_vision_head
from bluestar.modules import GlobalAvgPool2d

__all__ = ["VisionClassifier"]

class VisionClassifier(ModelBase):

    def __init__(
            self,
            backbone_name: str,
            head_name: str = "cls",
            pool2d: bool = False,
            **model_kwargs,
    ) -> None:
        """
        **model_kwargs
            "backbone": {...}
            "head": {...}
        """
        super().__init__()

        # -------- configuration ------------------------ #
        backbone_kwargs = model_kwargs["backbone"] if ("backbone" in model_kwargs) else {}
        head_kwargs = model_kwargs["head"] if ("head" in model_kwargs) else {}

        # -------- backbone ------------------------ #
        self.backbone = build_vision_backbone(name=backbone_name, **backbone_kwargs)
        self.feature_dim = self.backbone.features_ch[-1]

        self.output_features = False

        if "output_features" in model_kwargs.keys():
            if model_kwargs["output_features"] is True:
                self.output_features = True

        # -------- pooling ------------------------ #
        self.pool2d = GlobalAvgPool2d() if pool2d else None

        # -------- head ------------------------ #
        head_kwargs["feature_dim"] = self.feature_dim
        if "num_classes" not in head_kwargs:
            raise ValueError("[ERROR] Classifier head kwargs should have attribute `num_classes`.")
        self.head = build_vision_head(name=head_name, **head_kwargs)
        self.num_classes = self.head.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vision classifier forward
        :param x:   (batch_size, 3, h, w)
        :return:    (batch_size, num_classes)
        """
        if self.output_features:
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)

        if self.pool2d is not None:
            x = self.pool2d(x)

        x = self.head(x)
        return x
