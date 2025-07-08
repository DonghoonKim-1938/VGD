import torch
from torch import nn

import torch.nn.functional as F

from bluestar.models.base_model import ModelBase
from bluestar.models.vision.backbones import build_vision_backbone
from bluestar.models.vision.heads import build_vision_head
from bluestar.modules import GlobalAvgPool2d

__all__ = ["QueryGenerator"]

class QueryGenerator(ModelBase):

    def __init__(
            self,
            backbone_name: str,
            head_name: str = "cls",
            pool2d: bool = False,
            alpha_center: float = 0.0,
            alpha_temp: float = 1.0,
            center_decay: float = 0.9,
            **model_kwargs,
    ) -> None:
        """
        **model_kwargs
            "backbone": {...}
            "head": {...}
        """
        super().__init__()

        # ------------------------ configuration ------------------------ #
        backbone_kwargs = model_kwargs["backbone"] if ("backbone" in model_kwargs) else {}
        head_kwargs = model_kwargs["head"] if ("head" in model_kwargs) else {}

        # ------------------------ backbone ------------------------ #
        self.backbone = build_vision_backbone(name=backbone_name, **backbone_kwargs)
        self.feature_dim = self.backbone.features_ch[-1]

        # ------------------------ pooling ------------------------ #
        self.pool2d = GlobalAvgPool2d() if pool2d else None

        # ------------------------ head ------------------------ #
        head_kwargs["feature_dim"] = self.feature_dim
        if "num_classes" not in head_kwargs:
            raise ValueError("[ERROR] Classifier head kwargs should have attribute `num_classes`.")
        self.head = build_vision_head(name=head_name, **head_kwargs)
        self.num_classes = self.head.num_classes

        # ------------------------ QG head ------------------------ #
        self.qg_head = nn.Sequential(
            nn.Linear(self.feature_dim, int(self.feature_dim/2)),
            nn.GELU(),
            nn.Linear(int(self.feature_dim/2), 1),
        )

        self.center_decay = center_decay

        self.register_buffer("alpha_center", torch.tensor(alpha_center))
        self.register_buffer("alpha_temp", torch.tensor(alpha_temp))

    def update_center(self, mean):
        self.alpha_center = self.center_decay * self.alpha_center + (1.0 - self.center_decay) * mean

    def forward(self, x: torch.Tensor):
        """Vision classifier forward
        :param x:   (batch_size, 3, h, w)
        :return:    (batch_size, num_classes)
        """
        x = self.backbone(x)

        if self.pool2d is not None:
            x = self.pool2d(x)

        a = self.qg_head(x)
        a_mean = a.detach().mean()

        # Only update 50% of the sampled image.
        # Divided by temperature for widen the distribution.
        a = F.sigmoid((a-self.alpha_center)/self.alpha_temp)

        if self.training == True:
            self.update_center(a_mean)

        x = self.head(x)

        return x, a
