from typing import Union, Tuple
import torch
from bluestar.metrics.base_metric import MetricBase, register_metric
import torch.nn.functional as F

__all__ = ["Similarity"]

@register_metric("similarity")

class Similarity(MetricBase):


    def __init__(
            self,
            image_feature,
            text_feature,
            idx,
    ) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(
            self,
            image_feature: torch.Tensor,  # image_feature
            target: torch.Tensor,  # integer label
            idx : torch.Tensor, # index

    ):
        f"""Accuracy computation.
        :param output:      (batch_size, num_classes)
        :param target:      (batch_size,) or (batch_size, num_classes)
        :return:            dict
        """

        similarity = target.cpu().numpy() @ image_feature.cpu().numpy().T
        res = {"idx": idx, "similarity": similarity, "class": target}
        return res
