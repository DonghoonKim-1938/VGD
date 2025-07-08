from typing import Union, Tuple
import torch
import numpy as np
from bluestar.metrics.base_metric import MetricBase, register_metric
import torch.nn.functional as F
from bluestar.utils.dist_utils import set_dist
import sys
__all__ = ["Accuracy"]

@register_metric("accuracy")

class Accuracy(MetricBase):


    def __init__(
            self,
            top_k: Union[int, Tuple[int, ...]] = 1,
            ignore_index: int = -1
    ) -> None:
        super().__init__()
        if isinstance(top_k, int):
            top_k = (top_k,)
        else:
            top_k = sorted(top_k)
        self.top_k = top_k
        self.ignore_index = ignore_index

    def per_class_acc(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            indices: torch.Tensor = None, # For crawl (1) or not crawl dataset.
            num_classes: int = 1000
    ):
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
        pred_one_hot = F.one_hot(pred.to(torch.int64), num_classes=num_classes)

        score_table = pred_one_hot * target
        #print(f"pred: {pred}")

        #target_index = np.nonzero(target)
        #print(f"target index: {target_index}")
        #print(f"value of target index: {target[target_index]}")

        score_table_index = np.nonzero(score_table)
        #print(f"score_table index: {score_table_index}")
        #print(f"value of score_table index: {score_table[score_table_index]}")

        acc = score_table.sum().float()
        #print(f"acc: {acc}")

        acc_per_class = score_table.sum((0, 1)).float()
        #print(f"acc_per_class: {acc_per_class}")


        results = []
        results.append(acc)
        results.append(acc_per_class)

        if indices is not None:
            judge_crawl = indices.reshape(1, -1, 1)
            crawl_score_table = score_table * judge_crawl
            orig_score_table = score_table - crawl_score_table

            acc_indices = crawl_score_table.sum().float() # crawl data accuracy
            results.append(acc_indices)

            acc_indices = orig_score_table.sum().float()  # orig data accuracy
            results.append(acc_indices)

            acc_per_class_indices = crawl_score_table.sum((0, 1)).float() # crawl_per_class
            results.append(acc_per_class_indices)

            acc_per_class_indices = crawl_score_table.sum((0, 1)).float() # crawl_per_class
            results.append(acc_per_class_indices)
        return results

    @torch.no_grad()
    def forward(
            self,
            output: torch.Tensor,  # either before/after softmax
            target: torch.Tensor,  # integer label
            indices: torch.Tensor = None, # For crawl (1) or not crawl dataset.
            num_classes: int=1000
    ):
        f"""Accuracy computation.sd
        :param output:      (batch_size, num_classes)
        :param target:      (batch_size,) or (batch_size, num_classes)
        :return:            dict
        """

        res = {}

        target_idices = F.one_hot(target, num_classes=num_classes) if target.ndim == 1 \
            else target.gt_(0.1).to(torch.int64)

        if self.top_k == (1,):  # shortcut
            pred = torch.argmax(output, dim=1, keepdim=False)  # (n,)
            output = self.per_class_acc(pred, target_idices, indices, num_classes=num_classes)

            res["acc1"] = output[0]
            res["acc1_per_class"] = output[1]

            if indices is not None:
                res["acc1_crawl"] = output[2]
                res[f"acc1_per_class_crawl"] = output[3]
            return res

        max_k = max(self.top_k)
        _, pred = torch.topk(output, max_k, dim=1, largest=True, sorted=True)  # (n, k) sorted -> false
        pred = pred.t()  # (n, k) -> (k, n)

        for k in self.top_k:
            output = self.per_class_acc(pred[:k], target_idices, indices, num_classes=num_classes)
            res[f"acc{k}"] = output[0]
            res[f"acc{k}_per_class"] = output[1]

            if indices is not None:
                res[f"acc{k}_crawl"] = output[2]
                res[f"acc{k}_orig"] = output[3]
                res[f"acc{k}_per_class_crawl"] = output[4]

        return res

