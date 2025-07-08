from collections import defaultdict
import json
import os
from subprocess import Popen, PIPE, check_call
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch

from bluestar.metrics.base_metric import MetricBase, register_metric
from bluestar.metrics.cider import Cider
from bluestar.metrics.spice import Spice
from bluestar.modules.tokenizers import *


__all__ = ["CocoCaptionsEvaluator"]

'''
coco caption evaluator from the source code
'''
@register_metric("coco_caption")
class CocoCaptionsEvaluator(MetricBase):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cider = Cider()
        self.spice = Spice(cfg["assets_dir"])
        gt_annotations = json.load(open(cfg["ground_truth"]))["annotations"]
        self.ground_truth: Dict[int, List[str]] = defaultdict(list)

        for ann in gt_annotations:
            self.ground_truth[ann["image_id"]].append(ann["caption"])

        self.tokenizer = PennTreebankTokenizer(cfg["assets_dir"]+"/SPICE-1.0/lib/stanford-corenlp-3.4.1.jar")
        self.ground_truth = self.tokenizer.encode(self.ground_truth)

    @torch.no_grad()
    def forward(self, preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]= None) -> Dict[str, float]:
        r"""Compute CIDEr and SPICE scores for predictions.

        Args:
            preds: List of per instance predictions in COCO Captions format:
                ``[ {"image_id": int, "caption": str} ...]``.

        Returns:
            Computed metrics; a dict with keys ``{"CIDEr", "SPICE"}``.
        """
        if isinstance(preds, str):
            preds = json.load(open(preds))["annotations"]


        res = {ann["image_id"]: [ann["caption"]] for ann in preds}
        res = self.tokenizer.encode(res)

        if gts == None:
            gt = self.ground_truth
        else:
            gt = {ann["image_id"]: [ann["caption"]] for ann in gts}
            gt = self.tokenizer.encode(gt)

        # Remove IDs from predictions which are not in GT.
        common_image_ids = gt.keys() & res.keys()
        res = {k: v for k, v in res.items() if k in common_image_ids}

        # Add dummy entries for IDs absent in preds, but present in GT.
        for k in gt:
            res[k] = res.get(k, [""])

        cider_score = self.cider(res, gt)
        spice_score = self.spice(res, gt)

        return {"CIDEr": 100 * cider_score, "SPICE": 100 * spice_score}