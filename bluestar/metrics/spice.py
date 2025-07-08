import json
import os
from subprocess import check_call
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch

from bluestar.metrics.base_metric import MetricBase, register_metric

__all__ = ["Spice"]

r"""Compute SPICE score given ground truth captions and predictions."""

@register_metric("spice")
class Spice(MetricBase):
    def __init__(self, assets_dir):
        super().__init__()
        self.assets_dir = assets_dir

    @torch.no_grad()
    def forward(
            self,
            predictions: Dict[int, List[str]],
            ground_truth: Dict[int, List[str]],
    ):

        # Prepare temporary input file for the SPICE scorer.
        input_data = [
            {
                "image_id": image_id,
                "test": predictions[image_id][0],
                "refs": ground_truth[image_id],
            }
            for image_id in ground_truth
        ]
        # Create a temporary directory and dump input file to SPICE.
        temp_dir = tempfile.mkdtemp()
        INPUT_PATH = os.path.join(temp_dir, "input_file.json")
        OUTPUT_PATH = os.path.join(temp_dir, "output_file.json")
        json.dump(input_data, open(INPUT_PATH, "w"))

        # fmt: off
        # Run the command to execute SPICE jar.
        SPICE_JAR = f"{self.assets_dir}/SPICE-1.0/spice-1.0.jar"
        CACHE_DIR = f"{self.assets_dir}/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        spice_cmd = [
            "java", "-jar", "-Xmx8G", SPICE_JAR, INPUT_PATH,
            "-cache", CACHE_DIR, "-out", OUTPUT_PATH, "-subset", "-silent",
        ]
        check_call(spice_cmd, cwd=self.assets_dir)
        # fmt: on

        # Read and process results
        results = json.load(open(OUTPUT_PATH))
        image_id_to_scores = {item["image_id"]: item["scores"] for item in results}
        spice_scores = [
            np.array(item["scores"]["All"]["f"]).astype(float) for item in results
        ]
        return np.mean(spice_scores)
