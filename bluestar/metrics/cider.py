from collections import defaultdict
import json
import os
from subprocess import Popen, PIPE, check_call
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch

from bluestar.metrics.base_metric import MetricBase, register_metric

__all__ = ["Cider"]

r"""Compute CIDEr score given ground truth captions and predictions."""


@register_metric("cider")
class Cider(MetricBase):
    def __init__(
            self,
            n: int = 4,
            sigma: float = 6.0,
    ):
        super().__init__()
        self.n = n
        self.sigma = sigma

    def to_ngrams(
            self,
            sentence: str,
    ):
        r"""Convert a sentence into n-grams and their counts."""
        words = sentence.split()
        counts = defaultdict(int)  # type: ignore
        for k in range(1, self.n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i: i + k])
                counts[ngram] += 1
        return counts

    def counts2vec(
            self,
            cnts,
            document_frequency,
            log_reference_length
    ):
        r"""Function maps counts of ngram to vector of tfidf weights."""
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        for (ngram, term_freq) in cnts.items():
            df = np.log(max(1.0, document_frequency[ngram]))
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vec[len(ngram) - 1][ngram] = float(term_freq) * (
                    log_reference_length - df
            )
            # Compute norm for the vector: will be used for computing similarity
            norm[len(ngram) - 1] += pow(vec[len(ngram) - 1][ngram], 2)

            if len(ngram) == 2:
                length += term_freq
        norm = [np.sqrt(nn) for nn in norm]
        return vec, norm, length

    def sim(
            self,
            vec_hyp,
            vec_ref,
            norm_hyp,
            norm_ref,
            length_hyp,
            length_ref
    ):
        r"""Compute the cosine similarity of two vectors."""
        delta = float(length_hyp - length_ref)
        val = np.array([0.0 for _ in range(self.n)])
        for nn in range(self.n):
            for (ngram, count) in vec_hyp[nn].items():
                val[nn] += (
                        min(vec_hyp[nn][ngram], vec_ref[nn][ngram]) * vec_ref[nn][ngram]
                )

            val[nn] /= (norm_hyp[nn] * norm_ref[nn]) or 1
            val[nn] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    @torch.no_grad()
    def forward(
            self,
            predictions: Dict[int, List[str]],
            ground_truth: Dict[int, List[str]],
    ):

        ctest = [self.to_ngrams(predictions[image_id][0]) for image_id in ground_truth]
        crefs = [
            [self.to_ngrams(gt) for gt in ground_truth[image_id]] for image_id in ground_truth
        ]
        # Build document frequency and compute IDF.
        document_frequency = defaultdict(float)
        for refs in crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                document_frequency[ngram] += 1

        # Compute log reference length.
        log_reference_length = np.log(float(len(crefs)))

        scores = []
        for test, refs in zip(ctest, crefs):
            # Compute vector for test captions.
            vec, norm, length = self.counts2vec(
                test, document_frequency, log_reference_length
            )
            # Compute vector for ref captions.
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = self.counts2vec(
                    ref, document_frequency, log_reference_length
                )
                score += self.sim(vec, vec_ref, norm, norm_ref, length, length_ref)

            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)

        return np.mean(scores)
