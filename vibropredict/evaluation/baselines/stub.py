"""
Mean-Predictor Baseline

A trivial baseline that predicts the mean log(k_cat) of the training set.
Used in tests to verify the baseline ABC contract and registry mechanics.
"""

from __future__ import annotations

from typing import Any

from vibropredict.evaluation.baselines.base import BaselineModel, register_baseline


@register_baseline("mean_predictor")
class MeanPredictorBaseline(BaselineModel):
    """Baseline that always predicts a fixed mean value.

    Args:
        mean_value: The constant log(k_cat) prediction.
            Defaults to 1.5, which is roughly the mean of many
            enzyme kinetics datasets.
    """

    def __init__(self, mean_value: float = 1.5):
        self.mean_value = mean_value

    def predict(
        self,
        sequence: str,
        smiles: str,
    ) -> tuple[float, dict[str, Any]]:
        """Return the constant mean prediction.

        Args:
            sequence: Amino acid sequence (ignored).
            smiles: Substrate SMILES (ignored).

        Returns:
            Tuple of (mean_value, empty metadata dict).
        """
        return self.mean_value, {"method": "mean_predictor"}

    @property
    def name(self) -> str:
        return "MeanPredictor"
