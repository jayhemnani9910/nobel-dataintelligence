"""
CatPred Baseline Integration

Wraps the CatPred enzyme kinetics prediction model (Nature Communications 2025)
for use in the VibroPredict baseline evaluation harness.

CatPred reference:
    https://www.nature.com/articles/s41467-025-57215-9
    https://github.com/maranasgroup/CatPred

As of 2026-04-19, CatPred is not pip-installable and exposes no public HTTP
API. This wrapper therefore supports only local inference, via one of:

  1. An installed ``catpred`` package in the current environment.
  2. A vendored copy under
     ``vibropredict/evaluation/baselines/_vendored/catpred/`` exposing a
     ``CatPredModel`` class with a ``.predict(sequence, substrate)`` method.

If neither is present, registration still succeeds but calling ``predict()``
raises a ``RuntimeError`` with setup instructions. This lets the benchmark
runner skip CatPred gracefully in environments where it isn't installed.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from vibropredict.evaluation.baselines.base import BaselineModel, register_baseline

logger = logging.getLogger(__name__)


def _check_catpred_available() -> str:
    """Return 'local' if CatPred is importable, otherwise 'unavailable'."""
    try:
        spec = importlib.util.find_spec("vibropredict.evaluation.baselines._vendored.catpred")
        if spec is not None:
            return "local"
    except (ModuleNotFoundError, ValueError):
        pass

    if importlib.util.find_spec("catpred") is not None:
        return "local"

    return "unavailable"


@register_baseline("catpred")
class CatPredBaseline(BaselineModel):
    """CatPred baseline for enzyme k_cat prediction.

    Local inference only — CatPred has no public HTTP endpoint. If a vendored
    or installed ``catpred`` package is found, inference runs through it.
    Otherwise ``predict()`` raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._model = None
        self._available: str | None = None

    def setup(self) -> None:
        """Detect CatPred availability and load the model if possible."""
        self._available = _check_catpred_available()

        if self._available == "local":
            self._setup_local()
        else:
            logger.warning(
                "CatPred is not available. To enable this baseline:\n"
                "  1. Install CatPred (clone https://github.com/maranasgroup/CatPred\n"
                "     and install into your environment), OR\n"
                "  2. Vendor the source under "
                "vibropredict/evaluation/baselines/_vendored/catpred/ with a\n"
                "     CatPredModel class exposing .predict(sequence, substrate)."
            )

    def _setup_local(self) -> None:
        """Instantiate CatPred from a vendored copy or installed package."""
        try:
            from vibropredict.evaluation.baselines._vendored.catpred import (
                CatPredModel,
            )

            self._model = CatPredModel()
            logger.info("CatPred loaded from vendored source.")
            return
        except ImportError:
            pass

        try:
            from catpred import CatPredModel  # type: ignore[import-untyped]

            self._model = CatPredModel()
            logger.info("CatPred loaded from installed package.")
        except ImportError:
            self._available = "unavailable"
            logger.warning("CatPred import failed despite being detected.")

    def predict(
        self,
        sequence: str,
        smiles: str,
    ) -> tuple[float, dict[str, Any]]:
        """Predict log(k_cat) using CatPred.

        Raises:
            RuntimeError: If CatPred is not installed or vendored.
        """
        if self._available is None:
            self.setup()

        if self._available != "local" or self._model is None:
            raise RuntimeError(
                "CatPred is not available. Install it locally or vendor it — "
                "see the setup() warning for instructions."
            )

        result = self._model.predict(sequence=sequence, substrate=smiles)
        log_kcat = float(result.get("log_kcat", result.get("prediction", 0.0)))
        metadata = {
            "method": "catpred_local",
            "raw_result": result,
        }
        return log_kcat, metadata

    @property
    def name(self) -> str:
        return "CatPred"
