"""Pluggable baseline model registry for SOTA comparison."""

from __future__ import annotations

import logging

from vibropredict.evaluation.baselines.base import (
    BaselineModel,
    get_baseline,
    list_baselines,
    register_baseline,
)

__all__ = [
    "BaselineModel",
    "get_baseline",
    "list_baselines",
    "register_baseline",
]

# Auto-discover baselines by importing them.
from vibropredict.evaluation.baselines import stub  # noqa: F401

try:
    from vibropredict.evaluation.baselines import catpred  # noqa: F401
except ImportError as exc:
    # CatPred dependencies not installed — skip registration
    logging.getLogger(__name__).warning(f"CatPred baseline not registered: {exc}")
