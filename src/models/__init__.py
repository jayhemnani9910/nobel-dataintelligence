"""
Model architectures for Quantum Data Decoder.

This package includes optional graph components that depend on `torch_geometric`.
To keep imports lightweight (and to allow running tests without optional deps),
symbols are exposed via lazy attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Graph models (optional: torch_geometric)
    "ProteinGNN": (".gnn", "ProteinGNN"),
    "GraphConstruction": (".gnn", "GraphConstruction"),
    # Spectral models
    "SpectralCNN": (".cnn", "SpectralCNN"),
    "SpectralFeatureExtractor": (".cnn", "SpectralFeatureExtractor"),
    "MultiScaleSpectralCNN": (".cnn", "MultiScaleSpectralCNN"),
    # Multimodal models
    "VibroStructuralModel": (".multimodal", "VibroStructuralModel"),
    "VibroStructuralFusion": (".multimodal", "VibroStructuralFusion"),
    # Losses
    "MarginRankingLossCustom": (".losses", "MarginRankingLossCustom"),
    "SpearmanCorrelationLoss": (".losses", "SpearmanCorrelationLoss"),
    "FocalLoss": (".losses", "FocalLoss"),
    "ContrastiveLoss": (".losses", "ContrastiveLoss"),
    "WeightedBCELoss": (".losses", "WeightedBCELoss"),
    "CombinedLoss": (".losses", "CombinedLoss"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"Failed to import {__name__}.{name}") from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))

