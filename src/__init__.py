"""
Quantum Data Decoder - Vibrational Analysis for Protein Function Prediction

A comprehensive framework for predicting protein stability and function
through multimodal deep learning on structural and spectral data.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__version__ = "0.1.0"
__author__ = "Quantum Data Decoder Team"

__all__ = [
    'data_acquisition',
    'nma_analysis',
    'spectral_generation',
    'utils',
    'datasets',
    'training',
    'models',
]


def __getattr__(name: str) -> ModuleType:
    """Lazy-load submodules to avoid import-time optional dependency failures."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(f"{__name__}.{name}")
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Failed to import submodule '{name}'. This feature may require optional dependencies."
        ) from exc

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
