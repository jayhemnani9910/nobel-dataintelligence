"""Training infrastructure for VibroPredict."""

from __future__ import annotations
from importlib import import_module
from typing import Any

_EXPORTS = {
    "TrainerWithMMDrop": (".trainer", "TrainerWithMMDrop"),
    "MutantRankingLoss": (".losses", "MutantRankingLoss"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = _EXPORTS[name]
    module = import_module(mod_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
