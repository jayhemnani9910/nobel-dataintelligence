"""
BaselineModel ABC and Registry

Defines the abstract base class for pluggable baseline models and
a decorator-based registry for auto-discovery.

To add a new baseline:
1. Create a file under vibropredict/evaluation/baselines/
2. Subclass BaselineModel
3. Decorate with @register_baseline("name")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Global baseline registry
_BASELINE_REGISTRY: dict[str, type] = {}


class BaselineModel(ABC):
    """Abstract base class for baseline enzyme kinetics models.

    All baselines must implement the ``predict`` method and optionally
    override ``setup`` for initialization that requires downloads or
    heavy computation.
    """

    @abstractmethod
    def predict(
        self,
        sequence: str,
        smiles: str,
    ) -> tuple[float, dict[str, Any]]:
        """Predict log(k_cat) for an enzyme-substrate pair.

        Args:
            sequence: Amino acid sequence of the enzyme.
            smiles: SMILES string of the substrate.

        Returns:
            Tuple of:
                - log_kcat: Predicted log10(k_cat) value.
                - metadata: Dictionary with model-specific metadata
                  (e.g., confidence scores, feature importances).
        """
        ...

    def setup(self) -> None:  # noqa: B027
        """Optional setup hook for model initialization.

        Override this to download weights, initialize heavy dependencies,
        etc. Called once before the first ``predict`` call.
        """

    def predict_batch(
        self,
        sequences: list[str],
        smiles_list: list[str],
    ) -> list[tuple[float, dict[str, Any]]]:
        """Predict log(k_cat) for a batch of enzyme-substrate pairs.

        Default implementation calls ``predict`` in a loop. Override
        for models that support native batching.

        Args:
            sequences: List of amino acid sequences.
            smiles_list: List of substrate SMILES strings.

        Returns:
            List of (log_kcat, metadata) tuples.
        """
        return [self.predict(seq, smi) for seq, smi in zip(sequences, smiles_list, strict=True)]

    @property
    def name(self) -> str:
        """Human-readable name of the baseline model."""
        return self.__class__.__name__


def register_baseline(name: str):
    """Decorator to register a baseline model class.

    Usage::

        @register_baseline("my_model")
        class MyModel(BaselineModel):
            def predict(self, sequence, smiles):
                ...

    Args:
        name: Unique identifier for the baseline.
    """

    def decorator(cls):
        if not issubclass(cls, BaselineModel):
            raise TypeError(f"{cls.__name__} must subclass BaselineModel to be registered.")
        if name in _BASELINE_REGISTRY:
            logger.warning(
                f"Baseline '{name}' is already registered; overwriting with {cls.__name__}."
            )
        _BASELINE_REGISTRY[name] = cls
        logger.debug(f"Registered baseline: {name} -> {cls.__name__}")
        return cls

    return decorator


def get_baseline(name: str, **kwargs) -> BaselineModel:
    """Instantiate a registered baseline model by name.

    Args:
        name: Registered baseline identifier.
        **kwargs: Passed to the baseline's constructor.

    Returns:
        An instance of the requested BaselineModel subclass.

    Raises:
        KeyError: If no baseline with the given name is registered.
    """
    if name not in _BASELINE_REGISTRY:
        available = sorted(_BASELINE_REGISTRY.keys())
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")
    return _BASELINE_REGISTRY[name](**kwargs)


def list_baselines() -> list[str]:
    """Return a sorted list of all registered baseline names."""
    return sorted(_BASELINE_REGISTRY.keys())
