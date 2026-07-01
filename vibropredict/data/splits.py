"""
Data Splitting Strategies for Enzyme Kinetics

Provides pluggable split strategies for enzyme kinetics datasets:
- RandomSplit: Standard random train/val/test split (reproducible).
- ECHoldoutSplit: Out-of-distribution split by EC class — entire
  enzyme families are held out to test generalization.

Both implement a common Splitter protocol:
    .split(df) -> dict[str, pd.DataFrame]
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class Splitter(Protocol):
    """Protocol for data splitting strategies."""

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split a DataFrame into train/val/test subsets.

        Args:
            df: Input DataFrame with at least the columns required
                by the specific splitter implementation.

        Returns:
            Dictionary with keys 'train', 'val', 'test', each mapping
            to a DataFrame subset.
        """
        ...


class RandomSplit:
    """Random train/val/test split with configurable ratios.

    Reproduces the existing (pre-Task-B) behavior when given the
    same seed and default ratios.

    Args:
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        if not (0 < train_ratio + val_ratio < 1.0 + 1e-9):
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must sum to less than 1.0"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split DataFrame randomly.

        Args:
            df: Input DataFrame (any columns).

        Returns:
            Dict with 'train', 'val', 'test' DataFrames.
        """
        n = len(df)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        result = {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "val": df.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True),
        }

        logger.info(
            f"RandomSplit (seed={self.seed}): "
            f"train={len(result['train'])}, "
            f"val={len(result['val'])}, "
            f"test={len(result['test'])}"
        )
        return result


class ECHoldoutSplit:
    """Out-of-distribution split by EC class.

    Holds out entire EC classes so no enzyme family appears in more than one
    split. This tests whether a model generalizes to unseen enzyme families.

    If ``ec_column`` is not present in the DataFrame, it is derived from
    ``ec_number`` or ``ec`` columns by extracting the first ``ec_level`` digits.

    Default ``ec_level=2`` (EC subclass) balances generalization pressure with
    statistical robustness: EC top-level (level 1) has only ~6 classes on most
    enzyme corpora, which makes the test split statistically thin; EC subclass
    (level 2) typically gives 60+ groups on KinHub-scale data.

    Args:
        train_ratio: Approximate fraction of EC classes for training.
        val_ratio: Approximate fraction of EC classes for validation.
        ec_column: Column name containing the EC class label.
        ec_level: How many levels of the EC hierarchy to use for
            grouping (1 = top-level class, 2 = subclass, 3 = sub-subclass).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ec_column: str = "ec_class",
        ec_level: int = 2,
        seed: int = 42,
    ):
        if not (0 < train_ratio + val_ratio < 1.0 + 1e-9):
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must sum to less than 1.0"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.ec_column = ec_column
        self.ec_level = ec_level
        self.seed = seed

    def _ensure_ec_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive ec_class column if not already present."""
        if self.ec_column in df.columns:
            return df

        # Try common column names
        ec_source = None
        for col in ["ec_number", "ec", "EC", "EC_number"]:
            if col in df.columns:
                ec_source = col
                break

        if ec_source is None:
            raise ValueError(
                f"Cannot derive '{self.ec_column}': no EC number column found. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df[self.ec_column] = df[ec_source].apply(self._extract_ec_class)
        return df

    def _extract_ec_class(self, ec_number: str) -> str:
        """Extract EC class at the configured level.

        E.g., for ec_level=1: '2.7.1.1' -> '2'
              for ec_level=2: '2.7.1.1' -> '2.7'
        """
        try:
            parts = str(ec_number).split(".")
            return ".".join(parts[: self.ec_level])
        except (AttributeError, TypeError):
            return "unknown"

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split DataFrame by EC class holdout.

        Args:
            df: Input DataFrame with an EC number column.

        Returns:
            Dict with 'train', 'val', 'test' DataFrames.

        Raises:
            ValueError: If any split would be empty.
        """
        df = self._ensure_ec_column(df)

        # Get unique EC classes and shuffle
        ec_classes = sorted(df[self.ec_column].unique())
        rng = np.random.RandomState(self.seed)
        rng.shuffle(ec_classes)

        n_classes = len(ec_classes)
        n_train = max(1, int(n_classes * self.train_ratio))
        n_val = max(1, int(n_classes * self.val_ratio))
        # Ensure test gets at least 1 class
        n_val = min(n_val, n_classes - n_train - 1)

        train_classes = set(ec_classes[:n_train])
        val_classes = set(ec_classes[n_train : n_train + n_val])
        test_classes = set(ec_classes[n_train + n_val :])

        # Verify no overlap
        assert train_classes.isdisjoint(val_classes), "Train/val EC overlap"
        assert train_classes.isdisjoint(test_classes), "Train/test EC overlap"
        assert val_classes.isdisjoint(test_classes), "Val/test EC overlap"

        result = {
            "train": df[df[self.ec_column].isin(train_classes)].reset_index(drop=True),
            "val": df[df[self.ec_column].isin(val_classes)].reset_index(drop=True),
            "test": df[df[self.ec_column].isin(test_classes)].reset_index(drop=True),
        }

        # Verify no empty splits
        for split_name, split_df in result.items():
            if len(split_df) == 0:
                raise ValueError(
                    f"ECHoldoutSplit produced empty '{split_name}' split. "
                    f"EC classes: train={len(train_classes)}, "
                    f"val={len(val_classes)}, test={len(test_classes)}"
                )

        logger.info(
            f"ECHoldoutSplit (seed={self.seed}, level={self.ec_level}): "
            f"train={len(result['train'])} ({len(train_classes)} classes), "
            f"val={len(result['val'])} ({len(val_classes)} classes), "
            f"test={len(result['test'])} ({len(test_classes)} classes)"
        )
        return result
