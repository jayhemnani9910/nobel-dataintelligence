"""
Tests for data splitting strategies.

Verifies:
- RandomSplit reproduces existing behavior with the same seed.
- ECHoldoutSplit places no EC class in more than one split.
- ECHoldoutSplit produces no empty splits.
- Both implement the Splitter protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibropredict.data.splits import ECHoldoutSplit, RandomSplit, Splitter


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking KinHub data."""
    rng = np.random.RandomState(0)
    n = 200
    ec_numbers = [
        "1.1.1.1",
        "1.2.3.4",
        "1.3.1.1",
        "2.7.1.1",
        "2.1.3.1",
        "3.1.1.1",
        "3.2.1.1",
        "3.4.21.1",
        "4.1.1.1",
        "4.2.1.1",
        "5.1.1.1",
        "6.1.1.1",
        "6.3.2.1",
    ]
    return pd.DataFrame(
        {
            "uniprot_id": [f"P{i:05d}" for i in range(n)],
            "substrate_smiles": [f"CC{'C' * (i % 5)}" for i in range(n)],
            "k_cat": rng.uniform(0.1, 1000, n),
            "ec_number": [ec_numbers[i % len(ec_numbers)] for i in range(n)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 5] * n,
        }
    )


class TestSplitterProtocol:
    """Verify both split classes implement the Splitter protocol."""

    def test_random_split_is_splitter(self):
        assert isinstance(RandomSplit(), Splitter)

    def test_ec_holdout_is_splitter(self):
        assert isinstance(ECHoldoutSplit(), Splitter)


class TestRandomSplit:
    """Tests for RandomSplit."""

    def test_split_sizes(self, sample_df):
        """Check that split sizes match expected ratios."""
        splitter = RandomSplit(train_ratio=0.8, val_ratio=0.1, seed=42)
        splits = splitter.split(sample_df)

        total = sum(len(s) for s in splits.values())
        assert total == len(sample_df)

        assert len(splits["train"]) == int(len(sample_df) * 0.8)
        assert len(splits["val"]) == int(len(sample_df) * 0.1)

    def test_reproducibility(self, sample_df):
        """Same seed produces identical splits."""
        splitter1 = RandomSplit(seed=42)
        splitter2 = RandomSplit(seed=42)

        splits1 = splitter1.split(sample_df)
        splits2 = splitter2.split(sample_df)

        for key in ["train", "val", "test"]:
            pd.testing.assert_frame_equal(splits1[key], splits2[key])

    def test_different_seed_different_split(self, sample_df):
        """Different seeds produce different splits."""
        splits1 = RandomSplit(seed=42).split(sample_df)
        splits2 = RandomSplit(seed=123).split(sample_df)

        # At least one split should differ
        all_same = all(splits1[k].equals(splits2[k]) for k in ["train", "val", "test"])
        assert not all_same

    def test_no_overlap(self, sample_df):
        """No sample appears in multiple splits."""
        splitter = RandomSplit(seed=42)
        splits = splitter.split(sample_df)

        train_ids = set(splits["train"]["uniprot_id"])
        val_ids = set(splits["val"]["uniprot_id"])
        test_ids = set(splits["test"]["uniprot_id"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_splits_non_empty(self, sample_df):
        """All splits should be non-empty."""
        splitter = RandomSplit(seed=42)
        splits = splitter.split(sample_df)

        for key in ["train", "val", "test"]:
            assert len(splits[key]) > 0, f"Split '{key}' is empty"


class TestECHoldoutSplit:
    """Tests for ECHoldoutSplit."""

    def test_no_ec_class_in_multiple_splits(self, sample_df):
        """No EC class should appear in more than one split."""
        splitter = ECHoldoutSplit(seed=42, ec_level=1)
        splits = splitter.split(sample_df)

        # Derive EC class at level 1
        for key, split_df in splits.items():
            if "ec_class" not in split_df.columns:
                split_df = split_df.copy()
                split_df["ec_class"] = split_df["ec_number"].apply(lambda x: str(x).split(".")[0])
                splits[key] = split_df

        train_classes = set(splits["train"]["ec_class"])
        val_classes = set(splits["val"]["ec_class"])
        test_classes = set(splits["test"]["ec_class"])

        assert train_classes.isdisjoint(val_classes), (
            f"Train/val EC overlap: {train_classes & val_classes}"
        )
        assert train_classes.isdisjoint(test_classes), (
            f"Train/test EC overlap: {train_classes & test_classes}"
        )
        assert val_classes.isdisjoint(test_classes), (
            f"Val/test EC overlap: {val_classes & test_classes}"
        )

    def test_all_splits_non_empty(self, sample_df):
        """ECHoldoutSplit should produce no empty splits."""
        splitter = ECHoldoutSplit(seed=42, ec_level=1)
        splits = splitter.split(sample_df)

        for key in ["train", "val", "test"]:
            assert len(splits[key]) > 0, f"Split '{key}' is empty"

    def test_all_data_preserved(self, sample_df):
        """Total rows across splits should equal input."""
        splitter = ECHoldoutSplit(seed=42, ec_level=1)
        splits = splitter.split(sample_df)

        total = sum(len(s) for s in splits.values())
        assert total == len(sample_df)

    def test_reproducibility(self, sample_df):
        """Same seed produces identical splits."""
        splits1 = ECHoldoutSplit(seed=42).split(sample_df)
        splits2 = ECHoldoutSplit(seed=42).split(sample_df)

        for key in ["train", "val", "test"]:
            pd.testing.assert_frame_equal(splits1[key], splits2[key])

    def test_ec_level_2(self, sample_df):
        """EC level 2 produces finer-grained grouping."""
        splitter = ECHoldoutSplit(seed=42, ec_level=2)
        splits = splitter.split(sample_df)

        # Derive EC class at level 2
        for key, split_df in splits.items():
            if "ec_class" not in split_df.columns:
                split_df = split_df.copy()
                split_df["ec_class"] = split_df["ec_number"].apply(
                    lambda x: ".".join(str(x).split(".")[:2])
                )
                splits[key] = split_df

        train_classes = set(splits["train"]["ec_class"])
        val_classes = set(splits["val"]["ec_class"])
        test_classes = set(splits["test"]["ec_class"])

        assert train_classes.isdisjoint(val_classes)
        assert train_classes.isdisjoint(test_classes)
        assert val_classes.isdisjoint(test_classes)

    def test_derives_ec_class_from_ec_number(self, sample_df):
        """Should auto-derive ec_class from ec_number column."""
        # ec_class is not in the sample_df by default
        assert "ec_class" not in sample_df.columns
        assert "ec_number" in sample_df.columns

        splitter = ECHoldoutSplit(seed=42)
        # Should not raise
        splits = splitter.split(sample_df)
        assert len(splits["train"]) > 0

    def test_raises_without_ec_column(self):
        """Should raise ValueError if no EC column is found."""
        df = pd.DataFrame(
            {
                "uniprot_id": ["P00001"],
                "substrate_smiles": ["CC"],
                "k_cat": [1.0],
            }
        )
        splitter = ECHoldoutSplit(seed=42)
        with pytest.raises(ValueError, match="no EC number column found"):
            splitter.split(df)
