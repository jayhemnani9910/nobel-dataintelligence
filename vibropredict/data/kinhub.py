"""
KinHub Data Loader

Loads and validates enzyme kinetics data from KinHub CSV exports.
Handles deduplication of substrate-enzyme pairs via geometric mean aggregation.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["uniprot_id", "k_cat", "substrate_smiles"]


class KinHubLoader:
    """
    Loader for KinHub enzyme kinetics CSV data.

    Validates required columns, drops incomplete rows, and resolves
    ambiguous duplicate (enzyme, substrate) entries by geometric mean.
    """

    def __init__(self, csv_path: str):
        """
        Initialize loader.

        Args:
            csv_path: Path to KinHub CSV file.
        """
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        """
        Read CSV and validate required columns.

        Returns:
            Raw DataFrame with at least the required columns present.

        Raises:
            ValueError: If any required column is missing.
        """
        df = pd.read_csv(self.csv_path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"KinHub CSV is missing required columns: {sorted(missing)}"
            )
        logger.info(f"Loaded {len(df)} rows from {self.csv_path}")
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing uniprot_id or k_cat.

        Args:
            df: Input DataFrame.

        Returns:
            Cleaned DataFrame with no nulls in key columns.
        """
        before = len(df)
        df = df.dropna(subset=["uniprot_id", "k_cat"]).copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing uniprot_id or k_cat")
        return df

    def resolve_ambiguities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve duplicate (uniprot_id, substrate_smiles) pairs.

        For each duplicate group, the geometric mean of k_cat values
        is computed and a single representative row is kept.

        Args:
            df: Input DataFrame (should already be validated).

        Returns:
            DataFrame with unique (uniprot_id, substrate_smiles) pairs.
        """
        group_cols = ["uniprot_id", "substrate_smiles"]
        duplicated_mask = df.duplicated(subset=group_cols, keep=False)
        n_duplicated = duplicated_mask.sum()

        if n_duplicated == 0:
            return df

        # Geometric mean via log-space arithmetic
        def _geometric_mean(series: pd.Series) -> float:
            log_vals = np.log(series.clip(lower=1e-30))
            return float(np.exp(log_vals.mean()))

        # Split into unique and duplicated groups
        unique_df = df[~duplicated_mask].copy()
        dup_df = df[duplicated_mask].copy()

        # For each group, keep first row but replace k_cat with geometric mean
        merged = dup_df.groupby(group_cols, sort=False).agg(
            k_cat_geomean=("k_cat", _geometric_mean)
        ).reset_index()

        # Merge back non-numeric columns from the first occurrence
        first_rows = dup_df.drop_duplicates(subset=group_cols, keep="first").copy()
        first_rows = first_rows.drop(columns=["k_cat"])
        merged = first_rows.merge(merged, on=group_cols, how="left")
        merged = merged.rename(columns={"k_cat_geomean": "k_cat"})

        result = pd.concat([unique_df, merged], ignore_index=True)
        n_merged = len(dup_df) - len(merged)
        logger.info(
            f"Resolved {n_duplicated} duplicated rows into {len(merged)} groups "
            f"({n_merged} rows merged)"
        )
        return result
