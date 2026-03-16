"""
EnzyExtract Filter

Loads EnzyExtract CSV data and filters for high-confidence entries
that complement the KinHub training set.
"""

import logging
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)


def _is_valid_smiles(smiles: str) -> bool:
    """
    Check whether a SMILES string is chemically valid.

    Uses RDKit when available; falls back to a basic non-empty check otherwise.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # RDKit not installed -- accept any non-empty string
        return True


class EnzyExtractFilter:
    """
    Filter for EnzyExtract enzyme kinetics CSV data.

    Applies quality filters (confidence, valid identifiers, SMILES
    parseability) and removes entries already present in KinHub.
    """

    def __init__(self, csv_path: str):
        """
        Initialize filter.

        Args:
            csv_path: Path to EnzyExtract CSV file.
        """
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        """
        Read CSV into a DataFrame.

        Returns:
            Raw DataFrame from the EnzyExtract export.
        """
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows from {self.csv_path}")
        return df

    def filter(self, df: pd.DataFrame, kinhub_ids: Set[str]) -> pd.DataFrame:
        """
        Apply quality and deduplication filters.

        Keeps rows where:
        - uniprot_id is non-empty
        - substrate_smiles is parseable (via RDKit if available)
        - confidence > 0.9
        - uniprot_id is not already present in *kinhub_ids*

        Args:
            df: Raw EnzyExtract DataFrame.
            kinhub_ids: Set of UniProt IDs already in KinHub.

        Returns:
            Filtered DataFrame.
        """
        before = len(df)

        # Valid uniprot_id
        mask_uid = df["uniprot_id"].notna() & (df["uniprot_id"].astype(str).str.strip() != "")

        # Valid SMILES
        mask_smiles = df["substrate_smiles"].apply(_is_valid_smiles)

        # Confidence threshold
        mask_conf = df["confidence"] > 0.9

        # Not in KinHub
        mask_novel = ~df["uniprot_id"].astype(str).isin(kinhub_ids)

        filtered = df[mask_uid & mask_smiles & mask_conf & mask_novel].copy()
        filtered = filtered.reset_index(drop=True)

        logger.info(
            f"Filtered EnzyExtract: {before} -> {len(filtered)} rows "
            f"(dropped {before - len(filtered)})"
        )
        return filtered

    def merge(self, kinhub_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge KinHub and filtered EnzyExtract DataFrames.

        Args:
            kinhub_df: Validated KinHub DataFrame.
            filtered_df: Filtered EnzyExtract DataFrame.

        Returns:
            Concatenated DataFrame with a fresh integer index.
        """
        merged = pd.concat([kinhub_df, filtered_df], ignore_index=True)
        logger.info(
            f"Merged dataset: {len(kinhub_df)} + {len(filtered_df)} = {len(merged)} rows"
        )
        return merged
