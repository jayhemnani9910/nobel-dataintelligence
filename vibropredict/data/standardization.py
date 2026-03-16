"""
Data Standardization Utilities

Transformations for enzyme kinetics features: log-transform k_cat,
canonicalize SMILES, compute differential reaction fingerprints (DRFP),
and split datasets.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_transform_kcat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a log10-transformed k_cat column.

    Zeros and negative values are clipped to 1e-10 before the transform.

    Args:
        df: DataFrame with a 'k_cat' column.

    Returns:
        DataFrame with an additional 'log_kcat' column.
    """
    df = df.copy()
    clipped = df["k_cat"].clip(lower=1e-10)
    df["log_kcat"] = np.log10(clipped)
    logger.info(f"log10-transformed k_cat for {len(df)} rows")
    return df


def canonicalize_smiles(
    df: pd.DataFrame, col: str = "substrate_smiles"
) -> pd.DataFrame:
    """
    Canonicalize SMILES strings using RDKit.

    Invalid SMILES are set to None.

    Args:
        df: DataFrame containing a SMILES column.
        col: Name of the column to canonicalize.

    Returns:
        DataFrame with the SMILES column replaced by canonical forms.
    """
    from rdkit import Chem  # type: ignore

    df = df.copy()

    def _canon(smiles: Optional[str]) -> Optional[str]:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    before_invalid = df[col].isna().sum()
    df[col] = df[col].apply(_canon)
    after_invalid = df[col].isna().sum()
    new_invalid = after_invalid - before_invalid
    if new_invalid > 0:
        logger.info(f"Canonicalization set {new_invalid} invalid SMILES to None")
    return df


def compute_drfp(
    substrate_smiles: str,
    product_smiles: Optional[str],
    n_bits: int = 512,
) -> np.ndarray:
    """
    Compute a differential reaction fingerprint (DRFP).

    Uses the XOR of Morgan fingerprints for substrate and product.
    If the product SMILES is None or invalid, returns the substrate
    fingerprint (XOR with zeros).

    Args:
        substrate_smiles: SMILES of the substrate.
        product_smiles: SMILES of the product (may be None).
        n_bits: Length of the bit vector.

    Returns:
        NumPy array of shape (n_bits,) with float values.
    """
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    def _morgan(smiles: Optional[str]) -> Optional[np.ndarray]:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp, dtype=np.float32)

    sub_fp = _morgan(substrate_smiles)
    if sub_fp is None:
        return np.zeros(n_bits, dtype=np.float32)

    prod_fp = _morgan(product_smiles)
    if prod_fp is None:
        return sub_fp

    # XOR via absolute difference on binary vectors
    drfp = np.abs(sub_fp - prod_fp)
    return drfp


def cluster_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train / validation / test sets.

    This performs a simple random split. For proper sequence-based
    clustering (e.g. via MMseqs2), an external tool should be used
    to assign cluster IDs before splitting.

    Args:
        df: Input DataFrame.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    logger.info(
        f"Split {n} rows -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df
