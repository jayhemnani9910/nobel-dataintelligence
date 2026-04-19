#!/usr/bin/env python3
"""
KinHub vs RealKcat Dataset Audit

Joins a local copy of the RealKcat dataset against our KinHub CSV on
(uniprot_id, substrate_smiles) and writes an overlap report.

As of 2026-04-19, no verified public URL for RealKcat's (enzyme, substrate,
k_cat) table is known. This script therefore requires a local path via
``--realkcat-path``. If no path is given, the audit runs in blocked mode:
it writes a minimal report marking every KinHub row's RealKcat membership as
unknown and exits without error.

Usage (once RealKcat data is obtained):
    python scripts/audit_kinhub_vs_realkcat.py \
        --kinhub data/kaggle/kinhub.csv \
        --realkcat-path data/external/realkcat_dataset.csv \
        --output data/audits/kinhub_realkcat_overlap.csv

Usage (blocked mode, data not yet available):
    python scripts/audit_kinhub_vs_realkcat.py \
        --kinhub data/kaggle/kinhub.csv \
        --output data/audits/kinhub_realkcat_overlap.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REALKCAT_EXPECTED_COLUMNS = {"uniprot_id", "substrate_smiles", "kcat"}
REALKCAT_COLUMN_ALIASES = {
    "UniProt_ID": "uniprot_id",
    "uniprot": "uniprot_id",
    "Substrate_SMILES": "substrate_smiles",
    "SMILES": "substrate_smiles",
    "k_cat": "kcat",
    "Kcat": "kcat",
    "K_cat": "kcat",
}


def _load_realkcat_local(path: Path) -> pd.DataFrame | None:
    """Load a local RealKcat CSV and normalize its column names.

    Returns a DataFrame if the file exists and contains the expected
    columns (after alias mapping), otherwise None.
    """
    if not path.exists():
        logger.warning(f"RealKcat file not found at {path}")
        return None

    df = pd.read_csv(path)
    df = df.rename(columns=REALKCAT_COLUMN_ALIASES)
    missing = REALKCAT_EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error(
            f"RealKcat file at {path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
        return None

    logger.info(f"Loaded RealKcat from {path}: {len(df)} rows")
    return df


def _normalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES if RDKit is available, otherwise strip whitespace."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except ImportError:
        pass
    return str(smiles).strip()


def run_audit(
    kinhub_path: str,
    output_path: str,
    realkcat_path: str | None = None,
) -> dict:
    """Run the KinHub vs RealKcat overlap audit.

    Args:
        kinhub_path: Path to the KinHub CSV file.
        output_path: Path for the output overlap CSV.
        realkcat_path: Optional local path to a RealKcat CSV. If None or
            the file does not exist, the audit runs in blocked mode.

    Returns:
        Dictionary with audit results including overlap_percentage.
    """
    kinhub_path = Path(kinhub_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load KinHub
    if not kinhub_path.exists():
        logger.error(f"KinHub CSV not found at {kinhub_path}")
        sys.exit(1)

    kinhub_df = pd.read_csv(kinhub_path)
    logger.info(f"KinHub: {len(kinhub_df)} rows")

    # Normalize column names
    col_map = {"k_cat": "kcat_kinhub"}
    if "kcat" in kinhub_df.columns:
        col_map = {"kcat": "kcat_kinhub"}
    elif "k_cat" in kinhub_df.columns:
        col_map = {"k_cat": "kcat_kinhub"}
    elif "log_kcat" in kinhub_df.columns:
        # Convert back from log space for comparison

        kinhub_df["kcat_kinhub"] = 10 ** kinhub_df["log_kcat"]
        col_map = {}

    kinhub_df = kinhub_df.rename(columns=col_map)

    # Normalize SMILES
    if "substrate_smiles" in kinhub_df.columns:
        kinhub_df["substrate_smiles_norm"] = kinhub_df["substrate_smiles"].apply(_normalize_smiles)
    else:
        logger.error("KinHub CSV missing 'substrate_smiles' column.")
        sys.exit(1)

    # Load RealKcat if a path was provided
    realkcat_df: pd.DataFrame | None = None
    if realkcat_path is not None:
        realkcat_df = _load_realkcat_local(Path(realkcat_path))
    else:
        logger.info(
            "No --realkcat-path provided. Running in blocked mode; see "
            "docs/future/DATASET_AUDIT.md for how to obtain the dataset."
        )

    results = {
        "kinhub_rows": len(kinhub_df),
        "realkcat_available": realkcat_df is not None,
    }

    if realkcat_df is None:
        logger.warning(
            "RealKcat data is NOT publicly available. Writing audit report with blocked status."
        )
        results["overlap_percentage"] = None
        results["realkcat_rows"] = None
        results["overlap_count"] = None

        # Write a minimal CSV marking all KinHub rows as unknown overlap
        audit_df = kinhub_df[["uniprot_id", "substrate_smiles"]].copy()
        audit_df["in_kinhub"] = True
        audit_df["in_realkcat"] = None  # Unknown
        if "kcat_kinhub" in kinhub_df.columns:
            audit_df["kcat_kinhub"] = kinhub_df["kcat_kinhub"]
        else:
            audit_df["kcat_kinhub"] = None
        audit_df["kcat_realkcat"] = None
        audit_df.to_csv(output_path, index=False)
        logger.info(f"Audit CSV written (blocked) to {output_path}")
        return results

    # RealKcat available — perform join
    logger.info(f"RealKcat: {len(realkcat_df)} rows")
    results["realkcat_rows"] = len(realkcat_df)

    realkcat_df = realkcat_df.rename(columns={"kcat": "kcat_realkcat"})
    if "substrate_smiles" in realkcat_df.columns:
        realkcat_df["substrate_smiles_norm"] = realkcat_df["substrate_smiles"].apply(
            _normalize_smiles
        )
    else:
        logger.error("RealKcat CSV missing 'substrate_smiles' column.")
        sys.exit(1)

    # Outer join on (uniprot_id, normalized_smiles)
    kinhub_keys = kinhub_df[["uniprot_id", "substrate_smiles_norm", "substrate_smiles"]].copy()
    kinhub_keys["in_kinhub"] = True
    if "kcat_kinhub" in kinhub_df.columns:
        kinhub_keys["kcat_kinhub"] = kinhub_df["kcat_kinhub"]

    realkcat_keys = realkcat_df[["uniprot_id", "substrate_smiles_norm", "substrate_smiles"]].copy()
    realkcat_keys["in_realkcat"] = True
    if "kcat_realkcat" in realkcat_df.columns:
        realkcat_keys["kcat_realkcat"] = realkcat_df["kcat_realkcat"]

    merged = pd.merge(
        kinhub_keys,
        realkcat_keys,
        on=["uniprot_id", "substrate_smiles_norm"],
        how="outer",
        suffixes=("", "_rk"),
    )

    merged["in_kinhub"] = merged["in_kinhub"].fillna(False)
    merged["in_realkcat"] = merged["in_realkcat"].fillna(False)

    # Use substrate_smiles from whichever side has it
    if "substrate_smiles_rk" in merged.columns:
        merged["substrate_smiles"] = merged["substrate_smiles"].fillna(
            merged["substrate_smiles_rk"]
        )
        merged = merged.drop(columns=["substrate_smiles_rk"])

    merged = merged.drop(columns=["substrate_smiles_norm"])

    overlap = merged["in_kinhub"] & merged["in_realkcat"]
    overlap_count = int(overlap.sum())
    overlap_pct = (overlap_count / len(kinhub_df)) * 100 if len(kinhub_df) > 0 else 0.0

    results["overlap_count"] = overlap_count
    results["overlap_percentage"] = round(overlap_pct, 2)

    logger.info(
        f"Overlap: {overlap_count} entries "
        f"({overlap_pct:.1f}% of KinHub, "
        f"{(overlap_count / len(realkcat_df) * 100):.1f}% of RealKcat)"
    )

    # Write output
    output_cols = [
        "uniprot_id",
        "substrate_smiles",
        "in_kinhub",
        "in_realkcat",
    ]
    if "kcat_kinhub" in merged.columns:
        output_cols.append("kcat_kinhub")
    if "kcat_realkcat" in merged.columns:
        output_cols.append("kcat_realkcat")

    merged[output_cols].to_csv(output_path, index=False)
    logger.info(f"Audit CSV written to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit KinHub vs RealKcat dataset overlap.")
    parser.add_argument(
        "--kinhub",
        type=str,
        default="data/kaggle/kinhub.csv",
        help="Path to KinHub CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/audits/kinhub_realkcat_overlap.csv",
        help="Path for output overlap CSV.",
    )
    parser.add_argument(
        "--realkcat-path",
        type=str,
        default=None,
        help=(
            "Optional local path to RealKcat CSV. If omitted, the audit "
            "runs in blocked mode (no overlap computed)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_audit(args.kinhub, args.output, realkcat_path=args.realkcat_path)
    print("\n=== Audit Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
