"""
Structure Quality Control

Utilities for parsing pLDDT scores from PDB files, filtering
structures by quality thresholds, and flagging disordered regions.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def parse_plddt_from_pdb(pdb_path: str) -> np.ndarray:
    """
    Extract per-residue pLDDT scores from a PDB file.

    Reads the B-factor column of CA (C-alpha) ATOM records,
    which stores pLDDT in AlphaFold / ESMFold predictions.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        1-D array of pLDDT values, one per CA atom.
    """
    plddt_values: List[float] = []

    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    bfactor = float(line[60:66].strip())
                    plddt_values.append(bfactor)
                except (ValueError, IndexError):
                    continue

    arr = np.array(plddt_values, dtype=np.float64)
    logger.info(f"Parsed {len(arr)} pLDDT values from {pdb_path}")
    return arr


def filter_by_quality(
    pdb_path: str,
    min_global: float = 60.0,
    min_per_residue: float = 70.0,
) -> bool:
    """
    Check whether a predicted structure passes quality thresholds.

    A structure passes if:
    - Mean pLDDT >= *min_global*
    - More than 70 % of residues have pLDDT >= *min_per_residue*

    Args:
        pdb_path: Path to PDB file.
        min_global: Minimum acceptable mean pLDDT.
        min_per_residue: Per-residue pLDDT threshold for the
                         fraction check.

    Returns:
        True if the structure passes both checks.
    """
    plddt = parse_plddt_from_pdb(pdb_path)

    if plddt.size == 0:
        logger.warning(f"No pLDDT values found in {pdb_path}")
        return False

    mean_plddt = float(plddt.mean())
    frac_good = float((plddt >= min_per_residue).sum()) / plddt.size

    passed = mean_plddt >= min_global and frac_good > 0.7
    logger.info(
        f"Quality check {pdb_path}: mean_pLDDT={mean_plddt:.1f}, "
        f"frac>={min_per_residue}={frac_good:.2f}, passed={passed}"
    )
    return passed


def flag_disordered_regions(
    plddt: np.ndarray, threshold: float = 50.0
) -> List[Tuple[int, int]]:
    """
    Identify contiguous regions with low pLDDT (likely disordered).

    Args:
        plddt: 1-D array of per-residue pLDDT scores.
        threshold: Residues below this value are considered disordered.

    Returns:
        List of (start, end) tuples (inclusive indices) for each
        contiguous disordered region.
    """
    if plddt.size == 0:
        return []

    low_mask = plddt < threshold
    regions: List[Tuple[int, int]] = []
    in_region = False
    start = 0

    for i, is_low in enumerate(low_mask):
        if is_low and not in_region:
            start = i
            in_region = True
        elif not is_low and in_region:
            regions.append((start, i - 1))
            in_region = False

    # Close final region if it extends to the end
    if in_region:
        regions.append((start, len(plddt) - 1))

    if regions:
        logger.info(
            f"Found {len(regions)} disordered region(s) below pLDDT {threshold}"
        )
    return regions
