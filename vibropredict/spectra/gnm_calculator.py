"""
GNM Calculator for VibroPredict

Computes Gaussian Network Model eigenvalues and eigenvectors from
protein structures, and extracts thermodynamic features using ANM analysis.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_prody():
    """Lazy import for ProDy."""
    try:
        import prody as pr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ProDy is required for GNM calculations. Install it with `pip install prody`."
        ) from exc
    return pr


class GNMCalculator:
    """
    Gaussian Network Model calculator for protein vibrational analysis.

    Builds a GNM Kirchhoff matrix from C-alpha coordinates and computes
    normal mode eigenvalues and eigenvectors. Also provides thermodynamic
    feature extraction via ANM analysis.
    """

    def __init__(self, cutoff: float = 10.0):
        """
        Initialize GNM calculator.

        Args:
            cutoff: Distance cutoff for C-alpha contacts (Angstroms).
        """
        self.cutoff = cutoff

    def compute_from_pdb(self, pdb_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GNM eigenvalues and eigenvectors from a PDB file.

        Parses the PDB, selects C-alpha atoms, builds the GNM Kirchhoff
        matrix, and calculates normal modes.

        Args:
            pdb_path: Path to PDB file.

        Returns:
            Tuple of (eigenvalues, eigenvectors) with zero modes removed.
        """
        pr = _require_prody()

        structure = pr.parsePDB(pdb_path)
        ca_atoms = structure.select('ca')
        if ca_atoms is None:
            raise ValueError(f"No C-alpha atoms found in {pdb_path}")

        n_atoms = ca_atoms.numAtoms()
        logger.info(f"Parsed {pdb_path}: {n_atoms} C-alpha atoms")

        gnm = pr.GNM(f"GNM_{n_atoms}")
        gnm.buildKirchhoff(ca_atoms, cutoff=self.cutoff)

        n_modes = min(n_atoms - 1, n_atoms)
        gnm.calcModes(n_modes)

        eigenvalues = np.asarray(gnm.getEigvals())
        eigenvectors = np.asarray(gnm.getEigvecs())

        # Filter near-zero (rigid-body) modes
        zero_tol = 1e-8
        nonzero_idx = np.where(eigenvalues > zero_tol)[0]
        nonzero_idx = nonzero_idx[np.argsort(eigenvalues[nonzero_idx])]

        logger.info(f"Computed {len(nonzero_idx)} non-zero GNM modes")
        return eigenvalues[nonzero_idx], eigenvectors[:, nonzero_idx]

    def compute_from_coords(
        self, coords: np.ndarray, cutoff: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build GNM from a raw coordinate array and compute modes.

        Args:
            coords: Coordinate array of shape (n_atoms, 3).
            cutoff: Distance cutoff in Angstroms. Defaults to self.cutoff.

        Returns:
            Tuple of (eigenvalues, eigenvectors) with zero modes removed.
        """
        pr = _require_prody()

        if cutoff is None:
            cutoff = self.cutoff

        n_atoms = coords.shape[0]
        logger.info(f"Building GNM from {n_atoms} coordinate points")

        gnm = pr.GNM(f"GNM_{n_atoms}")
        gnm.buildKirchhoff(coords, cutoff=cutoff)

        n_modes = min(n_atoms - 1, n_atoms)
        gnm.calcModes(n_modes)

        eigenvalues = np.asarray(gnm.getEigvals())
        eigenvectors = np.asarray(gnm.getEigvecs())

        zero_tol = 1e-8
        nonzero_idx = np.where(eigenvalues > zero_tol)[0]
        nonzero_idx = nonzero_idx[np.argsort(eigenvalues[nonzero_idx])]

        logger.info(f"Computed {len(nonzero_idx)} non-zero GNM modes from coords")
        return eigenvalues[nonzero_idx], eigenvectors[:, nonzero_idx]

    def extract_thermodynamic_features(
        self, pdb_path: str, k: int = 100
    ) -> dict:
        """
        Extract thermodynamic features from a PDB structure using ANM analysis.

        Computes vibrational entropy, mean B-factor (residue fluctuations),
        and mode collectivity via ``src.nma_analysis.ANMAnalyzer``.

        Args:
            pdb_path: Path to PDB file.
            k: Number of modes to compute.

        Returns:
            Dictionary with keys:
                - vibrational_entropy: Vibrational entropy in J/(mol*K)
                - mean_bfactor: Mean residue fluctuation (Angstroms^2)
                - collectivity_mode0: Collectivity of the lowest-frequency mode
        """
        from src.nma_analysis import ANMAnalyzer

        logger.info(f"Extracting thermodynamic features from {pdb_path}")
        analyzer = ANMAnalyzer(pdb_path)

        vibrational_entropy = analyzer.compute_vibrational_entropy(k=k)
        fluctuations = analyzer.get_residue_fluctuations(k=k)
        mean_bfactor = float(np.mean(fluctuations))
        collectivity_mode0 = analyzer.get_mode_collectivity(mode_idx=0)

        features = {
            'vibrational_entropy': vibrational_entropy,
            'mean_bfactor': mean_bfactor,
            'collectivity_mode0': collectivity_mode0,
        }

        logger.info(
            f"Thermodynamic features: S_vib={vibrational_entropy:.2f} J/(mol*K), "
            f"mean_bfactor={mean_bfactor:.4f}, collectivity={collectivity_mode0:.4f}"
        )
        return features
