"""
VDOS Engine for VibroPredict

Generates Vibrational Density of States (VDOS) spectra from protein
structures using GNM normal modes and spectral broadening.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from vibropredict.spectra.gnm_calculator import GNMCalculator

logger = logging.getLogger(__name__)


class VibroEnzymePipeline:
    """
    End-to-end pipeline for generating VDOS spectra from PDB structures.

    Combines GNM mode calculation with spectral broadening to produce
    continuous vibrational density of states suitable for downstream
    machine learning.
    """

    def __init__(
        self,
        n_points: int = 1000,
        freq_max: float = 500.0,
        broadening: float = 5.0,
    ):
        """
        Initialize the VDOS generation pipeline.

        Args:
            n_points: Number of frequency bins in the output spectrum.
            freq_max: Maximum frequency in cm^-1.
            broadening: Lorentzian broadening parameter in cm^-1.
        """
        self.n_points = n_points
        self.freq_max = freq_max
        self.broadening = broadening
        self.gnm_calculator = GNMCalculator()

    def generate_vdos(self, pdb_path: str) -> tuple[np.ndarray, dict]:
        """
        Generate a VDOS spectrum from a PDB file.

        Uses GNMCalculator to obtain eigenvalues, converts them to
        vibrational frequencies, and applies Lorentzian broadening
        via ``src.spectral_generation.SpectralGenerator``.

        Args:
            pdb_path: Path to PDB file.

        Returns:
            Tuple of (vdos_array, auxiliary_features_dict) where
            vdos_array has shape (n_points,) and auxiliary_features_dict
            contains extracted spectral features.
        """
        from src.spectral_generation import SpectralGenerator

        logger.info(f"Generating VDOS for {pdb_path}")

        eigenvalues, _ = self.gnm_calculator.compute_from_pdb(pdb_path)

        # Convert GNM eigenvalues to approximate frequencies in cm^-1.
        # GNM eigenvalues are proportional to omega^2; apply the same
        # conversion factor used in ANMAnalyzer.
        conversion_factor = 1 / (2 * np.pi * 29979.2458)
        frequencies = np.sqrt(np.maximum(eigenvalues, 0)) * conversion_factor

        generator = SpectralGenerator(
            freq_min=0, freq_max=self.freq_max, n_points=self.n_points
        )
        vdos = generator.generate_dos(frequencies, broadening=self.broadening)
        features = generator.extract_spectral_features(vdos)

        logger.info(
            f"VDOS generated: {len(frequencies)} modes, "
            f"peak at {features.get('peak_frequency', 0):.1f} cm^-1"
        )
        return vdos, features

    def batch_generate(
        self,
        pdb_paths: list[str],
        output_dir: str,
        n_workers: int = 4,
    ) -> list[str]:
        """
        Generate VDOS spectra for multiple PDB files in parallel.

        Each spectrum is saved as ``{stem}_vdos.npy`` in *output_dir*.

        Args:
            pdb_paths: List of PDB file paths.
            output_dir: Directory to write output .npy files.
            n_workers: Number of parallel worker threads.

        Returns:
            List of output file paths.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        total = len(pdb_paths)
        logger.info(f"Batch VDOS generation: {total} structures, {n_workers} workers")

        output_paths: list[str] = []

        def _process(pdb_path: str) -> str:
            stem = Path(pdb_path).stem
            vdos, _ = self.generate_vdos(pdb_path)
            out_path = output_dir_path / f"{stem}_vdos.npy"
            np.save(str(out_path), vdos)
            return str(out_path)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process, p): p for p in pdb_paths
            }
            for i, future in enumerate(as_completed(futures), 1):
                pdb_path = futures[future]
                try:
                    result = future.result()
                    output_paths.append(result)
                    logger.info(f"Progress: {i}/{total} — saved {result}")
                except Exception:
                    logger.exception(f"Failed to process {pdb_path}")

        logger.info(f"Batch complete: {len(output_paths)}/{total} succeeded")
        return output_paths
