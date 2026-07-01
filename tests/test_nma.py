"""
Regression tests for Normal Mode Analysis (src/nma_analysis.py).

These tests guard the eigenvalue -> wavenumber conversion. A units bug in that
conversion (factor ~2e7 too small) previously collapsed every VDOS spectrum to
a single delta at 0 cm^-1, making all proteins indistinguishable through the
spectral branch while every shape/finiteness test still passed. The tests below
exercise ANMAnalyzer on real coordinate sets and assert that:

  1. mode wavenumbers land in the physical band for protein collective motions,
  2. the VDOS peak is NOT pinned to the zero-frequency bin, and
  3. two structurally distinct folds yield measurably different VDOS spectra.

They build ProDy AtomGroups in memory, so no PDB files are needed at test time.
"""

import numpy as np
import pytest

pr = pytest.importorskip("prody")

from src.nma_analysis import ENM_FREQ_CM1_PER_SQRT_EIGVAL, ANMAnalyzer  # noqa: E402


def _make_ca(coords: np.ndarray):
    """Build a minimal C-alpha-only ProDy AtomGroup from coordinates."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    ag = pr.AtomGroup("test")
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnums(np.arange(1, n + 1))
    ag.setResnames(["ALA"] * n)
    ag.setElements(["C"] * n)
    ag.setChids(["A"] * n)
    return ag


def _helix(n: int = 30) -> np.ndarray:
    """Idealized alpha-helix-like C-alpha trace."""
    return np.array([[1.5 * np.cos(i * 1.7), 1.5 * np.sin(i * 1.7), i * 1.5] for i in range(n)])


def _blob(n: int = 30, seed: int = 1) -> np.ndarray:
    """Compact globular C-alpha cloud."""
    return np.random.RandomState(seed).randn(n, 3) * 6.0


def test_conversion_constant_is_physical():
    """The reduced-ENM conversion should be ~108.6 cm^-1 per sqrt(eigval),
    not the ~5e-6 of the old bug."""
    assert 100.0 < ENM_FREQ_CM1_PER_SQRT_EIGVAL < 120.0


def test_frequencies_in_physical_band():
    """Mode wavenumbers should sit in the ~1-500 cm^-1 range of protein
    collective vibrations, not collapse to ~0."""
    anm = ANMAnalyzer(_make_ca(_helix()), cutoff=15.0)
    freqs, _ = anm.compute_modes(k=20)
    assert freqs.size > 0
    assert freqs.max() > 1.0, "all modes collapsed near 0 cm^-1 (conversion bug)"
    assert freqs.max() < 1000.0, "wavenumbers unphysically large"


def test_vdos_peak_not_at_zero_bin():
    """A correctly scaled VDOS peaks at a real frequency, not bin 0."""
    anm = ANMAnalyzer(_make_ca(_helix()), cutoff=15.0)
    vdos = anm.compute_vdos(k=20)
    assert int(np.argmax(vdos)) > 1, "VDOS pinned to the zero-frequency bin"


def test_distinct_structures_give_distinct_vdos():
    """The core regression: two different folds must not produce
    near-identical spectra. This is what the old units bug violated."""
    v_helix = ANMAnalyzer(_make_ca(_helix()), cutoff=15.0).compute_vdos(k=20)
    v_blob = ANMAnalyzer(_make_ca(_blob()), cutoff=15.0).compute_vdos(k=20)
    cos = float(v_helix @ v_blob / (np.linalg.norm(v_helix) * np.linalg.norm(v_blob)))
    assert cos < 0.999, f"distinct folds gave near-identical VDOS (cos={cos:.6f})"


def test_vibrational_entropy_finite_and_positive():
    """Entropy should be a finite, positive quantity in the corrected scale."""
    anm = ANMAnalyzer(_make_ca(_helix()), cutoff=15.0)
    s = anm.compute_vibrational_entropy(k=20)
    assert np.isfinite(s)
    assert s > 0.0
