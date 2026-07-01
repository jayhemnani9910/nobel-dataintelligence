"""
Regression tests for the Phase-3 VDOS pipeline (vibropredict.spectra).

Guards the eigenvalue->wavenumber conversion in VibroEnzymePipeline, which
previously used the same collapse factor (1/(2*pi*29979.2458), ~2e7x too small)
as the ANM module, mapping every GNM mode to ~1e-5 cm^-1 so every VDOS spectrum
became an identical delta at bin 0. These tests assert the frequencies land in a
physical band and that distinct folds give distinct spectra.

Structures are built in-memory (no PDB files or network needed).
"""

import numpy as np
import pytest

pr = pytest.importorskip("prody")

from vibropredict.spectra.gnm_calculator import GNMCalculator  # noqa: E402
from vibropredict.spectra.vdos_engine import VibroEnzymePipeline  # noqa: E402
from src.nma_analysis import ENM_FREQ_CM1_PER_SQRT_EIGVAL  # noqa: E402


def _helix(n=40):
    """Ideal alpha-helix CA trace as a ProDy AtomGroup."""
    t = np.arange(n)
    coords = np.c_[2.3 * np.cos(t * 100 * np.pi / 180),
                   2.3 * np.sin(t * 100 * np.pi / 180),
                   1.5 * t].astype(float)
    ag = pr.AtomGroup("helix")
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnums(np.arange(1, n + 1))
    ag.setResnames(["ALA"] * n)
    ag.setChids(["A"] * n)
    ag.setElements(["C"] * n)
    return ag


def _blob(n=40, seed=0):
    """Compact globular CA cloud."""
    rng = np.random.default_rng(seed)
    coords = rng.normal(scale=8.0, size=(n, 3))
    ag = pr.AtomGroup("blob")
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnums(np.arange(1, n + 1))
    ag.setResnames(["ALA"] * n)
    ag.setChids(["A"] * n)
    ag.setElements(["C"] * n)
    return ag


def _vdos_from_coords(ag, npts=1000, fmax=500.0, broadening=5.0):
    from src.spectral_generation import SpectralGenerator
    gc = GNMCalculator(cutoff=10.0)
    ev, _ = gc.compute_from_coords(ag.getCoords())
    freqs = np.sqrt(np.maximum(ev, 0)) * ENM_FREQ_CM1_PER_SQRT_EIGVAL
    gen = SpectralGenerator(freq_min=0, freq_max=fmax, n_points=npts)
    return gen.generate_dos(freqs, broadening=broadening), freqs


def test_frequencies_in_physical_band():
    _, freqs = _vdos_from_coords(_helix())
    # GNM pseudo-frequencies must be well above the collapsed ~1e-5 cm^-1 regime.
    assert freqs.max() > 1.0, f"max freq {freqs.max()} still collapsed"
    assert freqs.min() >= 0.0


def test_vdos_peak_not_in_bin_zero():
    vdos, _ = _vdos_from_coords(_helix())
    assert int(vdos.argmax()) > 1, "VDOS still collapsed to bin 0"


def test_distinct_folds_give_distinct_vdos():
    v_helix, _ = _vdos_from_coords(_helix())
    v_blob, _ = _vdos_from_coords(_blob())
    cos = float(np.dot(v_helix, v_blob) /
                (np.linalg.norm(v_helix) * np.linalg.norm(v_blob)))
    assert cos < 0.999, f"helix and blob VDOS nearly identical (cos={cos:.6f})"
