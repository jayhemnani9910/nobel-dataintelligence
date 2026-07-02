"""
Tests for inference VDOS routing and checkpoint resolution.

Guards two audit fixes:
  1. _generate_vdos_for_sequence uses REAL NMA VDOS when a structure is given
     (structure-specific), and only falls back to the length-only pseudo-VDOS
     when no structure is available.
  2. _resolve_checkpoint raises a helpful error naming the bundled toy stubs
     instead of silently trying to load an incompatible checkpoint.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody")
pytest.importorskip("torch")

from src import inference  # noqa: E402

_PDB_DIR = Path(__file__).resolve().parent.parent / "data" / "pdb"


def test_fallback_vdos_is_length_only():
    """Without a structure, VDOS depends only on sequence length (a warning
    path) — two equal-length sequences give identical pseudo-VDOS."""
    v1 = inference._generate_vdos_for_sequence("A" * 80)
    v2 = inference._generate_vdos_for_sequence("W" * 80)
    assert np.allclose(v1, v2)


@pytest.mark.skipif(
    not (_PDB_DIR / "6eqe.pdb").exists(), reason="bundled PDB not present"
)
def test_real_vdos_is_structure_specific():
    """With structures, two different proteins give distinct real VDOS."""
    v_a = inference._generate_vdos_for_sequence("X" * 265, pdb_path=str(_PDB_DIR / "6eqe.pdb"))
    v_b = inference._generate_vdos_for_sequence("X" * 258, pdb_path=str(_PDB_DIR / "6ths.pdb"))
    cos = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b)))
    assert cos < 0.999, f"structure VDOS not distinct (cos={cos:.6f})"
    assert int(v_a.argmax()) > 1  # not collapsed to bin 0


def test_missing_checkpoint_error_is_honest(tmp_path):
    """The checkpoint resolver names the toy stubs and lists what's present."""
    ckdir = tmp_path / "checkpoints"
    ckdir.mkdir()
    (ckdir / "best_model_epoch1.pt").write_bytes(b"stub")
    with pytest.raises(FileNotFoundError) as exc:
        inference._resolve_checkpoint(ckdir / "novozymes_best.pt")
    msg = str(exc.value)
    assert "best_model_epoch1.pt" in msg
    assert "stub" in msg.lower() or "smoke" in msg.lower()


_REAL_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model_epoch10.pt"


@pytest.mark.skipif(
    not (_REAL_CKPT.exists() and (_PDB_DIR / "6eqe.pdb").exists()),
    reason="real checkpoint / bundled PDB not present",
)
def test_predict_stability_loads_real_checkpoint_and_runs():
    """Regression for the audit CRITICAL + coords/features fixes:

    1. num_go_terms is inferred from the checkpoint (the real epoch10 weights
       use 10000; hardcoding 100 caused a cafa_head size-mismatch RuntimeError).
    2. When the sequence is shorter than the structure's CA count, coords AND
       features are clipped to a common length (previously only features were
       'truncated' to the larger value, leaving edge indices out of range and
       raising IndexError inside GATv2Conv).

    Only runs when torch_geometric (GNN backend) is importable.
    """
    pytest.importorskip("torch_geometric")
    # Intentionally use a short sequence to exercise the mismatch-alignment path.
    result = inference.predict_stability(
        sequence="MKTIIALSYIFCLVFA",
        pH=7.0,
        checkpoint_path=str(_REAL_CKPT),
        pdb_path=str(_PDB_DIR / "6eqe.pdb"),
    )
    assert "predicted_tm" in result
    assert isinstance(result["predicted_tm"], float)
    assert result["sequence_length"] == 16
