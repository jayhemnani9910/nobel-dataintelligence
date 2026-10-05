"""Regression tests for fixes from the 2026-10-06 audit (numbers refer to it)."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data_acquisition import KaggleDataAcquisition
from src.datasets import NovozymesDataset, ProteinStructureDataset
from src.models.gnn import ProteinGNN
from src.models.losses import CombinedLoss, MarginRankingLossCustom
from src.spectral_generation import SpectralGenerator
from src.training import Trainer
from src.utils import Logger

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_ca(n: int, name: str = "CA"):
    pr = pytest.importorskip("prody")
    coords = np.array(
        [[1.5 * np.cos(i * 1.7), 1.5 * np.sin(i * 1.7), i * 1.5] for i in range(n)]
    )
    ag = pr.AtomGroup("test")
    ag.setCoords(coords)
    ag.setNames([name] * n)
    ag.setResnums(np.arange(1, n + 1))
    ag.setResnames(["ALA"] * n)
    ag.setElements(["C"] * n)
    ag.setChids(["A"] * n)
    return ag


def _write_pdb(path: Path, n: int) -> Path:
    pr = pytest.importorskip("prody")
    pr.writePDB(str(path), _make_ca(n))
    return path


def _write_novozymes_csvs(tmp_path: Path) -> Path:
    csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "seq_id": [0, 1, 2],
            "protein_sequence": ["ACD", "EFG", "HIK"],
            "pH": [7.0, 7.0, 7.0],
            "data_source": ["a", "b", "c"],
            "tm": [50.0, 55.0, 60.0],
        }
    ).to_csv(csv, index=False)
    pd.DataFrame(
        {
            "seq_id": [1, 2],
            "protein_sequence": [None, "HIK"],
            "pH": [None, 5.0],
            "data_source": [None, "c"],
            "tm": [None, 70.0],
        }
    ).to_csv(tmp_path / "train_updates.csv", index=False)
    return csv


# --- crashes -----------------------------------------------------------------


def test_petase_script_runs_from_outside_repo(tmp_path):
    """#1 np.trapezoid, #3 sys.path, #16 freq keys."""
    pytest.importorskip("prody")
    (tmp_path / "data" / "pdb").mkdir(parents=True)
    _write_pdb(tmp_path / "data" / "pdb" / "6eqe.pdb", 30)
    _write_pdb(tmp_path / "data" / "pdb" / "6ths.pdb", 35)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "real_world_petase.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    summary = json.loads(
        (tmp_path / "benchmarks/real_world/petase_vs_lcc_summary.json").read_text()
    )
    assert "min_freq_cm1" in summary["IsPETase_wildtype"]


def test_structure_dataset_getitem_and_upper_case_metadata(tmp_path):
    """#2 ca.getSequence(), #75 metadata pdb_id case."""
    pdb = _write_pdb(tmp_path / "prot.pdb", 12)
    meta = pd.DataFrame({"pdb_id": ["PROT"], "label": [1.5]})
    sample = ProteinStructureDataset([str(pdb)], str(tmp_path), meta)[0]
    assert sample["graph"].x.shape[0] == 12 and float(sample["labels"]) == 1.5


def test_gnn_default_input_dim_matches_features():
    """#7"""
    assert ProteinGNN().input_proj.in_features == 24


def test_margin_ranking_loss_batch_of_one():
    """#8"""
    loss = MarginRankingLossCustom()(torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1))
    assert torch.isfinite(loss)


def test_gnm_without_ca_raises_value_error():
    """#9"""
    from src.nma_analysis import GNMAnalyzer

    with pytest.raises(ValueError):
        GNMAnalyzer(_make_ca(5, name="CB"))


def test_compare_structures_different_mode_counts(tmp_path):
    """#10"""
    from src.nma_analysis import compare_structures

    small = _write_pdb(tmp_path / "small.pdb", 8)
    big = _write_pdb(tmp_path / "big.pdb", 30)
    assert np.isfinite(compare_structures(str(small), str(big))["frequency_shift_cm1"])


def test_instrumental_response_on_zero_spectrum():
    """#11"""
    out = SpectralGenerator().apply_instrumental_response(np.zeros(1000))
    assert np.all(np.isfinite(out))


# --- wrong results -----------------------------------------------------------


def test_novozymes_dataset_applies_updates(tmp_path):
    """#22"""
    csv = _write_novozymes_csvs(tmp_path)
    ds = NovozymesDataset(str(csv), str(tmp_path / "missing.pdb"), str(tmp_path))
    assert ds.df[["seq_id", "pH"]].values.tolist() == [[0, 7.0], [2, 5.0]]


def test_load_novozymes_data_applies_updates(tmp_path):
    """#21"""
    _write_novozymes_csvs(tmp_path)
    df = KaggleDataAcquisition(str(tmp_path)).load_novozymes_data()
    assert df[["seq_id", "tm"]].values.tolist() == [[0, 50.0], [2, 70.0]]


def test_combined_loss_honours_initial_weights():
    """#27"""
    loss = CombinedLoss({"a": nn.MSELoss()}, initial_weights={"a": 2.0})
    assert torch.exp(loss.log_weights["a"]).item() == pytest.approx(2.0)


def test_spectral_std_dev_is_around_centroid():
    """#29"""
    gen = SpectralGenerator()
    spectrum = np.zeros(gen.n_points)
    spectrum[10] = 1.0
    assert gen.extract_spectral_features(spectrum)["std_dev"] == pytest.approx(0.0)


def _scripted_trainer(tmp_path, val_losses, scheduler_fn=None):
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None
    trainer = Trainer(model, optimizer, scheduler, "cpu", str(tmp_path))
    epoch = {"n": 0}

    def train_epoch(*args, **kwargs):
        epoch["n"] += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch["n"]))
        return {"train_loss": 0.0}

    def validate(*args, **kwargs):
        return {"val_loss": val_losses[epoch["n"] - 1]}

    trainer.train_epoch = train_epoch
    trainer.validate = validate
    return trainer


def test_fit_restores_best_weights(tmp_path):
    """#20 / #30"""
    trainer = _scripted_trainer(tmp_path, [1.0, 2.0, 3.0])
    trainer.fit(None, None, None, epochs=3, early_stopping_patience=10)
    assert trainer.model.weight.item() == 1.0


def test_fit_steps_epoch_schedulers_without_val_loss(tmp_path):
    """#80"""
    trainer = _scripted_trainer(
        tmp_path,
        [4.0, 3.0],
        lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5),
    )
    trainer.fit(None, None, None, epochs=2, early_stopping_patience=10)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


def test_logger_setup_twice_adds_one_handler():
    """#120"""
    Logger.setup("audit_fix_test")
    logger = Logger.setup("audit_fix_test", level=logging.DEBUG)
    assert len(logger.handlers) == 1
