"""Regression tests for fixes from the 2026-10-06 audit.

Each test names the finding it covers. No pretrained model is downloaded:
the transformers module is replaced by tiny fakes.
"""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fakes for the pretrained encoders
# ---------------------------------------------------------------------------


class _FakeTokens(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """One token per character (spaces ignored), padded with id 0."""

    def __call__(self, texts, **kwargs):
        ids = [[ord(c) % 50 + 1 for c in t.replace(" ", "")] for t in texts]
        n = max(len(i) for i in ids)
        return _FakeTokens(
            input_ids=torch.tensor([i + [0] * (n - len(i)) for i in ids]),
            attention_mask=torch.tensor(
                [[1] * len(i) + [0] * (n - len(i)) for i in ids]
            ),
        )


class _FakeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Embedding(64, dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


@pytest.fixture
def fake_transformers(monkeypatch):
    mod = types.ModuleType("transformers")
    tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTokenizer())
    mod.T5Tokenizer = tokenizer
    mod.AutoTokenizer = tokenizer
    mod.T5EncoderModel = SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeEncoder(8)
    )
    mod.AutoModel = SimpleNamespace(from_pretrained=lambda *a, **k: _FakeEncoder(768))
    monkeypatch.setitem(sys.modules, "transformers", mod)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Models (#6, #68, #35)
# ---------------------------------------------------------------------------


def test_prott5_state_dict_stable_after_forward(fake_transformers):
    """#6: the lazily loaded encoder must not appear in state_dict."""
    from vibropredict.models.sequence_encoder import ProtT5Encoder

    enc = ProtT5Encoder(output_dim=8)
    before = set(enc.state_dict())
    enc(["MKT"])
    assert set(enc.state_dict()) == before


def test_chemical_encoder_state_dict_stable_after_forward(fake_transformers):
    """#6: same for ChemBERTa in ChemicalEncoder."""
    pytest.importorskip("rdkit")
    from vibropredict.models.chemical_encoder import ChemicalEncoder

    enc = ChemicalEncoder(fp_dim=16, output_dim=4)
    before = set(enc.state_dict())
    enc(["CC"])
    assert set(enc.state_dict()) == before


def test_prott5_train_keeps_frozen_encoder_in_eval(fake_transformers):
    """#68: parent.train() must not re-enable dropout in the frozen encoder."""
    from vibropredict.models.sequence_encoder import ProtT5Encoder

    enc = ProtT5Encoder(output_dim=8)
    enc(["MKT"])
    enc.train()
    assert not enc._encoder.training


def test_prott5_pooling_ignores_padding(fake_transformers):
    """#35: a protein's embedding must not depend on the longest one in its batch."""
    from vibropredict.models.sequence_encoder import ProtT5Encoder

    torch.manual_seed(0)
    enc = ProtT5Encoder(output_dim=8)
    alone = enc(["MK"])
    batched = enc(["MK", "MKTLLAV"])
    assert torch.allclose(alone[0], batched[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Trainer (#38, #80)
# ---------------------------------------------------------------------------


def _scripted_trainer(tmp_path, val_losses, scheduler_factory=None):
    """Trainer whose epochs add 1 to the weight and report fixed val losses."""
    from vibropredict.training.trainer import TrainerWithMMDrop

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    trainer = TrainerWithMMDrop(
        model, optimizer, checkpoint_dir=str(tmp_path), scheduler=scheduler
    )
    losses = iter(val_losses)

    def train_epoch(*args, **kwargs):
        optimizer.step()  # no grads, so a no-op; keeps the scheduler order valid
        with torch.no_grad():
            model.weight.add_(1.0)
        return {"train_loss": 0.0, "gate_stats": {}}

    trainer.train_epoch = train_epoch
    trainer.validate = lambda *args, **kwargs: {"val_loss": next(losses)}
    return trainer, model, optimizer


def test_fit_restores_best_weights(tmp_path):
    """#38: after fit the model holds the best-val epoch's weights."""
    trainer, model, _ = _scripted_trainer(tmp_path, [1.0, 2.0, 3.0])
    trainer.fit(None, None, loss_fn=None, epochs=3, patience=10)
    assert model.weight.item() == pytest.approx(1.0)


def test_fit_steps_non_plateau_scheduler_without_val_loss(tmp_path):
    """#80: StepLR must be stepped per epoch, not with val_loss as epoch."""
    trainer, _, optimizer = _scripted_trainer(
        tmp_path,
        [5.0, 5.0],
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        ),
    )
    trainer.fit(None, None, loss_fn=None, epochs=2, patience=10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Metrics and loss (#90, #113, #131, #89)
# ---------------------------------------------------------------------------


def test_rmse_flattens_column_vector():
    """#90: (N,1) predictions vs (N,) targets must not broadcast to (N,N)."""
    from vibropredict.training.metrics import rmse

    assert rmse(np.array([[1.0], [2.0]]), np.array([1.0, 2.0])) == 0.0


def test_r_squared_constant_targets_is_nan():
    """#131: R^2 is undefined for constant targets."""
    from vibropredict.training.metrics import r_squared

    assert np.isnan(r_squared(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])))


def test_pearson_constant_predictions_is_nan():
    """#113: a collapsed (constant) model gives NaN, explicitly."""
    from vibropredict.training.metrics import pearson_correlation

    assert np.isnan(
        pearson_correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    )


def test_pearson_single_sample_is_nan():
    """#113: fewer than 2 samples returns NaN instead of raising."""
    from vibropredict.training.metrics import pearson_correlation

    assert np.isnan(pearson_correlation(np.array([1.0]), np.array([2.0])))


def test_ranking_loss_ignores_tied_pairs():
    """#89: tied pairs add no constant margin penalty."""
    from vibropredict.training.losses import MutantRankingLoss

    preds = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.5, 1.5])
    loss = MutantRankingLoss(lambda_rank=1.0)(preds, targets, torch.tensor([[0, 1]]))
    assert loss.item() == pytest.approx(
        torch.nn.functional.mse_loss(preds, targets).item()
    )


# ---------------------------------------------------------------------------
# Data (#32, #33, #13, #124, #111, #82)
# ---------------------------------------------------------------------------


def test_kinhub_validate_drops_non_positive_kcat():
    """#32/#64: k_cat <= 0 rows are dropped, not clipped to 1e-30."""
    from vibropredict.data.kinhub import KinHubLoader

    df = pd.DataFrame(
        {
            "uniprot_id": ["A", "A"],
            "k_cat": [0.0, 100.0],
            "substrate_smiles": ["CC", "CC"],
        }
    )
    assert list(KinHubLoader("unused").validate(df)["k_cat"]) == [100.0]


def test_kinhub_validate_drops_missing_smiles():
    """#33: rows with missing substrate_smiles are dropped."""
    from vibropredict.data.kinhub import KinHubLoader

    df = pd.DataFrame(
        {
            "uniprot_id": ["A", "A"],
            "k_cat": [1.0, 4.0],
            "substrate_smiles": [None, "CC"],
        }
    )
    assert len(KinHubLoader("unused").validate(df)) == 1


def test_ec_holdout_needs_three_classes():
    """#13: too few EC classes raises a clear ValueError."""
    from vibropredict.data.splits import ECHoldoutSplit

    df = pd.DataFrame({"ec_class": ["1.1", "1.1", "2.1", "2.1"]})
    with pytest.raises(ValueError, match="at least 3 EC classes"):
        ECHoldoutSplit().split(df)


def test_split_ratios_summing_to_one_rejected():
    """#124: train + val == 1.0 would leave test empty."""
    from vibropredict.data.splits import RandomSplit

    with pytest.raises(ValueError):
        RandomSplit(train_ratio=0.9, val_ratio=0.1)


def test_missing_ec_number_is_unknown():
    """#111: NaN EC numbers map to 'unknown', not 'nan'."""
    from vibropredict.data.splits import ECHoldoutSplit

    assert ECHoldoutSplit()._extract_ec_class(float("nan")) == "unknown"


def test_dataset_missing_sequence_is_empty_string(tmp_path):
    """#82: a NaN sequence must not become the string 'nan'."""
    from vibropredict.data.enzyme_kinetics_dataset import EnzymeKineticsDataset

    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "uniprot_id": ["P1"],
            "sequence": [None],
            "log_kcat": [1.0],
            "substrate_smiles": ["CC"],
        }
    ).to_csv(csv, index=False)
    assert EnzymeKineticsDataset(str(csv), str(tmp_path))[0]["sequence"] == ""


# ---------------------------------------------------------------------------
# Spectra and structures (#14, #87, #88, #130)
# ---------------------------------------------------------------------------


def test_gnm_single_atom_raises_value_error():
    """#14: a 1-atom structure gives a clear ValueError."""
    pytest.importorskip("prody")
    from vibropredict.spectra.gnm_calculator import GNMCalculator

    with pytest.raises(ValueError, match="at least 2"):
        GNMCalculator().compute_from_coords(np.zeros((1, 3)))


def _pdb_with_bfactors(bfactors):
    return "\n".join(
        f"ATOM  {i + 1:5d} {'CA':^4s} ALA A{i + 1:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{b:6.2f}           C"
        for i, b in enumerate(bfactors)
    )


def test_parse_plddt_rescales_unit_scale(tmp_path):
    """#88: 0-1 pLDDT (HF ESMFold) is rescaled to 0-100."""
    from vibropredict.structures.quality_control import parse_plddt_from_pdb

    pdb = tmp_path / "s.pdb"
    pdb.write_text(_pdb_with_bfactors([0.9, 0.8]))
    np.testing.assert_allclose(parse_plddt_from_pdb(str(pdb)), [90.0, 80.0])


def test_validate_plddt_accepts_unit_scale():
    """#87: a confident 0-1 scale structure passes the 70 threshold."""
    from vibropredict.structures.esmfold_runner import ESMFoldPredictor

    assert ESMFoldPredictor().validate_plddt(_pdb_with_bfactors([0.9, 0.8]))


def test_sifts_rejects_path_like_uniprot_id(tmp_path, monkeypatch):
    """#130: an ID with '/' or '..' is rejected before any request or cache write."""
    from vibropredict.structures import sifts_mapper

    def _no_request(*args, **kwargs):
        raise AssertionError("request must not be made")

    monkeypatch.setattr(sifts_mapper.requests, "get", _no_request)
    mapper = sifts_mapper.SIFTSMapper(cache_dir=str(tmp_path / "cache"))
    assert mapper._fetch_mapping("../evil") == {}


# ---------------------------------------------------------------------------
# SOTA comparison (#34, #84)
# ---------------------------------------------------------------------------


def test_live_comparison_lists_vibropredict_once():
    """#34: VibroPredict appears exactly once in the comparison table."""
    from vibropredict.evaluation.baselines import list_baselines
    from vibropredict.evaluation.sota_comparison import run_live_comparison

    df = run_live_comparison(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.5]),
        ["MK"] * 3,
        ["CC"] * 3,
        skip_baselines=list_baselines(),
    )
    assert (df["model"] == "VibroPredict").sum() == 1


def test_failed_baseline_marked_failed():
    """#84: a crashed baseline is not labelled 'live'."""
    from vibropredict.evaluation.sota_comparison import compare_with_baselines

    df = compare_with_baselines({}, {"X": {"rmse": float("nan"), "error": "boom"}})
    assert df.loc[df["model"] == "X", "source"].item() == "failed"


# ---------------------------------------------------------------------------
# scripts/run_benchmarks.py (#4, #17, #40, #41, #103)
# ---------------------------------------------------------------------------


def test_run_benchmarks_runs_as_plain_script(tmp_path):
    """#4: `python scripts/run_benchmarks.py` finds the vibropredict package."""
    kinhub = tmp_path / "kinhub.csv"
    pd.DataFrame(
        {
            "uniprot_id": [f"P{i}" for i in range(12)],
            "sequence": ["MKT"] * 12,
            "substrate_smiles": ["CC"] * 12,
            "log_kcat": np.linspace(0, 2, 12),
            "ec_number": [f"{i % 6 + 1}.1.1.1" for i in range(12)],
        }
    ).to_csv(kinhub, index=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_benchmarks.py"),
            "--kinhub",
            str(kinhub),
            "--skip-vibropredict",
            "--skip-baselines",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_benchmarks_metrics_include_mae():
    """#17: the MAE column has a value to show."""
    mod = _load_script("run_benchmarks")
    metrics = mod._compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.5]))
    assert metrics["mae"] == pytest.approx(0.5)


def test_run_benchmarks_failed_run_rendered_as_failed():
    """#41: a crashed run shows 'failed: <error>', not 'pending'."""
    mod = _load_script("run_benchmarks")
    lines = mod._split_section("t", None, "random", {"random": {"error": "boom"}}, {})
    assert any("failed: boom" in line for line in lines)


def test_run_benchmarks_passes_product_smiles(tmp_path, monkeypatch):
    """#40: inference passes product_smiles like the training dataset does."""
    from vibropredict.models import vibropredict_hybrid

    seen = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            pass

        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, sequences, vdos, substrate_smiles, product_smiles=None):
            seen["product_smiles"] = product_smiles
            return torch.zeros(1), None

    monkeypatch.setattr(vibropredict_hybrid, "VibroPredictHybrid", _FakeModel)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1"],
            "sequence": ["MK"],
            "log_kcat": [1.0],
            "substrate_smiles": ["CC"],
            "product_smiles": ["CCO"],
        }
    )
    _load_script("run_benchmarks")._run_vibropredict(str(ckpt), df, str(tmp_path))
    assert seen["product_smiles"] == ["CCO"]


def test_dry_run_does_not_overwrite_real_report(tmp_path, monkeypatch):
    """#103: --dry-run writes to a separate file name."""
    mod = _load_script("run_benchmarks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mod,
        "_get_env_dump",
        lambda: {
            "git_sha": "x",
            "timestamp": "t",
            "python_version": "3",
            "platform": "p",
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["run_benchmarks.py", "--dry-run", "--output-dir", str(tmp_path)]
    )
    mod.main()
    assert sorted(p.name for p in tmp_path.glob("benchmarks*.json")) == [
        "benchmarks_dry_run.json"
    ]


# ---------------------------------------------------------------------------
# scripts/audit_kinhub_vs_realkcat.py (#15, #71)
# ---------------------------------------------------------------------------


def _write_overlap_inputs(tmp_path):
    kinhub = tmp_path / "kinhub.csv"
    realkcat = tmp_path / "realkcat.csv"
    pd.DataFrame(
        {
            "uniprot_id": ["A", "A", "B"],
            "k_cat": [1.0, 2.0, 3.0],
            "substrate_smiles": ["CC", "CC", None],
        }
    ).to_csv(kinhub, index=False)
    pd.DataFrame(
        {
            "uniprot_id": ["A", "A", "B"],
            "kcat": [1.0, 2.0, 3.0],
            "substrate_smiles": ["CC", "CC", None],
        }
    ).to_csv(realkcat, index=False)
    return kinhub, realkcat


def test_audit_overlap_not_inflated_by_duplicate_keys(tmp_path):
    """#15: duplicate keys do not push overlap past the KinHub row count."""
    kinhub, realkcat = _write_overlap_inputs(tmp_path)
    results = _load_script("audit_kinhub_vs_realkcat").run_audit(
        str(kinhub), str(tmp_path / "out.csv"), realkcat_path=str(realkcat)
    )
    assert results["overlap_percentage"] <= 100.0


def test_audit_missing_smiles_never_overlaps(tmp_path):
    """#71: NaN SMILES rows do not match each other as the string 'nan'."""
    kinhub, realkcat = _write_overlap_inputs(tmp_path)
    results = _load_script("audit_kinhub_vs_realkcat").run_audit(
        str(kinhub), str(tmp_path / "out.csv"), realkcat_path=str(realkcat)
    )
    assert results["overlap_count"] == 2


def test_hybrid_loads_checkpoint_that_still_has_frozen_encoder_weights():
    from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

    model = VibroPredictHybrid(fusion_dim=64, dropout=0.0)
    state = dict(model.state_dict())
    state["seq_encoder._encoder.shared.weight"] = torch.zeros(2, 2)
    state["chem_encoder._smiles_encoder.embeddings.weight"] = torch.zeros(2, 2)

    model.load_state_dict(state)  # strict; old encoder keys are ignored
