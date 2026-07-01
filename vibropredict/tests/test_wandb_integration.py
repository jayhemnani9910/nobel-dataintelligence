"""
Tests for W&B integration in TrainerWithMMDrop.

Verifies that:
- Training works unchanged without wandb when log_to_wandb=False (default).
- wandb.init and wandb.log are called when log_to_wandb=True.
- wandb.finish is called at the end of training.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader


def _make_dummy_model():
    """Create a minimal model that matches VibroPredictHybrid's interface."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, sequences, vdos, substrate_smiles, product_smiles, drop_spectral):
            batch_size = vdos.shape[0]
            x = vdos.mean(dim=-1).mean(dim=-1, keepdim=True)
            logkcat = self.linear(torch.cat([x] * 10, dim=-1)).squeeze(-1)
            gates = torch.ones(batch_size, 3) / 3.0
            return logkcat, gates

    return DummyModel()


def _make_dummy_loader(n=20, batch_size=4):
    """Create a DataLoader that mimics the enzyme kinetics batch format."""
    vdos = torch.randn(n, 1, 1000)
    log_kcat = torch.randn(n)

    class DictDataset:
        def __init__(self):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "sequences": ["ACDEFGH"] * 1,
                "vdos": vdos[idx],
                "substrate_smiles": ["CC(=O)O"] * 1,
                "product_smiles": None,
                "log_kcat": log_kcat[idx],
            }

    def collate_fn(batch):
        return {
            "sequences": [b["sequences"][0] for b in batch],
            "vdos": torch.stack([b["vdos"] for b in batch]),
            "substrate_smiles": [b["substrate_smiles"][0] for b in batch],
            "product_smiles": None,
            "log_kcat": torch.stack([b["log_kcat"] for b in batch]),
        }

    return DataLoader(DictDataset(), batch_size=batch_size, collate_fn=collate_fn)


class TestWandBDisabled:
    """Tests that training works with log_to_wandb=False (default)."""

    def test_default_no_wandb(self):
        """TrainerWithMMDrop initializes with log_to_wandb=False by default."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        model = _make_dummy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = TrainerWithMMDrop(model, optimizer, device="cpu")

        assert trainer.log_to_wandb is False
        assert trainer._wandb is None

    def test_training_works_without_wandb(self):
        """Full fit() works without wandb installed."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        model = _make_dummy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        trainer = TrainerWithMMDrop(model, optimizer, device="cpu")
        loader = _make_dummy_loader()
        best_loss = trainer.fit(loader, loader, loss_fn, epochs=1, p_drop=0.0, patience=5)

        assert isinstance(best_loss, float)
        assert best_loss >= 0


class TestWandBEnabled:
    """Tests that wandb integration works correctly when enabled."""

    def test_wandb_init_called(self):
        """wandb.init is called when log_to_wandb=True."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        mock_wandb = MagicMock()
        mock_wandb.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            model = _make_dummy_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = TrainerWithMMDrop(model, optimizer, device="cpu", log_to_wandb=True)

            mock_wandb.init.assert_called_once()
            assert trainer.log_to_wandb is True

    def test_wandb_log_called_per_epoch(self):
        """wandb.log is called once per epoch during fit()."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        mock_wandb = MagicMock()
        mock_wandb.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            model = _make_dummy_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            trainer = TrainerWithMMDrop(model, optimizer, device="cpu", log_to_wandb=True)
            loader = _make_dummy_loader()
            trainer.fit(loader, loader, loss_fn, epochs=3, p_drop=0.0, patience=5)

            # wandb.log should be called once per epoch
            assert mock_wandb.log.call_count == 3

            # Check logged keys contain expected fields
            first_call_args = mock_wandb.log.call_args_list[0]
            log_dict = first_call_args[0][0]
            assert "epoch" in log_dict
            assert "train/loss" in log_dict
            assert "val/loss" in log_dict
            assert "lr" in log_dict

    def test_wandb_finish_called(self):
        """wandb.finish is called when fit() completes."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        mock_wandb = MagicMock()
        mock_wandb.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            model = _make_dummy_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            trainer = TrainerWithMMDrop(model, optimizer, device="cpu", log_to_wandb=True)
            loader = _make_dummy_loader()
            trainer.fit(loader, loader, loss_fn, epochs=1, p_drop=0.0, patience=5)

            mock_wandb.finish.assert_called_once()

    def test_gate_stats_logged(self):
        """Gate statistics are included in wandb.log calls."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        mock_wandb = MagicMock()
        mock_wandb.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            model = _make_dummy_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            trainer = TrainerWithMMDrop(model, optimizer, device="cpu", log_to_wandb=True)
            loader = _make_dummy_loader()
            trainer.fit(loader, loader, loss_fn, epochs=1, p_drop=0.0, patience=5)

            log_dict = mock_wandb.log.call_args_list[0][0][0]
            # Gate stats should be present
            assert "gates/gate_seq_mean" in log_dict
            assert "gates/gate_spec_mean" in log_dict
            assert "gates/gate_chem_mean" in log_dict
