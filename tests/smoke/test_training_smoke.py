"""
Training Smoke Test

Runs TrainerWithMMDrop.fit for 1 epoch on 100 rows of synthetic data
to catch training-loop regressions that unit tests miss.

This test:
- Does NOT download real data or hit any external API.
- Runs on CPU in under 5 minutes.
- Verifies loss decreases and a checkpoint is written.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@pytest.mark.timeout(300)
class TestTrainingSmokeTest:
    """Smoke test for the VibroPredict training loop."""

    def test_loss_decreases_after_one_epoch(self, synthetic_loader, dummy_model, tmp_path):
        """Loss at epoch 1 should be strictly lower than initial loss."""
        import torch

        from vibropredict.training.trainer import TrainerWithMMDrop

        train_loader, val_loader = synthetic_loader
        model = dummy_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        checkpoint_dir = str(tmp_path / "checkpoints")
        trainer = TrainerWithMMDrop(
            model,
            optimizer,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        # Compute initial loss before any training
        initial_metrics = trainer.validate(val_loader, loss_fn)
        initial_loss = initial_metrics["val_loss"]

        # Train for 2 epochs (so we can compare epoch 0 vs epoch 1)
        best_loss = trainer.fit(
            train_loader,
            val_loader,
            loss_fn,
            epochs=2,
            p_drop=0.0,
            patience=10,
        )

        # Loss should decrease after training
        assert best_loss < initial_loss, (
            f"Training did not reduce loss: initial={initial_loss:.4f}, best={best_loss:.4f}"
        )

    def test_checkpoint_written(self, synthetic_loader, dummy_model, tmp_path):
        """A checkpoint file should be written during training."""
        import torch

        from vibropredict.training.trainer import TrainerWithMMDrop

        train_loader, val_loader = synthetic_loader
        model = dummy_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        checkpoint_dir = tmp_path / "checkpoints"
        trainer = TrainerWithMMDrop(
            model,
            optimizer,
            device="cpu",
            checkpoint_dir=str(checkpoint_dir),
        )

        trainer.fit(
            train_loader,
            val_loader,
            loss_fn,
            epochs=1,
            p_drop=0.0,
            patience=10,
        )

        # At least one checkpoint should exist
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0, "No checkpoint file was written"

    def test_mm_drop_does_not_crash(self, synthetic_loader, dummy_model, tmp_path):
        """Training with MM-Drop (p_drop > 0) should complete without errors."""
        import torch

        from vibropredict.training.trainer import TrainerWithMMDrop

        train_loader, val_loader = synthetic_loader
        model = dummy_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        trainer = TrainerWithMMDrop(
            model,
            optimizer,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # This should not raise
        best_loss = trainer.fit(
            train_loader,
            val_loader,
            loss_fn,
            epochs=1,
            p_drop=0.25,  # MM-Drop enabled
            patience=10,
        )

        assert isinstance(best_loss, float)
        assert best_loss >= 0

    def test_gate_stats_in_train_metrics(self, synthetic_loader, dummy_model, tmp_path):
        """train_epoch should return gate statistics."""
        import torch

        from vibropredict.training.trainer import TrainerWithMMDrop

        train_loader, _ = synthetic_loader
        model = dummy_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        trainer = TrainerWithMMDrop(
            model,
            optimizer,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        metrics = trainer.train_epoch(train_loader, loss_fn, p_drop=0.0)

        assert "train_loss" in metrics
        assert "gate_stats" in metrics
        gate_stats = metrics["gate_stats"]
        assert "gate_seq_mean" in gate_stats
        assert "gate_spec_mean" in gate_stats
        assert "gate_chem_mean" in gate_stats
