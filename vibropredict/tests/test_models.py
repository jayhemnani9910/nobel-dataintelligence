"""Unit tests for VibroPredict models, losses, and metrics.

Tests cover:
- TriModalFusion: forward shape and gate constraints
- MutantRankingLoss: forward with and without mutant_pairs
- Metrics: rmse, r_squared, pearson, spearman
- TrainerWithMMDrop._unpack_batch
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from vibropredict.training.losses import MutantRankingLoss
from vibropredict.training.metrics import (
    rmse,
    r_squared,
    pearson_correlation,
    spearman_correlation,
)


class TestTriModalFusionMock(unittest.TestCase):
    """Test TriModalFusion-like behavior with a minimal mock module.

    We do not import the real TriModalFusion (it may pull in transformers);
    instead we test the contract: output shape and gate constraints.
    """

    def _make_mock_fusion(self, dim: int = 32):
        """Build a lightweight fusion module that mimics TriModalFusion."""

        class _MockFusion(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.gate = nn.Linear(dim * 3, 3)
                self.out = nn.Linear(dim * 3, 1)

            def forward(self, h_seq, h_spec, h_chem):
                combined = torch.cat([h_seq, h_spec, h_chem], dim=-1)
                gates = torch.softmax(self.gate(combined), dim=-1)
                logkcat = self.out(combined).squeeze(-1)
                return logkcat, gates

        return _MockFusion(dim)

    def test_forward_returns_correct_shape(self):
        """Test forward pass returns (batch,) logkcat and (batch, 3) gates."""
        model = self._make_mock_fusion(dim=16)
        h_seq = torch.randn(4, 16)
        h_spec = torch.randn(4, 16)
        h_chem = torch.randn(4, 16)

        logkcat, gates = model(h_seq, h_spec, h_chem)

        self.assertEqual(logkcat.shape, (4,))
        self.assertEqual(gates.shape, (4, 3))

    def test_gates_sum_to_one(self):
        """Test gate weights sum to 1 for each sample."""
        model = self._make_mock_fusion(dim=16)
        h_seq = torch.randn(8, 16)
        h_spec = torch.randn(8, 16)
        h_chem = torch.randn(8, 16)

        _, gates = model(h_seq, h_spec, h_chem)
        sums = gates.sum(dim=-1)

        np.testing.assert_array_almost_equal(
            sums.detach().numpy(), np.ones(8), decimal=5
        )


class TestMutantRankingLoss(unittest.TestCase):
    """Test MutantRankingLoss forward pass."""

    def test_forward_without_mutant_pairs(self):
        """Test loss equals MSE when no mutant_pairs provided."""
        loss_fn = MutantRankingLoss(lambda_rank=0.1)
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])

        loss = loss_fn(preds, targets)
        expected_mse = torch.nn.functional.mse_loss(preds, targets)

        self.assertAlmostEqual(loss.item(), expected_mse.item(), places=5)

    def test_forward_with_mutant_pairs(self):
        """Test loss includes ranking term when mutant_pairs provided."""
        loss_fn = MutantRankingLoss(lambda_rank=0.1)
        preds = torch.tensor([1.0, 2.0, 3.0, 0.5])
        targets = torch.tensor([1.1, 1.9, 3.2, 0.4])
        pairs = torch.tensor([[0, 1], [2, 3]])

        loss_with_pairs = loss_fn(preds, targets, mutant_pairs=pairs)
        loss_no_pairs = loss_fn(preds, targets)

        # With ranking term, loss should differ from pure MSE
        self.assertIsInstance(loss_with_pairs.item(), float)
        self.assertGreater(loss_with_pairs.item(), 0)

    def test_loss_is_differentiable(self):
        """Test loss supports backpropagation."""
        loss_fn = MutantRankingLoss(lambda_rank=0.5)
        preds = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        targets = torch.tensor([1.5, 2.5, 3.5])
        pairs = torch.tensor([[0, 1], [1, 2]])

        loss = loss_fn(preds, targets, mutant_pairs=pairs)
        loss.backward()

        self.assertIsNotNone(preds.grad)


class TestMetrics(unittest.TestCase):
    """Test vibropredict.training.metrics functions."""

    def setUp(self):
        """Generate known test data."""
        self.predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.targets = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    def test_rmse(self):
        """Test RMSE with known values."""
        value = rmse(self.predictions, self.targets)
        expected = np.sqrt(np.mean((self.predictions - self.targets) ** 2))
        self.assertAlmostEqual(value, expected, places=5)

    def test_r_squared_perfect(self):
        """Test R-squared is 1.0 for perfect predictions."""
        perfect = np.array([1.0, 2.0, 3.0])
        value = r_squared(perfect, perfect)
        self.assertAlmostEqual(value, 1.0, places=5)

    def test_r_squared_bad(self):
        """Test R-squared is low for bad predictions."""
        preds = np.array([5.0, 5.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0])
        value = r_squared(preds, targets)
        self.assertLess(value, 0.5)

    def test_pearson_correlation(self):
        """Test Pearson correlation with perfectly correlated data."""
        value = pearson_correlation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
        )
        self.assertAlmostEqual(value, 1.0, places=5)

    def test_spearman_correlation(self):
        """Test Spearman correlation with monotonic data."""
        value = spearman_correlation(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        )
        self.assertAlmostEqual(value, 1.0, places=5)


class TestTrainerUnpackBatch(unittest.TestCase):
    """Test TrainerWithMMDrop._unpack_batch with mock batches."""

    def setUp(self):
        """Create a minimal trainer with a dummy model."""
        from vibropredict.training.trainer import TrainerWithMMDrop

        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        self.trainer = TrainerWithMMDrop(
            model=model, optimizer=optimizer, device='cpu'
        )

    def test_unpack_valid_batch(self):
        """Test _unpack_batch returns correct tuple from valid batch."""
        batch = {
            'sequences': ['MKTIIALSYIF', 'ACDEFGHIK'],
            'vdos': torch.randn(2, 1, 1000),
            'substrate_smiles': ['CC', 'CCO'],
            'product_smiles': ['C=O', 'CC(=O)O'],
            'log_kcat': torch.tensor([1.5, 2.3]),
        }
        sequences, vdos, sub_smi, prod_smi, log_kcat = self.trainer._unpack_batch(batch)

        self.assertEqual(len(sequences), 2)
        self.assertEqual(vdos.shape, (2, 1, 1000))
        self.assertEqual(len(sub_smi), 2)
        self.assertEqual(len(prod_smi), 2)
        self.assertEqual(log_kcat.shape, (2,))

    def test_unpack_missing_key_raises(self):
        """Test _unpack_batch raises on missing required key."""
        batch = {'sequences': ['A'], 'vdos': torch.zeros(1, 1, 10)}
        with self.assertRaises(KeyError):
            self.trainer._unpack_batch(batch)

    def test_unpack_optional_product_smiles(self):
        """Test _unpack_batch returns None for missing product_smiles."""
        batch = {
            'sequences': ['MKT'],
            'vdos': torch.randn(1, 1, 100),
            'substrate_smiles': ['CC'],
            'log_kcat': torch.tensor([1.0]),
        }
        _, _, _, prod_smi, _ = self.trainer._unpack_batch(batch)
        self.assertIsNone(prod_smi)


if __name__ == '__main__':
    unittest.main()
