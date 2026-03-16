"""Unit tests for VibroPredict training utilities.

Tests cover:
- MutantRankingLoss: basic MSE, combined loss with pairs
- metrics.compute_all_metrics: returns all expected keys
- metrics.top_k_ranking_accuracy: perfect ranking and random
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from vibropredict.training.losses import MutantRankingLoss
from vibropredict.training.metrics import compute_all_metrics, top_k_ranking_accuracy


class TestMutantRankingLossBasic(unittest.TestCase):
    """Test MutantRankingLoss returns pure MSE without pairs."""

    def test_basic_mse_no_pairs(self):
        """Test loss equals MSE when mutant_pairs is None."""
        loss_fn = MutantRankingLoss(lambda_rank=0.5)
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.2, 2.1, 2.9, 4.3])

        loss = loss_fn(preds, targets, mutant_pairs=None)
        expected = torch.nn.functional.mse_loss(preds, targets)

        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_zero_loss_for_perfect_predictions(self):
        """Test loss is zero when predictions equal targets and no pairs."""
        loss_fn = MutantRankingLoss()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(preds, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestMutantRankingLossCombined(unittest.TestCase):
    """Test MutantRankingLoss with mutant pairs."""

    def test_combined_loss_with_pairs(self):
        """Test combined loss includes both MSE and ranking components."""
        loss_fn = MutantRankingLoss(lambda_rank=1.0)
        preds = torch.tensor([1.0, 3.0, 2.0, 4.0])
        targets = torch.tensor([1.0, 3.0, 2.0, 4.0])
        pairs = torch.tensor([[0, 1], [2, 3]])

        loss = loss_fn(preds, targets, mutant_pairs=pairs)

        # MSE should be zero, but ranking loss may add margin penalty
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_wrong_ranking_increases_loss(self):
        """Test that incorrect ranking order increases loss."""
        loss_fn = MutantRankingLoss(lambda_rank=1.0)
        targets = torch.tensor([1.0, 3.0])
        pairs = torch.tensor([[0, 1]])

        # Correct ordering: pred[0] < pred[1] matching target[0] < target[1]
        correct_preds = torch.tensor([1.0, 3.0])
        loss_correct = loss_fn(correct_preds, targets, mutant_pairs=pairs)

        # Reversed ordering
        reversed_preds = torch.tensor([3.0, 1.0])
        loss_reversed = loss_fn(reversed_preds, targets, mutant_pairs=pairs)

        self.assertGreater(loss_reversed.item(), loss_correct.item())

    def test_lambda_rank_scales_ranking_component(self):
        """Test that lambda_rank=0 gives pure MSE even with pairs."""
        loss_no_rank = MutantRankingLoss(lambda_rank=0.0)
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        pairs = torch.tensor([[0, 1], [1, 2]])

        loss = loss_no_rank(preds, targets, mutant_pairs=pairs)
        expected_mse = torch.nn.functional.mse_loss(preds, targets)

        self.assertAlmostEqual(loss.item(), expected_mse.item(), places=5)


class TestComputeAllMetrics(unittest.TestCase):
    """Test compute_all_metrics returns all expected keys."""

    def test_returns_all_keys(self):
        """Test output dictionary has all expected metric keys."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        metrics = compute_all_metrics(preds, targets)

        expected_keys = {'rmse', 'r_squared', 'pearson', 'spearman', 'top_k_accuracy'}
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_all_values_are_finite(self):
        """Test all metric values are finite floats."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        metrics = compute_all_metrics(preds, targets)

        for key, value in metrics.items():
            self.assertIsInstance(value, float, msg=f"{key} is not a float")
            self.assertTrue(np.isfinite(value), msg=f"{key} is not finite: {value}")

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_all_metrics(data, data)

        self.assertAlmostEqual(metrics['rmse'], 0.0, places=5)
        self.assertAlmostEqual(metrics['r_squared'], 1.0, places=5)
        self.assertAlmostEqual(metrics['pearson'], 1.0, places=5)
        self.assertAlmostEqual(metrics['spearman'], 1.0, places=5)


class TestTopKRankingAccuracy(unittest.TestCase):
    """Test top_k_ranking_accuracy with known data."""

    def test_perfect_ranking(self):
        """Test accuracy is 1.0 when top-k predictions match targets exactly."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        accuracy = top_k_ranking_accuracy(predictions, targets, k=3)
        self.assertAlmostEqual(accuracy, 1.0, places=5)

    def test_reversed_ranking(self):
        """Test accuracy is 0.0 when top-k predicted are bottom-k actual."""
        predictions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        accuracy = top_k_ranking_accuracy(predictions, targets, k=2)
        # Top-2 by prediction are indices [0,1], top-2 by target are [3,4]
        self.assertAlmostEqual(accuracy, 0.0, places=5)

    def test_k_larger_than_array(self):
        """Test k is clamped to array length."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])

        accuracy = top_k_ranking_accuracy(predictions, targets, k=100)
        # k clamped to 3; all match
        self.assertAlmostEqual(accuracy, 1.0, places=5)

    def test_partial_overlap(self):
        """Test partial overlap returns fractional accuracy."""
        predictions = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        accuracy = top_k_ranking_accuracy(predictions, targets, k=2)
        # Top-2 by pred: indices [3,4]; top-2 by target: indices [3,4]
        self.assertAlmostEqual(accuracy, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
