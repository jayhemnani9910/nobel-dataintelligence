"""Tests for custom loss functions."""

import unittest
import torch
import torch.nn as nn

from src.models.losses import (
    PearsonCorrelationLoss,
    SpearmanCorrelationLoss,
    ContrastiveLoss,
    FocalLoss,
    WeightedBCELoss,
    CombinedLoss,
    MarginRankingLossCustom,
)


class TestPearsonCorrelationLoss(unittest.TestCase):
    """Test PearsonCorrelationLoss (and its SpearmanCorrelationLoss alias)."""

    def test_perfect_correlation(self):
        loss_fn = PearsonCorrelationLoss()
        # Use enough elements so Bessel's correction in std() is negligible
        preds = torch.arange(100, dtype=torch.float32)
        targets = torch.arange(100, dtype=torch.float32)
        loss = loss_fn(preds, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=2)

    def test_anti_correlation(self):
        loss_fn = PearsonCorrelationLoss()
        preds = torch.arange(100, dtype=torch.float32)
        targets = torch.arange(99, -1, -1, dtype=torch.float32)
        loss = loss_fn(preds, targets)
        self.assertAlmostEqual(loss.item(), 2.0, places=2)

    def test_alias_works(self):
        self.assertIs(SpearmanCorrelationLoss, PearsonCorrelationLoss)

    def test_gradient_flows(self):
        loss_fn = PearsonCorrelationLoss()
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)
        loss = loss_fn(preds, targets)
        loss.backward()
        self.assertIsNotNone(preds.grad)


class TestContrastiveLoss(unittest.TestCase):
    """Test ContrastiveLoss with numerical stability."""

    def test_basic_forward(self):
        loss_fn = ContrastiveLoss(temperature=0.07, margin=1.0)
        emb1 = torch.randn(4, 32)
        emb2 = torch.randn(4, 32)
        labels = torch.ones(4)
        loss = loss_fn(emb1, emb2, labels)
        self.assertTrue(torch.isfinite(loss))

    def test_large_similarity_no_overflow(self):
        loss_fn = ContrastiveLoss(temperature=0.01, margin=1.0)
        # Embeddings that produce very large similarity values
        emb = torch.ones(4, 32) * 10.0
        labels = torch.ones(4)
        loss = loss_fn(emb, emb, labels)
        self.assertTrue(torch.isfinite(loss), "Loss overflowed with large similarities")

    def test_gradient_flows(self):
        loss_fn = ContrastiveLoss()
        emb1 = torch.randn(4, 16, requires_grad=True)
        emb2 = torch.randn(4, 16)
        labels = torch.ones(4)
        loss = loss_fn(emb1, emb2, labels)
        loss.backward()
        self.assertIsNotNone(emb1.grad)


class TestFocalLoss(unittest.TestCase):
    """Test FocalLoss."""

    def test_all_zeros_targets(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)

    def test_all_ones_targets(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4, 10)
        targets = torch.ones(4, 10)
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss))

    def test_perfect_predictions_low_loss(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # Very confident correct predictions
        logits = torch.ones(4, 10) * 10.0
        targets = torch.ones(4, 10)
        loss_correct = loss_fn(logits, targets)
        # Random predictions
        logits_random = torch.randn(4, 10)
        loss_random = loss_fn(logits_random, targets)
        self.assertLess(loss_correct.item(), loss_random.item())


class TestCombinedLoss(unittest.TestCase):
    """Test CombinedLoss for multi-task training."""

    def test_multi_task_forward(self):
        loss_fns = {
            "task_a": nn.MSELoss(),
            "task_b": nn.MSELoss(),
        }
        combined = CombinedLoss(loss_fns)

        total, individual = combined(
            task_a={"input": torch.randn(4), "target": torch.randn(4)},
            task_b={"input": torch.randn(4), "target": torch.randn(4)},
        )
        self.assertTrue(torch.isfinite(total))
        self.assertIn("task_a", individual)
        self.assertIn("task_b", individual)

    def test_partial_tasks(self):
        loss_fns = {
            "task_a": nn.MSELoss(),
            "task_b": nn.MSELoss(),
        }
        combined = CombinedLoss(loss_fns)

        # Only provide task_a
        total, individual = combined(
            task_a={"input": torch.randn(4), "target": torch.randn(4)},
        )
        self.assertIn("task_a", individual)
        self.assertNotIn("task_b", individual)


if __name__ == "__main__":
    unittest.main()
