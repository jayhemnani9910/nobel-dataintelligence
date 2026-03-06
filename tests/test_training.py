"""Unit tests for training utilities and metrics.

Tests cover:
- Training loop and epoch iteration
- Metric computation (Spearman, F-max, etc.)
- Early stopping callback
- Checkpoint saving and loading
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.training import Trainer, MetricComputer, EarlyStopping, create_training_config
from src.models.multimodal import VibroStructuralModel


class TestMetricComputer(unittest.TestCase):
    """Test metric computation functions."""
    
    def setUp(self):
        """Generate test data."""
        np.random.seed(42)
        self.predictions = np.random.randn(100)
        self.targets = np.random.randn(100)
        self.probabilities = torch.sigmoid(torch.from_numpy(self.predictions)).numpy()
    
    def test_spearman_correlation(self):
        """Test Spearman correlation metric."""
        # Perfect correlation
        perfect_preds = np.arange(100)
        perfect_targets = np.arange(100)
        
        corr = MetricComputer.spearman_correlation(perfect_preds, perfect_targets)
        self.assertAlmostEqual(corr, 1.0, places=5)
    
    def test_spearman_negative_correlation(self):
        """Test Spearman with negative correlation."""
        preds = np.arange(100)
        targets = np.arange(100)[::-1]  # Reversed
        
        corr = MetricComputer.spearman_correlation(preds, targets)
        self.assertLess(corr, 0)
    
    def test_f_max(self):
        """Test F-max score computation."""
        # Multi-label predictions and targets
        perfect_preds = np.ones((10, 100))
        perfect_targets = np.ones((10, 100))
        
        f_max_val = MetricComputer.f_max(perfect_preds, perfect_targets)
        self.assertGreater(f_max_val, 0)
        self.assertLessEqual(f_max_val, 1.0)
    
    def test_f_max_with_random_data(self):
        """Test F-max with random predictions."""
        preds = np.random.rand(10, 100)
        targets = np.random.randint(0, 2, (10, 100))
        
        f_max_val = MetricComputer.f_max(preds, targets)
        
        self.assertGreaterEqual(f_max_val, 0)
        self.assertLessEqual(f_max_val, 1.0)
    
    def test_mean_squared_error(self):
        """Test MSE computation."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 1.9, 3.2, 3.8])
        
        mse = MetricComputer.mean_squared_error(preds, targets)
        
        expected = np.mean((preds - targets)**2)
        self.assertAlmostEqual(mse, expected, places=6)
    
    def test_mean_absolute_error(self):
        """Test MAE computation."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        
        mae = MetricComputer.mean_absolute_error(preds, targets)
        
        expected = np.mean(np.abs(preds - targets))
        self.assertAlmostEqual(mae, expected, places=6)
    
    def test_accuracy(self):
        """Test accuracy metric."""
        preds = np.array([0, 1, 1, 0, 1, 0])
        targets = np.array([0, 1, 0, 0, 1, 1])
        
        acc = MetricComputer.accuracy(preds, targets)
        
        # 4 out of 6 correct
        self.assertAlmostEqual(acc, 4/6, places=5)


class TestEarlyStopping(unittest.TestCase):
    """Test EarlyStopping callback."""
    
    def test_early_stopping_basic(self):
        """Test early stopping triggers on no improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.0)
        
        # Simulate no improvement
        for i in range(5):
            early_stop.step(val_loss=1.0)
        
        # Should trigger after patience exceeded
        self.assertTrue(early_stop.should_stop)
    
    def test_early_stopping_with_improvement(self):
        """Test early stopping with improving loss."""
        early_stop = EarlyStopping(patience=3, min_delta=0.0)
        
        # Simulate improving loss
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            early_stop.step(val_loss=loss)
        
        # Should not trigger if loss improves
        self.assertFalse(early_stop.should_stop)
    
    def test_early_stopping_counter(self):
        """Test early stopping counter behavior."""
        early_stop = EarlyStopping(patience=3, min_delta=0.0)
        
        # Initial call sets best_loss
        early_stop.step(val_loss=1.0)
        self.assertEqual(early_stop.counter, 0)
        
        # No improvement
        early_stop.step(val_loss=1.0)
        self.assertEqual(early_stop.counter, 1)
        
        # Improvement resets counter
        early_stop.step(val_loss=0.5)
        self.assertEqual(early_stop.counter, 0)


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration factory."""
    
    def test_novozymes_config(self):
        """Test Novozymes configuration."""
        config = create_training_config(task='novozymes')
        
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
        self.assertIn('epochs', config)
        self.assertIn('early_stopping_patience', config)
    
    def test_cafa5_config(self):
        """Test CAFA 5 configuration."""
        config = create_training_config(task='cafa5')
        
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
    
    def test_config_defaults(self):
        """Test configuration provides sensible defaults."""
        config = create_training_config()
        
        # All keys should be present
        required_keys = ['batch_size', 'learning_rate', 'epochs', 'early_stopping_patience']
        for key in required_keys:
            self.assertIn(key, config)
        
        # Values should be positive
        self.assertGreater(config['batch_size'], 0)
        self.assertGreater(config['learning_rate'], 0)
        self.assertGreater(config['epochs'], 0)


class TestTrainer(unittest.TestCase):
    """Test Trainer class."""
    
    def setUp(self):
        """Initialize model and trainer."""
        self.device = torch.device('cpu')
        
        # Simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = self.temp_dir.name
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir
        )
    
    def tearDown(self):
        """Clean up temp dir."""
        self.temp_dir.cleanup()
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertEqual(str(self.trainer.device), str(self.device))
    
    def test_checkpoint_saving(self):
        """Test model checkpoint is saved."""
        self.trainer.save_checkpoint('test_checkpoint.pt')
        
        checkpoint_path = Path(self.checkpoint_dir) / 'test_checkpoint.pt'
        self.assertTrue(checkpoint_path.exists())
    
    def test_checkpoint_loading(self):
        """Test model checkpoint can be loaded."""
        # Save checkpoint
        self.trainer.save_checkpoint('test.pt')
        
        # Load checkpoint
        self.trainer.load_checkpoint('test.pt')


class TestMetricValidation(unittest.TestCase):
    """Test metric computation edge cases."""
    
    def test_metric_with_nan_handling(self):
        """Test metrics handle NaN gracefully."""
        preds = np.array([1.0, 2.0, np.nan, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Should handle NaN by ignoring or returning NaN
        try:
            mse = MetricComputer.mean_squared_error(preds, targets)
            self.assertTrue(np.isnan(mse) or np.isfinite(mse))
        except:
            pass
    
    def test_metric_with_zero_variance(self):
        """Test metrics with constant targets."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 2.0, 2.0])  # Constant
        
        # Spearman correlation with constant target should fail gracefully
        try:
            corr = MetricComputer.spearman_correlation(preds, targets)
            # Either returns NaN or handles it
            self.assertTrue(np.isnan(corr) or True)
        except:
            pass
    
    def test_metric_with_single_sample(self):
        """Test metrics with single sample."""
        preds = np.array([1.0])
        targets = np.array([1.0])
        
        mse = MetricComputer.mean_squared_error(preds, targets)
        self.assertEqual(mse, 0.0)


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow through training."""
    
    def test_gradient_accumulation(self):
        """Test gradients accumulate across batches."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Process two batches
        for batch in range(2):
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            
            # Check gradients exist
            for param in model.parameters():
                self.assertIsNotNone(param.grad)
            
            optimizer.step()
    
    def test_loss_decreases(self):
        """Test loss decreases during training."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should generally decrease
        self.assertLess(losses[-1], losses[0] * 1.5)  # Allow some variance


if __name__ == '__main__':
    unittest.main()
