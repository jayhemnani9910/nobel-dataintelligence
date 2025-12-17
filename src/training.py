"""
Training Pipeline for Quantum Data Decoder

Implements training loops, validation, evaluation, and checkpointing
for multimodal stability and function prediction tasks.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for Quantum Data Decoder models.
    
    Features:
    - Automatic checkpointing and early stopping
    - Learning rate scheduling
    - Metric computation and tracking
    - Multi-GPU support (distributed training ready)
    """
    
    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda', checkpoint_dir: str = './checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer (Adam, SGD, etc.)
            scheduler: Learning rate scheduler
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory for saving checkpoints
        """
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            device = "cpu"

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(self, train_loader: DataLoader,
                   loss_fn: Callable,
                   task: str = 'novozymes') -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            loss_fn: Loss function
            task: 'novozymes' or 'cafa5'
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            if not isinstance(batch, dict):
                raise TypeError(f"Expected batch to be a dict, got {type(batch)}. Did you set the correct collate_fn?")

            if "labels" in batch:
                labels = batch["labels"]
            elif "label" in batch:  # legacy key
                labels = batch["label"]
            else:
                raise KeyError("Batch missing required key 'labels' (or legacy 'label').")

            if "spectra" in batch:
                spectra = batch["spectra"]
            elif "spectrum" in batch:  # legacy key
                spectra = batch["spectrum"]
            else:
                raise KeyError("Batch missing required key 'spectra' (or legacy 'spectrum').")

            if "graph" not in batch:
                raise KeyError("Batch missing required key 'graph'.")

            # Move batch to device
            graph = batch['graph'].to(self.device)
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)
            
            if 'global_features' in batch:
                global_features = batch['global_features'].to(self.device)
            else:
                global_features = None
            
            # Forward pass
            outputs = self.model(
                graph, spectra, global_features=global_features, task=task
            )
            
            # Compute loss
            if task == 'novozymes':
                loss = loss_fn(outputs.squeeze(), labels.squeeze())
            else:  # cafa5
                loss = loss_fn(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        if batch_count == 0:
            raise ValueError("Training DataLoader produced 0 usable batches.")

        avg_loss = total_loss / batch_count
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader,
                loss_fn: Callable,
                metric_fn: Optional[Callable] = None,
                task: str = 'novozymes') -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation DataLoader
            loss_fn: Loss function
            metric_fn: Optional metric function
            task: 'novozymes' or 'cafa5'
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                if not isinstance(batch, dict):
                    raise TypeError(
                        f"Expected batch to be a dict, got {type(batch)}. Did you set the correct collate_fn?"
                    )

                if "labels" in batch:
                    labels = batch["labels"]
                elif "label" in batch:  # legacy key
                    labels = batch["label"]
                else:
                    raise KeyError("Batch missing required key 'labels' (or legacy 'label').")

                if "spectra" in batch:
                    spectra = batch["spectra"]
                elif "spectrum" in batch:  # legacy key
                    spectra = batch["spectrum"]
                else:
                    raise KeyError("Batch missing required key 'spectra' (or legacy 'spectrum').")

                if "graph" not in batch:
                    raise KeyError("Batch missing required key 'graph'.")

                graph = batch['graph'].to(self.device)
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)
                
                if 'global_features' in batch:
                    global_features = batch['global_features'].to(self.device)
                else:
                    global_features = None
                
                # Forward pass
                outputs = self.model(
                    graph, spectra, global_features=global_features, task=task
                )
                
                # Compute loss
                if task == 'novozymes':
                    loss = loss_fn(outputs.squeeze(), labels.squeeze())
                else:
                    loss = loss_fn(outputs, labels)
                
                total_loss += loss.item()
                batch_count += 1
                
                preds_for_metric = outputs
                if task == 'cafa5':
                    preds_for_metric = torch.sigmoid(outputs)
                all_preds.append(preds_for_metric.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if batch_count == 0:
            raise ValueError("Validation DataLoader produced 0 usable batches.")

        avg_loss = total_loss / batch_count
        
        # Compute additional metrics if provided
        metrics = {'val_loss': avg_loss}
        
        if metric_fn is not None:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            metric_value = metric_fn(all_preds, all_labels)
            metrics['metric'] = metric_value
        
        return metrics
    
    def fit(self, train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: Callable,
            epochs: int = 100,
            metric_fn: Optional[Callable] = None,
            early_stopping_patience: int = 10,
            task: str = 'novozymes'):
        """
        Train for multiple epochs with validation and early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            loss_fn: Loss function
            epochs: Number of epochs
            metric_fn: Optional metric function
            early_stopping_patience: Early stopping patience
            task: 'novozymes' or 'cafa5'
        """
        logger.info(f"Training for {epochs} epochs on {self.device}")
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, task=task)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader, loss_fn, metric_fn, task=task)
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            if 'metric' in val_metrics:
                logger.info(f"Metric: {val_metrics['metric']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_loss'])
                logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Checkpointing
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                self.save_checkpoint(f"best_model_epoch{epoch}.pt")
                logger.info("✓ New best validation loss!")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                logger.info(f"Best epoch: {self.best_epoch} with loss {self.best_val_loss:.4f}")
                break
        
        return self.best_val_loss
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_val_loss,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_loss']
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint


class MetricComputer:
    """Compute task-specific metrics."""
    
    @staticmethod
    def spearman_correlation(predictions: np.ndarray, 
                            targets: np.ndarray) -> float:
        """
        Compute Spearman rank correlation (Novozymes metric).
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Spearman correlation coefficient
        """
        from scipy.stats import spearmanr
        
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        corr, _ = spearmanr(targets, predictions)
        return corr
    
    @staticmethod
    def f_max(predictions: np.ndarray,
             targets: np.ndarray,
             threshold_range: np.ndarray = None) -> float:
        """
        Compute F-max score (CAFA 5 metric).
        
        CAFA competitions use hierarchical F-max over GO terms.
        
        Args:
            predictions: Predicted probabilities (batch_size, n_terms)
            targets: Ground truth binary labels (batch_size, n_terms)
            threshold_range: Thresholds to evaluate (default: [0.1, 0.9])
            
        Returns:
            Maximum F-score
        """
        if threshold_range is None:
            threshold_range = np.linspace(0.1, 0.9, 9)
        
        max_f = 0
        for threshold in threshold_range:
            # Binarize predictions
            pred_binary = (predictions >= threshold).astype(int)
            
            # Compute F-score per sample
            tp = np.sum((pred_binary == 1) & (targets == 1), axis=1)
            fp = np.sum((pred_binary == 1) & (targets == 0), axis=1)
            fn = np.sum((pred_binary == 0) & (targets == 1), axis=1)
            
            # Compute precision and recall
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # Compute F-score
            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            mean_f = np.mean(f_score)
            
            max_f = max(max_f, mean_f)
        
        return max_f

    @staticmethod
    def f_max_score(predictions: np.ndarray,
                    targets: np.ndarray,
                    threshold_range: np.ndarray = None) -> float:
        """Backward-compatible alias for :meth:`f_max`."""
        return MetricComputer.f_max(predictions, targets, threshold_range=threshold_range)
    
    @staticmethod
    def mean_squared_error(predictions: np.ndarray,
                          targets: np.ndarray) -> float:
        """Compute MSE."""
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mean_absolute_error(predictions: np.ndarray,
                           targets: np.ndarray) -> float:
        """Compute MAE."""
        return np.mean(np.abs(predictions - targets))
    
    @staticmethod
    def accuracy(predictions: np.ndarray,
                targets: np.ndarray,
                threshold: float = 0.5) -> float:
        """Compute classification accuracy (for binary/multi-label)."""
        pred_binary = (predictions >= threshold).astype(int)
        return np.mean(pred_binary == targets)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement
            min_delta: Minimum improvement threshold
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def step(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


def create_training_config(task: str = 'novozymes') -> Dict:
    """Create default training configuration."""
    configs = {
        'novozymes': {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 100,
            'early_stopping_patience': 15,
            'weight_decay': 1e-5,
            'dropout': 0.2,
        },
        'cafa5': {
            'batch_size': 16,
            'learning_rate': 5e-4,
            'epochs': 50,
            'early_stopping_patience': 10,
            'weight_decay': 1e-4,
            'dropout': 0.3,
        }
    }
    return configs.get(task, configs['novozymes'])


def main():
    """Test training pipeline."""
    logger.info("Training pipeline module loaded successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
