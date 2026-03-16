"""
Trainer with Modality-Masking Dropout (MM-Drop)

Training loop for VibroPredictHybrid that randomly drops the spectral
modality during training to improve robustness and prevent over-reliance
on any single input channel.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.training import EarlyStopping

logger = logging.getLogger(__name__)


class TrainerWithMMDrop:
    """
    Training orchestrator with Modality-Masking Dropout.

    Randomly masks the spectral modality during training to encourage
    the model to leverage all input channels. During validation the
    full model (no dropping) is used.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu',
        checkpoint_dir: str = './checkpoints',
        scheduler: _LRScheduler = None,
    ):
        """
        Initialize trainer.

        Args:
            model: VibroPredictHybrid model.
            optimizer: Optimizer (Adam, SGD, etc.).
            device: 'cuda' or 'cpu'.
            checkpoint_dir: Directory for saving checkpoints.
            scheduler: Optional learning rate scheduler.
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

        logger.info(f"TrainerWithMMDrop initialized on {device}")

    def _unpack_batch(self, batch: dict) -> tuple:
        """
        Extract fields from a batch dictionary and move tensors to device.

        Args:
            batch: Dictionary with keys: sequences, vdos, substrate_smiles,
                   product_smiles, log_kcat.

        Returns:
            Tuple of (sequences, vdos, substrate_smiles, product_smiles, log_kcat).
        """
        sequences = batch['sequences']
        vdos = batch['vdos'].to(self.device)
        substrate_smiles = batch['substrate_smiles']
        product_smiles = batch.get('product_smiles')
        log_kcat = batch['log_kcat'].to(self.device)
        return sequences, vdos, substrate_smiles, product_smiles, log_kcat

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn,
        p_drop: float = 0.25,
    ) -> Dict[str, float]:
        """
        Train for one epoch with modality-masking dropout.

        Args:
            train_loader: DataLoader for training data.
            loss_fn: Loss function (e.g. MutantRankingLoss).
            p_drop: Probability of dropping the spectral modality per batch.

        Returns:
            Dictionary with 'train_loss' key.
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            sequences, vdos, substrate_smiles, product_smiles, log_kcat = (
                self._unpack_batch(batch)
            )

            # Randomly decide whether to drop spectral modality
            drop_spectral = bool(np.random.rand() < p_drop)

            # Forward pass
            logkcat, gates = self.model(
                sequences, vdos, substrate_smiles, product_smiles, drop_spectral
            )

            loss = loss_fn(logkcat, log_kcat)

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

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn,
    ) -> Dict[str, float]:
        """
        Validate model on validation set (no MM-Drop).

        Args:
            val_loader: Validation DataLoader.
            loss_fn: Loss function.

        Returns:
            Dictionary with 'val_loss', 'predictions', and 'targets' keys.
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                sequences, vdos, substrate_smiles, product_smiles, log_kcat = (
                    self._unpack_batch(batch)
                )

                logkcat, gates = self.model(
                    sequences, vdos, substrate_smiles, product_smiles, False
                )

                loss = loss_fn(logkcat, log_kcat)

                total_loss += loss.item()
                batch_count += 1

                all_preds.append(logkcat.cpu().numpy())
                all_targets.append(log_kcat.cpu().numpy())

        if batch_count == 0:
            raise ValueError("Validation DataLoader produced 0 usable batches.")

        avg_loss = total_loss / batch_count
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return {
            'val_loss': avg_loss,
            'predictions': predictions,
            'targets': targets,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn,
        epochs: int = 50,
        p_drop: float = 0.25,
        patience: int = 10,
    ) -> float:
        """
        Train for multiple epochs with validation and early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            loss_fn: Loss function.
            epochs: Maximum number of training epochs.
            p_drop: Spectral modality drop probability.
            patience: Early stopping patience.

        Returns:
            Best validation loss achieved.
        """
        logger.info(f"Training for {epochs} epochs on {self.device}")
        stopper = EarlyStopping(patience=patience)

        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"{'=' * 60}")

            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, p_drop=p_drop)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")

            # Validate
            val_metrics = self.validate(val_loader, loss_fn)
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_loss'])
                logger.info(
                    f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Checkpointing and early stopping
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(f"best_model_epoch{epoch}.pt")
                logger.info("New best validation loss!")

            if stopper.step(val_loss):
                logger.info(f"\nEarly stopping at epoch {epoch}")
                logger.info(
                    f"Best epoch: {self.best_epoch} with loss {self.best_val_loss:.4f}"
                )
                break
            else:
                logger.info(f"Patience: {stopper.counter}/{patience}")

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_loss']
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
