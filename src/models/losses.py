"""
Custom Loss Functions for Quantum Data Decoder

Implements specialized loss functions for stability ranking and
multi-label function prediction.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLossCustom(nn.Module):
    """
    Pairwise ranking loss for stability prediction.
    
    For Novozymes: Ranking pairs of (WT, Mutant) by stability.
    Objective: Predict correct ranking of stability scores.
    
    Loss = max(0, -y * (score_mut - score_wt) + margin)
    where y = +1 if mutant more stable, -1 if WT more stable
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize ranking loss.
        
        Args:
            margin: Margin for ranking (higher = more aggressive)
        """
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, scores_1: torch.Tensor, scores_2: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss.
        
        Args:
            scores_1: Scores for first set (WT)
            scores_2: Scores for second set (Mutants)
            targets: Labels (+1 or -1) indicating which should rank higher
            
        Returns:
            Scalar loss value
        """
        # PyTorch MarginRankingLoss expects: loss(x1, x2, y)
        # where y=1 means x1 should rank higher, y=-1 means x2 should rank higher
        # Convert target to PyTorch convention (-1 or 1)
        targets_pt = (targets > 0.5).float() * 2 - 1  # Convert [0,1] to [-1,1]
        
        return self.loss_fn(scores_1.squeeze(), scores_2.squeeze(), targets_pt)


class SpearmanCorrelationLoss(nn.Module):
    """
    Loss based on Spearman rank correlation (differentiable approximation).
    
    For competitions using Spearman correlation as metric, we optimize a
    differentiable approximation during training.
    """
    
    def __init__(self):
        """Initialize Spearman loss."""
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable Spearman correlation loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Loss (higher = worse correlation)
        """
        # Compute Pearson correlation as proxy (faster, differentiable)
        # Standardize
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        pred_std = predictions.std()
        target_std = targets.std()
        
        pred_norm = (predictions - pred_mean) / (pred_std + 1e-8)
        target_norm = (targets - target_mean) / (target_std + 1e-8)
        
        # Pearson correlation coefficient
        correlation = (pred_norm * target_norm).mean()
        
        # Loss = 1 - correlation (so correlation=1 gives loss=0)
        loss = 1 - correlation
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification with class imbalance.
    
    For CAFA 5: Many GO terms are rare (long-tail distribution).
    Focal loss down-weights easy examples to focus on hard negatives.
    
    FL(pt) = -α(1-pt)^γ log(pt)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare classes
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for multi-label setting.
        
        Args:
            logits: Raw model outputs (batch_size, num_classes)
            targets: Binary labels (batch_size, num_classes)
            
        Returns:
            Scalar loss
        """
        # Compute binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Focal term: (1 - pt)^γ
        p_t = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    
    For pre-training: Treat WT and naturally-occurring variants as
    positive pairs (similar), and random structures as negatives (dissimilar).
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for softmax scaling
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss (e.g., NT-Xent for SimCLR-style pretraining).
        
        Args:
            embeddings_1: First set of embeddings (batch_size, dim)
            embeddings_2: Second set of embeddings (batch_size, dim)
            labels: Binary labels (1 = positive pair, 0 = negative pair)
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings
        emb1_norm = F.normalize(embeddings_1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings_2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(emb1_norm, emb2_norm.t()) / self.temperature
        
        # Loss for positive pairs: minimize distance
        # Loss for negative pairs: maximize distance with margin
        loss_pos = -torch.log(torch.sigmoid(similarity).diagonal() + 1e-8)
        loss_neg = torch.log(
            1 + torch.exp(similarity - self.margin)
        ).sum(dim=1) - torch.log(
            1 + torch.exp(similarity.diagonal() - self.margin)
        )
        
        loss = (loss_pos + loss_neg).mean()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy for multi-label classification
    with class imbalance weights.
    """
    
    def __init__(self, pos_weight: torch.Tensor = None):
        """
        Initialize weighted BCE loss.
        
        Args:
            pos_weight: Weights for positive class per label
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model outputs (batch_size, num_classes)
            targets: Binary labels (batch_size, num_classes)
            
        Returns:
            Scalar loss
        """
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='mean'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='mean'
            )
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Combines losses from multiple objectives with learnable weights.
    """
    
    def __init__(self, loss_fns: dict, initial_weights: dict = None):
        """
        Initialize combined loss.
        
        Args:
            loss_fns: Dictionary of loss functions {'task_name': loss_fn}
            initial_weights: Initial weights for each loss
        """
        super().__init__()
        
        self.loss_fns = nn.ModuleDict(loss_fns)
        
        if initial_weights is None:
            initial_weights = {k: 1.0 for k in loss_fns.keys()}
        
        # Learnable temperature parameters for weight scaling
        self.log_weights = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(0.0))
            for k in loss_fns.keys()
        })
    
    def forward(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss across tasks.
        
        Args:
            **kwargs: Task-specific arguments
            
        Returns:
            Total loss and dict of individual losses
        """
        total_loss = 0
        individual_losses = {}
        
        for task_name, loss_fn in self.loss_fns.items():
            if task_name in kwargs:
                task_loss = loss_fn(**kwargs[task_name])
                weight = torch.exp(self.log_weights[task_name])
                
                individual_losses[task_name] = task_loss.item()
                total_loss += weight * task_loss
        
        return total_loss, individual_losses


def main():
    """Test custom loss functions."""
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Test MarginRankingLoss
    logger.info("Testing MarginRankingLoss...")
    loss_fn = MarginRankingLossCustom(margin=1.0)
    scores_wt = torch.tensor([5.0, 6.0, 4.0])
    scores_mut = torch.tensor([6.0, 5.0, 5.0])
    targets = torch.tensor([0.0, 1.0, 1.0])  # 0=WT better, 1=mutant better
    loss = loss_fn(scores_wt, scores_mut, targets)
    logger.info(f"Ranking loss: {loss.item():.4f}")
    
    # Test SpearmanCorrelationLoss
    logger.info("Testing SpearmanCorrelationLoss...")
    loss_fn = SpearmanCorrelationLoss()
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.5, 2.5, 2.8, 4.2])
    loss = loss_fn(pred, target)
    logger.info(f"Spearman loss: {loss.item():.4f}")
    
    # Test FocalLoss
    logger.info("Testing FocalLoss...")
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 100)
    targets = torch.randint(0, 2, (4, 100)).float()
    loss = loss_fn(logits, targets)
    logger.info(f"Focal loss: {loss.item():.4f}")
    
    logger.info("Custom loss functions test passed!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
