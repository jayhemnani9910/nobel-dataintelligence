"""
Mutant Ranking Loss

Combines MSE regression loss with a pairwise ranking term
to preserve relative ordering among enzyme mutants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MutantRankingLoss(nn.Module):
    """
    Combined MSE + pairwise ranking loss for enzyme kinetics prediction.

    When mutant_pairs are provided, the loss encourages correct relative
    ordering of paired predictions in addition to absolute accuracy.

    Args:
        lambda_rank: Weight for the ranking loss term.
    """

    def __init__(self, lambda_rank: float = 0.1):
        """
        Initialize MutantRankingLoss.

        Args:
            lambda_rank: Weight for the ranking loss term.
        """
        super().__init__()
        self.lambda_rank = lambda_rank

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mutant_pairs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Predicted values, shape (N,).
            targets: Ground truth values, shape (N,).
            mutant_pairs: Optional index pairs of shape (M, 2) into predictions.

        Returns:
            Scalar loss tensor.
        """
        mse_loss = F.mse_loss(predictions, targets)

        if mutant_pairs is not None:
            pred_diff = (
                predictions[mutant_pairs[:, 0]] - predictions[mutant_pairs[:, 1]]
            )
            target_diff = (
                targets[mutant_pairs[:, 0]] - targets[mutant_pairs[:, 1]]
            )
            target_sign = torch.sign(target_diff)
            ranking_loss = F.margin_ranking_loss(
                predictions[mutant_pairs[:, 0]],
                predictions[mutant_pairs[:, 1]],
                target_sign,
                margin=0.1,
            )
            return mse_loss + self.lambda_rank * ranking_loss

        return mse_loss
