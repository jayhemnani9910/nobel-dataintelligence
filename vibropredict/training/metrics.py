"""
Evaluation Metrics for Enzyme Kinetics Prediction

Provides regression and ranking metrics for assessing model performance
on log(k_cat) prediction tasks.
"""

import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def _flatten_pair(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both arrays so (N, 1) vs (N,) cannot broadcast to (N, N)."""
    predictions = np.ravel(predictions)
    targets = np.ravel(targets)
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have the same length"
        )
    return predictions, targets


def _correlation_undefined(predictions: np.ndarray, targets: np.ndarray) -> bool:
    """True when a correlation cannot be computed (n < 2 or constant input)."""
    if len(predictions) < 2 or np.ptp(predictions) == 0 or np.ptp(targets) == 0:
        logger.warning("Correlation undefined (n < 2 or constant input); returning NaN")
        return True
    return False


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        RMSE value.
    """
    predictions, targets = _flatten_pair(predictions, targets)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def r_squared(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute coefficient of determination (R-squared).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        R-squared value (NaN when targets are constant).
    """
    predictions, targets = _flatten_pair(predictions, targets)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Pearson correlation coefficient.
    """
    predictions, targets = _flatten_pair(predictions, targets)
    if _correlation_undefined(predictions, targets):
        return float("nan")
    corr, _ = pearsonr(predictions, targets)
    return float(corr)


def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Spearman correlation coefficient.
    """
    predictions, targets = _flatten_pair(predictions, targets)
    if _correlation_undefined(predictions, targets):
        return float("nan")
    corr, _ = spearmanr(predictions, targets)
    return float(corr)


def top_k_ranking_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute top-k ranking accuracy.

    Measures the fraction of the top-k predicted samples that are
    actually in the top-k by ground truth.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        k: Number of top entries to compare.

    Returns:
        Fraction of overlap between predicted and actual top-k.
    """
    k = min(k, len(predictions))
    pred_top_k = set(np.argsort(predictions)[-k:])
    target_top_k = set(np.argsort(targets)[-k:])
    overlap = len(pred_top_k & target_top_k)
    return float(overlap / k)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Compute all regression and ranking metrics.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Dictionary with keys: rmse, r_squared, pearson, spearman, top_k_accuracy.
    """
    predictions, targets = _flatten_pair(predictions, targets)
    return {
        "rmse": rmse(predictions, targets),
        "r_squared": r_squared(predictions, targets),
        "pearson": pearson_correlation(predictions, targets),
        "spearman": spearman_correlation(predictions, targets),
        "top_k_accuracy": top_k_ranking_accuracy(predictions, targets),
    }
