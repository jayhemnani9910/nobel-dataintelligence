"""
Visualization Utilities for VibroPredict

Provides plotting functions for model evaluation: correlation plots,
ablation bar charts, gate weight distributions, and error histograms.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vibropredict.training.metrics import r_squared

logger = logging.getLogger(__name__)


def plot_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter plot of predictions vs targets with regression line.

    Annotates the plot with the R-squared value.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    r2 = r_squared(predictions, targets)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.5, s=10)

    # Regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    x_range = np.linspace(targets.min(), targets.max(), 100)
    ax.plot(x_range, p(x_range), 'r--', linewidth=1.5)

    ax.set_xlabel("Targets")
    ax.set_ylabel("Predictions")
    ax.set_title(title)
    ax.annotate(f"$R^2 = {r2:.3f}$", xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Correlation plot saved to {save_path}")
    plt.close(fig)


def plot_ablation_results(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart comparing ablation variants.

    Args:
        results_df: DataFrame from run_ablation with columns:
                    variant, rmse, r_squared, pearson, spearman.
        save_path: If provided, save figure to this path.
    """
    metrics = ['rmse', 'r_squared', 'pearson', 'spearman']
    variants = results_df['variant'].tolist()

    x = np.arange(len(variants))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(variants, rotation=15)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Ablation plot saved to {save_path}")
    plt.close(fig)


def plot_gate_weights(
    gate_weights: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart showing mean attention per modality.

    Args:
        gate_weights: Array of shape (N, 3) with columns for
                      sequence, spectral, and chemical gates.
        save_path: If provided, save figure to this path.
    """
    labels = ['Sequence', 'Spectral', 'Chemical']
    means = gate_weights.mean(axis=0)
    stds = gate_weights.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['#2196F3', '#4CAF50', '#FF9800'])

    ax.set_ylabel("Mean Gate Weight")
    ax.set_title("Modality Gate Weights")
    ax.set_ylim(0, 1)

    # Annotate bars with values
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Gate weights plot saved to {save_path}")
    plt.close(fig)


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Histogram of prediction errors (predictions - targets).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        save_path: If provided, save figure to this path.
    """
    errors = predictions - targets

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

    ax.set_xlabel("Error (Predicted - Target)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.annotate(f"Mean: {errors.mean():.3f}\nStd: {errors.std():.3f}",
                xy=(0.72, 0.85), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error distribution plot saved to {save_path}")
    plt.close(fig)
