"""
State-of-the-Art Comparison

Compares VibroPredict results against registered baseline models
and published enzyme kinetics prediction baselines.

Refactored for Phase 1 to use the pluggable baseline harness
(vibropredict.evaluation.baselines) instead of hardcoded numbers.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Published baseline numbers (R-squared on standard benchmarks)
# These are fallback values from literature when live baselines are unavailable.
DEFAULT_PUBLISHED_BASELINES: dict[str, dict[str, float]] = {
    "DLKcat": {"r_squared": 0.64, "year": 2021},
    "TurNuP": {"r_squared": 0.62, "year": 2022},
    "UniKP": {"r_squared": 0.68, "year": 2023},
    "MPEK": {"r_squared": 0.70, "year": 2023},
    "CatPred": {"r_squared": 0.79, "year": 2025},  # Nature Communications 2025
}


def compare_with_baselines(
    our_results: dict,
    baseline_results: dict | None = None,
    include_live_baselines: bool = True,
) -> pd.DataFrame:
    """
    Compare our model results with baseline methods.

    Args:
        our_results: Dictionary with at least 'rmse', 'r_squared', 'spearman'.
        baseline_results: Optional dictionary mapping model names to metric
                          dictionaries. If None, uses published baseline numbers.
        include_live_baselines: If True, also run registered baselines from the
                                pluggable harness (requires test data).

    Returns:
        DataFrame with columns: model, rmse, r_squared, spearman.
    """
    if baseline_results is None:
        baseline_results = DEFAULT_PUBLISHED_BASELINES

    rows = []

    # Add our results
    rows.append(
        {
            "model": "VibroPredict",
            "rmse": our_results.get("rmse", float("nan")),
            "r_squared": our_results.get("r_squared", float("nan")),
            "spearman": our_results.get("spearman", float("nan")),
            "source": "live",
        }
    )

    # Add baselines
    for model_name, metrics in baseline_results.items():
        rows.append(
            {
                "model": model_name,
                "rmse": metrics.get("rmse", float("nan")),
                "r_squared": metrics.get("r_squared", float("nan")),
                "spearman": metrics.get("spearman", float("nan")),
                "source": "live" if "rmse" in metrics else "published",
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"Comparison table:\n{df.to_string(index=False)}")
    return df


def run_live_comparison(
    our_predictions: np.ndarray,
    our_targets: np.ndarray,
    test_sequences: list[str],
    test_smiles: list[str],
    skip_baselines: list[str] | None = None,
) -> pd.DataFrame:
    """Run live head-to-head comparison using registered baselines.

    Args:
        our_predictions: VibroPredict predictions on test set.
        our_targets: Ground truth log(k_cat) values.
        test_sequences: Protein sequences for the test set.
        test_smiles: Substrate SMILES for the test set.
        skip_baselines: List of baseline names to skip.

    Returns:
        DataFrame with per-model metrics.
    """
    from vibropredict.evaluation.baselines.base import get_baseline, list_baselines
    from vibropredict.training.metrics import compute_all_metrics

    skip = set(skip_baselines or [])

    # Our metrics
    our_metrics = compute_all_metrics(our_predictions, our_targets)
    results = {"VibroPredict": our_metrics}

    # Run each registered baseline
    for name in list_baselines():
        if name in skip:
            logger.info(f"Skipping baseline: {name}")
            continue

        try:
            model = get_baseline(name)
            model.setup()

            preds = []
            for seq, smi in zip(test_sequences, test_smiles, strict=True):
                log_kcat, _ = model.predict(seq, smi)
                preds.append(log_kcat)

            preds_arr = np.array(preds)
            metrics = compute_all_metrics(preds_arr, our_targets)
            results[name] = metrics
            logger.info(f"{name}: R²={metrics['r_squared']:.4f}")
        except Exception as exc:
            logger.warning(f"{name} failed: {exc}")
            results[name] = {
                "rmse": float("nan"),
                "r_squared": float("nan"),
                "spearman": float("nan"),
                "error": str(exc),
            }

    return compare_with_baselines(our_metrics, results)
