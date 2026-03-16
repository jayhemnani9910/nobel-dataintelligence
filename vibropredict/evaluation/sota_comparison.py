"""
State-of-the-Art Comparison

Compares VibroPredict results against published enzyme kinetics
prediction baselines (DLKcat, TurNuP, UniKP, MPEK).
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Published baseline numbers (R-squared on standard benchmarks)
DEFAULT_BASELINES: Dict[str, Dict[str, float]] = {
    'DLKcat': {'r_squared': 0.64},
    'TurNuP': {'r_squared': 0.62},
    'UniKP': {'r_squared': 0.68},
    'MPEK': {'r_squared': 0.70},
}


def compare_with_baselines(
    our_results: dict,
    baseline_results: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compare our model results with baseline methods.

    Args:
        our_results: Dictionary with at least 'rmse', 'r_squared', 'spearman'.
        baseline_results: Optional dictionary mapping model names to metric
                          dictionaries. If None, uses published baseline numbers.

    Returns:
        DataFrame with columns: model, rmse, r_squared, spearman.
    """
    if baseline_results is None:
        baseline_results = DEFAULT_BASELINES

    rows = []

    # Add our results
    rows.append({
        'model': 'VibroPredict',
        'rmse': our_results.get('rmse', float('nan')),
        'r_squared': our_results.get('r_squared', float('nan')),
        'spearman': our_results.get('spearman', float('nan')),
    })

    # Add baselines
    for model_name, metrics in baseline_results.items():
        rows.append({
            'model': model_name,
            'rmse': metrics.get('rmse', float('nan')),
            'r_squared': metrics.get('r_squared', float('nan')),
            'spearman': metrics.get('spearman', float('nan')),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Comparison table:\n{df.to_string(index=False)}")
    return df
