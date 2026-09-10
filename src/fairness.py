"""Fairness auditing utilities for model predictions."""
import numpy as np
import pandas as pd
from typing import Dict, List, Union


def audit_fairness(y_true: np.ndarray, y_pred: np.ndarray,
                   protected_attr: pd.Series) -> Dict[str, Union[float, Dict]]:
    """
    Audit fairness metrics by protected attribute groups.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (log-salary).
    y_pred : np.ndarray
        Predicted values (log-salary).
    protected_attr : pd.Series
        Protected attribute (e.g., experience_level, gender, region).

    Returns
    -------
    dict with MAE disparity and other fairness metrics by group.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Create dataframe for analysis
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'protected': protected_attr.values
    })

    # Calculate per-group metrics
    groups = df['protected'].unique()
    group_metrics = {}

    for group in groups:
        mask = df['protected'] == group
        group_true = df.loc[mask, 'y_true']
        group_pred = df.loc[mask, 'y_pred']

        mae = float(np.mean(np.abs(group_true - group_pred)))
        rmse = float(np.sqrt(np.mean((group_true - group_pred) ** 2)))
        mape = float(np.mean(np.abs((group_true - group_pred) / (group_true + 1e-8))) * 100)
        count = int(mask.sum())

        group_metrics[str(group)] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'count': count
        }

    # Calculate disparity metrics
    maes = [m['mae'] for m in group_metrics.values()]
    max_mae = max(maes)
    min_mae = min(maes)
    mean_mae = np.mean(maes)

    disparity = {
        'max_mae_diff': float(max_mae - min_mae),
        'max_mae_ratio': float(max_mae / min_mae) if min_mae > 0 else float('inf'),
        'coefficient_of_variation': float(np.std(maes) / mean_mae) if mean_mae > 0 else 0,
        'num_groups': len(groups)
    }

    return {
        'group_metrics': group_metrics,
        'disparity': disparity,
        'overall_mae': float(np.mean(np.abs(y_true - y_pred))),
        'protected_attribute': protected_attr.name if hasattr(protected_attr, 'name') else 'unknown'
    }


def get_fairness_summary(audit_result: Dict) -> str:
    """Generate human-readable summary of fairness audit."""
    lines = [
        f"Fairness Audit for '{audit_result['protected_attribute']}':",
        f"  Overall MAE: {audit_result['overall_mae']:.4f}",
        f"  Groups analyzed: {audit_result['disparity']['num_groups']}",
        f"  Max MAE disparity: {audit_result['disparity']['max_mae_diff']:.4f}",
        f"  Max/Min MAE ratio: {audit_result['disparity']['max_mae_ratio']:.2f}",
        "  Per-group MAE:",
    ]
    for group, metrics in audit_result['group_metrics'].items():
        lines.append(f"    {group}: MAE={metrics['mae']:.4f} (n={metrics['count']})")
    return '\n'.join(lines)
