"""
project/utils/metrics.py
Evaluation metrics for point regression and uncertainty quantification (UQ).
"""

from typing import Dict
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression performance metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Filter out NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-12 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def uq_calibration_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    nominal_confidence: float = 0.95
) -> Dict[str, float]:
    """
    Evaluates quality of uncertainty estimation:
    - PICP: Prediction Interval Coverage Probability (fraction within confidence interval).
    - MPIW: Mean Prediction Interval Width.
    - NLL: Gaussian negative log-likelihood.
    - z_score_std: Empirical standard deviation of standardized residuals (should be ~1.0).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ym = np.asarray(y_pred_mean, dtype=np.float64).ravel()
    ys = np.asarray(y_pred_std, dtype=np.float64).ravel()

    mask = ~np.isnan(yt) & ~np.isnan(ym) & ~np.isnan(ys) & (ys > 1e-12)
    if np.sum(mask) == 0:
        return {"picp": float("nan"), "mpiw": float("nan"), "nll": float("nan"), "z_score_std": float("nan")}

    yt = yt[mask]
    ym = ym[mask]
    ys = ys[mask]

    # Two-tailed standard normal critical value
    alpha = 1.0 - nominal_confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    lower_bound = ym - z_crit * ys
    upper_bound = ym + z_crit * ys

    # PICP: fraction of ground truth values within bounds
    covered = (yt >= lower_bound) & (yt <= upper_bound)
    picp = float(np.mean(covered))

    # MPIW: average interval width
    interval_widths = upper_bound - lower_bound
    mpiw = float(np.mean(interval_widths))

    # Gaussian NLL
    variance = ys ** 2
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + (yt - ym) ** 2 / (2.0 * variance)))

    # Standardized residuals z = (y - mu) / sigma
    z_residuals = (yt - ym) / ys
    z_std = float(np.std(z_residuals))

    return {
        "picp": picp,
        "nominal_confidence": nominal_confidence,
        "mpiw": mpiw,
        "nll": nll,
        "z_score_std": z_std,
    }
