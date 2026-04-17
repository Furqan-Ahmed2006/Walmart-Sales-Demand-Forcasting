"""
Evaluation metrics for the Walmart demand forecasting project.

The Kaggle competition uses Weighted Mean Absolute Error (WMAE):

    WMAE = (1 / sum(w_i)) * sum(w_i * |y_i - y_hat_i|)

where w_i = 5 for holiday weeks and w_i = 1 otherwise.
"""

import numpy as np
import pandas as pd


def wmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
) -> float:
    """
    Compute the Weighted Mean Absolute Error used in the Kaggle competition.

    Parameters
    ----------
    y_true     : array-like, actual weekly sales
    y_pred     : array-like, predicted weekly sales
    is_holiday : array-like of 0/1 indicating holiday weeks

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics and return them as a dict.

    Parameters
    ----------
    y_true     : actual values
    y_pred     : predicted values
    is_holiday : optional holiday indicator for WMAE; zeros assumed if None
    """
    if is_holiday is None:
        is_holiday = np.zeros(len(y_true), dtype=int)

    return {
        "WMAE": wmae(y_true, y_pred, is_holiday),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def print_metrics(metrics: dict[str, float], model_name: str = "") -> None:
    """Pretty-print evaluation metrics."""
    header = f"{'Model: ' + model_name if model_name else 'Evaluation'}"
    print(f"\n{'='*50}")
    print(f"  {header}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"  {metric:<8}: {value:>12,.4f}")
    print(f"{'='*50}\n")
