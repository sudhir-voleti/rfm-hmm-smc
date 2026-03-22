"""
Performance Metrics for State Recovery and Managerial Utility
ARI, CLV, Whale Detection, Lead Time, etc.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from typing import Dict, List, Tuple, Optional


def compute_ari(true_states: np.ndarray, pred_states: np.ndarray) -> float:
    """
    Adjusted Rand Index for state recovery accuracy.

    Parameters
    ----------
    true_states : ndarray, shape (N, T)
    pred_states : ndarray, shape (N, T)

    Returns
    -------
    ari : float in [-1, 1]
        1.0 = perfect recovery, 0.0 = random, negative = worse than random
    """
    # Flatten across time
    true_flat = true_states.flatten()
    pred_flat = pred_states.flatten()

    # Handle missing values
    valid = ~np.isnan(true_flat) & ~np.isnan(pred_flat)

    if valid.sum() < 2:
        return np.nan

    return adjusted_rand_score(true_flat[valid], pred_flat[valid])


def compute_clv_metrics(spend_matrix: np.ndarray, 
                        states: np.ndarray,
                        discount: float = 0.99,
                        horizon: int = 52) -> Dict:
    """
    Compute CLV by state and CLV ratio.

    Parameters
    ----------
    spend_matrix : ndarray, shape (N, T)
    states : ndarray, shape (N, T)
        Decoded latent states
    discount : float
        Weekly discount factor (default 0.99 ~ 50% annual)
    horizon : int
        CLV horizon in weeks

    Returns
    -------
    dict with 'clv_by_state', 'clv_ratio', 'clv_total'
    """
    N, T = spend_matrix.shape
    K = int(states.max()) + 1

    # Compute CLV per customer (discounted sum)
    discounts = discount ** np.arange(T)
    clv_per_customer = (spend_matrix * discounts).sum(axis=1)

    # CLV by state (average of customers ever in that state)
    clv_by_state = []
    for k in range(K):
        mask = (states == k).any(axis=1)  # Customers who visited state k
        if mask.sum() > 0:
            clv_by_state.append(clv_per_customer[mask].mean())
        else:
            clv_by_state.append(0.0)

    clv_by_state = np.array(clv_by_state)

    # CLV ratio: max / min (non-zero)
    valid_clv = clv_by_state[clv_by_state > 0]
    if len(valid_clv) >= 2:
        clv_ratio = valid_clv.max() / valid_clv.min()
    else:
        clv_ratio = np.nan

    return {
        'clv_by_state': clv_by_state,
        'clv_ratio': clv_ratio,
        'clv_total': clv_per_customer.sum(),
    }


def compute_whale_metrics(spend_matrix: np.ndarray,
                         predicted_clv: np.ndarray,
                         percentile: float = 90.0) -> Dict:
    """
    Whale detection metrics: precision, recall, F1.

    Defines whales as top percentile by actual spend.
    Tests if model predicts them via top percentile by CLV.

    Parameters
    ----------
    spend_matrix : ndarray, shape (N, T)
    predicted_clv : ndarray, shape (N,)
    percentile : float
        Threshold for whale definition (default 90)

    Returns
    -------
    dict with 'precision', 'recall', 'f1', 'threshold_spend', 'threshold_clv'
    """
    # Actual whales: top percentile by total spend
    total_spend = spend_matrix.sum(axis=1)
    spend_threshold = np.percentile(total_spend, percentile)
    actual_whales = total_spend >= spend_threshold

    # Predicted whales: top percentile by predicted CLV
    clv_threshold = np.percentile(predicted_clv, percentile)
    predicted_whales = predicted_clv >= clv_threshold

    # Confusion matrix
    tp = (actual_whales & predicted_whales).sum()
    fp = (~actual_whales & predicted_whales).sum()
    fn = (actual_whales & ~predicted_whales).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_whales_true': actual_whales.sum(),
        'n_whales_pred': predicted_whales.sum(),
        'threshold_spend': spend_threshold,
        'threshold_clv': clv_threshold,
    }


def compute_lead_time(states: np.ndarray,
                       spend_matrix: np.ndarray,
                       surge_threshold: float = None,
                       validation_window: int = 12) -> Dict:
    """
    Compute mean lead time: weeks from S0->S1 transition to spend surge.

    Parameters
    ----------
    states : ndarray, shape (N, T)
        Decoded states (0 = dormant, 1 = active, etc.)
    spend_matrix : ndarray, shape (N, T)
    surge_threshold : float
        Spend threshold defining a surge (default: 90th percentile)
    validation_window : int
        Weeks after transition to check for surge

    Returns
    -------
    dict with 'mean_lead_time', 'n_transitions', 'validations_rate'
    """
    N, T = states.shape

    if surge_threshold is None:
        surge_threshold = np.percentile(spend_matrix[spend_matrix > 0], 90)

    lead_times = []
    n_transitions = 0
    n_validated = 0

    for i in range(N):
        # Find S0 -> S1 transitions
        for t in range(1, T):
            if states[i, t-1] == 0 and states[i, t] == 1:
                n_transitions += 1

                # Check for surge in validation window
                end_window = min(t + validation_window, T)
                future_spend = spend_matrix[i, t:end_window].sum()

                if future_spend > surge_threshold:
                    # Find exact week of surge
                    cumsum = np.cumsum(spend_matrix[i, t:end_window])
                    surge_week = t + np.where(cumsum > surge_threshold)[0][0]
                    lead_times.append(surge_week - t)
                    n_validated += 1

    return {
        'mean_lead_time': np.mean(lead_times) if lead_times else np.nan,
        'std_lead_time': np.std(lead_times) if lead_times else np.nan,
        'n_transitions': n_transitions,
        'n_validated': n_validated,
        'validation_rate': n_validated / n_transitions if n_transitions > 0 else 0.0,
    }


def compute_oos_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Out-of-sample prediction metrics.

    Parameters
    ----------
    y_true : ndarray, shape (N, T_test)
    y_pred : ndarray, shape (N, T_test)

    Returns
    -------
    dict with 'rmse', 'mae', 'mape'
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

    errors = y_true[mask] - y_pred[mask]

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / y_true[mask])) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
    }


======================================================================
Save as: src/utils/metrics.py
======================================================================
