"""
SMC Tempering Schedule Utilities
Beta annealing from prior (beta=0) to posterior (beta=1).
"""

import numpy as np
from typing import List, Callable


def linear_schedule(n_stages: int) -> np.ndarray:
    """Linear tempering: beta_j = j / J"""
    return np.linspace(0, 1, n_stages)


def exponential_schedule(n_stages: int, rate: float = 5.0) -> np.ndarray:
    """Exponential tempering: beta_j = (exp(rate * j/J) - 1) / (exp(rate) - 1)"""
    j = np.arange(n_stages) / n_stages
    return (np.exp(rate * j) - 1) / (np.exp(rate) - 1)


def adaptive_schedule(log_likelihoods: np.ndarray, target_ess: float = 0.8) -> np.ndarray:
    """
    Adaptive tempering based on Effective Sample Size (ESS).

    Parameters
    ----------
    log_likelihoods : ndarray, shape (n_particles,)
        Log p(y | theta) for each particle
    target_ess : float
        Target ESS ratio (ESS / N)

    Returns
    -------
    beta_increment : float
        Increment to add to current beta
    """
    # Find beta such that ESS = target_ess * N
    # Using bisection search
    from scipy.optimize import brentq

    def ess_ratio(beta):
        log_weights = beta * log_likelihoods
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()
        ess = 1.0 / np.sum(weights ** 2)
        return ess / len(log_likelihoods) - target_ess

    # Solve for beta in [0, 1]
    try:
        beta_new = brentq(ess_ratio, 0, 1)
    except ValueError:
        beta_new = 1.0  # If can't find, jump to end

    return beta_new


def compute_ess(log_weights: np.ndarray) -> float:
    """
    Compute Effective Sample Size from log weights.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights
    """
    log_weights = log_weights - np.max(log_weights)  # stabilize
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return 1.0 / np.sum(weights ** 2)


def tempering_diagnostics(beta_history: List[float], ess_history: List[float]) -> dict:
    """
    Compute diagnostics for tempering schedule.

    Returns
    -------
    dict with 'n_stages', 'mean_ess', 'min_ess', 'beta_annealing_rate'
    """
    beta_arr = np.array(beta_history)
    ess_arr = np.array(ess_history)

    return {
        'n_stages': len(beta_history),
        'mean_ess': np.mean(ess_arr),
        'min_ess': np.min(ess_arr),
        'beta_annealing_rate': np.mean(np.diff(beta_arr)),
        'total_ess_drop': ess_arr[0] - ess_arr[-1] if len(ess_arr) > 1 else 0,
    }


======================================================================
Save as: src/core/tempering.py
======================================================================
