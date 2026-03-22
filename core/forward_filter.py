"""
Batched Forward-Filtering Algorithm (Web Appendix A)
Log-space stabilization for numerical stability.
"""

import numpy as np
import pytensor.tensor as pt
from typing import Callable


def forward_filter_log_batched(Y, log_Gamma, log_pi0, log_likelihood_fn):
    """
    Batched forward filtering with log-space stabilization.

    Parameters
    ----------
    Y : tensor, shape (N, T)
        Observed transaction data
    log_Gamma : tensor, shape (K, K)
        Log transition matrix
    log_pi0 : tensor, shape (K,)
        Log initial state probabilities
    log_likelihood_fn : callable
        Function computing log p(y_t | S_t=k) for all k

    Returns
    -------
    log_alpha : tensor, shape (N, T, K)
        Log filtered probabilities
    """
    N, T = Y.shape
    K = log_pi0.shape[0]

    # Initialize: broadcast prior to all customers
    log_alpha = pt.full((N, K), log_pi0)
    log_alphas = []

    for t in range(T):
        # (a) Emission
        log_l_t = log_likelihood_fn(Y[:, t])

        # (b) Transition: logsumexp over previous states
        log_alpha_trans = pt.logsumexp(
            log_alpha.dimshuffle(0, 1, 'x') + log_Gamma.dimshuffle('x', 0, 1),
            axis=1
        )

        # (c) Update
        log_alpha = log_l_t + log_alpha_trans

        # (d) Normalize
        log_alpha = log_alpha - pt.logsumexp(log_alpha, axis=1, keepdims=True)

        log_alphas.append(log_alpha)

    return pt.stack(log_alphas, axis=1)


def forward_filter_numpy(Y, Gamma, pi0, emission_probs):
    """
    NumPy version for reference/testing (no GPU).

    Parameters
    ----------
    Y : ndarray, shape (N, T)
    Gamma : ndarray, shape (K, K)
    pi0 : ndarray, shape (K,)
    emission_probs : ndarray, shape (N, T, K)
        p(y_t | S_t=k) for all n,t,k

    Returns
    -------
    alpha : ndarray, shape (N, T, K)
        Filtered probabilities
    """
    N, T = Y.shape
    K = len(pi0)

    alpha = np.zeros((N, T, K))

    # Initialize
    alpha[:, 0, :] = pi0 * emission_probs[:, 0, :]
    alpha[:, 0, :] /= alpha[:, 0, :].sum(axis=1, keepdims=True)

    # Forward pass
    for t in range(1, T):
        # Predict: sum over previous states
        pred = alpha[:, t-1, :] @ Gamma  # (N, K)

        # Update: multiply by emission
        alpha[:, t, :] = pred * emission_probs[:, t, :]

        # Normalize
        alpha[:, t, :] /= alpha[:, t, :].sum(axis=1, keepdims=True)

    return alpha


======================================================================
Save as: src/core/forward_filter.py
======================================================================
