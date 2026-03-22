src/core/__init__.py:
"""Core SMC algorithms (forward filtering, tempering)."""
from .forward_filter import forward_filter_log_batched, forward_filter_numpy
from .tempering import linear_schedule, exponential_schedule, compute_ess

__all__ = [
    'forward_filter_log_batched',
    'forward_filter_numpy', 
    'linear_schedule',
    'exponential_schedule',
    'compute_ess',
]


----------------------------------------------------------------------

src/utils/__init__.py:
"""Utility functions (data loading, metrics)."""
from .data_utils import load_simulation_data, load_uci_data, create_rfm_panel, train_test_split
from .metrics import compute_ari, compute_clv_metrics, compute_whale_metrics, compute_lead_time, compute_oos_metrics

__all__ = [
    'load_simulation_data',
    'load_uci_data',
    'create_rfm_panel',
    'train_test_split',
    'compute_ari',
    'compute_clv_metrics',
    'compute_whale_metrics',
    'compute_lead_time',
    'compute_oos_metrics',
]


======================================================================
Also create empty src/__init__.py and src/models/__init__.py
======================================================================
