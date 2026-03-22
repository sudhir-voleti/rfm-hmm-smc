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


