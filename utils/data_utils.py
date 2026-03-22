"""
Data Loading and Preprocessing Utilities
UCI Online Retail, simulation data, CDNOW.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

def load_simulation_data(world: str, N: int, T: int, seed: int, data_dir: Path):
    """
    Load simulation data from separate .npy (states) and .csv (spend) files.

    Expected files (with seed suffix for demo subsets):
      - true_states_{world}_N{N}_T{T}_seed{seed}.npy  (for N=50, T=20 subsets)
      - hmm_{world}_N{N}_T{T}_seed{seed}.csv

    Or (without seed suffix for full data):
      - true_states_{world}_N{N}_T{T}.npy  (for N=1000, T=52 full data)
      - hmm_{world}_N{N}_T{T}.csv
    """
    world_cap = world.capitalize()

    # Try with seed suffix first (for demo subsets)
    states_file_seed = data_dir / f"true_states_{world_cap}_N{N}_T{T}_seed{seed}.npy"
    csv_file_seed = data_dir / f"hmm_{world_cap}_N{N}_T{T}_seed{seed}.csv"

    # Fallback to without seed (for full production data)
    states_file_noseed = data_dir / f"true_states_{world_cap}_N{N}_T{T}.npy"
    csv_file_noseed = data_dir / f"hmm_{world_cap}_N{N}_T{T}.csv"

    # Determine which files to use
    if states_file_seed.exists() and csv_file_seed.exists():
        states_file = states_file_seed
        csv_file = csv_file_seed
    elif states_file_noseed.exists() and csv_file_noseed.exists():
        states_file = states_file_noseed
        csv_file = csv_file_noseed
    else:
        # Report what we were looking for
        missing = []
        if not states_file_seed.exists():
            missing.append(str(states_file_seed))
        if not csv_file_seed.exists():
            missing.append(str(csv_file_seed))
        if not states_file_noseed.exists():
            missing.append(str(states_file_noseed))
        if not csv_file_noseed.exists():
            missing.append(str(csv_file_noseed))
        raise FileNotFoundError(
            f"Simulation data not found for {world} N={N} T={T} seed={seed}.\n"
            f"Tried:\n"
            f"  With seed: {states_file_seed.name}, {csv_file_seed.name}\n"
            f"  Without seed: {states_file_noseed.name}, {csv_file_noseed.name}\n"
            f"Missing files: {missing}\n"
            f"Generate via: python data/simulation/create_subset.py --world {world} --N {N} --T {T} --seed {seed}"
        )

    # Load states
    true_states = np.load(states_file, allow_pickle=True)

    # Load spend data from CSV
    df = pd.read_csv(csv_file)

    # Pivot to wide format (N, T)
    y = df.pivot(index='customer_id', columns='t', values='y').values

    # Verify shapes match
    if y.shape != true_states.shape:
        raise ValueError(f"Shape mismatch: y={y.shape}, states={true_states.shape}")

    return {
        'y': y,
        'true_states': true_states,
        'params': {
            'world': world,
            'N': N,
            'T': T,
            'seed': seed,
        }
    }
    
def load_uci_data(csv_path: Path, min_transactions: int = 5) -> pd.DataFrame:
    """
    Load and clean UCI Online Retail dataset.

    Parameters
    ----------
    csv_path : Path
        Path to data.csv (ID 352) or online_retail_II.csv (ID 502)
    min_transactions : int
        Minimum transactions per customer for inclusion

    Returns
    -------
    df : DataFrame with columns [customer_id, week, spend, n_transactions]
    """
    # Load raw data
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Standardize column names (ID 352 vs 502 have different names)
    col_map = {
        'Customer ID': 'customer_id',
        'CustomerID': 'customer_id',
        'InvoiceDate': 'date',
        'Invoice': 'invoice',
        'Quantity': 'quantity',
        'Price': 'price',
        'UnitPrice': 'price',
    }
    df = df.rename(columns=col_map)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])

    # Compute spend
    df['spend'] = df['quantity'] * df['price']

    # Filter valid transactions (positive quantity and price)
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]

    # Aggregate to customer-week level
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.isocalendar().year
    df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)

    weekly = df.groupby(['customer_id', 'year_week']).agg({
        'spend': 'sum',
        'invoice': 'nunique',  # number of transactions
    }).reset_index()
    weekly = weekly.rename(columns={'invoice': 'n_transactions'})

    # Filter customers with sufficient transactions
    cust_counts = weekly['customer_id'].value_counts()
    valid_custs = cust_counts[cust_counts >= min_transactions].index
    weekly = weekly[weekly['customer_id'].isin(valid_custs)]

    return weekly


def create_rfm_panel(weekly_df: pd.DataFrame, T: int = 52) -> Dict:
    """
    Convert weekly transactions to RFM panel format.

    Parameters
    ----------
    weekly_df : DataFrame from load_uci_data()
    T : int
        Number of time periods (weeks)

    Returns
    -------
    Dict with keys: y (N,T), R (N,T), F (N,T), M (N,T), customer_ids
    """
    # Pivot to wide format
    all_weeks = sorted(weekly_df['year_week'].unique())[:T]

    # Spend matrix
    spend_pivot = weekly_df.pivot_table(
        index='customer_id', 
        columns='year_week', 
        values='spend', 
        fill_value=0
    )

    # Ensure all weeks present
    for w in all_weeks:
        if w not in spend_pivot.columns:
            spend_pivot[w] = 0
    spend_pivot = spend_pivot[all_weeks]

    N = len(spend_pivot)

    # Compute R, F, M features
    y = spend_pivot.values

    # Recency (weeks since last purchase)
    R = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            if y[i, t] > 0:
                R[i, t] = 0
            elif t == 0:
                R[i, t] = 999  # Never purchased
            else:
                R[i, t] = R[i, t-1] + 1

    # Frequency (cumulative transactions)
    F = np.cumsum(y > 0, axis=1)

    # Monetary (average spend when positive)
    M = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            if y[i, t] > 0:
                M[i, t] = y[i, t]
            elif t > 0:
                M[i, t] = M[i, t-1]

    return {
        'y': y,
        'R': R,
        'F': F,
        'M': M,
        'customer_ids': spend_pivot.index.values,
        'N': N,
        'T': T,
    }


def train_test_split(data: Dict, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Split panel data into train/test by time.

    Returns
    -------
    train_data, test_data : Dicts with 'y', 'T', etc.
    """
    np.random.seed(random_seed)

    N, T_full = data['y'].shape
    T_train = int(T_full * train_ratio)
    T_test = T_full - T_train

    train = {
        'y': data['y'][:, :T_train],
        'R': data['R'][:, :T_train],
        'F': data['F'][:, :T_train],
        'M': data['M'][:, :T_train],
        'N': N,
        'T': T_train,
        'T_full': T_full,
    }

    test = {
        'y': data['y'][:, T_train:],
        'R': data['R'][:, T_train:],
        'F': data['F'][:, T_train:],
        'M': data['M'][:, T_train:],
        'N': N,
        'T': T_test,
    }

    return train, test


======================================================================
Save as: src/utils/data_utils.py
======================================================================
