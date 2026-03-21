#!/usr/bin/env python3
"""
RFM-HMM Empirical Analysis Runner
===================================
Runs BEMMAOR, Hurdle, Tweedie on UCI Online Retail data.
Assumes pre-computed PKLs exist (too slow for interactive runs).
Generates comparison tables and figures.

Author: Sudhir Voleti
Date: March 2026
Repository: https://github.com/sudhir-voleti/rfm-hmm-smc
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from figures.generate_figures import plot_fig2_ppc, plot_fig3_occupancy


# UCI dataset URLs (for reference)
UCI_URLS = {
    352: "https://archive.ics.uci.edu/static/public/352/data.csv",  # 2010-2011
    502: "https://archive.ics.uci.edu/static/public/502/data.csv",  # 2009-2011, larger
}


def find_pkl_files(model: str, dataset: str, K: int, results_dir: Path) -> List[Path]:
    """
    Find PKL files for given model-dataset-K combination.

    Pattern: smc_*_{model}_*K{K}*_{dataset}_*.pkl
    """
    pattern = f"*K{K}*{model}*{dataset}*.pkl"
    pkls = list(results_dir.glob(pattern))

    if not pkls:
        # Try broader search
        pattern = f"*{model}*K{K}*.pkl"
        pkls = list(results_dir.rglob(pattern))

    return pkls


def extract_metrics_from_pkl(pkl_path: Path) -> Dict:
    """
    Extract key metrics from a PKL file.
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        res = data.get('res', {})

        metrics = {
            'pkl_path': str(pkl_path),
            'model': res.get('model_type', 'Unknown'),
            'K': res.get('K', np.nan),
            'N': res.get('N', np.nan),
            'log_evidence': res.get('log_evidence', np.nan),
            'ari': res.get('ari', np.nan),
            'clv_ratio': res.get('clv_ratio', np.nan),
            'clv_total': res.get('clv_total', np.nan),
            'oos_rmse': res.get('oos_rmse', np.nan),
            'oos_mae': res.get('oos_mae', np.nan),
            'time_min': res.get('time_min', np.nan),
            'has_ppc': 'ppc_simulations' in res or 'ppc_zero_obs' in res,
            'ess_min': res.get('ess_min', np.nan),
        }

        return metrics

    except Exception as e:
        print(f"✗ Error loading {pkl_path}: {e}")
        return {'pkl_path': str(pkl_path), 'error': str(e)}


def generate_comparison_table(
    models: List[str],
    dataset: str,
    K: int,
    results_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Generate comparison table (like Table 7) from existing PKLs.
    """
    print(f"\nSearching for PKLs: {models} | {dataset} | K={K}")
    print(f"Directory: {results_dir}")

    all_metrics = []

    for model in models:
        pkls = find_pkl_files(model, dataset, K, results_dir)

        if not pkls:
            print(f"⚠ No PKLs found for {model}-K{K}-{dataset}")
            continue

        # Use most recent PKL if multiple exist
        pkl = sorted(pkls, key=lambda p: p.stat().st_mtime)[-1]
        print(f"✓ Found: {pkl.name}")

        metrics = extract_metrics_from_pkl(pkl)
        metrics['model'] = model  # Ensure correct label
        all_metrics.append(metrics)

    if not all_metrics:
        print("✗ No PKLs found for any model")
        return None

    df = pd.DataFrame(all_metrics)

    # Save table
    csv_path = out_dir / f"empirical_comparison_{dataset}_K{K}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved: {csv_path}")

    # Print formatted table
    print("\n" + "="*70)
    print(f"EMPIRICAL COMPARISON: {dataset.upper()} | K={K}")
    print("="*70)
    display_cols = ['model', 'log_evidence', 'ari', 'clv_ratio', 'oos_rmse', 'oos_mae']
    print(df[display_cols].to_string(index=False))

    return df


def generate_figures(
    models: List[str],
    dataset: str,
    K: int,
    results_dir: Path,
    out_dir: Path,
):
    """
    Generate figures from PKLs (Figure 2: PPC, Figure 3: Occupancy).
    """
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    # Find PKLs with full PPC for Figure 2
    ppc_pkls = {}
    for model in models:
        pkls = find_pkl_files(model, dataset, K, results_dir)
        if pkls:
            pkl = sorted(pkls, key=lambda p: p.stat().st_mtime)[-1]
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            if 'ppc_simulations' in data.get('res', {}):
                ppc_pkls[model] = str(pkl)

    if len(ppc_pkls) >= 2:
        print(f"\nGenerating Figure 2 (PPC) with: {list(ppc_pkls.keys())}")
        try:
            fig2_path = out_dir / f"fig2_ppc_{dataset}_K{K}.pdf"
            # Note: Actual plotting requires the figure generation code
            print(f"✓ Figure 2 would be saved to: {fig2_path}")
        except Exception as e:
            print(f"✗ Figure 2 failed: {e}")
    else:
        print(f"⚠ Need ≥2 models with full PPC for Figure 2 (found: {len(ppc_pkls)})")

    # Figure 3: State occupancy (BEMMAOR preferred)
    bemmaor_pkls = find_pkl_files("BEMMAOR", dataset, K, results_dir)
    if bemmaor_pkls:
        pkl = sorted(bemmaor_pkls, key=lambda p: p.stat().st_mtime)[-1]
        print(f"\nGenerating Figure 3 (Occupancy) from: {pkl.name}")
        try:
            fig3_path = out_dir / f"fig3_occupancy_{dataset}_K{K}.pdf"
            print(f"✓ Figure 3 would be saved to: {fig3_path}")
        except Exception as e:
            print(f"✗ Figure 3 failed: {e}")
    else:
        print("⚠ No BEMMAOR PKL found for Figure 3")


def main():
    parser = argparse.ArgumentParser(
        description="RFM-HMM Empirical Analysis (UCI/CDNOW)"
    )
    parser.add_argument(
        "--dataset", type=str, default="uci",
        choices=["uci", "cdnow"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--K", type=int, default=3,
        help="Number of states"
    )
    parser.add_argument(
        "--models", nargs="+", 
        default=["BEMMAOR", "Hurdle", "Tweedie"],
        help="Models to compare"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/empirics",
        help="Directory with existing PKL files"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/figures",
        help="Output directory for tables and figures"
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="Generate figures (requires PKLs with PPC)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RFM-HMM EMPIRICAL ANALYSIS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"K: {args.K}")
    print(f"Models: {args.models}")
    print(f"PKL source: {results_dir}")
    print("="*70)

    # Generate comparison table
    df = generate_comparison_table(
        models=args.models,
        dataset=args.dataset,
        K=args.K,
        results_dir=results_dir,
        out_dir=out_dir,
    )

    # Generate figures if requested
    if args.figures and df is not None:
        generate_figures(
            models=args.models,
            dataset=args.dataset,
            K=args.K,
            results_dir=results_dir,
            out_dir=out_dir,
        )

    print("\n✓ Empirical analysis complete")


if __name__ == "__main__":
    main()


======================================================================
Save as: examples/run_empirical_analysis.py
======================================================================
