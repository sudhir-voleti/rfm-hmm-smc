#!/usr/bin/env python3
"""
RFM-HMM Simulation Study Runner
================================
Reproduces 4-world simulation results (Table 7, Web Appendix K).
Single-threaded execution. For parallel runs, use multiple terminals
with different --world arguments.

Author: Sudhir Voleti
Date: March 2026
Repository: https://github.com/sudhir-voleti/rfm-hmm-smc
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.smc_hmm_bemmaor import run_smc_bemmaor
from models.smc_hmm_hurdle import run_smc_hurdle
from models.smc_hmm_tweedie import run_smc_tweedie


# Configuration
WORLDS = ["Poisson", "Gamma", "Clumpy", "Sporadic"]
MODELS = {
    "BEMMAOR": run_smc_bemmaor,
    "Hurdle": run_smc_hurdle,
    "Tweedie": run_smc_tweedie,
}

DEFAULT_CONFIG = {
    "N": 500,
    "T": 104,
    "K_values": [2, 3],
    "draws": 1000,
    "chains": 4,
    "cores": 4,
}


def load_simulation_data(world: str, N: int, T: int, seed: int, data_dir: Path) -> Dict:
    """
    Load simulation data from .npy files.

    Expected files: {world}_N{N}_T{T}_seed{seed}.npy
    Contains: y (N,T), true_states (N,T), params (dict)
    """
    filename = data_dir / f"{world.lower()}_N{N}_T{T}_seed{seed}.npy"

    if not filename.exists():
        raise FileNotFoundError(
            f"Simulation data not found: {filename}\n"
            f"Generate via: python data/simulation/generate_simulation.py "
            f"--world {world} --N {N} --T {T} --seed {seed}"
        )

    data = np.load(filename, allow_pickle=True).item()
    return {
        "y": data["y"],
        "true_states": data["true_states"],
        "params": data.get("params", {}),
        "world": world,
        "N": N,
        "T": T,
    }


def run_single_config(
    model_name: str,
    world: str,
    K: int,
    config: Dict,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> Dict:
    """
    Run single model-world-K configuration.

    Returns
    -------
    results : dict with metrics (log_ev, ARI, CLV_ratio, OOS_RMSE, etc.)
    """
    print(f"\n{'='*70}")
    print(f"Running: {model_name} | {world} | K={K} | seed={seed}")
    print(f"{'='*70}")

    # Load data
    try:
        data = load_simulation_data(world, config["N"], config["T"], seed, data_dir)
    except FileNotFoundError as e:
        print(f"✗ SKIPPED: {e}")
        return None

    # Run model
    model_func = MODELS[model_name]

    start_time = time.time()
    try:
        result = model_func(
            data=data,
            K=K,
            draws=config["draws"],
            chains=config["chains"],
            cores=config["cores"],
            seed=seed,
            out_dir=out_dir,
        )
        elapsed = time.time() - start_time

        # Extract key metrics
        metrics = {
            "model": model_name,
            "world": world,
            "K": K,
            "seed": seed,
            "log_evidence": result.get("log_evidence", np.nan),
            "ari": result.get("ari", np.nan),
            "clv_ratio": result.get("clv_ratio", np.nan),
            "oos_rmse": result.get("oos_rmse", np.nan),
            "oos_mae": result.get("oos_mae", np.nan),
            "time_min": elapsed / 60,
            "status": "SUCCESS",
        }

        print(f"✓ Completed in {elapsed/60:.1f} min")
        print(f"  Log-Ev: {metrics['log_evidence']:.1f}, ARI: {metrics['ari']:.3f}")

        return metrics

    except Exception as e:
        print(f"✗ FAILED: {str(e)[:100]}")
        return {
            "model": model_name,
            "world": world,
            "K": K,
            "seed": seed,
            "status": f"FAILED: {str(e)[:50]}",
        }


def run_full_study(
    worlds: List[str],
    models: List[str],
    K_values: List[int],
    config: Dict,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run complete simulation study across all configurations.
    """
    results = []

    for world in worlds:
        for model in models:
            for K in K_values:
                result = run_single_config(
                    model_name=model,
                    world=world,
                    K=K,
                    config=config,
                    data_dir=data_dir,
                    out_dir=out_dir,
                    seed=seed,
                )
                if result:
                    results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = out_dir / f"simulation_results_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Mean ARI by Model-World")
    print("="*70)
    summary = df.groupby(["model", "world"])["ari"].mean().unstack()
    print(summary.round(3))

    return df


def main():
    parser = argparse.ArgumentParser(
        description="RFM-HMM Simulation Study (4 Worlds × 3 Models × K=2,3)"
    )
    parser.add_argument(
        "--worlds", nargs="+", choices=WORLDS + ["all"],
        default=["all"], help="Simulation worlds to run"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()) + ["all"],
        default=["all"], help="Models to run"
    )
    parser.add_argument(
        "--K", nargs="+", type=int, default=[2, 3],
        help="State-space cardinalities"
    )
    parser.add_argument(
        "--N", type=int, default=500, help="Number of customers"
    )
    parser.add_argument(
        "--T", type=int, default=104, help="Number of time periods"
    )
    parser.add_argument(
        "--draws", type=int, default=1000, help="SMC particle draws"
    )
    parser.add_argument(
        "--chains", type=int, default=4, help="SMC chains"
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="CPU cores for sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (in PKL filename)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/simulation",
        help="Directory with .npy simulation files"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/simulation",
        help="Output directory for PKLs and CSV"
    )

    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand "all"
    worlds = WORLDS if "all" in args.worlds else args.worlds
    models = list(MODELS.keys()) if "all" in args.models else args.models

    # Build config
    config = {
        "N": args.N,
        "T": args.T,
        "K_values": args.K,
        "draws": args.draws,
        "chains": args.chains,
        "cores": args.cores,
    }

    print("="*70)
    print("RFM-HMM SIMULATION STUDY")
    print("="*70)
    print(f"Worlds: {worlds}")
    print(f"Models: {models}")
    print(f"K: {args.K}")
    print(f"N={args.N}, T={args.T}, draws={args.draws}")
    print(f"Seed: {args.seed} (in PKL filename)")
    print(f"Data: {data_dir}")
    print(f"Output: {out_dir}")
    print("="*70)

    # Run study
    df = run_full_study(
        worlds=worlds,
        models=models,
        K_values=args.K,
        config=config,
        data_dir=data_dir,
        out_dir=out_dir,
        seed=args.seed,
    )

    print("\n✓ Simulation study complete")
    print(f"Results: {out_dir}/simulation_results_seed{args.seed}.csv")


if __name__ == "__main__":
    main()


======================================================================
Save as: examples/run_simulation_study.py
======================================================================
