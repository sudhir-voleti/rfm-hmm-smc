#!/usr/bin/env python3
"""
RFM-HMM Demo Mode - Quick Pilot Runs
======================================
Fast demonstration with configurable N, draws, etc.
WARNING: Not converged - for structure demonstration only.

Usage: python examples/run_demo.py --model BEMMAOR --world Poisson --K 2 --N 50 --draws 200 --seed 42

Author: Sudhir Voleti
Date: March 2026
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add repo root to path (flat structure - no src/ folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.smc_hmm_bemmaor import run_smc_bemmaor
from models.smc_hmm_hurdle import run_smc_hurdle
from models.smc_hmm_tweedie import run_smc
from utils.data_utils import load_simulation_data

def run_demo(model: str, world: str, K: int, N: int, draws: int, 
             chains: int, seed: int, out_dir: Path):
    """
    Run quick demo with warnings about non-convergence.
    """
    print("=" * 70)
    print("⚠️  DEMO MODE - NOT FOR PRODUCTION")
    print("=" * 70)
    print(f"Model: {model} | World: {world} | K={K} | N={N}")
    print(f"Draws: {draws} (vs 1000+ for production)")
    print(f"Chains: {chains}")
    print(f"Seed: {seed}")
    print("⚠️  WARNING: Results are NOT converged.")
    print(" Use only for: structure check, metric validation, debugging.")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "simulation"

    # Auto-detect T from available data
    # Try to find matching file
    world_cap = world.capitalize()
    import os
    sim_files = list((Path(__file__).parent.parent / "data" / "simulation").glob(f"hmm_{world_cap}_N{N}_T*_seed{seed}.csv"))

    if sim_files:
        # Extract T from filename
        fname = sim_files[0].name
        T = int(fname.split(f"_N{N}_T")[1].split("_")[0])
        print(f"  Detected T={T} from data file")
    else:
        T = 20  # Default
        print(f"  Using default T={T}")

    try:
        data = load_simulation_data(world, N, T, seed, data_dir)
    except FileNotFoundError as e:
        print(f"✗ Data not found: {e}")
        print(f"\nGenerate data first:")
        print(f"  python data/simulation/create_subset.py --world {world} --N {N} --T 20 --seed {seed}")
        return None

    # DEBUG: Check data
    print("\n🔍 Data check:")
    print(f"  y shape: {data['y'].shape}, range: [{data['y'].min():.2f}, {data['y'].max():.2f}]")
    print(f"  y NaN: {np.isnan(data['y']).sum()}, Inf: {np.isinf(data['y']).sum()}")
    print(f"  zero rate: {(data['y'] == 0).mean():.1%}")

    # Run model
    model_funcs = {
        "BEMMAOR": run_smc_bemmaor,
        "Hurdle": run_smc_hurdle,
        "Tweedie": run_smc,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Running {model}...")
    try:
        result = model_funcs[model](
            data=data,
            K=K,
            draws=draws,
            chains=chains,
            seed=seed,
            out_dir=out_dir,
        )

        print(f"\n✓ Demo complete!")
        print(f" PKL: {result[0]}")  # pkl_path

        res = result[1]  # res dict
        print(f"\n📊 Quick metrics (UNCONVERGED):")
        print(f" Log-Ev: {res.get('log_evidence', 'N/A')}")
        print(f" Time: {res.get('time_min', 'N/A'):.1f} min")

        if 'ari' in res:
            print(f" ARI: {res.get('ari', 'N/A')}")

        return result

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Demo mode: Quick pilot runs (NOT converged)"
    )
    parser.add_argument(
        "--model", type=str, default="BEMMAOR",
        choices=["BEMMAOR", "Hurdle", "Tweedie"],
        help="Model to demo"
    )
    parser.add_argument(
        "--world", type=str, default="Poisson",
        choices=["Poisson", "Gamma", "Clumpy", "Sporadic"],
        help="Simulation world"
    )
    parser.add_argument(
        "--K", type=int, default=2,
        help="Number of states"
    )
    parser.add_argument(
        "--N", type=int, default=50,
        help="Number of customers (default: 50)"
    )
    parser.add_argument(
        "--draws", type=int, default=200,
        help="Number of SMC draws/particles (default: 200)"
    )
    parser.add_argument(
        "--chains", type=int, default=2,
        help="Number of chains (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: results/demo)"
    )

    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = Path(__file__).parent.parent / "results" / "demo"
    else:
        out_dir = Path(args.out_dir)

    run_demo(args.model, args.world, args.K, args.N, args.draws, 
             args.chains, args.seed, out_dir)

if __name__ == "__main__":
    main()
