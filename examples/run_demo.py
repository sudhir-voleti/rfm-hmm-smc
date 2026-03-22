#!/usr/bin/env python3
"""
RFM-HMM Demo Mode - Quick Pilot Runs
======================================
Fast demonstration with N=50, D=200 (runs in ~2-3 minutes).
WARNING: Not converged - for structure demonstration only.

Use this to:
- Verify PKL/idata structure
- Test metric computations
- Validate installation
- Preview results before full runs

Usage: python examples/run_demo.py --model BEMMAOR --world Poisson --K 3

Author: Sudhir Voleti
Date: March 2026
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.smc_hmm_bemmaor import run_smc_bemmaor
from models.smc_hmm_hurdle import run_smc_hurdle
from models.smc_hmm_tweedie import run_smc_tweedie
from utils.data_utils import load_simulation_data


DEMO_CONFIG = {
    "N": 50,
    "T": 20,  # Shorter time series
    "draws": 200,  # Fewer particles
    "chains": 2,   # Fewer chains
    "cores": 2,
}


def run_demo(model: str, world: str, K: int, seed: int = 42):
    """
    Run quick demo with warnings about non-convergence.
    """
    print("="*70)
    print("⚠️  DEMO MODE - NOT FOR PRODUCTION")
    print("="*70)
    print(f"Model: {model} | World: {world} | K={K} | N={DEMO_CONFIG['N']}")
    print(f"Draws: {DEMO_CONFIG['draws']} (vs 1000 for production)")
    print(f"T: {DEMO_CONFIG['T']} (vs 104 for production)")
    print("
⚠️  WARNING: Results are NOT converged.")
    print("   Use only for: structure check, metric validation, debugging.")
    print("="*70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "simulation"
    try:
        data = load_simulation_data(world, DEMO_CONFIG["N"], DEMO_CONFIG["T"], seed, data_dir)
    except FileNotFoundError:
        print(f"✗ Data not found. Generate first:")
        print(f"   python data/simulation/generate_simulation.py \")
        print(f"       --world {world} --N {DEMO_CONFIG['N']} --T {DEMO_CONFIG['T']} --seed {seed}")
        return

    # Run model
    model_funcs = {
        "BEMMAOR": run_smc_bemmaor,
        "Hurdle": run_smc_hurdle,
        "Tweedie": run_smc_tweedie,
    }

    out_dir = Path(__file__).parent.parent / "results" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Running {model}...")
    try:
        result = model_funcs[model](
            data=data,
            K=K,
            draws=DEMO_CONFIG["draws"],
            chains=DEMO_CONFIG["chains"],
            cores=DEMO_CONFIG["cores"],
            seed=seed,
            out_dir=out_dir,
        )

        print(f"\n✓ Demo complete!")
        print(f"   PKL: {out_dir}/smc_K{K}_{model}_N{DEMO_CONFIG['N']}_demo.pkl")
        print(f"\n📊 Quick metrics (UNCONVERGED):")
        print(f"   Log-Ev: {result.get('log_evidence', 'N/A'):.1f}")
        print(f"   ARI: {result.get('ari', 'N/A'):.3f}")
        print(f"   Time: {result.get('time_min', 'N/A'):.1f} min")

        print(f"\n📁 To inspect idata structure:")
        print(f"   python -c \"")
        print(f"   import pickle;")
        print(f"   data = pickle.load(open('{out_dir}/smc_K{K}_{model}_N{DEMO_CONFIG['N']}_demo.pkl', 'rb'));")
        print(f"   print(data['idata'].groups())")
        print(f"   \"")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Demo mode: Quick pilot runs (2-3 min, NOT converged)"
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
        "--K", type=int, default=3,
        help="Number of states"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    run_demo(args.model, args.world, args.K, args.seed)


if __name__ == "__main__":
    main()
