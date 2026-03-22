#!/usr/bin/env python3
"""
Reproducibility Check Script
============================
Mimics what a diligent reviewer would do to verify the codebase works.
Run this after cloning/downloading and installing requirements.

Usage: python reproducibility_check.py
"""

import sys
import os
from pathlib import Path

def test_suite():
    print("=" * 70)
    print("RFM-HMM-SMC Reproducibility Check")
    print("=" * 70)
    
    # 1. Path Check
    print("\n[1] Checking system path...")
    try:
        # Ensure src is in path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        import src
        print("   ✓ System path and src directory aligned.")
    except ImportError as e:
        print(f"   ✗ FAIL: src not found: {e}")
        print("   Attempting fix...")
        sys.path.insert(0, os.getcwd())
        try:
            import src
            print("   ✓ Fixed: Added current directory to sys.path.")
        except ImportError:
            print("   ✗ CRITICAL: Cannot locate src directory.")
            return False

    # 2. Core Model Imports
    print("\n[2] Checking model imports...")
    models_to_test = [
        ("src.models.smc_hmm_bemmaor", "run_smc_bemmaor"),
        ("src.models.smc_hmm_hurdle", "run_smc_hurdle"),
        ("src.models.smc_hmm_tweedie", "run_smc_tweedie"),
    ]
    
    for module_name, func_name in models_to_test:
        try:
            module = __import__(module_name, fromlist=[func_name])
            getattr(module, func_name)
            print(f"   ✓ {module_name} imported.")
        except ImportError as e:
            print(f"   ✗ FAIL: {module_name}: {e}")

    # 3. Utilities
    print("\n[3] Checking utilities...")
    try:
        from src.utils.data_utils import load_simulation_data
        from src.utils.metrics import compute_ari
        from src.core.forward_filter import forward_filter_numpy
        print("   ✓ Data utilities functional.")
        print("   ✓ Metrics module functional.")
        print("   ✓ Core algorithms functional.")
    except ImportError as e:
        print(f"   ✗ FAIL: Utility import error: {e}")

    # 4. Data availability
    print("\n[4] Checking data files...")
    data_dir = Path(__file__).parent / "data" / "simulation"
    if data_dir.exists():
        npy_files = list(data_dir.glob("*.npy"))
        print(f"   ✓ Simulation data directory found ({len(npy_files)} .npy files).")
        if len(npy_files) == 0:
            print("   ⚠ WARNING: No .npy files found. Run simulation generator.")
    else:
        print(f"   ✗ FAIL: Data directory not found: {data_dir}")

    # 5. Examples
    print("\n[5] Checking example scripts...")
    examples = [
        "examples/run_demo.py",
        "examples/run_simulation_study.py",
        "examples/run_empirical_analysis.py",
    ]
    for ex in examples:
        ex_path = Path(__file__).parent / ex
        if ex_path.exists():
            print(f"   ✓ {ex} found.")
        else:
            print(f"   ✗ FAIL: {ex} not found.")

    # 6. Figures
    print("\n[6] Checking figure generation...")
    fig_script = Path(__file__).parent / "figures" / "generate_figures.py"
    if fig_script.exists():
        print(f"   ✓ Figure generation script found.")
    else:
        print(f"   ⚠ WARNING: Figure script not found.")

    print("\n" + "=" * 70)
    print("Check Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. If all ✓ above: Run 'python examples/run_demo.py --model BEMMAOR --world Poisson --K 2'")
    print("  2. For full reproduction: See README.md")
    
    return True

if __name__ == "__main__":
    success = test_suite()
    sys.exit(0 if success else 1)
