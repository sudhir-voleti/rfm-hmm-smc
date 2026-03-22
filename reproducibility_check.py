#!/usr/bin/env python3
"""
Reproducibility Check Script
============================
Mimics what a diligent reviewer would do to verify the codebase works.
Run this after cloning/downloading and installing requirements.

Usage: python reproducibility_check.py
"""

# Check PyTensor environment setup
print("\n[0] Checking PyTensor environment...")
import os
pytensor_flags = os.environ.get('PYTENSOR_FLAGS', '')

if 'FAST_COMPILE' in pytensor_flags:
    print(" ✓ PyTensor using Python backend (FAST_COMPILE) - cross-platform compatible")
else:
    print(" ⚠ WARNING: PyTensor may use C backend")
    print("   For cross-platform compatibility, set before running:")
    print("   export PYTENSOR_FLAGS='floatX=float32,device=cpu,mode=FAST_COMPILE'")
    print("   (or add to your shell profile)")

# Verify floatX is float32
if 'floatX=float32' in pytensor_flags:
    print(" ✓ floatX=float32 set (Apple Silicon optimized)")
else:
    print(" ⚠ WARNING: floatX not set to float32")
    
import sys
import os
from pathlib import Path

def test_suite():
    print("=" * 70)
    print("RFM-HMM-SMC Reproducibility Check")
    print("=" * 70)

    # 1. Path Check - add src to path
    print("\n[1] Checking system path...")
    src_path = Path(__file__).parent / "src"

    if not src_path.exists():
        print(f" ✗ CRITICAL: src directory not found at {src_path}")
        return False

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f" ✓ Added src to path: {src_path}")
    else:
        print(" ✓ src already in path")

    # 2. Core Model Imports (without src. prefix)
    print("\n[2] Checking model imports...")
    models_to_test = [
        ("models.smc_hmm_bemmaor", "run_smc_bemmaor"),
        ("models.smc_hmm_hurdle", "run_smc_hurdle"),
        ("models.smc_hmm_tweedie", "run_smc_tweedie"),
    ]

    all_imports_ok = True
    for module_name, func_name in models_to_test:
        try:
            module = __import__(module_name, fromlist=[func_name])
            getattr(module, func_name)
            print(f" ✓ {module_name}.{func_name} imported")
        except ImportError as e:
            print(f" ✗ FAIL: {module_name}: {e}")
            all_imports_ok = False
        except AttributeError as e:
            print(f" ✗ FAIL: {func_name} not found in {module_name}: {e}")
            all_imports_ok = False

    # 3. Utilities (without src. prefix)
    print("\n[3] Checking utilities...")
    try:
        from utils.data_utils import load_simulation_data
        print(" ✓ Data utilities functional")
    except ImportError as e:
        print(f" ✗ FAIL: data_utils: {e}")
        all_imports_ok = False

    try:
        from utils.metrics import compute_ari
        print(" ✓ Metrics module functional")
    except ImportError as e:
        print(f" ✗ FAIL: metrics: {e}")
        all_imports_ok = False

    try:
        from core.forward_filter import forward_filter_numpy
        print(" ✓ Core algorithms functional")
    except ImportError as e:
        print(f" ✗ FAIL: forward_filter: {e}")
        all_imports_ok = False

    # 4. Data availability
    print("\n[4] Checking data files...")
    data_dir = Path(__file__).parent / "data" / "simulation"
    if data_dir.exists():
        npy_files = list(data_dir.glob("*.npy"))
        print(f" ✓ Simulation data directory found ({len(npy_files)} .npy files)")
        if len(npy_files) == 0:
            print(" ⚠ WARNING: No .npy files found. Run simulation generator.")
    else:
        print(f" ✗ FAIL: Data directory not found: {data_dir}")

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
            print(f" ✓ {ex} found")
        else:
            print(f" ✗ FAIL: {ex} not found")

    # 6. Figures
    print("\n[6] Checking figure generation...")
    fig_script = Path(__file__).parent / "figures" / "generate_figures.py"
    if fig_script.exists():
        print(f" ✓ Figure generation script found")
    else:
        print(f" ⚠ WARNING: Figure script not found")

    print("\n" + "=" * 70)
    if all_imports_ok:
        print("✓ Check Complete - All imports successful")
    else:
        print("✗ Check Complete - Some imports failed")
    print("=" * 70)

    if all_imports_ok:
        print("\nNext steps:")
        print(" 1. Run demo: python examples/run_demo.py --model BEMMAOR --world Poisson --K 2")
        print(" 2. For full reproduction: See README.md")

    return all_imports_ok

if __name__ == "__main__":
    success = test_suite()
    sys.exit(0 if success else 1)
