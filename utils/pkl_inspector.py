# pkl_inspector.py
import pickle
import numpy as np

def inspect_pkl(pkl_path):
    """Display table of contents for a PKL file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"PKL: {pkl_path}")
    print(f"{'='*60}")
    
    print(f"\nTop-level keys: {list(data.keys())}")
    
    if 'res' in data:
        print(f"\n--- Results (res) ---")
        for k, v in data['res'].items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {type(v).__name__} {v.shape}")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"  {k}: {type(v).__name__}")
    
    if 'idata' in data:
        print(f"\n--- InferenceData (idata) ---")
        print(f"Groups: {list(data['idata'].groups())}")
        for group in data['idata'].groups():
            grp = getattr(data['idata'], group)
            print(f"  {group}: {list(grp.data_vars)[:5]}...")
    
    if 'data' in data:
        print(f"\n--- Data ---")
        for k, v in data['data'].items():
            shape = f" {v.shape}" if hasattr(v, 'shape') else ""
            print(f"  {k}: {type(v).__name__}{shape}")

if __name__ == "__main__":
    import sys
    inspect_pkl(sys.argv[1])
