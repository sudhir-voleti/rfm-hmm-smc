# RFM-HMM-SMC

Reproducible Code for "Physics of RFM: Hidden Markov Models for Customer Dynamics"

## Quick Start (2 minutes)

```bash
# Clone and enter directory
cd rfm-hmm-smc

# Install dependencies
pip install -r requirements.txt

# Run demo (BEMMAOR model, 50 customers, 2 states, ~2 minutes)
python models/smc_hmm_bemmaor.py \
    --csv_path data/simulation/hmm_Poisson_N1000_T52.csv \
    --K 2 --T 20 --N 50 --draws 200
```

**Expected output:** PKL file in `results/demo/` with log-evidence, CLV, and whale detection metrics.

## Repository Structure

```
rfm-hmm-smc/
├── models/              # Core model implementations
│   ├── smc_hmm_bemmaor.py    # BEMMAOR (NBD-Gamma HMM)
│   ├── smc_hmm_hurdle.py     # Hurdle-Gamma HMM
│   └── smc_hmm_tweedie.py    # Tweedie HMM
├── data/
│   ├── simulation/      # Simulation worlds (Poisson, Gamma, Sporadic, Clumpy)
│   └── empirics/        # UCI Online Retail data
├── utils/               # Helper functions
├── figures/             # Figure generation scripts
├── reproducibility_check.py  # Verify installation
└── README.md           # This file
```

## Requirements

- Python 3.11
- PyMC 5.26.1
- PyTensor 2.35.1
- NumPy, Pandas, SciPy, ArviZ

See `requirements.txt` for complete list.

## Usage

### Simulation Study

```bash
# BEMMAOR with different worlds
python models/smc_hmm_bemmaor.py --csv_path data/simulation/hmm_Poisson_N1000_T52.csv --K 2 --T 20 --N 50 --draws 200
python models/smc_hmm_bemmaor.py --csv_path data/simulation/hmm_Gamma_N1000_T52.csv --K 3 --T 20 --N 50 --draws 200

# Hurdle model
python models/smc_hmm_hurdle.py --csv_path data/simulation/hmm_Poisson_N1000_T52.csv --K 2 --T 20 --N 50 --draws 200

# Tweedie model
python models/smc_hmm_tweedie.py --csv_path data/simulation/hmm_Poisson_N1000_T52.csv --K 2 --T 20 --N 50 --draws 200
```

### Empirical Analysis (UCI)

```bash
# UCI Online Retail (1MB subset included)
python models/smc_hmm_bemmaor.py \
    --csv_path data/empirics/uci_500.csv \
    --K 3 --T 42 --N 500 --draws 200 --dataset uci
```

### Verify Installation

```bash
python reproducibility_check.py
```

Should show: ✓ All model imports successful

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv_path` | Path to data CSV | Required |
| `--K` | Number of states (2-4) | Required |
| `--T` | Time periods | 20 |
| `--N` | Number of customers | 50 |
| `--draws` | SMC particles | 200 |
| `--chains` | Parallel chains | 4 |
| `--seed` | Random seed | 42 |
| `--out_dir` | Output directory | results/demo |

## Output

Each run produces:
- **PKL file**: Full model results (idata, parameters, predictions)
- **Log-evidence**: Model fit metric
- **CLV by state**: Customer Lifetime Value per latent state
- **Whale detection**: Precision/Recall for high-value customer identification
- **PPC**: Posterior predictive simulations

## Data Availability

- **Simulation data**: Generated from HMM DGP with known ground truth
- **UCI data**: Subset of UCI Machine Learning Repository Online Retail dataset (1MB)
  - Original: 541,909 transactions, 4,339 customers
  - Subset: 500 customers, 53 weeks
  - Source: https://archive.ics.uci.edu/ml/datasets/online+retail

## Citation

If using this code, please cite the associated paper (under review at Marketing Science).

## License

MIT License - See LICENSE file

## Contact

For questions about replication: sudhir.voleti@isb.edu
