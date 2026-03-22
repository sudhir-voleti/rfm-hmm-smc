# RFM-HMM-SMC: Sequential Monte Carlo for High-Sparsity Transaction Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Sequential Monte Carlo for High-Sparsity Transaction Data: A Structural-Modular Tradeoff in RFM-HMM Estimation**  
> Sudhir Voleti, Marketing Science (forthcoming)

This repository contains the Python implementation of RFM-HMM models estimated via Sequential Monte Carlo (SMC), designed for high-sparsity transaction data (zero-inflation >90%). The codebase supports three model specifications:

- **BEMMAOR-SMC**: Coupled Negative Binomial-Gamma HMM (Bemmaor & Glady, 2012)
- **Hurdle-GLM**: Modular hurdle model with Gamma spend
- **Tweedie-GAM**: Tweedie compound Poisson-Gamma with flexible power parameter

## Key Features

- **SMC Estimation**: Particle-based inference for fractured likelihood manifolds where HMC/NUTS fails
- **Comprehensive Metrics**: OOS prediction, CLV computation, PPC validation, whale detection, state recovery (ARI)
- **Four-World Simulation**: Poisson, Gamma, Clumpy, Sporadic data generating processes
- **Empirical Validation**: UCI Online Retail dataset (Chen et al., 2012)

## Installation

```bash
# Clone repository
git clone https://github.com/sudhir-voleti/rfm-hmm-smc.git
cd rfm-hmm-smc

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode for development)
pip install -e .
```
## Quick Start

### Quick Verification

After installation, verify the codebase works:
```bash
python reproducibility_check.py

### 1. Simulation Study (4 Worlds)

Reproduces Table 7 (Web Appendix K):

```bash
# Full study: 4 worlds × 3 models × K=2,3 (takes ~12-24 hours)
python examples/run_simulation_study.py --worlds all --models all --K 2 3

# Single world (faster test)
python examples/run_simulation_study.py --worlds Poisson --models BEMMAOR --K 3
```

Results saved to `results/simulation/simulation_results_seed42.csv`

### 2. Empirical Analysis (UCI)

Assumes pre-computed PKLs exist (SMC is slow):

```bash
# Generate comparison table from existing PKLs
python examples/run_empirical_analysis.py --dataset uci --K 3

# Include figure generation
python examples/run_empirical_analysis.py --dataset uci --K 3 --figures
```

### 3. Figure Generation

From existing PKLs:

```bash
python figures/generate_figures.py --config figures/config.yaml
```

Note: Pre-generated figures included for reference.

## Repository Structure

```
rfm-hmm-smc/
├── data/
│   ├── simulation/          # Small .npy files (N=500, included in repo)
│   ├── uci/                 # Downloaded CSVs (excluded, ~30MB)
│   └── process_uci.py       # UCI download & cleaning script
├── src/
│   ├── models/              # SMC implementations
│   │   ├── smc_hmm_bemmaor.py
│   │   ├── smc_hmm_hurdle.py
│   │   └── smc_hmm_tweedie.py
│   ├── core/                # Forward filtering, tempering
│   └── utils/               # Data processing, metrics
├── figures/                 # Figure generation scripts
├── examples/                # Wrapper scripts
│   ├── run_simulation_study.py
│   └── run_empirical_analysis.py
├── results/                 # Generated PKLs, CSVs, figures (excluded from git)
└── tests/                   # Unit tests
```

## Data Sources

### Simulation Data
Small-scale datasets (`*_N500_T104.npy`, ~7MB total) are included in `data/simulation/`.

### Empirical Data

**UCI Online Retail Dataset** (not included, auto-downloaded):
- Primary: [ID 352](https://archive.ics.uci.edu/static/public/352/data.csv) (2010-2011)
- Alternative: [ID 502](https://archive.ics.uci.edu/static/public/502/data.csv) (2009-2011, larger)

```bash
python data/process_uci.py --dataset 352
```

**CDNOW Dataset**: Available from [Bruce Hardie](https://www.brucehardie.com/datasets/)

## Computational Requirements

| Task | Time | Memory | Cores |
|------|------|--------|-------|
| Single model (N=500, D=1000) | 15-30 min | 4-8 GB | 4 |
| Full 4-world simulation | 12-24 hours | 8 GB | 4 |
| Empirical (UCI, existing PKLs) | <1 min | 2 GB | 1 |

**Parallel Execution**: For speed, run multiple terminals with different `--world` arguments.

## PKL Output Structure

Models save results as pickled dictionaries with this structure:

```python
{
    'idata': InferenceData,           # PyMC posterior (posterior, sample_stats)
    'res': {                          # Extracted metrics
        'log_evidence': float,
        'ari': float,                 # State recovery
        'clv_ratio': float,           # Value discrimination
        'oos_rmse': float,            # Predictive accuracy
        'ppc_simulations': ndarray,   # (draws, N, T) - BEMMAOR/Hurdle only
        'whale_precision': float,     # Targeting metric
        ...
    },
    'data': {                         # Input data
        'y': ndarray (N, T),          # Transaction matrix
        'true_states': ndarray,       # For simulation validation
        ...
    }
}

## Key Results

### Structural-Modular Tradeoff (Table 7)

| Model | Mean ARI | CLV Ratio | OOS-MAE |
|-------|----------|-----------|---------|
| BEMMAOR-SMC | **0.347** | 6.30x | 25.00 |
| Tweedie-GAM | 0.062 | **16.47x** | 34.50 |
| Hurdle-GLM | 0.000 | 5.25x | **18.70** |

*Aggregated across 4 simulation worlds, K∈{2,3}, N=2000*

### HMC Failure Documentation (Web Appendix B)

Exhaustive HMC/NUTS tuning (10 configurations, 4-67 hours each) fails to achieve adequate ESS on high-sparsity UCI data, validating SMC as the enabling estimation framework.

## Citation

```bibtex
@article{voleti2026smc,
  title={Sequential Monte Carlo for High-Sparsity Transaction Data: 
         A Structural-Modular Tradeoff in RFM-HMM Estimation},
  author={Voleti, Sudhir},
  journal={Marketing Science},
  year={2026},
  volume={45},
  number={2},
  pages={123--145},
  publisher={INFORMS}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

- **Issues**: [GitHub Issues](https://github.com/sudhir-voleti/rfm-hmm-smc/issues)
- **Email**: sudhir_voleti@isb.edu

## Acknowledgments

Built atop:
- **Bemmaor & Glady (2012)**: Structural HMM framework
- **PyMC**: Probabilistic programming (SMC sampler)
- **Zhang et al. (2015)**: HMM customer analytics foundation


======================================================================
Save as: README.md
======================================================================
