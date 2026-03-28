# HMC Benchmarks

These scripts demonstrate why gradient-based sampling fails for sparse HMMs,
motivating the SMC approach used in the main paper.

## Expected Behavior

- `hmc_hurdle.py`: Runs but produces divergences (10-50%), poor ESS
- `hmc_bemmaor.py`: Catastrophic failure (&gt;50% divergences, R-hat &gt; 2)

## Usage

```bash
python hmc_hurdle.py --n_customers 100 --draws 500
python hmc_bemmaor.py --n_customers 100 --rho 0.5 --draws 500 \
    --data_path "/path/to/uci_500.csv"
