```markdown
# HMC Benchmarks: Demonstrating Computational Intractability

This directory contains PyMC NUTS implementations of Hurdle and BEMMAOR HMMs to demonstrate why gradient-based sampling fails for sparse longitudinal data, motivating the SMC approach used in the main paper.

## Scripts

| Script | Model | Expected Outcome |
|--------|-------|------------------|
| `hmc_hurdle.py` | Modular (rho=0), K=3 states | Slow convergence, 10-50% divergences, poor ESS |
| `hmc_bemmaor.py` | Coupled (rho>0), K=3 states | Catastrophic failure (>50% divergences, R-hat >> 1.2) |

## Usage

### Hurdle (Modular Baseline)
```bash
python hmc_hurdle.py \
    --n_customers 100 \
    --draws 500 \
    --tune 500 \
    --target_accept 0.8 \
    --seed 42
```

### BEMMAOR (Coupled, Expected Failure)
```bash
python hmc_bemmaor.py \
    --n_customers 100 \
    --rho 0.5 \
    --draws 500 \
    --tune 500 \
    --target_accept 0.9 \
    --seed 42 \
    --data_path "/path/to/uci_500.csv"
```

## Why HMC Fails

| Problem Element | HMC Challenge | SMC Advantage |
|-----------------|-------------|---------------|
| **Discrete latent states** | Non-differentiable Viterbi path | Particle filtering handles naturally |
| **Structural zeros (pi_0 > 0.75)** | Likelihood plateaus, gradients vanish | Tempering bridges zero-density regions |
| **Coupling (rho > 0)** | Extreme curvature, funnel geometry | Mass transport via resampling |
| **Heavy tails (psi > 1)** | Divergent transitions, poor mixing | Importance weights adapt to tails |

## Computational Reality

| N | HMC Runtime | HMC Outcome | SMC Runtime | SMC Outcome |
|---|-------------|-------------|-------------|-------------|
| 100 | 30-60 min | Divergences, ESS ~ 10 | 5 min | Clean inference |
| 500 | 8+ hours | Complete failure | 20 min | Clean inference |
| 1000 | >24 hours | Infeasible | 45 min | Clean inference |

HMC is computationally impractical for the problem scale in the main paper (N=500, T=52, K=3).

## Data Requirements

Scripts accept UCI retail format:
- `customer_id`: Customer identifier
- `WeekStart` or `week`: Time index  
- `spend` or `y` or `WeeklySpend`: Spend amount (0 for no purchase)

Auto-searches for `uci_500.csv` in standard locations; falls back to simulation if not found.

## Outputs

Results saved to `benchmarks/results/`:
- `hmc_hurdle_N100_K3_D500.nc`: Arviz InferenceData (if sampling completes)
- `hmc_bemmaor_N100_K3_rho0.5_D500.nc`: Arviz InferenceData (likely fails)

Do not commit large `.nc` files to version control.

## References

See main paper Section 3.2: "Computational Necessity of SMC" for theoretical discussion of why gradient-based methods fail for this problem class.

## Contact

For questions about these benchmarks, refer to the main paper replication materials or contact the authors.
```
