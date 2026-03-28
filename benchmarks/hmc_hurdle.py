"""
hmc_hurdle.py
Minimal PyMC NUTS implementation of Hurdle HMM for benchmarking.
Expected behavior: Slow convergence, divergences, poor ESS in high-sparsity regimes.
Demonstrates why SMC is necessary for this problem class.

Usage:
    python hmc_hurdle.py --n_customers 100 --n_timepoints 52 --K 3 --draws 500 --tune 500

Output:
    Saves to benchmarks/results/hmc_hurdle_*.nc (Arviz InferenceData)
"""

import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_or_simulate_data(n_customers=100, n_timepoints=52, K=3, seed=42):
    """
    Load UCI data if available, else simulate sparse panel.
    For HMC benchmarking, simulation is preferred (known ground truth).
    """
    rng = np.random.default_rng(seed)
    
    # Simulate sparse data: pi_0 ~ 0.75
    # Simplified: no state dynamics, just population heterogeneity
    # For HMM with K=3, we'd need transition matrix - keeping it simple for HMC baseline
    
    print(f"Simulating data: N={n_customers}, T={n_timepoints}, sparsity~0.75")
    
    # Latent states (simplified: random assignment for HMC test)
    S = rng.integers(0, K, size=(n_customers, n_timepoints))
    
    # Timing (binary activity)
    lambda_by_state = np.array([0.5, 2.0, 5.0])  # Dormant, Active, Whale
    z = rng.poisson(lambda_by_state[S]) > 0
    
    # Spend (gamma, only when active)
    mu_by_state = np.array([10.0, 50.0, 200.0])
    y = np.zeros((n_customers, n_timepoints))
    for i in range(n_customers):
        for t in range(n_timepoints):
            if z[i, t]:
                y[i, t] = rng.gamma(shape=2.0, scale=mu_by_state[S[i, t]] / 2.0)
    
    pi_0_obs = (y == 0).mean()
    print(f"Observed sparsity: {pi_0_obs:.3f}")
    
    return z.astype(int), y, S


def build_hurdle_hmm(z_obs, y_obs, n_customers, n_timepoints, K=3):
    """
    Hurdle HMM: Independent timing and spend, conditional on latent state.
    
    Timing: Poisson(lambda_k) -> binary z
    Spend: Gamma(alpha, beta_k) if z > 0, else 0
    
    This is the MODULAR specification (rho = 0).
    """
    with pm.Model() as model:
        # Priors on state-specific parameters
        # Timing rates
        lambda_ = pm.Gamma('lambda', alpha=2.0, beta=1.0, shape=K, 
                          initval=np.array([0.5, 2.0, 5.0]))
        
        # Spend parameters (Gamma mean = alpha/beta, so beta = alpha/mean)
        alpha_spend = pm.Gamma('alpha_spend', alpha=2.0, beta=1.0, 
                               initval=2.0)  # Shared shape
        mu_spend = pm.Gamma('mu_spend', alpha=2.0, beta=0.1, shape=K,
                           initval=np.array([10.0, 50.0, 200.0]))
        beta_spend = alpha_spend / mu_spend
        
        # Transition matrix (sticky)
        gamma_diag = pm.Beta('gamma_diag', alpha=5.0, beta=1.0, 
                            initval=0.85)
        gamma_off = (1 - gamma_diag) / (K - 1)
        
        gamma = pt.eye(K) * gamma_diag + (1 - pt.eye(K)) * gamma_off
        
        # Initial state distribution
        pi_0 = pm.Dirichlet('pi_0', a=np.ones(K), initval=np.ones(K)/K)
        
        # HMM likelihood via forward algorithm
        # This is where HMC struggles: discrete latent states
        logp = 0
        for i in range(n_customers):
            # Forward pass for customer i
            log_alpha = pt.log(pi_0)  # log forward message
            
            for t in range(n_timepoints):
                # Emission: P(z, y | S=k)
                z_it = z_obs[i, t]
                y_it = y_obs[i, t]
                
                # Timing likelihood
                log_p_z = pm.Poisson.logp(z_it, lambda_)
                
                # Spend likelihood (hurdle)
                log_p_y = pt.switch(
                    pt.eq(y_it, 0),
                    0,  # If z=0, y must be 0 (deterministic)
                    pm.Gamma.logp(y_it, alpha_spend, beta_spend)
                )
                
                log_emission = log_p_z + log_p_y
                
                # Forward update
                log_alpha = log_emission + pt.logsumexp(
                    pt.log(gamma) + log_alpha[:, None], axis=0
                )
            
            logp += pt.logsumexp(log_alpha)
        
        pm.Potential('loglik', logp)
        
    return model


def main():
    parser = argparse.ArgumentParser(description='HMC Hurdle HMM Benchmark')
    parser.add_argument('--n_customers', type=int, default=100,
                       help='Number of customers (reduce for HMC sanity)')
    parser.add_argument('--n_timepoints', type=int, default=52,
                       help='Number of timepoints')
    parser.add_argument('--K', type=int, default=3,
                       help='Number of hidden states')
    parser.add_argument('--draws', type=int, default=500,
                       help='NUTS draws (reduce if divergences)')
    parser.add_argument('--tune', type=int, default=500,
                       help='NUTS tuning steps')
    parser.add_argument('--target_accept', type=float, default=0.8,
                       help='Target acceptance rate (increase if divergences)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='benchmarks/results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HMC Hurdle HMM Benchmark")
    print("=" * 60)
    print(f"Config: N={args.n_customers}, T={args.n_timepoints}, K={args.K}")
    print(f"NUTS: {args.draws} draws, {args.tune} tune, target_accept={args.target_accept}")
    print("\nWARNING: This will likely produce divergences and poor ESS.")
    print("This is expected behavior demonstrating HMC limitations.\\n")
    
    # Load/simulate data
    z_obs, y_obs, S_true = load_or_simulate_data(
        args.n_customers, args.n_timepoints, args.K, args.seed
    )
    
    # Build model
    print("Building Hurdle HMM model...")
    model = build_hurdle_hmm(z_obs, y_obs, args.n_customers, args.n_timepoints, args.K)
    
    # Sample with NUTS
    print(f"\\nStarting NUTS sampling...")
    print(f"Expected runtime: 30+ minutes for N=100, much longer for N=500")
    
    try:
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                target_accept=args.target_accept,
                random_seed=args.seed,
                nuts_sampler="pymc",  # Default, slower but stable
                # nuts_sampler="nutpie",  # Faster if installed
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True}
            )
        
        # Diagnostics
        print("\\n" + "=" * 60)
        print("SAMPLING COMPLETE")
        print("=" * 60)
        
        divergences = idata.sample_stats.diverging.sum().values
        print(f"Divergences: {divergences}/{args.draws} ({100*divergences/args.draws:.1f}%)")
        
        try:
            ess = az.ess(idata)
            print(f"Min ESS: {ess.min().values:.1f}")
            print(f"Mean ESS: {ess.mean().values:.1f}")
        except:
            print("ESS calculation failed (common with poor mixing)")
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"hmc_hurdle_N{args.n_customers}_K{args.K}_D{args.draws}.nc"
        filepath = output_path / filename
        idata.to_netcdf(filepath)
        print(f"\\nSaved to: {filepath}")
        
        # Summary for comparison
        summary = az.summary(idata, var_names=["lambda", "mu_spend"])
        print("\\nParameter Summary:")
        print(summary[["mean", "sd", "r_hat"]].head(10))
        
        if divergences > 0.1 * args.draws:
            print("\\n" + "!" * 60)
            print("HIGH DIVERGENCE RATE - HMC FAILED")
            print("This demonstrates why SMC is necessary for this problem.")
            print("!" * 60)
        
    except Exception as e:
        print(f"\\nSAMPLING FAILED: {e}")
        print("This is expected for HMC on HMMs with sparse data.")
        print("SMC is the recommended alternative.")
        raise


if __name__ == "__main__":
    main()
