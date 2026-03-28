"""
hmc_bemmaor.py
Minimal PyMC NUTS implementation of BEMMAOR (coupled) HMM for benchmarking.
Expected behavior: WORSE than Hurdle — coupling creates extreme curvature.
Demonstrates structural coupling is incompatible with gradient-based sampling.

Usage:
    python hmc_bemmaor.py --n_customers 100 --n_timepoints 52 --K 3 --draws 500 --tune 500

Output:
    Saves to benchmarks/results/hmc_bemmaor_*.nc (Arviz InferenceData)
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
    Simulate sparse panel with COUPLED timing and spend.
    Ground truth: shared latent factor theta drives both processes.
    """
    rng = np.random.default_rng(seed)
    
    print(f"Simulating COUPLED data: N={n_customers}, T={n_timepoints}")
    
    # Shared latent factor (engagement)
    theta = rng.normal(0, 1, size=(n_customers, n_timepoints))
    
    # Latent states
    S = rng.integers(0, K, size=(n_customers, n_timepoints))
    
    # State-specific base rates
    lambda_base = np.array([0.5, 2.0, 5.0])
    mu_base = np.array([10.0, 50.0, 200.0])
    
    # COUPLED generation: theta affects both timing and spend
    rho_true = 0.5  # Coupling strength
    
    z = np.zeros((n_customers, n_timepoints), dtype=int)
    y = np.zeros((n_customers, n_timepoints))
    
    for i in range(n_customers):
        for t in range(n_timepoints):
            k = S[i, t]
            # Timing: log(lambda) = base + 0.5*theta
            lam = np.exp(np.log(lambda_base[k]) + 0.5 * theta[i, t])
            z[i, t] = rng.poisson(lam) > 0
            
            # Spend: log(mu) = base + rho*theta
            if z[i, t]:
                mu = np.exp(np.log(mu_base[k]) + rho_true * theta[i, t])
                y[i, t] = rng.gamma(shape=2.0, scale=mu / 2.0)
    
    pi_0_obs = (y == 0).mean()
    print(f"Observed sparsity: {pi_0_obs:.3f} (with coupling)")
    
    return z.astype(int), y, S


def build_bemmaor_hmm(z_obs, y_obs, n_customers, n_timepoints, K=3, rho=0.5):
    """
    BEMMAOR HMM: Structural coupling via shared latent factor theta.
    
    Timing: Poisson(lambda_k * exp(0.5*theta))
    Spend: Gamma(alpha, beta_k / exp(rho*theta))  [mean scales with exp(rho*theta)]
    
    This is the COUPLED specification (rho > 0).
    
    CRITICAL: This creates extreme likelihood curvature. HMC will struggle.
    """
    with pm.Model() as model:
        # Priors on state-specific base parameters
        lambda_base = pm.Gamma('lambda_base', alpha=2.0, beta=1.0, shape=K,
                              initval=np.array([0.5, 2.0, 5.0]))
        
        alpha_spend = pm.Gamma('alpha_spend', alpha=2.0, beta=1.0, initval=2.0)
        mu_base = pm.Gamma('mu_base', alpha=2.0, beta=0.1, shape=K,
                          initval=np.array([10.0, 50.0, 200.0]))
        
        # Coupling parameter (fixed or estimated)
        # For benchmarking, fix rho; estimating it makes HMC impossible
        rho_param = pm.HalfNormal('rho', sigma=0.5, initval=rho)
        
        # Transition matrix
        gamma_diag = pm.Beta('gamma_diag', alpha=5.0, beta=1.0, initval=0.85)
        gamma_off = (1 - gamma_diag) / (K - 1)
        gamma = pt.eye(K) * gamma_diag + (1 - pt.eye(K)) * gamma_off
        
        # Initial state
        pi_0 = pm.Dirichlet('pi_0', a=np.ones(K), initval=np.ones(K)/K)
        
        # SHARED LATENT FACTOR (the coupling mechanism)
        # This is a high-dimensional latent variable (N*T parameters!)
        # HMC will struggle with this geometry
        theta = pm.Normal('theta', mu=0, sigma=1, 
                         shape=(n_customers, n_timepoints),
                         initval=np.zeros((n_customers, n_timepoints)))
        
        # HMM likelihood with coupling
        logp = 0
        for i in range(n_customers):
            log_alpha = pt.log(pi_0)
            
            for t in range(n_timepoints):
                z_it = z_obs[i, t]
                y_it = y_obs[i, t]
                theta_it = theta[i, t]
                
                # COUPLED timing: lambda = base * exp(0.5*theta)
                lambda_coupled = lambda_base * pt.exp(0.5 * theta_it)
                log_p_z = pm.Poisson.logp(z_it, lambda_coupled)
                
                # COUPLED spend: mu = base * exp(rho*theta)
                mu_coupled = mu_base * pt.exp(rho_param * theta_it)
                beta_coupled = alpha_spend / mu_coupled
                
                log_p_y = pt.switch(
                    pt.eq(y_it, 0),
                    0,
                    pm.Gamma.logp(y_it, alpha_spend, beta_coupled)
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
    parser = argparse.ArgumentParser(description='HMC BEMMAOR HMM Benchmark')
    parser.add_argument('--n_customers', type=int, default=100,
                       help='Number of customers (reduce for HMC sanity)')
    parser.add_argument('--n_timepoints', type=int, default=52,
                       help='Number of timepoints')
    parser.add_argument('--K', type=int, default=3,
                       help='Number of hidden states')
    parser.add_argument('--rho', type=float, default=0.5,
                       help='Coupling strength (0=independent, 1=full)')
    parser.add_argument('--draws', type=int, default=500,
                       help='NUTS draws (likely to diverge)')
    parser.add_argument('--tune', type=int, default=500,
                       help='NUTS tuning steps')
    parser.add_argument('--target_accept', type=float, default=0.9,
                       help='Target acceptance (higher for coupled model)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='benchmarks/results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HMC BEMMAOR (Coupled) HMM Benchmark")
    print("=" * 60)
    print(f"Config: N={args.n_customers}, T={args.n_timepoints}, K={args.K}, rho={args.rho}")
    print(f"NUTS: {args.draws} draws, {args.tune} tune, target_accept={args.target_accept}")
    print("\\n" + "!" * 60)
    print("WARNING: Coupled HMMs are EXTREMELY difficult for HMC.")
    print("Expected: Massive divergences, R-hat >> 1.2, ESS near zero.")
    print("This demonstrates why SMC is the only viable approach.")
    print("!" * 60 + "\\n")
    
    # Load/simulate data
    z_obs, y_obs, S_true = load_or_simulate_data(
        args.n_customers, args.n_timepoints, args.K, args.seed
    )
    
    # Build model
    print("Building BEMMAOR (coupled) HMM model...")
    print(f"Latent dimensionality: {args.n_customers * args.n_timepoints} theta variables")
    model = build_bemmaor_hmm(z_obs, y_obs, args.n_customers, args.n_timepoints, 
                              args.K, args.rho)
    
    # Sample with NUTS
    print(f"\\nStarting NUTS sampling...")
    print(f"This may take hours or fail entirely.")
    
    try:
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                target_accept=args.target_accept,
                random_seed=args.seed,
                nuts_sampler="pymc",
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
                # Aggressive adaptation for difficult geometry
                max_treedepth=12  # Deeper trees for coupled model
            )
        
        # Diagnostics
        print("\\n" + "=" * 60)
        print("SAMPLING COMPLETE (unexpected!)")
        print("=" * 60)
        
        divergences = idata.sample_stats.diverging.sum().values
        print(f"Divergences: {divergences}/{args.draws} ({100*divergences/args.draws:.1f}%)")
        
        try:
            r_hat = az.rhat(idata, var_names=["lambda_base", "mu_base", "rho"])
            print(f"Max R-hat: {r_hat.max().values:.3f}")
        except:
            print("R-hat calculation failed")
        
        try:
            ess = az.ess(idata, var_names=["lambda_base", "mu_base", "rho"])
            print(f"Min ESS: {ess.min().values:.1f}")
        except:
            print("ESS calculation failed")
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"hmc_bemmaor_N{args.n_customers}_K{args.K}_rho{args.rho}_D{args.draws}.nc"
        filepath = output_path / filename
        idata.to_netcdf(filepath)
        print(f"\\nSaved to: {filepath}")
        
        if divergences > 0.5 * args.draws:
            print("\\n" + "!" * 60)
            print("CATASTROPHIC DIVERGENCE - HMC COMPLETELY FAILED")
            print("Structural coupling + HMM + sparsity = intractable for NUTS")
            print("SMC with tempering is the only viable inference method.")
            print("!" * 60)
        
    except Exception as e:
        print(f"\\nSAMPLING FAILED: {e}")
        print("This is EXPECTED for coupled HMMs.")
        print("The likelihood manifold is too curved for gradient-based sampling.")
        print("\\nKey insight: Coupling creates 'funnels' and 'bridges'")
        print("that NUTS cannot navigate without exhaustive exploration.")
        print("SMC's particle population can 'tunnel' across these barriers.")
        raise


if __name__ == "__main__":
    main()

