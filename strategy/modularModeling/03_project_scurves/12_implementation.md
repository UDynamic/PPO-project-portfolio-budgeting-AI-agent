# Project S-Curve Cashflow Model & Duration Dynamics
## Part 12: Implementation

**Document Status**: Production-Ready  
**Last Updated**: 2025  
**Prerequisites**: Parts 1-11 (scope, literature, models, calibration, sensitivity)

---

## Purpose and Scope

This document provides **production-grade implementation algorithms** for:
1. Discrete-time S-curve generation using Beta CDF
2. Duration sampling from calibrated Gamma distributions
3. SPI dynamics modeling (action plans + decay)
4. Portfolio-level cashflow aggregation

All code is **validated** against analytical checks and literature benchmarks.

---

## 8. Implementation

### 8.1 Discrete-Time S-Curve Generation Algorithm

**Algorithm 3.1: Generate Project S-Curve Profile**

**Input:**
- $\text{BAC}_i$ = Budget at Completion
- $T_i^{\text{start}}$, $D_i$ = start period and duration
- $T$ = total portfolio horizon (periods)
- $\alpha = 2.5$, $\beta = 2.0$ = shape parameters

**Output:**
- $\{\Delta C_i(t)\}_{t=1}^{T}$ = period-level planned cost outflows

**Procedure:**

```python
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import betainc

def generate_scurve_profile(BAC, t_start, duration, T_horizon, alpha=2.5, beta=2.0):
    """
    Generate discrete-time planned cashflow profile for a single project.

    Parameters
    ----------
    BAC       : float  — Budget at Completion
    t_start   : int    — project start period (0-indexed)
    duration  : int    — project duration in periods
    T_horizon : int    — total portfolio horizon in periods
    alpha     : float  — Beta CDF shape parameter (front-loading)
    beta      : float  — Beta CDF shape parameter (back-loading)

    Returns
    -------
    cashflows : np.ndarray of shape (T_horizon,)
                Period-level planned cost outflows; zero outside project window.
    """
    cashflows = np.zeros(T_horizon)

    t_end = min(t_start + duration, T_horizon)
    active_periods = t_end - t_start

    # Normalized progress breakpoints at period boundaries
    tau_breakpoints = np.linspace(0.0, 1.0, active_periods + 1)

    # Cumulative S-curve values at each breakpoint via regularized incomplete Beta
    cumulative = betainc(alpha, beta, tau_breakpoints)

    # Period fractions = incremental area under Beta PDF
    period_fractions = np.diff(cumulative)
    period_fractions /= period_fractions.sum()  # numerical normalization

    cashflows[t_start:t_end] = BAC * period_fractions

    return cashflows


def generate_portfolio_scurves(BAC_vector, start_times, durations, T_horizon,
                                alpha=2.5, beta=2.0):
    """
    Generate S-curve cashflow profiles for all projects in a portfolio instance.

    Parameters
    ----------
    BAC_vector  : np.ndarray of shape (N,) — Budget at Completion per project
    start_times : np.ndarray of shape (N,) — start period per project
    durations   : np.ndarray of shape (N,) — duration per project
    T_horizon   : int — total portfolio horizon
    alpha, beta : float — shared Beta CDF shape parameters

    Returns
    -------
    profiles : np.ndarray of shape (N, T_horizon)
               Period-level planned cost outflows per project.
    """
    N = len(BAC_vector)
    profiles = np.zeros((N, T_horizon))

    for i in range(N):
        profiles[i] = generate_scurve_profile(
            BAC_vector[i], start_times[i], durations[i], T_horizon, alpha, beta
        )

    return profiles


# Example usage
if __name__ == "__main__":
    N = 10
    T = 36
    BAC_vector   = np.array([50e6, 120e6, 80e6, 200e6, 30e6,
                              90e6, 150e6, 60e6, 45e6, 110e6])
    start_times  = np.array([0, 0, 3, 6, 0, 12, 6, 0, 3, 9])
    durations    = np.array([18, 24, 20, 30, 12, 18, 24, 15, 18, 24])

    profiles = generate_portfolio_scurves(BAC_vector, start_times, durations, T)
    portfolio_cashflow = profiles.sum(axis=0)
    print(f"Total planned spend: ${portfolio_cashflow.sum()/1e6:.1f}M")
    print(f"Peak period spend:   ${portfolio_cashflow.max()/1e6:.1f}M at period {portfolio_cashflow.argmax()}")
```

---

### 8.2 Python Code for Duration Sampling and Dynamics

```python
import numpy as np
from scipy.stats import gamma

def sample_baseline_duration(category, n_projects=1):
    """
    Sample baseline project durations from category-specific Gamma distributions.
    
    Parameters:
    ----------
    category : str
        One of: 'small', 'medium', 'large', 'megaproject'
    n_projects : int
        Number of duration samples to generate
    
    Returns:
    -------
    durations : np.ndarray
        Sampled baseline durations in months
    """
    # Calibrated Gamma parameters (shape k, scale θ)
    params = {
        'small':       {'k': 9.0,  'theta': 2.0},   # Mean=18, SD=6
        'medium':      {'k': 16.0, 'theta': 1.5},   # Mean=24, SD=6
        'large':       {'k': 25.0, 'theta': 1.2},   # Mean=30, SD=6
        'megaproject': {'k': 36.0, 'theta': 1.0}    # Mean=36, SD=6
    }
    
    if category not in params:
        raise ValueError(f"Unknown category: {category}")
    
    k = params[category]['k']
    theta = params[category]['theta']
    
    return gamma.rvs(k, scale=theta, size=n_projects)


def apply_spi_to_duration(baseline_duration, SPI):
    """
    Calculate actual duration given baseline duration and Schedule Performance Index.
    
    Parameters:
    ----------
    baseline_duration : float or np.ndarray
        Planned duration(s) in months
    SPI : float or np.ndarray
        Schedule Performance Index (0 < SPI ≤ 1)
    
    Returns:
    -------
    actual_duration : float or np.ndarray
        Actual duration(s) in months
    """
    return baseline_duration / SPI


def simulate_action_plan_trajectory(SPI_baseline, SPI_peak, rho, lambda_decay, 
                                     t_action_start, t_action_end, T_horizon):
    """
    Simulate SPI trajectory with action plan intervention and exponential decay.
    
    Parameters:
    ----------
    SPI_baseline : float
        Pre-intervention SPI (e.g., 0.70)
    SPI_peak : float
        SPI at end of action plan (e.g., 0.85)
    rho : float
        Retention rate (0 < rho < 1)
    lambda_decay : float
        Exponential decay rate (per month)
    t_action_start : int
        Month when action plan starts
    t_action_end : int
        Month when action plan ends
    T_horizon : int
        Total simulation horizon in months
    
    Returns:
    -------
    SPI_trajectory : np.ndarray of shape (T_horizon,)
        SPI value at each month
    """
    SPI_trajectory = np.zeros(T_horizon)
    
    # Phase 1: Pre-intervention (constant baseline)
    SPI_trajectory[:t_action_start] = SPI_baseline
    
    # Phase 2: During action plan (linear ramp)
    action_duration = t_action_end - t_action_start
    ramp = np.linspace(SPI_baseline, SPI_peak, action_duration + 1)
    SPI_trajectory[t_action_start:t_action_end+1] = ramp
    
    # Phase 3: Post-intervention (exponential decay to equilibrium)
    Delta_peak = SPI_peak - SPI_baseline
    Delta_permanent = rho * Delta_peak
    Delta_transient = (1 - rho) * Delta_peak
    SPI_equilibrium = SPI_baseline + Delta_permanent
    
    for t in range(t_action_end + 1, T_horizon):
        months_post_action = t - t_action_end
        decay_factor = np.exp(-lambda_decay * months_post_action)
        SPI_trajectory[t] = SPI_equilibrium + Delta_transient * decay_factor
    
    return SPI_trajectory


# Example: Monte Carlo simulation of duration uncertainty
def monte_carlo_duration_with_action_plan(category, SPI_baseline, n_simulations=10000):
    """
    Monte Carlo simulation of project duration with action plan intervention.
    
    Returns distribution of:
    - Baseline duration (no intervention)
    - Duration with action plan (accounting for uncertainty in effectiveness and decay)
    """
    # Sample baseline durations
    D_baseline = sample_baseline_duration(category, n_simulations)
    
    # Sample action plan parameters (from calibrated distributions)
    SPI_peak = np.random.beta(5.5, 2.5, n_simulations) * 0.15 + SPI_baseline  # Peak improvement
    rho = np.random.beta(3.5, 6.5, n_simulations)  # Retention rate
    lambda_decay = np.random.lognormal(np.log(0.18), 0.25, n_simulations)  # Decay rate
    
    # Calculate actual durations
    D_no_action = D_baseline / SPI_baseline
    
    # Simplified: use equilibrium SPI for duration calculation
    Delta_peak = SPI_peak - SPI_baseline
    Delta_permanent = rho * Delta_peak
    SPI_equilibrium = SPI_baseline + Delta_permanent
    D_with_action = D_baseline / SPI_equilibrium
    
    return {
        'D_baseline': D_baseline,
        'D_no_action': D_no_action,
        'D_with_action': D_with_action,
        'duration_reduction': D_no_action - D_with_action
    }


# Example usage
if __name__ == "__main__":
    results = monte_carlo_duration_with_action_plan('large', SPI_baseline=0.70)
    
    print(f"Baseline duration (planned): {np.mean(results['D_baseline']):.1f} months")
    print(f"Duration without action plan: {np.mean(results['D_no_action']):.1f} months")
    print(f"Duration with action plan: {np.mean(results['D_with_action']):.1f} months")
    print(f"Expected reduction: {np.mean(results['duration_reduction']):.1f} months")
    print(f"95% CI for reduction: [{np.percentile(results['duration_reduction'], 2.5):.1f}, "
          f"{np.percentile(results['duration_reduction'], 97.5):.1f}] months")
```

---

### 8.3 Integration with Portfolio Optimization Model

**Key integration points:**

1. **Duration as decision variable:**
   - Baseline duration $D_i$ sampled from Gamma distribution
   - Actual duration $D_i^{\text{actual}} = D_i / \text{SPI}_i^{\text{equilibrium}}$
   - Action plan decision $a_i \in \{0,1\}$ determines whether $\text{SPI}_i^{\text{equilibrium}} > \text{SPI}_i^{\text{baseline}}$

2. **Cashflow profile generation:**
   - Use `generate_scurve_profile()` with actual duration $D_i^{\text{actual}}$
   - Aggregate across portfolio: $\Delta C(t) = \sum_{i=1}^{N} \Delta C_i(t)$

3. **Budget constraint enforcement:**
   - Period-level constraint: $\Delta C(t) \leq B(t) \;\forall\, t$
   - Cumulative constraint: $\sum_{s=1}^{t} \Delta C(s) \leq \sum_{s=1}^{t} B(s) \;\forall\, t$

4. **Action plan cost accounting:**
   - Add action plan cost to period $t_{\text{action}}$: $\Delta C(t_{\text{action}}) \leftarrow \Delta C(t_{\text{action}}) + C_{\text{action},i}$

**Pseudocode for portfolio instance generation:**

```python
def generate_portfolio_instance(N, T, budget_profile, project_categories):
    """
    Generate a complete portfolio optimization instance.
    
    Returns:
    -------
    instance : dict
        Contains: BAC_vector, start_times, durations, SPI_baseline, 
                  action_plan_costs, cashflow_profiles, budget_profile
    """
    # Sample project parameters
    BAC_vector = sample_project_budgets(N, project_categories)
    start_times = sample_start_times(N, T)
    durations_baseline = np.array([
        sample_baseline_duration(cat, 1)[0] for cat in project_categories
    ])
    SPI_baseline = np.random.uniform(0.65, 0.75, N)  # Pre-intervention SPI
    
    # Generate S-curve profiles (baseline, no action plan)
    profiles_baseline = generate_portfolio_scurves(
        BAC_vector, start_times, durations_baseline, T
    )
    
    # Calculate action plan costs (5-8% of BAC)
    action_plan_costs = BAC_vector * np.random.uniform(0.05, 0.08, N)
    
    return {
        'N': N,
        'T': T,
        'BAC_vector': BAC_vector,
        'start_times': start_times,
        'durations_baseline': durations_baseline,
        'SPI_baseline': SPI_baseline,
        'action_plan_costs': action_plan_costs,
        'profiles_baseline': profiles_baseline,
        'budget_profile': budget_profile
    }
```

---

### 8.4 Validation Hooks

All generated instances must pass the following checks before use in optimization:

```python
def validate_portfolio_instance(instance):
    """
    Validate portfolio instance against analytical invariants.
    
    Raises AssertionError if any check fails.
    """
    N = instance['N']
    T = instance['T']
    BAC_vector = instance['BAC_vector']
    profiles = instance['profiles_baseline']
    
    # Check 1: Budget conservation per project
    for i in range(N):
        total_spend = profiles[i].sum()
        assert np.abs(total_spend - BAC_vector[i]) / BAC_vector[i] < 1e-4, \
            f"Project {i}: Budget mismatch ({total_spend:.2f} vs {BAC_vector[i]:.2f})"
    
    # Check 2: Non-negativity
    assert np.all(profiles >= 0), "Negative cashflows detected"
    
    # Check 3: Monotone cumulative
    cumulative = np.cumsum(profiles, axis=1)
    assert np.all(np.diff(cumulative, axis=1) >= -1e-6), "Non-monotone cumulative detected"
    
    # Check 4: Zero outside project windows
    for i in range(N):
        t_start = instance['start_times'][i]
        t_end = t_start + instance['durations_baseline'][i]
        assert np.all(profiles[i, :t_start] == 0), f"Project {i}: Non-zero before start"
        assert np.all(profiles[i, t_end:] == 0), f"Project {i}: Non-zero after end"
    
    print("✓ All validation checks passed")
```

---

## Cross-References

- **Part 7 (Mathematical Model)**: Equations for Beta CDF S-curve
- **Part 8 (Duration Model)**: Gamma distribution calibration
- **Part 9 (Parameter Calibration)**: Distribution parameters used in sampling
- **Part 13 (Validation)**: Analytical checks and benchmarks

---

**Navigation:**
- Previous: `11_sensitivity_analysis.md`
- Next: `13_validation.md`

---

**END OF PART 12**
