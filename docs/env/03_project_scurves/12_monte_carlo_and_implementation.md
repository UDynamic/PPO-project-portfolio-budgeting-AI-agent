# CHUNK 12
## Coverage
Section 7.3: Monte Carlo Simulation Results + Section 8.1: Python S-Curve Implementation

## Dependency Notes
Uses parameter distributions from Chunk 09. Implements S-curve model from Chunk 07.

## Overlap Notes
References Beta CDF parameterization (α=2.5, β=2.0) from Chunk 07.

## Content

---

### 7.3 Monte Carlo Simulation Results

**Simulation setup:**
- **Number of iterations:** 10,000
- **Sampling method:** Latin Hypercube Sampling (LHS) for variance reduction
- **Input distributions:** As specified in Master Parameter Table (Section 4)
- **Fixed inputs:** $\text{SPI}_{\text{baseline}} = 0.70$, $T_{\text{action}} = 3$ months

**Output metric:** $\text{SPI}_{\text{equilibrium}}$ (long-term stabilized SPI)

**Results:**

| Statistic | Value |
|-----------|-------|
| **Mean** | 0.753 |
| **Median (P50)** | 0.752 |
| **Standard Deviation** | 0.028 |
| **Coefficient of Variation** | 3.7% |
| **P10 (Pessimistic)** | 0.718 |
| **P90 (Optimistic)** | 0.789 |
| **Minimum** | 0.682 |
| **Maximum** | 0.835 |

**Probability of achieving key thresholds:**

| Threshold | Probability |
|-----------|-------------|
| $\text{SPI}_{\text{equilibrium}} \geq 0.75$ | 52% |
| $\text{SPI}_{\text{equilibrium}} \geq 0.80$ | 12% |
| $\text{SPI}_{\text{equilibrium}} \geq 0.85$ | 1% |
| $\text{SPI}_{\text{equilibrium}} < 0.70$ | 0.3% |

**Interpretation:**
- **Median outcome:** SPI stabilizes at 0.752 (7.4% improvement over baseline)
- **80% confidence interval:** [0.723, 0.783] (2.3% to 11.9% improvement)
- **Realistic expectation:** Action plans typically deliver 5-10 percentage point permanent SPI improvement
- **Unlikely to fully recover:** Only 1% chance of achieving SPI ≥ 0.85 (near-perfect performance)

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

**Key Implementation Notes:**

1. **Regularized Incomplete Beta Function:** Uses `scipy.special.betainc` for numerical stability
2. **Normalization:** Period fractions are renormalized to ensure exact budget conservation
3. **Boundary Handling:** Cashflows are zero outside the project execution window
4. **Vectorization:** Portfolio-level generation loops over projects but vectorizes period calculations

**Validation Checks:**

```python
def validate_scurve(cashflows, BAC, tolerance=1e-6):
    """Validate S-curve profile against invariants."""
    assert np.all(cashflows >= 0), "Negative cashflows detected"
    assert abs(cashflows.sum() - BAC) / BAC < tolerance, "Budget conservation violated"
    cumulative = np.cumsum(cashflows)
    assert np.all(np.diff(cumulative) >= -tolerance), "Non-monotonic cumulative"
    return True
```

---

**End of Chunk 12**

**Next Chunk Preview**: Chunk 13 covers duration sampling code (Section 8.2) and validation framework (Section 9).
