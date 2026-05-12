# CHUNK 09
## Coverage
Section 5: Parameter Calibration (Master Tables & Distributions)

## Dependency Notes
Consolidates parameters from all previous chunks. Reference for implementation.

## Overlap Notes
None (comprehensive reference table)

## Content

---

## 5. Parameter Calibration

### 5.1 Master Parameter Table — S-Curve

| Parameter | Symbol | Baseline Value | Range | Distribution | Source |
|-----------|--------|----------------|-------|--------------|--------|
| **S-Curve Shape** |
| Alpha (front-loading) | α | 2.0 | [1.5, 3.0] | Fixed | Barraza (2011) |
| Beta (tail-off) | β | 2.5 | [2.0, 4.0] | Fixed | Mubarak (2015) |
| **Duration Model** |
| Scaling constant | γ | 2.5 | [2.0, 3.5] | Fixed | Merrow (2011) |
| Elasticity | δ | 0.35 | [0.30, 0.45] | Fixed | AACE (2020) |
| Log-duration std dev | σ_ln | 0.40 | [0.30, 0.50] | Fixed | Flyvbjerg et al. (2018) |
| **Start Time** |
| Start time distribution | t_start | Uniform(0,12) | - | Uniform | Khanzadi et al. (2018) |

**Notes**:
- Baseline values are **deterministic** in foundational model
- Ranges provided for sensitivity analysis (Section 7)
- All duration parameters in months, BAC in millions USD

---

### 5.2 Sensitivity Ranges

For Monte Carlo sensitivity analysis, the following parameter variations are recommended:

| Scenario | α | β | γ | δ | σ_ln | Interpretation |
|----------|---|---|---|---|------|----------------|
| **Baseline** | 2.0 | 2.5 | 2.5 | 0.35 | 0.40 | Industry standard |
| **Front-loaded** | 1.5 | 2.0 | 2.5 | 0.35 | 0.40 | Aggressive early spending |
| **Back-loaded** | 3.0 | 4.0 | 2.5 | 0.35 | 0.40 | Conservative ramp-up |
| **Fast execution** | 2.0 | 2.5 | 2.0 | 0.30 | 0.30 | Optimistic durations |
| **Slow execution** | 2.0 | 2.5 | 3.5 | 0.45 | 0.50 | Pessimistic durations |

---

### 5.3 Probability Distributions for Model Parameters

For **stochastic parameter modeling** (advanced extension), the following distributions are recommended:

#### 5.3.1 Retention Rate (ρ)

**Distribution**: Beta distribution
$$\rho \sim \text{Beta}(\alpha_\rho, \beta_\rho)$$

**Calibration**:
- Mean: $E[\rho] = 0.45$
- Standard deviation: $\sigma_\rho = 0.10$
- Implied parameters: $\alpha_\rho = 8.1, \beta_\rho = 9.9$
- Support: [0, 1]

**Rationale**: Beta distribution naturally bounded on [0,1], flexible shape.

#### 5.3.2 Decay Rate (λ_decay)

**Distribution**: Gamma distribution
$$\lambda_{\text{decay}} \sim \text{Gamma}(k, \theta)$$

**Calibration**:
- Mean: $E[\lambda_{\text{decay}}] = 0.12$ per month
- Standard deviation: $\sigma = 0.03$
- Implied parameters: $k = 16, \theta = 0.0075$
- Support: (0, ∞)

**Rationale**: Gamma distribution ensures positive values, right-skewed (allows for occasional fast decay).

#### 5.3.3 Action Plan Effectiveness (η)

**Distribution**: Truncated normal
$$\eta \sim \mathcal{N}(\mu_\eta, \sigma_\eta) \text{ truncated to } [0.10, 0.30]$$

**Calibration**:
- Mean: $\mu_\eta = 0.18$
- Standard deviation: $\sigma_\eta = 0.04$
- Truncation: [0.10, 0.30]

**Rationale**: Normal distribution reflects measurement uncertainty, truncation enforces physical bounds.

#### 5.3.4 Action Plan Cost Factor (κ)

**Distribution**: Lognormal
$$\kappa \sim \text{Lognormal}(\mu_{\ln,\kappa}, \sigma_{\ln,\kappa})$$

**Calibration**:
- Median: $\exp(\mu_{\ln,\kappa}) = 0.05$ (5% of remaining BAC)
- Coefficient of variation: 0.40
- Implied parameters: $\mu_{\ln,\kappa} = -2.996, \sigma_{\ln,\kappa} = 0.385$

**Rationale**: Cost factors are naturally right-skewed (occasional high-cost interventions).

#### 5.3.5 Stabilization Period (T_stabilize)

**Distribution**: Discrete uniform
$$T_{\text{stabilize}} \sim \text{DiscreteUniform}(4, 9) \text{ months}$$

**Rationale**: Limited empirical data, uniform distribution reflects epistemic uncertainty.

---

### 5.4 Master Parameter Table — Post-Action Plan Dynamics

| Parameter | Symbol | Baseline | Range | Distribution | Source |
|-----------|--------|----------|-------|--------------|--------|
| **Action Plan** |
| Effectiveness | η | 0.18 | [0.12, 0.25] | Truncated Normal | Kim et al. (2003) |
| Duration | T_action | 3 months | [2, 4] | Fixed | PMI (2019) |
| Ramp-up rate | λ_ramp | 0.50/month | [0.30, 0.70] | Fixed | Calibrated |
| Cost factor | κ | 0.05 | [0.03, 0.08] | Lognormal | Fleming & Koppelman (2016) |
| **Post-Action Dynamics** |
| Retention rate | ρ | 0.45 | [0.35, 0.55] | Beta | Cross-study synthesis |
| Decay rate | λ_decay | 0.12/month | [0.08, 0.18] | Gamma | Kim et al. (2003) |
| Stabilization period | T_stabilize | 6 months | [4, 9] | Discrete Uniform | Fleming & Koppelman (2016) |
| Half-life | t_1/2 | 6 months | [4, 8] | Derived | $\ln(2)/\lambda_{\text{decay}}$ |

**Implementation notes**:
- Baseline values are **point estimates** for deterministic simulation
- Distributions are for **probabilistic sensitivity analysis** (PSA)
- All time parameters in months
- Cost factor κ is fraction of remaining project BAC

---

**End of Chunk 09**

**Next Chunk Preview**: Chunk 10 covers action plan financial justification (Section 6), including cost-benefit analysis and BCR calculations.
