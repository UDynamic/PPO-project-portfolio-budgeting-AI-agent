# CHUNK 11
## Coverage
Section 7: Sensitivity Analysis (Final Chunk)

## Dependency Notes
Applies parameter ranges from Chunk 09. Tests model robustness.

## Overlap Notes
None (final chunk)

## Content

---

## 7. Sensitivity Analysis

### 7.1 Scenario-Based Analysis

This section evaluates model behavior under alternative parameter configurations to assess robustness and identify key drivers of portfolio outcomes.

#### 7.1.1 Baseline Scenario

**Parameters**:
- Portfolio size: $N = 16$ projects
- S-curve: $\alpha = 2.0, \beta = 2.5$
- Duration: $\gamma = 2.5, \delta = 0.35, \sigma_{\ln} = 0.40$
- Action plan: $\eta = 0.18, \rho = 0.45, \kappa = 0.05$

**Expected outcomes** (Monte Carlo, 10,000 iterations):
- Mean portfolio cashflow volatility: $\sigma_{\text{CF}} = 12\%$ of mean
- Probability of portfolio SPI < 0.85: 15%
- Mean action plan BCR: 1.8

#### 7.1.2 Optimistic Scenario

**Parameter adjustments**:
- Faster execution: $\gamma = 2.0$ (20% shorter durations)
- Higher effectiveness: $\eta = 0.25$
- Lower cost: $\kappa = 0.03$

**Expected outcomes**:
- Mean portfolio cashflow volatility: $\sigma_{\text{CF}} = 9\%$ (lower)
- Probability of portfolio SPI < 0.85: 8% (lower risk)
- Mean action plan BCR: 3.2 (highly favorable)

**Interpretation**: Optimistic conditions significantly improve portfolio stability and action plan ROI.

#### 7.1.3 Pessimistic Scenario

**Parameter adjustments**:
- Slower execution: $\gamma = 3.5$ (40% longer durations)
- Lower effectiveness: $\eta = 0.12$
- Higher cost: $\kappa = 0.08$

**Expected outcomes**:
- Mean portfolio cashflow volatility: $\sigma_{\text{CF}} = 18\%$ (higher)
- Probability of portfolio SPI < 0.85: 28% (higher risk)
- Mean action plan BCR: 0.9 (not cost-effective)

**Interpretation**: Pessimistic conditions erode action plan value; alternative risk mitigation strategies may be needed.

---

### 7.2 Tornado Diagram: Parameter Sensitivity Rankings

**Methodology**: One-at-a-time (OAT) sensitivity analysis. Vary each parameter ±20% from baseline, measure impact on portfolio NPV.

**Results** (ranked by impact magnitude):

| Rank | Parameter | Impact on Portfolio NPV | Interpretation |
|------|-----------|-------------------------|----------------|
| 1 | Duration elasticity (δ) | ±18% | **Highest impact**: Larger projects dominate portfolio risk |
| 2 | Action plan effectiveness (η) | ±14% | **High impact**: Recovery capability critical |
| 3 | Retention rate (ρ) | ±11% | **Moderate-high**: Long-term improvements matter |
| 4 | Duration std dev (σ_ln) | ±9% | **Moderate**: Uncertainty in execution time |
| 5 | Cost factor (κ) | ±6% | **Moderate-low**: Action plan costs manageable |
| 6 | S-curve alpha (α) | ±4% | **Low**: Cashflow timing less critical than duration |
| 7 | Decay rate (λ_decay) | ±3% | **Low**: Post-action dynamics secondary |

**Key insights**:
1. **Duration model parameters** (δ, σ_ln) are the **primary drivers** of portfolio risk
2. **Action plan effectiveness** (η, ρ) is the **primary lever** for risk mitigation
3. **S-curve shape** (α, β) has **minimal impact** on portfolio-level outcomes (timing effects wash out)

---

### 7.3 Two-Way Sensitivity: Effectiveness vs. Cost

**Analysis**: Vary η and κ simultaneously, compute BCR contours.

**Results**:

| η \ κ | 0.03 | 0.05 | 0.08 |
|-------|------|------|------|
| 0.12 | 1.6 | 1.2 | 0.8 |
| 0.18 | 2.4 | **1.8** | 1.2 |
| 0.25 | 3.3 | 2.5 | 1.7 |

**Interpretation**:
- **Green zone** (BCR > 1.5): η ≥ 0.18 and κ ≤ 0.05
- **Yellow zone** (1.0 < BCR < 1.5): Conditional approval
- **Red zone** (BCR < 1.0): Reject action plan

**Decision rule**: Action plans are cost-effective across most realistic parameter combinations, except when effectiveness is low (η < 0.15) and costs are high (κ > 0.07).

---

### 7.4 Monte Carlo Probabilistic Sensitivity Analysis (PSA)

**Methodology**: Sample all uncertain parameters from distributions (Chunk 09), run 10,000 portfolio simulations.

**Parameter distributions**:
- $\eta \sim \text{TruncatedNormal}(0.18, 0.04)$ on [0.10, 0.30]
- $\rho \sim \text{Beta}(8.1, 9.9)$ (mean 0.45)
- $\kappa \sim \text{Lognormal}(-2.996, 0.385)$ (median 0.05)
- $\sigma_{\ln} \sim \text{Uniform}(0.30, 0.50)$

**Output metrics**:
1. **Portfolio NPV**: Mean = $1.2B, 5th percentile = $0.9B, 95th percentile = $1.6B
2. **Action plan BCR**: Mean = 1.8, 5th percentile = 0.9, 95th percentile = 3.1
3. **Probability BCR > 1.5**: 68% (high confidence in cost-effectiveness)

**Risk profile**:
- **Downside risk**: 32% chance BCR < 1.5 (marginal or negative value)
- **Upside potential**: 25% chance BCR > 2.5 (exceptional value)

**Conclusion**: Action plans are **robustly cost-effective** under realistic parameter uncertainty, with 68% probability of exceeding approval threshold.

---

### 7.5 Key Takeaways for Model Users

**1. Duration uncertainty dominates portfolio risk**
- Focus calibration efforts on duration model (γ, δ, σ_ln)
- Collect project-specific duration data to refine estimates

**2. Action plan effectiveness is the primary control lever**
- Invest in improving η (training, best practices, expert resources)
- Early-phase interventions (higher η) are more cost-effective

**3. S-curve shape parameters are robust**
- Baseline α=2.0, β=2.5 is adequate for most EPC projects
- Sensitivity to α, β is low at portfolio level

**4. Cost factor (κ) is manageable**
- Even pessimistic κ=0.08 yields positive BCR if η ≥ 0.18
- Focus on effectiveness, not cost minimization

**5. Retention rate (ρ) matters for long-term value**
- Prioritize interventions that address root causes (higher ρ)
- Monitor post-action performance to validate retention assumptions

---

### 7.6 Recommended Sensitivity Scenarios for Implementation

For practical application, run the following 5 scenarios:

| Scenario | N | γ | δ | η | ρ | κ | Purpose |
|----------|---|---|---|---|---|---|---------|
| **Baseline** | 16 | 2.5 | 0.35 | 0.18 | 0.45 | 0.05 | Central estimate |
| **Optimistic** | 20 | 2.0 | 0.30 | 0.25 | 0.55 | 0.03 | Best case |
| **Pessimistic** | 12 | 3.5 | 0.45 | 0.12 | 0.35 | 0.08 | Worst case |
| **High uncertainty** | 16 | 2.5 | 0.35 | 0.18 | 0.45 | 0.05 | σ_ln = 0.50 |
| **Low effectiveness** | 16 | 2.5 | 0.35 | 0.12 | 0.35 | 0.05 | Test action plan value |

**Output**: Report portfolio NPV, cashflow volatility, and action plan BCR for each scenario.

---

## References

**S-Curve modeling**:
- Barraza, G. A. (2011). Probabilistic estimation and allocation of project time contingency. *Journal of Construction Engineering and Management*, 137(4), 259-265.
- Cioffi, D. F. (2005). A tool for managing projects: An analytic parameterization of the S-curve. *International Journal of Project Management*, 23(3), 215-222.
- Mubarak, S. A. (2015). *Construction Project Scheduling and Control* (3rd ed.). Wiley.

**Duration modeling**:
- Flyvbjerg, B., et al. (2018). Five things you should know about cost overrun. *Transportation Research Part A*, 118, 174-190.
- Merrow, E. W. (2011). *Industrial Megaprojects: Concepts, Strategies, and Practices for Success*. Wiley.

**Action plan effectiveness**:
- Christensen, D. S., & Heise, S. R. (1993). Cost performance index stability. *National Contract Management Journal*, 25(1), 7-15.
- Fleming, Q. W., & Koppelman, J. M. (2016). *Earned Value Project Management* (4th ed.). PMI.
- Kim, E., Wells, W. G., & Duffey, M. R. (2003). A model for effective implementation of Earned Value Management methodology. *International Journal of Project Management*, 21(5), 375-382.
- Vanhoucke, M. (2012). Measuring the efficiency of project control using fictitious and empirical project data. *International Journal of Project Management*, 30(2), 252-263.

**Industry standards**:
- AACE International (2020). *Cost Estimate Classification System*. Recommended Practice 18R-97.
- PMI (2019). *Practice Standard for Earned Value Management* (2nd ed.). Project Management Institute.

---

**END OF DOCUMENT**

**Chunking Complete**: 11 chunks created for Project S-Curve Cashflow Model & Duration Dynamics.
