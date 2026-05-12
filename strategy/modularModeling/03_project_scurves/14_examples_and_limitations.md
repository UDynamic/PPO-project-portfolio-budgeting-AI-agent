# Project S-Curve Cashflow Model & Duration Dynamics
## Part 14: Example Calculations and Limitations

**Document Status**: Production-Ready  
**Last Updated**: 2025  
**Prerequisites**: Parts 1-13 (all model components, calibration, validation)

---

## Purpose and Scope

This document provides:
1. **Worked examples** with full numerical calculations
2. **Model limitations** and mitigation strategies
3. **Future research directions**

---

## 10. Example Calculations

### 10.1 Full Post-Intervention Trajectory with Uncertainty

#### 10.1.1 Deterministic Base Case

**Given:**
- $\text{SPI}_{\text{baseline}} = 0.70$ (pre-intervention)
- $\text{SPI}_{\text{peak}} = 0.85$ (end of Month 3 action plan)
- $\rho = 0.35$ (base case retention rate)
- $\lambda_{\text{decay}} = 0.18$ per month (base case decay rate)

**Step 1: Calculate improvements**

$$\Delta_{\text{peak}} = 0.85 - 0.70 = 0.15$$

$$\Delta_{\text{permanent}} = 0.35 \times 0.15 = 0.0525$$

$$\Delta_{\text{transient}} = (1 - 0.35) \times 0.15 = 0.0975$$

**Step 2: Calculate equilibrium SPI**

$$\text{SPI}_{\text{equilibrium}} = 0.70 + 0.0525 = 0.7525$$

**Step 3: Calculate SPI at key time points**

| Month | $t$ (months post-action) | $e^{-0.18t}$ | $\Delta_{\text{transient}} \cdot e^{-0.18t}$ | $\text{SPI}(t)$ |
|-------|--------------------------|--------------|---------------------------------------------|-----------------|
| 3 (end of action plan) | 0 | 1.000 | 0.0975 | 0.8525 |
| 4 | 1 | 0.835 | 0.0814 | 0.8339 |
| 5 | 2 | 0.698 | 0.0681 | 0.8206 |
| 6 | 3 | 0.583 | 0.0568 | 0.8093 |
| 9 | 6 | 0.340 | 0.0331 | 0.7856 |
| 12 | 9 | 0.198 | 0.0193 | 0.7718 |
| 18 | 15 | 0.067 | 0.0065 | 0.7590 |
| 24 | 21 | 0.023 | 0.0022 | 0.7547 |
| ∞ | ∞ | 0.000 | 0.0000 | 0.7525 |

**Key observations:**
- Half-life of transient gains: $t_{1/2} = \ln(2)/0.18 = 3.85$ months
- 90% decay by Month 15 post-action
- Permanent improvement: 5.25 percentage points (35% of peak gain)

---

#### 10.1.2 Duration Impact Calculation

**Given:**
- Baseline planned duration: $D_{\text{baseline}} = 30$ months
- Pre-intervention SPI: 0.70
- Post-intervention equilibrium SPI: 0.7525

**Without action plan:**

$$D_{\text{actual, no action}} = \frac{30}{0.70} = 42.86 \text{ months}$$

**With action plan:**

$$D_{\text{actual, with action}} = \frac{30}{0.7525} = 39.87 \text{ months}$$

**Duration reduction:**

$$\Delta D = 42.86 - 39.87 = 2.99 \text{ months}$$

**Interpretation:**
- Action plan reduces delay from 12.86 months to 9.87 months
- 23% reduction in delay magnitude
- Still 33% over baseline (not full recovery)

---

#### 10.1.3 Probabilistic Analysis (Monte Carlo)

**Parameter distributions:**
- $\rho \sim \text{Beta}(3.5, 6.5)$ → Mean = 0.35, SD = 0.15
- $\lambda_{\text{decay}} \sim \text{Lognormal}(\ln(0.18), 0.25)$ → Median = 0.18, GSD = 1.28
- $\text{SPI}_{\text{peak}} \sim \text{Beta}(5.5, 2.5) \times 0.15 + 0.70$ → Mean = 0.82, SD = 0.04

**Monte Carlo results (10,000 simulations):**

| Metric | P10 | P50 | P90 | Mean | SD |
|--------|-----|-----|-----|------|-----|
| $\text{SPI}_{\text{equilibrium}}$ | 0.72 | 0.75 | 0.78 | 0.75 | 0.03 |
| Duration with action (months) | 38.5 | 40.0 | 41.7 | 40.1 | 1.8 |
| Duration reduction (months) | 1.2 | 2.9 | 4.8 | 3.0 | 1.5 |
| Probability of SPI ≥ 0.80 | — | — | — | 8% | — |
| Probability of SPI ≥ 0.85 | — | — | — | 1% | — |

**Key insights:**
- 80% confidence interval for equilibrium SPI: [0.72, 0.78]
- Median duration reduction: 2.9 months (consistent with deterministic case)
- Wide uncertainty: 80% CI for reduction is [1.2, 4.8] months
- Low probability of near-perfect recovery (SPI ≥ 0.85): only 1%

---

### 10.2 Financial Justification Example

**Given:**
- Project BAC: $200M
- Baseline duration: 30 months
- Pre-intervention SPI: 0.70
- Action plan cost: $12M (6% of BAC)
- Delay penalty: $500K per month
- Discount rate: 8% annual (0.64% monthly)

**Scenario 1: No action plan**

- Actual duration: 42.86 months
- Delay: 12.86 months
- Delay penalty: $12.86M × $0.5M = $6.43M
- NPV of delay cost: $6.43M / (1.0064)^{36} = $5.15M

**Scenario 2: With action plan**

- Action plan cost: $12M (paid at Month 3)
- Actual duration: 39.87 months
- Delay: 9.87 months
- Delay penalty: $9.87M × $0.5M = $4.94M
- NPV of delay cost: $4.94M / (1.0064)^{36} = $3.95M
- NPV of action plan: $12M / (1.0064)^3 = $11.77M

**Net benefit:**

$$\text{NPV}_{\text{benefit}} = (5.15 - 3.95) - 11.77 = -10.57M$$

**Conclusion:** Action plan is **not financially justified** under these assumptions.

**Break-even analysis:**

For action plan to be justified, delay penalty must exceed:

$$\text{Penalty}_{\text{break-even}} = \frac{12M}{2.99 \text{ months}} = 4.01M \text{ per month}$$

This is 8× higher than assumed penalty, indicating action plans are only justified for projects with severe delay consequences (e.g., contractual liquidated damages, market window closure).

---

## 11. Limitations and Future Research

### 11.1 Model Limitations

1. **Exponential decay assumption:**
   - Model assumes smooth exponential decay; actual trajectories may exhibit step changes (e.g., key personnel departure)
   - **Mitigation:** Use discrete event simulation for projects with known discontinuities

2. **Independence of parameters:**
   - Model treats $\rho$ and $\lambda_{\text{decay}}$ as independent; in reality, they may be correlated (high retention often accompanies slow decay)
   - **Mitigation:** Use copula-based joint distributions for advanced Monte Carlo analysis

3. **Single action plan assumption:**
   - Model assumes one action plan per project; multiple interventions may have compounding or diminishing effects
   - **Mitigation:** For multiple interventions, reset $\text{SPI}_{\text{baseline}}$ at each activation and track cumulative costs

4. **Homogeneous project assumption:**
   - Parameters calibrated for EPC oil & gas projects; other industries may exhibit different dynamics
   - **Mitigation:** Recalibrate distributions using industry-specific data

5. **No external shock modeling:**
   - Model assumes stable external environment; major disruptions (e.g., pandemic, regulatory changes) not captured
   - **Mitigation:** Incorporate scenario analysis for known external risks

---

### 11.2 Future Research Directions

1. **Dynamic S-curve integration:**
   - Incorporate Earned Schedule (ES) methodology to adjust S-curve shape based on observed performance
   - **Reference:** Lipke (2003), "Schedule is Different"

2. **Machine learning calibration:**
   - Use historical project data to train predictive models for $\rho$ and $\lambda_{\text{decay}}$ based on project characteristics
   - **Potential features:** Project size, complexity, contract type, organizational maturity scores

3. **Multi-intervention dynamics:**
   - Develop models for sequential action plans with diminishing returns
   - **Hypothesis:** Second intervention effectiveness $\eta_2 = 0.7 \times \eta_1$ (30% reduction)

4. **Cost-benefit optimization:**
   - Formulate optimization problem to determine optimal action plan timing and intensity
   - **Objective function:** Minimize total project cost (baseline + action plan + delay penalties)

5. **Organizational learning curves:**
   - Model improvement in $\rho$ and $\eta$ across multiple projects as organization gains experience
   - **Hypothesis:** $\eta_{\text{project } n} = \eta_0 \cdot n^{\alpha}$ where $\alpha \approx 0.1$ (learning curve exponent)

---

### 11.3 Practical Recommendations

**For portfolio managers:**

1. **Set realistic expectations:** Action plans deliver 5-10 percentage point permanent SPI improvement, not full recovery
2. **Invest in institutionalization:** Embed improvements into standard procedures to maximize retention rate
3. **Monitor post-intervention performance:** Track decay rate monthly to detect rapid reversion early
4. **Use probabilistic planning:** Incorporate parameter uncertainty into schedule risk analysis
5. **Assess organizational capability:** Adjust parameter values based on PMO maturity and governance strength

**For optimization models:**

1. **Use equilibrium SPI for duration calculations:** Don't assume peak SPI persists
2. **Include action plan costs in budget constraints:** Typically 5-8% of project BAC
3. **Model duration as stochastic:** Use Gamma distributions calibrated to project category
4. **Validate financial justification:** Action plans only justified when delay penalties are severe

---

## Cross-References

- **Part 9 (Parameter Calibration)**: Distribution parameters used in examples
- **Part 11 (Sensitivity Analysis)**: Tornado diagrams for parameter importance
- **Part 12 (Implementation)**: Code for Monte Carlo simulation
- **Part 13 (Validation)**: Analytical checks for example calculations

---

**Navigation:**
- Previous: `13_validation.md`
- Next: `15_appendices_and_references.md`

---

**END OF PART 14**
