# CHUNK 14
## Coverage
Section 10: Example Calculations + Section 11: Limitations and Future Research

## Dependency Notes
Uses all parameters from Chunks 04-09. Demonstrates complete model application.

## Overlap Notes
References decay dynamics from Chunk 06 and parameter distributions from Chunk 09.

## Content

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
| 9 | 6 | 0.340 | 0.0332 | 0.7857 |
| 12 | 9 | 0.198 | 0.0193 | 0.7718 |
| 15 | 12 | 0.116 | 0.0113 | 0.7638 |
| 18 | 15 | 0.068 | 0.0066 | 0.7591 |
| 21 | 18 | 0.040 | 0.0039 | 0.7564 |
| 24 | 21 | 0.023 | 0.0022 | 0.7547 |

**Interpretation:**
- **Month 3:** SPI peaks at 0.85 (21% improvement over baseline)
- **Month 6:** SPI = 0.81 (58% of transient gain remains)
- **Month 9:** SPI = 0.79 (34% of transient gain remains)
- **Month 12:** SPI = 0.77 (20% of transient gain remains)
- **Month 15+:** SPI stabilizes at 0.75 (7.5% permanent improvement over baseline)

**Final outcome:**
- **Permanent gain:** 5.25 percentage points in SPI
- **Peak temporary gain:** 9.75 percentage points (decays over 12 months)
- **Total peak improvement:** 15 percentage points (Month 3)

---

#### 10.1.2 Probabilistic Analysis (Three-Point Estimate)

**Scenario planning using P10, P50, P90 values:**

| Scenario | $\rho$ | $\lambda_{\text{decay}}$ | $\eta$ | $\text{SPI}_{\text{peak}}$ | $\text{SPI}_{\text{equilibrium}}$ | $\text{SPI}(15)$ |
|----------|--------|-------------------------|--------|---------------------------|----------------------------------|-----------------|
| **Pessimistic (P10)** | 0.22 | 0.32 | 0.32 | 0.796 | 0.721 | 0.721 |
| **Base Case (P50)** | 0.35 | 0.18 | 0.50 | 0.850 | 0.753 | 0.759 |
| **Optimistic (P90)** | 0.50 | 0.10 | 0.68 | 0.904 | 0.802 | 0.807 |

**Interpretation:**
- **Pessimistic case:** Action plan delivers only 2.1 percentage point permanent improvement (weak organizational capability)
- **Base case:** Action plan delivers 5.3 percentage point permanent improvement (typical EPC project)
- **Optimistic case:** Action plan delivers 10.2 percentage point permanent improvement (strong organizational capability)

**Risk assessment:**
- **Downside risk:** 10% chance of achieving less than 2.1 percentage point improvement
- **Upside potential:** 10% chance of achieving more than 8.0 percentage point improvement
- **Expected value:** 5.3 percentage point improvement (base case)

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

**End of Chunk 14**

**Next Chunk Preview**: Chunk 15 (final) covers appendices (Section 12) and complete references (Section 13).
