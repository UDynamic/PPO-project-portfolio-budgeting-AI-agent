# CHUNK 11
## Coverage
Section 7: Sensitivity Analysis (7.1 Scenario-Based Analysis + 7.2 Tornado Diagram)

## Dependency Notes
Uses retention rate ρ, decay rate λ, and effectiveness η from Chunks 04-06. Applies post-action dynamics from Chunk 06.

## Overlap Notes
References parameter distributions from Chunk 09 (Master Parameter Table).

## Content

---

## 7. Sensitivity Analysis

### 7.1 Scenario-Based Analysis

**Scenario 1: Strong Organizational Capability (P90)**
- $\rho = 0.50$ (high retention, 90th percentile)
- $\lambda_{\text{decay}} = 0.10$ per month (slow decay, 10th percentile)
- $\eta = 0.68$ (strong action plan effectiveness, 90th percentile)

**Results:**
- **Permanent gain:** 50% of peak improvement persists
- **Half-life of transient gains:** $t_{1/2} = \frac{\ln(2)}{0.10} = 6.93$ months
- **Stabilization time:** ~10-12 months
- **Interpretation:** Projects with mature PMOs, formal change management, and strong governance structures

**Example trajectory:**
- $\text{SPI}_{\text{baseline}} = 0.70$
- $\text{SPI}_{\text{peak}} = 0.70 + 0.68 \times (1 - 0.70) = 0.904$
- $\Delta_{\text{peak}} = 0.204$
- $\text{SPI}_{\text{equilibrium}} = 0.70 + 0.50 \times 0.204 = 0.802$

| Month | $t$ (post-action) | $\text{SPI}(t)$ | Retention % |
|-------|-------------------|-----------------|-------------|
| 3 | 0 | 0.904 | 100% |
| 6 | 3 | 0.878 | 87% |
| 9 | 6 | 0.853 | 75% |
| 12 | 9 | 0.829 | 63% |
| 15 | 12 | 0.814 | 56% |
| 18 | 15 | 0.807 | 52% |

---

**Scenario 2: Weak Organizational Capability (P10)**
- $\rho = 0.22$ (low retention, 10th percentile)
- $\lambda_{\text{decay}} = 0.32$ per month (fast decay, 90th percentile)
- $\eta = 0.32$ (weak action plan effectiveness, 10th percentile)

**Results:**
- **Permanent gain:** Only 22% of peak improvement persists
- **Half-life of transient gains:** $t_{1/2} = \frac{\ln(2)}{0.32} = 2.17$ months
- **Stabilization time:** ~4-6 months
- **Interpretation:** Projects with ad-hoc governance, high personnel turnover, and weak project controls

**Example trajectory:**
- $\text{SPI}_{\text{baseline}} = 0.70$
- $\text{SPI}_{\text{peak}} = 0.70 + 0.32 \times (1 - 0.70) = 0.796$
- $\Delta_{\text{peak}} = 0.096$
- $\text{SPI}_{\text{equilibrium}} = 0.70 + 0.22 \times 0.096 = 0.721$

| Month | $t$ (post-action) | $\text{SPI}(t)$ | Retention % |
|-------|-------------------|-----------------|-------------|
| 3 | 0 | 0.796 | 100% |
| 4 | 1 | 0.775 | 78% |
| 5 | 2 | 0.755 | 57% |
| 6 | 3 | 0.738 | 40% |
| 9 | 6 | 0.724 | 24% |
| 12 | 9 | 0.721 | 22% |

---

**Scenario 3: Base Case (P50 - Recommended)**
- $\rho = 0.35$ (moderate retention, median)
- $\lambda_{\text{decay}} = 0.18$ per month (moderate decay, median)
- $\eta = 0.50$ (average action plan effectiveness, median)

**Results:**
- **Permanent gain:** 35% of peak improvement persists
- **Half-life of transient gains:** $t_{1/2} = \frac{\ln(2)}{0.18} = 3.85$ months
- **Stabilization time:** ~8-9 months
- **Interpretation:** Typical EPC project with standard project controls

**Example trajectory:**
- $\text{SPI}_{\text{baseline}} = 0.70$
- $\text{SPI}_{\text{peak}} = 0.70 + 0.50 \times (1 - 0.70) = 0.85$
- $\Delta_{\text{peak}} = 0.15$
- $\text{SPI}_{\text{equilibrium}} = 0.70 + 0.35 \times 0.15 = 0.7525$

| Month | $t$ (post-action) | $\text{SPI}(t)$ | Retention % |
|-------|-------------------|-----------------|-------------|
| 3 | 0 | 0.850 | 100% |
| 4 | 1 | 0.834 | 89% |
| 5 | 2 | 0.821 | 81% |
| 6 | 3 | 0.809 | 73% |
| 9 | 6 | 0.786 | 57% |
| 12 | 9 | 0.772 | 48% |
| 15 | 12 | 0.764 | 41% |
| 18 | 15 | 0.759 | 37% |

---

### 7.2 Tornado Diagram: Parameter Sensitivity Rankings

To quantify the relative impact of each parameter on final project outcomes, we perform a one-at-a-time (OAT) sensitivity analysis.

**Metric:** Final SPI at Month 18 (15 months post-action plan)

**Base case inputs:**
- $\text{SPI}_{\text{baseline}} = 0.70$
- $\rho = 0.35$
- $\lambda_{\text{decay}} = 0.18$ per month
- $\eta = 0.50$

**Base case output:**
- $\text{SPI}(15) = 0.759$

**Sensitivity analysis results:**

| Parameter | Low Value (P10) | High Value (P90) | $\text{SPI}(15)$ at Low | $\text{SPI}(15)$ at High | Range | Rank |
|-----------|-----------------|------------------|------------------------|-------------------------|-------|------|
| **Action Plan Effectiveness** ($\eta$) | 0.32 | 0.68 | 0.734 | 0.789 | 0.055 | 1 |
| **Retention Rate** ($\rho$) | 0.22 | 0.50 | 0.742 | 0.780 | 0.038 | 2 |
| **Decay Rate** ($\lambda_{\text{decay}}$) | 0.10 | 0.32 | 0.772 | 0.748 | 0.024 | 3 |

**Interpretation:**
1. **Action plan effectiveness ($\eta$) is the most influential parameter** — improving organizational response capability during the intervention has the largest impact on long-term outcomes
2. **Retention rate ($\rho$) is the second most important** — institutionalizing improvements determines how much of the gain persists
3. **Decay rate ($\lambda_{\text{decay}}$) has moderate impact** — faster decay reduces long-term benefits, but the effect is smaller than the first two parameters

**Management implications:**
- **Priority 1:** Invest in action plan execution quality (training, resources, expert support) to maximize $\eta$
- **Priority 2:** Embed improvements into standard procedures to maximize $\rho$
- **Priority 3:** Monitor post-intervention performance to detect rapid decay early

---

**End of Chunk 11**

**Next Chunk Preview**: Chunk 12 covers Monte Carlo simulation results (Section 7.3) and Python implementation code (Section 8.1).
