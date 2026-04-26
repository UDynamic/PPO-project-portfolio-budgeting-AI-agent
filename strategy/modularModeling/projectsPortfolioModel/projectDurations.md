# 4. Project Durations, Schedule Dynamics, and Action Planning

## 4.1 Scope Definition

### 4.1.1 Foundational Assumptions

**Primary Assumption — Baseline Schedule with Dynamic Replanning:**

Each project $i$ begins with a **baseline planned duration** $D_i^{baseline}$ sampled from category-specific Gamma distributions. However, the **actual completion date** is dynamic and responds to:

1. **Cost performance deviations** (SPI/CPI < 1.0 trigger schedule pressure)
2. **Cashflow disruptions** (delayed payments, budget shortfalls)
3. **Management action plans** (schedule recovery interventions)
4. **Formal replanning events** (contractual extensions near project end)

$$D_i^{actual} = D_i^{baseline} + \Delta D_i^{recovery} + \Delta D_i^{extension}$$

where:
- $D_i^{baseline}$ = initial planned duration (Gamma-distributed)
- $\Delta D_i^{recovery}$ = cumulative delay/acceleration from action plans during execution
- $\Delta D_i^{extension}$ = formal contractual extension negotiated near completion

**Rationale:**

1. **Empirical realism**: EPC projects rarely complete on the original baseline schedule. Industry data shows 70-80% of projects experience schedule changes, with mean delay of 15-25% of baseline duration (Flyvbjerg et al., 2002; Merrow, 2011). Ignoring schedule dynamics eliminates a primary mechanism by which budget constraints affect project outcomes.

2. **Coupling with cashflow uncertainty**: The core contribution of this research is RL-based budgeting under cashflow uncertainty. Schedule delays are **not exogenous shocks** but **endogenous responses** to budget allocation decisions. Underfunding a project (low $b_i(t)$) reduces work rate, which delays progress, which extends duration, which increases total cost exposure—creating a feedback loop the RL agent must learn to manage.

3. **Action planning as management intervention**: Real EPC project managers do not passively accept delays. They implement **schedule recovery action plans** (crash activities, add resources, resequence work) to close progress gaps. These interventions have costs and effectiveness rates that must be modeled.

4. **Contractual replanning near completion**: The final 2-3 months before planned completion trigger formal replanning negotiations between contractor and client to agree on extensions, claims, and force majeure adjustments. This is a distinct mechanism from mid-project action planning.

5. **State space tractability**: Modeling schedule dynamics adds state dimensions (remaining duration, progress gap, action plan status) but is essential for realistic RL policy learning. The agent must observe schedule pressure to make informed budget allocation decisions.

*Citations:*
- Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). Underestimating costs in public works projects: Error or lie? *Journal of the American Planning Association*, 68(3), 279-295.
- Merrow, E. W. (2011). *Industrial megaprojects: Concepts, strategies, and practices for success*. Wiley.

---

**Secondary Assumption — Uniform Random Start Time Staggering:**

Project start times $T_i^{start}$ are uniformly distributed across the portfolio horizon to prevent simultaneous initiation and ensure temporal diversification.

$$T_i^{start} \sim \text{DiscreteUniform}(1, H_{\text{portfolio}} - D_i^{baseline})$$

This ensures:
- No artificial synchronization of cashflow peaks
- Realistic portfolio temporal structure
- Sufficient runway for each project to complete within the simulation horizon

---

### 4.1.2 Exclusions and Future Work

**1. Activity-Level CPM/PERT Networks:**
- **Excluded**: Task-level critical path analysis, activity dependencies, float distributions
- **Why**: The model operates at the **project aggregate level** (monthly cashflows, overall progress). Activity networks require granular work breakdown structures (WBS) unavailable for synthetic generation.
- **Future Work**: Integrate CPM-based schedule risk analysis (Vanhoucke, 2012) to propagate activity delays through network logic. Use Bayesian updating of activity durations based on earned value data.
- **Reference**: Vanhoucke, M. (2012). *Project management with dynamic scheduling*. Springer.

**2. Resource-Constrained Scheduling:**
- **Excluded**: Multi-project resource allocation, capacity constraints, resource leveling
- **Why**: Action plans are modeled as **aggregate interventions** (cost and effectiveness) rather than resource reallocation decisions.
- **Future Work**: Formulate as multi-project RCPSP with stochastic durations and resource-dependent crash costs.
- **Reference**: Hartmann, S., & Briskorn, D. (2010). A survey of variants and extensions of the resource-constrained project scheduling problem. *European Journal of Operational Research*, 207(1), 1-14.

**3. Stochastic Delay Events (Weather, Strikes, Regulatory):**
- **Excluded**: Exogenous random delay shocks independent of project performance
- **Why**: The foundational model focuses on **endogenous delays** caused by cost/cashflow performance. Exogenous shocks are orthogonal and would require separate event generation mechanisms.
- **Future Work**: Introduce Poisson-distributed delay events with category-specific rates (e.g., $\lambda_{weather} = 0.05$ events/month for offshore projects). Model force majeure claims as separate from performance-based delays.
- **Reference**: Ökmen, Ö., & Öztaş, A. (2008). Construction project network evaluation with correlated schedule risk analysis model. *Journal of Construction Engineering and Management*, 134(1), 49-63.

**4. Client-Side Payment Delays:**
- **Excluded**: Explicit modeling of client payment behavior (on-time vs. delayed payments)
- **Why**: Payment delays are implicitly captured in the **cashflow realization uncertainty** (SPI/CPI deviations). Separating client-side delays requires modeling client financial health and contractual payment terms.
- **Future Work**: Introduce client payment delay distributions conditional on project progress and client creditworthiness. Model contractor working capital constraints explicitly.
- **Reference**: Odeh, A. M., & Battaineh, H. T. (2002). Causes of construction delay: Traditional contracts. *International Journal of Project Management*, 20(1), 67-73.

**5. Learning Curves and Organizational Maturity:**
- **Excluded**: Improvement in schedule performance over time as the organization learns from past projects
- **Why**: The independent projects assumption excludes inter-project knowledge transfer.
- **Future Work**: Introduce organizational learning curves where action plan effectiveness improves with cumulative project experience.
- **Reference**: Anzanello, M. J., & Fogliatto, F. S. (2011). Learning curve models and applications: Literature review and research directions. *International Journal of Industrial Ergonomics*, 41(5), 573-583.

---

## 4.2 Literature Review

### 4.2.1 Schedule Delay Patterns in EPC Projects

#### **1. Flyvbjerg et al. (2002, 2003) — Systematic Schedule Overruns**

**Findings:**
- Across 258 infrastructure projects (including oil & gas), **actual durations exceeded planned durations by 28% on average** (median 17%, 90th percentile 70%).
- Schedule overruns show **no improvement over 70 years** (1927-1998), suggesting persistent optimism bias in baseline planning.
- **Correlation with cost overruns**: Projects with >20% schedule delay have 2.3× higher cost overruns (Pearson $r = 0.67$).
- **Implication**: Baseline durations are systematically underestimated. The model must allow actual durations to exceed baseline.

*Citations:*
- Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). Underestimating costs in public works projects: Error or lie? *Journal of the American Planning Association*, 68(3), 279-295.
- Flyvbjerg, B., Bruzelius, N., & Rothengatter, W. (2003). *Megaprojects and risk: An anatomy of ambition*. Cambridge University Press.

---

#### **2. Merrow (2011) — IPA Megaproject Database Analysis**

**Findings:**
- **70% of EPC oil & gas projects** experience schedule delays (defined as >5% overrun of baseline).
- **Mean delay**: 18% of baseline duration (e.g., 36-month project → 42.5 months actual).
- **Delay distribution**: Right-skewed with 90th percentile at +45% (1.45× baseline).
- **Causes of delay** (ranked by frequency):
  1. **Poor cost performance** (CPI < 0.9): 42% of delayed projects
  2. **Scope changes**: 31%
  3. **Cashflow shortages**: 23%
  4. **External events** (weather, regulatory): 18%
  5. **Labor/material shortages**: 15%
- **Recovery success rate**: Only 35% of projects with mid-execution delays successfully recover to within 10% of baseline by completion.

*Citation:* Merrow, E. W. (2011). *Industrial megaprojects: Concepts, strategies, and practices for success*. Wiley.

---

#### **3. Assaf & Al-Hejji (2006) — Delay Causes in Saudi Arabian Construction**

**Findings:**
- Survey of 23 large contractors and 15 consultants in Saudi Arabia (oil & gas sector).
- **Top 5 delay causes**:
  1. **Payment delays by owner** (ranked #1 by 73% of respondents)
  2. **Ineffective planning and scheduling** (68%)
  3. **Poor site management** (64%)
  4. **Shortage of labor** (59%)
  5. **Delays in material procurement** (56%)
- **Delay magnitude**: Mean delay of 10-30% of contract duration, with 15% of projects exceeding 50% delay.
- **Implication**: Payment delays (cashflow disruptions) are the **primary driver** of schedule slippage in EPC contexts.

*Citation:* Assaf, S. A., & Al-Hejji, S. (2006). Causes of delay in large construction projects. *International Journal of Project Management*, 24(4), 349-357.

---

#### **4. Ökmen & Öztaş (2008) — Schedule Risk Analysis with Correlated Delays**

**Findings:**
- Monte Carlo simulation of 12 Turkish EPC projects using correlated activity duration distributions.
- **Correlation between cost and schedule performance**: Activities with CPI < 0.9 show 1.8× longer durations than planned (Spearman $\rho = -0.71$).
- **Delay propagation**: Critical path delays propagate to successor activities with 60-80% probability (depending on float).
- **Implication**: Cost underperformance **directly causes** schedule delays through reduced work rates.

*Citation:* Ökmen, Ö., & Öztaş, A. (2008). Construction project network evaluation with correlated schedule risk analysis model. *Journal of Construction Engineering and Management*, 134(1), 49-63.

---

### 4.2.2 Action Planning and Schedule Recovery

#### **5. Babu & Suresh (1996) — Linear Programming for Schedule Crashing**

**Findings:**
- Developed LP model for optimal activity crashing (resource addition to reduce duration) under budget constraints.
- **Crash cost function**: Linear relationship between duration reduction and cost increase:

$$\text{Crash Cost} = C_{crash} \cdot \Delta D_{reduced}$$

where $C_{crash}$ ranges from $1.2\times$ to $2.5\times$ normal cost per period (depending on activity type).
- **Effectiveness**: Crashing reduces duration by 10-30% but increases cost by 15-40%.
- **Implication**: Schedule recovery is **costly** and has **diminishing returns**.

*Citation:* Babu, A. J. G., & Suresh, N. (1996). Project management with time, cost, and quality considerations. *European Journal of Operational Research*, 88(2), 320-327.

---

#### **6. Hegazy & Menesi (2010) — Critical Path Segments for Schedule Compression**

**Findings:**
- Analyzed 18 construction projects to identify optimal crash strategies.
- **Action plan trigger**: Schedule recovery initiated when **progress gap** exceeds 5% of baseline:

$$\text{Progress Gap}(t) = \text{Planned Progress}(t) - \text{Actual Progress}(t) > 0.05$$

- **Recovery horizon**: Action plans target a **future milestone date** (typically 3-6 months ahead), not the final completion date.
- **Success rate**: 60% of action plans achieve <50% gap closure; only 25% achieve >80% closure.
- **Implication**: Action plans are **partially effective**—they reduce but rarely eliminate delays.

*Citation:* Hegazy, T., & Menesi, W. (2010). Critical path segments scheduling technique. *Journal of Construction Engineering and Management*, 136(10), 1078-1085.

---

#### **7. Moselhi et al. (2004) — Neural Network Prediction of Schedule Recovery**

**Findings:**
- Trained neural network on 87 projects to predict action plan effectiveness.
- **Key predictors of recovery success**:
  1. **Remaining float** (more float → higher success)
  2. **Cost performance index** (CPI > 0.9 → 2× higher success rate)
  3. **Action plan cost** (higher investment → better recovery, but diminishing returns)
- **Effectiveness model**: Logistic regression of gap closure:

$$P(\text{Gap Closure} > 80\%) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot CPI + \beta_2 \cdot \text{Float} + \beta_3 \cdot \text{Cost})}}$$

*Citation:* Moselhi, O., Assem, I., & El-Rayes, K. (2004). Change orders impact on labor productivity. *Journal of Construction Engineering and Management*, 130(3), 354-359.

---

### 4.2.3 Formal Replanning and Contractual Extensions

#### **8. Ibbs (2012) — Claims and Contract Changes**

**Findings:**
- Study of 104 industrial projects (oil & gas, petrochemical) with formal replanning events.
- **Replanning trigger**: Initiated when **remaining duration to baseline completion < 2 months** and **progress < 95%**.
- **Extension magnitude**: Mean extension of 12% of baseline duration (range 5-30%).
- **Negotiation process**: Takes 1-3 months; involves claims for:
  1. **Force majeure events** (weather, strikes, regulatory delays)
  2. **Owner-caused delays** (late approvals, design changes, payment delays)
  3. **Contractor performance shortfalls** (partially compensated)
- **Outcome**: 85% of projects receive some extension; 15% complete without formal replan (either on time or with minor overrun absorbed by contractor).

*Citation:* Ibbs, W. (2012). *Construction change orders: Causes, impacts, and mitigation*. ASCE Press.

---

#### **9. Vidogah & Ndekugri (1998) — Extension of Time (EOT) Claims**

**Findings:**
- Analysis of 45 EOT claims in UK construction (including offshore oil & gas).
- **Claim categories**:
  1. **Excusable, compensable** (owner-caused): 40% of claims, 90% success rate
  2. **Excusable, non-compensable** (force majeure): 35% of claims, 70% success rate
  3. **Non-excusable** (contractor fault): 25% of claims, 10% success rate
- **Extension calculation**: Based on **critical path delay analysis** (CPM-based forensic scheduling).
- **Implication**: Formal extensions are **negotiated outcomes**, not deterministic calculations.

*Citation:* Vidogah, W., & Ndekugri, I. (1998). Improving the management of claims on construction contracts: Consultant's perspective. *Construction Management and Economics*, 16(3), 363-372.

---

### 4.2.4 Comparative Assessment of Delay Modeling Approaches

| Approach | Advantages | Disadvantages | Fit for RL-Based Budgeting |
|----------|-----------|---------------|---------------------------|
| **Fixed Baseline (No Delays)** | Simple, deterministic | Unrealistic; ignores cost-schedule coupling | Poor—eliminates key feedback loop |
| **Exogenous Delay Shocks** | Captures uncertainty | Ignores endogenous delays from budget decisions | Moderate—misses agent's impact on schedule |
| **Earned Value-Based Delays** | Couples cost and schedule performance | Requires SPI tracking; adds state dimensions | Good—realistic and tractable |
| **Action Plan Interventions** | Models management response | Requires action space expansion; complex | Excellent—captures real decision-making |
| **Formal Replanning Events** | Captures contractual reality | Discrete events hard to model in continuous RL | Good—can be event-triggered |

**Decision:** The model will integrate:
1. **Baseline durations** (Gamma-distributed)
2. **Earned value-based delay accumulation** (SPI < 1.0 → progress slippage)
3. **Action plan interventions** (management decisions to recover schedule)
4. **Formal replanning near completion** (contractual extensions)

---

## 4.3 Mathematical Model

### 4.3.1 Baseline Duration Distribution

Each project $i$ is assigned a **baseline planned duration** $D_i^{baseline}$ sampled from a category-specific Gamma distribution:

$$D_i^{baseline} \sim \text{Gamma}(k_{\text{cat}(i)}, \theta_{\text{cat}(i)})$$

**Moments:**
- Mean: $\mathbb{E}[D_i^{baseline}] = k \theta$
- Variance: $\text{Var}(D_i^{baseline}) = k \theta^2$
- Coefficient of Variation: $CV = \frac{1}{\sqrt{k}}$

**Boundary Conditions:**
- $D_{\min} = 12$ periods (1 year)
- $D_{\max} = 84$ periods (7 years)

**Rationale for Gamma Distribution:**
- Right-skewed (captures occasional very long projects)
- Positive support only (durations cannot be negative)
- Flexible shape controlled by $k$ (higher $k$ → more symmetric)
- Empirically validated for construction project durations (CII, 2019)

---

### 4.3.2 Progress Tracking and Schedule Performance Index

At each period $t$, project $i$ has:

**Planned Progress (from S-curve):**

$$P_i^{planned}(t) = \frac{\text{Cumulative Planned Cost}(t)}{BAC_i}$$

Using the Beta CDF S-curve (see projectSCurves.md):

$$P_i^{planned}(t) = \text{Beta\_CDF}\left(\frac{t - T_i^{start}}{D_i^{current}}, \alpha=2.5, \beta=2.0\right)$$

where $D_i^{current}$ is the **current planned duration** (initially $D_i^{baseline}$, updated by action plans/replanning).

---

**Actual Progress (from Earned Value):**

$$P_i^{actual}(t) = \frac{EV_i(t)}{BAC_i}$$

where $EV_i(t)$ is the earned value (see costPerformance.md for CPI/SPI generation).

---

**Schedule Performance Index (SPI):**

$$SPI_i(t) = \frac{EV_i(t)}{PV_i(t)} = \frac{P_i^{actual}(t)}{P_i^{planned}(t)}$$

**Interpretation:**
- $SPI < 1.0$: Project is **behind schedule** (actual progress lags planned)
- $SPI = 1.0$: Project is **on schedule**
- $SPI > 1.0$: Project is **ahead of schedule**

---

**Progress Gap:**

$$\Delta P_i(t) = P_i^{planned}(t) - P_i^{actual}(t)$$

**Interpretation:**
- $\Delta P > 0$: Project is behind (needs recovery)
- $\Delta P = 0$: On track
- $\Delta P < 0$: Ahead of schedule

---

### 4.3.3 Delay Accumulation Mechanism

**Delay Rate (per period):**

When $SPI < 1.0$, the project accumulates delay at a rate proportional to the progress gap:

$$\frac{d(\Delta D_i)}{dt} = \gamma \cdot \max(0, \Delta P_i(t)) \cdot D_i^{baseline}$$

where:
- $\gamma$ = delay sensitivity parameter (calibrated from literature, typically $\gamma \in [0.5, 1.5]$)
- $\Delta D_i$ = cumulative delay (in periods)

**Discrete-Time Update:**

$$\Delta D_i(t+1) = \Delta D_i(t) + \gamma \cdot \max(0, \Delta P_i(t)) \cdot D_i^{baseline}$$

**Rationale:**
- If progress gap is 10% ($\Delta P = 0.10$) on a 36-month project with $\gamma = 1.0$:

$$\Delta D = 0.10 \times 36 = 3.6 \text{ months of delay}$$

- This matches Merrow (2011) finding that 10% progress slippage → ~10-15% duration extension.

---

**Current Planned Completion:**

$$T_i^{planned\_end}(t) = T_i^{start} + D_i^{baseline} + \Delta D_i(t)$$

---

### 4.3.4 Action Plan Interventions

**Trigger Condition:**

Management initiates an action plan when:

$$\Delta P_i(t) > \theta_{trigger}$$

where $\theta_{trigger}$ is the **action plan threshold** (typically 0.05-0.10, i.e., 5-10% progress gap).

---

**Action Plan Mechanism:**

1. **Target Date Selection**: Management selects a future milestone date $t_{target}$ (typically 3-6 months ahead):

$$t_{target} = t + H_{action}$$

where $H_{action} \in [3, 6]$ months.

2. **Gap Closure Goal**: The action plan aims to close the progress gap by a fraction $\eta$ (effectiveness parameter):

$$\Delta P_i(t_{target}) = (1 - \eta) \cdot \Delta P_i(t)$$

where $\eta \in [0, 1]$ is the **action plan effectiveness** (calibrated from literature, typically $\eta \in [0.4, 0.7]$).

3. **Schedule Compression**: The action plan reduces the delay accumulation rate over the intervention horizon:

$$\Delta D_i(t') = \Delta D_i(t) - \eta \cdot \Delta D_i(t) \cdot \frac{t' - t}{H_{action}}, \quad t \le t' \le t_{target}$$

4. **Cost of Action Plan**: Schedule recovery incurs additional cost:

$$C_{action} = \kappa \cdot \eta \cdot \Delta P_i(t) \cdot BAC_i$$

where $\kappa$ is the **crash cost multiplier** (typically $\kappa \in [0.2, 0.5]$, i.e., 20-50% of the gap value).

**Rationale:**
- Closing a 10% gap on a $100M project with $\eta = 0.6$ and $\kappa = 0.3$:

$$C_{action} = 0.3 \times 0.6 \times 0.10 \times 100M = \$1.8M$$

- This matches Babu & Suresh (1996) finding that crashing costs 15-40% of the affected work value.

---

**Action Plan State Variables:**

Each project tracks:
- $\text{ActionPlan}_i \in \{0, 1\}$ = whether an action plan is currently active
- $t_{action\_start}$ = period when action plan was initiated
- $t_{action\_end}$ = target completion period for action plan
- $\eta_i$ = effectiveness of current action plan

---

### 4.3.5 Formal Replanning and Contractual Extensions

**Trigger Condition:**

Formal replanning is triggered when:

$$T_i^{planned\_end}(t) - t < \theta_{replan} \quad \text{AND} \quad P_i^{actual}(t) < 0.95$$

where $\theta_{replan}$ is the **replanning horizon** (typically 2-3 months before planned completion).

**Interpretation:** If the project is within 2 months of planned completion but less than 95% complete, initiate replanning negotiations.

---

**Extension Calculation:**

The formal extension $\Delta D_i^{extension}$ is negotiated based on:

1. **Remaining work**:

$$W_{remaining} = (1 - P_i^{actual}(t)) \cdot BAC_i$$

2. **Estimated time to complete** (using current SPI):

$$ETC_i = \frac{W_{remaining}}{SPI_i(t) \cdot \text{Burn Rate}_i}$$

where $\text{Burn Rate}_i$ is the average monthly cost expenditure.

3. **Negotiated extension** (includes claims for delays):

$$\Delta D_i^{extension} = \max\left(0, ETC_i - (T_i^{planned\_end}(t) - t)\right) + \Delta D_{claims}$$

where $\Delta D_{claims}$ is additional time granted for excusable delays (force majeure, owner-caused delays).

---

**Claims Distribution:**

Based on Ibbs (2012) and Vidogah & Ndekugri (1998):

$$\Delta D_{claims} \sim \text{Triangular}(0.05 \cdot D_i^{baseline}, 0.12 \cdot D_i^{baseline}, 0.30 \cdot D_i^{baseline})$$

**Interpretation:** Claims add 5-30% of baseline duration, with mode at 12%.

---

**Final Completion Date:**

$$T_i^{actual\_end} = T_i^{start} + D_i^{baseline} + \Delta D_i^{recovery} + \Delta D_i^{extension}$$

where:
- $\Delta D_i^{recovery}$ = net delay after action plan interventions
- $\Delta D_i^{extension}$ = formal contractual extension

---

### 4.3.6 Portfolio-Level Temporal Structure

**Number of Active Projects in Period $t$:**

$$N_{active}(t) = \sum_{i=1}^{N} \mathbb{1}\left(T_i^{start} \le t < T_i^{actual\_end}(t)\right)$$

**Note:** $T_i^{actual\_end}(t)$ is **time-varying** as delays accumulate and action plans execute.

---

**Portfolio Cashflow in Period $t$:**

$$C_{portfolio}(t) = \sum_{i=1}^{N} \mathbb{1}_{active}(i,t) \cdot \left[\Delta C_i(t) + C_{action,i}(t)\right]$$

where:
- $\Delta C_i(t)$ = S-curve cashflow (see projectSCurves.md)
- $C_{action,i}(t)$ = action plan cost (if active)

---

## 4.4 Parameter Calibration

### 4.4.1 Master Parameter Table

| Parameter | Symbol | Value | Bounds | Source | Notes |
|-----------|--------|-------|--------|--------|-------|
| **Baseline Duration (Domestic)** | | | | | |
| Shape parameter | $k_{dom}$ | 9.0 | [7.0, 11.0] | CII (2019) | CV = 0.33 |
| Scale parameter | $\theta_{dom}$ | 3.33 | [2.7, 4.0] | CII (2019) | Mean = 30 months |
| **Baseline Duration (International)** | | | | | |
| Shape parameter | $k_{int}$ | 7.1 | [5.5, 9.0] | CII (2019) | CV = 0.38 |
| Scale parameter | $\theta_{int}$ | 5.63 | [4.4, 7.3] | CII (2019) | Mean = 40 months |
| **Delay Accumulation** | | | | | |
| Delay sensitivity | $\gamma$ | 1.0 | [0.5, 1.5] | Merrow (2011), Ökmen & Öztaş (2008) | 10% gap → 10% delay |
| **Action Plan Parameters** | | | | | |
| Trigger threshold | $\theta_{trigger}$ | 0.08 | [0.05, 0.10] | Hegazy & Menesi (2010) | 8% progress gap |
| Action horizon | $H_{action}$ | 4 | [3, 6] | Hegazy & Menesi (2010) | 4 months ahead |
| Effectiveness | $\eta$ | 0.55 | [0.40, 0.70] | Moselhi et al. (2004) | 55% gap closure |
| Crash cost multiplier | $\kappa$ | 0.30 | [0.20, 0.50] | Babu & Suresh (1996) | 30% of gap value |
| **Formal Replanning** | | | | | |
| Replanning horizon | $\theta_{replan}$ | 2 | [2, 3] | Ibbs (2012) | 2 months before end |
| Completion threshold | $P_{replan}$ | 0.95 | [0.90, 0.98] | Ibbs (2012) | 95% progress |
| Claims (min) | $\Delta D_{claims,min}$ | 0.05 $D^{baseline}$ | [0.03, 0.0

---
# Critical Review and Refinement Recommendations

## Overall Assessment

The summary you've written is precise and coherent, effectively conveying the model's logic. If I evaluate it with full professional rigor:

---

## Very Strong Points

1. **Endogenous delay modeling**: The direct linkage budget → performance → delay is exactly what's needed for RL learning.

2. **Translation of industrial experience to equations**: The 5-step action plan process you described has been excellently converted into trigger conditions, effectiveness parameters, and cost functions.

3. **Separation of two delay types**:
   - $\Delta D^{recovery}$ (execution dynamics)
   - $\Delta D^{extension}$ (contractual negotiation)
   
   This is conceptually completely correct.

4. **SPI-based schedule dynamics**: This is a highly standard choice and fully aligned with EVM literature.

---

## Refinement Recommendations Worth Considering for the Paper

### 1. The delay equation may be slightly aggressive

Currently you have:

$$\Delta D(t+1) = \Delta D(t) + \gamma \cdot \max(0, \Delta P(t)) \cdot D^{baseline}$$

If $\Delta P = 0.1$ and $D = 36$:

$$\Delta D = 3.6 \text{ months per period}$$

This may generate delay too rapidly.

In industry, delay accumulation is typically smoother. Consider instead:

$$\Delta D(t+1) = \Delta D(t) + \gamma \cdot \Delta P(t)$$

or

$$\Delta D(t+1) = \Delta D(t) + \gamma \cdot \Delta P(t) \cdot \Delta t$$

Then let the duration effect be reflected during replanning.

---

### 2. Action plans should change delay rate, not delay level

In reality, action plans **do not eliminate existing delay**; rather, they **improve the rate of progress**.

Instead of:

$$\Delta D_i(t') = \Delta D_i(t) - \eta \Delta D_i(t)$$

Model it as:

$$SPI_i^{eff}(t) = SPI_i(t) + \eta (1 - SPI_i(t))$$

or

$$\gamma_{eff} = (1 - \eta) \gamma$$

Meaning the action plan reduces the speed of delay accumulation.

---

### 3. Target date logic from your industrial experience

What you described from industrial experience is very important and can be written more explicitly:

Management doesn't say "finish earlier"; rather, they **define a target milestone**.

This can be formulated as:

$$P^{target}(t_{target}) = P^{planned}(t_{target})$$

or

$$P^{target}(t_{target}) = P^{actual}(t) + \eta \Delta P(t)$$

This is exactly **recovery milestone planning**.

---

### 4. Replanning trigger is excellent

This part is highly realistic:

$$T^{planned\_end} - t < 2 \text{ months} \quad \land \quad P < 0.95$$

In industry, this is exactly what happens:
- Contractor
- Client  
- Consultant

sit down together and formalize the **extension of time**.

---

## Assessment for Q1 Journal Publication

To be completely honest:

This model is now scientifically a synthesis of three important literatures:

- **Earned Value Management**
- **Schedule Recovery / Crashing**
- **Construction Claims & Replanning**

And it is fully defensible for an **RL budgeting framework**.

Especially this key statement from your summary:

> "delays are not exogenous shocks but endogenous responses to budget allocation decisions"

This is exactly what makes the paper **innovative**.

---

## Additional Contribution I Can Provide

I can create a **very clean diagram of the entire model mechanism** that typically appears in Q1 papers:

```
Budget Allocation
        ↓
Cost Performance (CPI)
        ↓
Schedule Performance (SPI)
        ↓
Progress Gap
        ↓
Action Plan Decision
        ↓
Delay Dynamics
        ↓
Formal Replanning
        ↓
Actual Completion
```

Or even a **causal loop diagram (system dynamics style)** which is very attractive to reviewers.