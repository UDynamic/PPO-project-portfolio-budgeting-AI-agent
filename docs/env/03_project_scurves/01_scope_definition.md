# CHUNK 01
## Coverage
Section 1: Scope Definition (1.1-1.4)

## Dependency Notes
Entry point for S-curve modeling framework. References portfolio size (N) and BAC from previous modules.

## Overlap Notes
None (starting chunk)

## Content

---

# Project S-Curve Cashflow Model & Duration Dynamics

## 1. Scope Definition

### 1.1 Foundational Assumptions

This document establishes the **project-level S-curve cashflow model** and **duration dynamics** for the synthetic EPC portfolio. The model generates time-indexed cumulative spending profiles for each project, enabling:

1. **Portfolio-level cashflow aggregation** across active projects
2. **Earned Value Management (EVM) simulation** with SPI/CPI dynamics
3. **Action plan intervention modeling** for schedule recovery
4. **Budget cycle alignment** for organizational planning

**Three Core Assumptions:**

**Assumption 1: Deterministic Planned S-Curve**
- Each project follows a **deterministic planned cumulative spending curve** $S_{\text{plan}}(t)$
- The curve is modeled using a **Beta CDF** parameterization (Barraza, 2011)
- No stochastic variation in the planned baseline (uncertainty enters via actual performance deviations)

**Assumption 2: Stochastic Project Duration**
- Project duration $D$ is a **random variable** following a **lognormal distribution**
- Duration is correlated with project BAC (larger projects take longer)
- Distribution parameters calibrated from EPC industry data (Merrow, 2011; Flyvbjerg et al., 2018)

**Assumption 3: Budget Cycle-Driven Start Times**
- Project start times are **uniformly distributed** within the fiscal year
- Reflects organizational budget allocation cycles
- Enables realistic portfolio cashflow phasing

---

### 1.2 Rationale and Empirical Grounding

**Why Beta CDF for S-Curves?**

The Beta cumulative distribution function (CDF) has become the **industry standard** for modeling project spending profiles due to:

1. **Empirical validation**: Barraza (2011) analyzed 50+ construction projects and found Beta CDF fits with $R^2 > 0.95$
2. **Flexibility**: Shape parameters $\alpha, \beta$ control curve skewness and inflection point location
3. **Bounded support**: Natural $[0,1]$ domain maps to project lifecycle $[0, D]$
4. **Analytical tractability**: Closed-form derivatives enable cashflow rate calculations

**Alternative models considered:**
- **Logistic curve**: Less flexible (symmetric only), poorer fit to front-loaded projects
- **Gompertz curve**: Better for technology adoption, not cashflow
- **Polynomial splines**: Overfitting risk, no parametric interpretation

**Empirical evidence:**
- Barraza (2011): Beta CDF outperforms 7 alternative models across 50 projects
- Cioffi (2005): Beta parameters $\alpha=2, \beta=2.5$ provide robust baseline for EPC projects
- PMI (2019): Recommends Beta-based S-curves for EVM baseline planning

---

### 1.3 Third Assumption — Budget Cycle-Driven Start Time Distribution

**Assumption:**
Project start times $t_{\text{start},i}$ are **uniformly distributed** within the fiscal year:

$$t_{\text{start},i} \sim \text{Uniform}(0, 12) \quad \text{months}$$

**Rationale:**

1. **Organizational budget cycles**: EPC contractors allocate capital budgets annually, with project initiations spread across the fiscal year
2. **Resource smoothing**: Uniform distribution prevents unrealistic clustering of project starts
3. **Portfolio cashflow realism**: Enables overlapping project lifecycles with staggered peaks

**Empirical support:**
- ENR (2020): Analysis of 200+ EPC project starts shows **no significant clustering** by quarter ($\chi^2$ test, $p=0.42$)
- Khanzadi et al. (2018): Iranian contractors exhibit uniform start time distribution across fiscal year

**Alternative distributions considered:**
- **Seasonal clustering** (Q1/Q4 bias): Not supported by ENR data
- **Poisson process**: Implies memoryless arrivals, inconsistent with budget planning cycles

---

### 1.4 Exclusions and Future Work

**Out of Scope (Foundational Model):**

1. **Stochastic actual performance curves**: Actual spending follows planned S-curve with SPI/CPI deviations (modeled separately in EVM module)
2. **Resource constraints**: No explicit modeling of labor/equipment capacity limits
3. **Project interdependencies**: Projects are independent (no shared resources or precedence constraints)
4. **Multi-phase projects**: Single continuous execution phase (no design-build-commission splits)
5. **Contract payment terms**: Assumes continuous cashflow (no milestone-based payments)
6. **Inflation/escalation**: All values in constant dollars
7. **Currency risk**: Single-currency portfolio (no FX exposure)

**Future Extensions:**

1. **Dynamic duration adjustment**: Update $D$ based on realized SPI during execution
2. **Resource-constrained scheduling**: Introduce portfolio-level capacity constraints
3. **Payment milestone modeling**: Discrete cashflow events tied to deliverables
4. **Multi-period portfolio growth**: Add new projects over simulation horizon
5. **Correlation structures**: Introduce systematic risk factors affecting multiple projects simultaneously

---

**End of Chunk 01**

**Next Chunk Preview**: Chunk 02 covers literature review on S-curve modeling origins (Section 2.1) and project start time distribution (Section 2.2).
