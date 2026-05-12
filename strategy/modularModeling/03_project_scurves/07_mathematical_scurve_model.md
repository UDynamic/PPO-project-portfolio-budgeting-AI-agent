# CHUNK 07
## Coverage
Section 4.1-4.4: Mathematical S-Curve Model (Beta CDF formulation)

## Dependency Notes
Implements Beta CDF from literature review (Chunks 02-03). Uses BAC from module 01.

## Overlap Notes
References α=2, β=2.5 parameters from Chunk 02.

## Content

---

## 4. Mathematical Model

### 4.1 Planned Cumulative Spending S-Curve

The planned cumulative spending for project $i$ at time $t$ is modeled using the **Beta CDF**:

$$S_i(t) = \text{BAC}_i \cdot I_x(\alpha, \beta)$$

where:
- $S_i(t)$ = cumulative spending at time $t$ (in dollars)
- $\text{BAC}_i$ = Budget at Completion for project $i$ (from module 01)
- $x = \frac{t - t_{\text{start},i}}{D_i}$ = normalized time (0 at start, 1 at completion)
- $I_x(\alpha, \beta)$ = regularized incomplete beta function
- $\alpha, \beta$ = shape parameters

**Regularized incomplete beta function**:
$$I_x(\alpha, \beta) = \frac{B(x; \alpha, \beta)}{B(\alpha, \beta)}$$

where:
$$B(x; \alpha, \beta) = \int_0^x u^{\alpha-1} (1-u)^{\beta-1} \, du$$

**Baseline parameters** (from Barraza, 2011; Mubarak, 2015):
- $\alpha = 2.0$ (moderate front-loading)
- $\beta = 2.5$ (gradual tail-off)

**Boundary conditions**:
- $S_i(t_{\text{start},i}) = 0$ (no spending before project starts)
- $S_i(t_{\text{start},i} + D_i) = \text{BAC}_i$ (full budget spent at completion)

---

### 4.2 Incremental Spending Rate (Cashflow Velocity)

The **instantaneous spending rate** (cashflow per unit time) is the derivative of the cumulative curve:

$$\frac{dS_i}{dt}(t) = \frac{\text{BAC}_i}{D_i} \cdot \text{Beta}(x; \alpha, \beta)$$

where:
$$\text{Beta}(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}$$

is the **Beta probability density function**.

**Interpretation**:
- Peak spending rate occurs at $x^* = \frac{\alpha - 1}{\alpha + \beta - 2}$
- For $\alpha=2, \beta=2.5$: $x^* = \frac{1}{3.5} \approx 0.29$ (29% into project lifecycle)
- Maximum rate: $\frac{dS_i}{dt}\bigg|_{x=x^*} = \frac{\text{BAC}_i}{D_i} \cdot 1.56$ (56% above average rate)

**Practical implication**: Projects spend fastest in the first third of execution, then gradually taper off.

---

### 4.3 Analytical Properties of the Baseline Parameterization

For $\alpha=2, \beta=2.5$:

**1. Inflection point**:
$$x_{\text{inflection}} = \frac{\alpha - 1}{\alpha + \beta - 2} = \frac{1}{3.5} \approx 0.29$$

**2. Cumulative spending at key milestones**:
- 25% duration: $S(0.25) \approx 0.18 \cdot \text{BAC}$ (18% spent)
- 50% duration: $S(0.50) \approx 0.52 \cdot \text{BAC}$ (52% spent)
- 75% duration: $S(0.75) \approx 0.84 \cdot \text{BAC}$ (84% spent)

**3. Skewness**:
$$\gamma_1 = \frac{2(\beta - \alpha)\sqrt{\alpha + \beta + 1}}{(\alpha + \beta + 2)\sqrt{\alpha\beta}} \approx 0.18$$

Positive skewness indicates **front-loaded spending** (more spending early than late).

**4. Variance**:
$$\text{Var}(x) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)} \approx 0.044$$

Low variance indicates **tight concentration** around the mean (predictable spending pattern).

---

### 4.4 Portfolio-Level Planned Spending

The **aggregate portfolio cashflow** at time $t$ is the sum across all active projects:

$$S_{\text{portfolio}}(t) = \sum_{i=1}^{N} S_i(t) \cdot \mathbb{1}_{[t_{\text{start},i}, \, t_{\text{start},i} + D_i]}(t)$$

where:
- $N$ = portfolio size (from module 01)
- $\mathbb{1}_{[a,b]}(t)$ = indicator function (1 if $t \in [a,b]$, else 0)

**Interpretation**: Only projects that have started but not yet completed contribute to portfolio cashflow at time $t$.

**Portfolio cashflow rate**:
$$\frac{dS_{\text{portfolio}}}{dt}(t) = \sum_{i=1}^{N} \frac{dS_i}{dt}(t) \cdot \mathbb{1}_{[t_{\text{start},i}, \, t_{\text{start},i} + D_i]}(t)$$

**Key properties**:
1. **Smoothing effect**: Staggered project starts (uniform distribution) reduce portfolio cashflow volatility
2. **Steady-state approximation**: For large $N$ (>15 projects), portfolio cashflow rate approaches constant (law of large numbers)
3. **Peak portfolio exposure**: Occurs when maximum number of projects overlap in execution phase

**Simulation algorithm**:
1. Sample $N$ from discrete uniform (module 01)
2. For each project $i=1,\ldots,N$:
   - Sample $\text{BAC}_i$ from lognormal (module 01)
   - Sample $D_i$ from lognormal (conditional on $\text{BAC}_i$)
   - Sample $t_{\text{start},i}$ from uniform(0, 12)
3. Compute $S_i(t)$ for all $t$ in simulation horizon
4. Aggregate: $S_{\text{portfolio}}(t) = \sum_i S_i(t)$

---

**End of Chunk 07**

**Next Chunk Preview**: Chunk 08 covers project duration model (Section 4.5) and post-intervention SPI dynamics (Section 4.6).
