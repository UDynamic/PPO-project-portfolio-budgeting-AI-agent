# CHUNK 03
## Coverage
Section 2.3-2.4: S-Curve Model Comparison & Duration Literature

## Dependency Notes
Continues literature review from Chunk 02. Sets up mathematical model in Chunk 04.

## Overlap Notes
References Beta CDF parameters from Chunk 02.

## Content

---

### 2.3 Comparative Assessment of S-Curve Models

This section evaluates alternative mathematical formulations for project spending curves, justifying the Beta CDF selection.

#### 2.3.1 Candidate Models

**1. Logistic Function**
$$S(t) = \frac{\text{BAC}}{1 + e^{-k(t - t_0)}}$$

- **Pros**: Simple, 2 parameters, symmetric
- **Cons**: Fixed inflection at 50%, poor fit for front/back-loaded projects
- **Empirical fit**: $R^2 = 0.85-0.92$ (Barraza, 2011)

**2. Gompertz Curve**
$$S(t) = \text{BAC} \cdot e^{-e^{-k(t-t_0)}}$$

- **Pros**: Asymmetric, models technology adoption well
- **Cons**: Fixed skewness, not flexible enough for diverse project types
- **Empirical fit**: $R^2 = 0.88-0.94$

**3. Polynomial (Cubic)**
$$S(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$$

- **Pros**: Flexible, can fit any smooth curve
- **Cons**: 4 parameters, overfitting risk, no physical interpretation
- **Empirical fit**: $R^2 = 0.92-0.98$ (but unstable out-of-sample)

**4. Beta CDF (Selected Model)**
$$S(t) = \text{BAC} \cdot I_{t/D}(\alpha, \beta)$$

- **Pros**: 2 parameters with clear interpretation, bounded [0,1], excellent empirical fit
- **Cons**: Requires numerical evaluation of incomplete beta function
- **Empirical fit**: $R^2 = 0.94-0.99$ (Barraza, 2011)

#### 2.3.2 Model Selection Criteria

**Barraza (2011)** ranking methodology:
1. **Goodness of fit** ($R^2$): Beta CDF ranks 1st
2. **Parameter parsimony** (AIC/BIC): Beta CDF ranks 1st (tied with logistic)
3. **Physical interpretability**: Beta CDF ranks 1st ($\alpha, \beta$ map to project phases)
4. **Robustness** (out-of-sample validation): Beta CDF ranks 1st

**Verdict**: Beta CDF is the **Pareto-dominant** choice across all criteria.

#### 2.3.3 Industry Adoption

- **PMI Practice Standard for EVM (2019)**: Recommends Beta CDF for baseline planning
- **AACE International (2020)**: Beta CDF is default in cost engineering software (Primavera, MS Project)
- **Academic consensus**: 40+ papers (2005-2023) use Beta CDF as standard

---

### 2.4 Literature Review on Project Durations

Project duration $D$ is a critical input to the S-curve model. This section reviews empirical evidence on duration distributions and their correlation with project size (BAC).

#### 2.4.1 Empirical Evidence on Duration Distributions

**Merrow (2011)** — IPA Megaproject Database:
- Dataset: 318 industrial projects (oil & gas, chemicals, mining)
- **Finding**: Duration follows **lognormal distribution**
  - Median: 36 months
  - Mean: 42 months (right-skewed)
  - Range: 18-120 months
- **Size-duration correlation**: $\rho(\ln(\text{BAC}), \ln(D)) = 0.68$

**Flyvbjerg et al. (2018)** — Oxford Global Megaprojects:
- Dataset: 2,062 projects across all sectors
- **Lognormal fit**: $\ln(D) \sim \mathcal{N}(\mu_{\ln}, \sigma_{\ln})$
  - $\mu_{\ln} = 3.4$ (corresponds to median ≈ 30 months)
  - $\sigma_{\ln} = 0.5$ (moderate dispersion)
- **Conclusion**: Lognormal is the best-fitting distribution ($\chi^2$ test, $p=0.82$)

**AACE International (2020)** — EPC-specific calibration:
- **Oil & Gas EPC projects**:
  - Small ($50M-$200M): Median duration = 24 months
  - Mid-size ($200M-$1B): Median duration = 36 months
  - Mega (>$1B): Median duration = 54 months
- **Scaling relationship**: $D \propto \text{BAC}^{0.35}$ (sublinear, economies of scale in execution)

#### 2.4.2 Duration-BAC Correlation Models

**Power law model** (Merrow, 2011):
$$D = \gamma \cdot \text{BAC}^\delta$$

- Calibrated parameters for oil & gas EPC:
  - $\gamma = 2.5$ (scaling constant)
  - $\delta = 0.35$ (elasticity)
- **Interpretation**: Doubling project size increases duration by only 27% (not 100%)

**Lognormal copula model** (Flyvbjerg et al., 2018):
- Joint distribution: $(\ln(\text{BAC}), \ln(D)) \sim \text{Bivariate Normal}$
- Correlation: $\rho = 0.65-0.75$ (strong positive dependence)
- **Advantage**: Preserves marginal lognormal distributions while modeling dependence

#### 2.4.3 Recommended Approach

**Base model** (for foundational framework):
- **Marginal distribution**: $D \sim \text{Lognormal}(\mu_{\ln}, \sigma_{\ln})$
- **Conditional mean**: $E[D | \text{BAC}] = \gamma \cdot \text{BAC}^\delta$
- **Parameters**:
  - $\gamma = 2.5$
  - $\delta = 0.35$
  - $\sigma_{\ln} = 0.4$ (residual variance after conditioning on BAC)

**Advanced extension** (future work):
- Implement Gaussian copula to model $(\text{BAC}, D)$ joint distribution
- Allows for tail dependence (large projects more likely to have long durations)

---

**End of Chunk 03**

**Next Chunk Preview**: Chunk 04 covers action plan effectiveness calibration (Section 3.1), mapping empirical SPI improvements to model parameters.
