# Chunk 06: Distribution Selection Justification

**Source**: `deprecated/modularModeling/projectsPortfolioModel/projectRevenuePlans.md`  
**Lines**: 423-500  
**Section**: 1.3 (all subsections)  
**Dependencies**: Chunks 01-05 (literature foundation + category calibration)  
**Next**: Chunk 07 (Validation and sensitivity analysis)

---

## 1.3 Distribution Selection Justification

### 1.3.1 Advance Payment Percentage: Truncated Normal

**Rationale**:
- Park et al. (2005) reports approximately normal distribution of advance percentages
- Mean 12.3%, SD 4.2% suggests normal distribution
- Truncation necessary to enforce contractual bounds (5-20% typical range)

**Mathematical form**:
$$\alpha \sim \text{TruncNormal}(\mu, \sigma, a, b)$$

where:
- $\mu$ = category-specific mean (8%, 10%, 13%, 15%)
- $\sigma$ = category-specific SD (2%, 2.5%, 3%, 3.5%)
- $a$ = lower bound (5%, 6%, 10%, 10%)
- $b$ = upper bound (12%, 15%, 18%, 20%)

**Literature support**:
- Park et al. (2005): Normal distribution fit (Kolmogorov-Smirnov test, p=0.23)
- Khanzadi et al. (2018): Iranian data shows similar normal pattern

### 1.3.2 Milestone Count: Discrete Uniform

**Rationale**:
- Elazouni & Gab-Allah (2004) shows relatively uniform distribution within client-type ranges
- No strong evidence for skewness toward specific counts
- Discrete uniform reflects lack of strong preference within contractual norms

**Mathematical form**:
$$N-1 \sim \text{DiscreteUniform}(n_{\min}, n_{\max})$$

**Literature support**:
- Elazouni & Gab-Allah (2004): Government 8-12 (uniform), Private 4-6 (uniform)
- Cui et al. (2010): Median 5-7, no significant skewness reported

### 1.3.3 Front-Loading Parameter: Exponential Decay

**Rationale**:
- Kenley & Wilson (1986) demonstrates exponential decay pattern in payment fractions
- Park et al. (2005) confirms front-loading in Middle East projects
- Exponential form captures diminishing payment amounts over project lifecycle

**Mathematical form**:
$$w_j = \exp\left(-\lambda \cdot \frac{j-1}{N-2}\right)$$

**Literature support**:
- Kenley & Wilson (1986): Exponential fit (R²=0.89) for Australian projects
- Park et al. (2005): 35-40% in first 30% progress implies λ≈0.25-0.35

### 1.3.4 Payment Delay: Log-Normal Distribution

**Rationale**:
- Ramachandra & Rotimi (2015) demonstrates log-normal fit for payment delays
- Delays cannot be negative (lower bound at 0)
- Right-skewed distribution reflects occasional long delays

**Mathematical form**:
$$\Delta t \sim \text{LogNormal}(\mu, \sigma)$$

**Literature support**:
- Ramachandra & Rotimi (2015): Log-normal fit (Anderson-Darling test, p=0.18)
- Parameters: μ=3.5, σ=0.6 (mean 42 days, SD 28 days)

### 1.3.5 Retention Rate: Truncated Normal

**Rationale**:
- Boussabaine & Elhag (1999) reports approximately normal distribution
- Mean 5.2%, SD 1.8%
- Truncation at 3-10% reflects contractual norms (FIDIC 5-10%)

**Mathematical form**:
$$\rho \sim \text{TruncNormal}(0.05, 0.018, 0.03, 0.10)$$

**Literature support**:
- Boussabaine & Elhag (1999): Normal fit (Shapiro-Wilk test, p=0.31)
- FIDIC (2017): Standard range 5-10%

---

## Key Takeaways

**Distribution Choices Summary**:

1. **Advance Payment %**: Truncated Normal - reflects empirical normal pattern with contractual bounds
2. **Milestone Count**: Discrete Uniform - no strong preference within client-type ranges
3. **Front-Loading λ**: Exponential Decay - captures diminishing payment pattern over project lifecycle
4. **Payment Delay**: Log-Normal - right-skewed, non-negative, reflects occasional long delays
5. **Retention Rate**: Truncated Normal - normal pattern with FIDIC contractual bounds (5-10%)

**Statistical Validation**:
- All distributions have empirical support from literature
- Goodness-of-fit tests confirm appropriate distribution choices
- Parameters calibrated from multiple studies across different regions

---

**End of Chunk 06**
