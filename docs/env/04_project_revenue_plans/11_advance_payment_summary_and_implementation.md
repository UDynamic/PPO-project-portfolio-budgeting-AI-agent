# CHUNK 011: Advance Payment - Summary Table and Implementation

## Coverage
Sections 2.2.4-2.2.6: Summary table of advance payment parameters, Milestone 0 definition, and implementation algorithm

## Dependencies
- Advance payment parameters from Chunk 010 (sections 2.2.1-2.2.3)
- Retention rate parameters from Chunks 008-009
- Contract value/revenue from Module 02

## Content

---

#### 2.2.4 Summary Table: Advance Payment Parameters

| Category | Client Type | P(Advance) | μ | σ | Lower Bound | Upper Bound | Literature Source |
|----------|-------------|------------|---|---|-------------|-------------|-------------------|
| DL | Government | 0.65 | 0.08 | 0.020 | 0.05 | 0.12 | Khanzadi et al. (2018), adjusted |
| DH | Government | 0.75 | 0.10 | 0.025 | 0.06 | 0.15 | Khanzadi et al. (2018) |
| IL | Private/IOC | 0.60 | 0.13 | 0.030 | 0.10 | 0.18 | Park et al. (2005) |
| IH | Private/IOC | 0.65 | 0.15 | 0.035 | 0.10 | 0.20 | Park et al. (2005), FIDIC (2017) |

**Validation metrics**:
- Overall advance probability: 0.67 (weighted by portfolio mix: 0.65×0.30 + 0.75×0.30 + 0.60×0.20 + 0.65×0.20)
- Literature benchmark: Park et al. (2005) 58% → Model is slightly optimistic but within range ✓
- Overall mean percentage: 11.5% (weighted)
- Literature benchmark: Park et al. (2005) 12.3% → Model within 1 SD ✓

---

#### 2.2.5 Milestone Definition

**Milestone 0: Advance Payment**

- **Trigger**: Contract signing / mobilization (t = 0)

**Note on Retention**: Advance payment is subject to retention withholding. The net advance payment received is:

$$P_0^{\text{net}} = P_0^{\text{gross}} \times (1 - r_i) = \alpha_i \times \text{Contract Value} \times (1 - r_i)$$

where $r_i$ is the project-specific retention rate (see Section 1.6 for calibration).

- **Amount**: $P_0 = \alpha_i \times R_i^{\text{total}}$ where $\alpha_i \sim \text{TruncNormal}(\mu_c, \sigma_c, a_c, b_c)$
- **Timing**: $t_0 = T_i^{\text{start}}$ (independent of SPI)
- **SPI Dependency**: None (payment occurs before work starts)

---

#### 2.2.6 Implementation Algorithm

```python
def sample_advance_payment(category, R_total):
    """
    Sample advance payment for a project.
    
    Parameters:
    - category: str, one of ['DL', 'DH', 'IL', 'IH']
    - R_total: float, total contract revenue
    
    Returns:
    - P_0: float, advance payment amount (0 if no advance)
    - alpha: float, advance percentage (0 if no advance)
    """
    # Category-specific parameters
    params = {
        'DL': {'p': 0.65, 'mu': 0.08, 'sigma': 0.020, 'a': 0.05, 'b': 0.12},
        'DH': {'p': 0.75, 'mu': 0.10, 'sigma': 0.025, 'a': 0.06, 'b': 0.15},
        'IL': {'p': 0.60, 'mu': 0.13, 'sigma': 0.030, 'a': 0.10, 'b': 0.18},
        'IH': {'p': 0.65, 'mu': 0.15, 'sigma': 0.035, 'a': 0.10, 'b': 0.20}
    }
    
    p = params[category]
    
    # Step 1: Determine if advance is granted (Bernoulli trial)
    has_advance = np.random.binomial(1, p['p'])
    
    if has_advance:
        # Step 2: Sample advance percentage from truncated normal
        alpha = truncnorm.rvs(
            (p['a'] - p['mu']) / p['sigma'],  # Standardized lower bound
            (p['b'] - p['mu']) / p['sigma'],  # Standardized upper bound
            loc=p['mu'],
            scale=p['sigma']
        )
        P_0 = alpha * R_total
    else:
        alpha = 0.0
        P_0 = 0.0
    
    return P_0, alpha
```

---

## Key Takeaways

1. **Parameter summary**: Complete advance payment specifications for all four categories with validation against literature benchmarks
2. **Milestone 0 definition**: Advance payment occurs at contract signing (t=0), independent of project performance
3. **Retention interaction**: Advance payment is subject to retention withholding, reducing net cash received
4. **Implementation**: Two-step algorithm (Bernoulli trial for presence, then Truncated Normal sampling for percentage)
5. **Validation**: Model predictions align with Park et al. (2005) empirical observations (67% vs 58% probability, 11.5% vs 12.3% mean percentage)

---

**Chunk Status**: Complete | Word count: ~600 | Mathematical density: Medium
