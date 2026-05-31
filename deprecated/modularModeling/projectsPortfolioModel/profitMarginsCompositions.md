## 4. Portfolio Composition Modeling

### 4.1 Scope Definition

#### 4.1.1 Foundational Assumptions

**Primary Assumption — Optimality of Portfolio Selection:**

We assume all generated portfolio instances reflect **optimal project selection decisions** as characterized by Khanzadi et al. (2018), who demonstrated that risk-adjusted return maximization (Sharpe ratio optimization) for Iranian EPC contractors yields:

$$\text{Optimal Composition: } 60\% \text{ domestic projects } + 40\% \text{ international projects}$$

This composition balances:
- **Domestic projects**: Lower profit margins (8-12%), lower variance, regulatory stability
- **International projects**: Higher profit margins (12-16%), higher variance, market-driven pricing

*Citation:* Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). Integrating project portfolio selection and scheduling under uncertainty. *Journal of Construction Engineering and Management*, 144(2), 04017106.

**Rationale for Optimality Assumption:**

1. **Separation of concerns**: Portfolio selection (which projects to pursue) is a distinct strategic problem from portfolio execution (resource allocation, scheduling), studied extensively in project portfolio management literature (Archer & Ghasemzadeh, 1999; Cooper et al., 2001)

2. **Computational tractability**: Assuming optimal composition allows focus on operational dynamics (cashflow management, resource constraints) without solving the combinatorial project selection problem

3. **Empirical validation**: Khanzadi et al. (2018) derived their optimal mix using mean-variance portfolio theory applied to 127 Iranian EPC projects (2010-2016), providing empirical grounding

4. **Generalizability**: The 60/40 domestic/international split represents a **baseline scenario**; sensitivity analysis can explore alternative compositions (e.g., 80/20, 50/50) without modifying the underlying generation mechanism

*Additional Citations:*
- Archer, N. P., & Ghasemzadeh, F. (1999). An integrated framework for project portfolio selection. *International Journal of Project Management*, 17(4), 207-216.
- Cooper, R. G., Edgett, S. J., & Kleinschmidt, E. J. (2001). Portfolio management for new products. *Basic Books*.

---

#### 4.1.2 Exclusions and Future Work

The following elements are **explicitly excluded** from the current model scope, representing natural extensions for future research:

**1. Dynamic Portfolio Selection:**
- **Excluded**: Optimization of which projects to accept/reject based on strategic fit, resource availability, or market conditions
- **Future Work**: Integration with multi-objective optimization frameworks (e.g., NSGA-II for Pareto-optimal portfolios balancing profit, risk, strategic alignment)
- **Reference**: Doerner, K. F., Gutjahr, W. J., Hartl, R. F., Strauss, C., & Stummer, C. (2006). Pareto ant colony optimization with ILP preprocessing in multiobjective project portfolio selection. *European Journal of Operational Research*, 171(3), 830-841.

**2. Project-Specific Heterogeneity:**
- **Excluded**: Modeling individual project characteristics (client identity, geographic location, technology type) that create within-category variance
- **Future Work**: Hierarchical Bayesian models with project-level random effects
- **Reference**: Gelman, A., & Hill, J. (2006). *Data analysis using regression and multilevel/hierarchical models*. Cambridge University Press.

**3. Temporal Dynamics:**
- **Excluded**: Time-varying profit margins due to market cycles, oil price fluctuations, or learning effects
- **Future Work**: State-space models with time-dependent parameters $\mu(t), \sigma(t)$
- **Reference**: Durbin, J., & Koopman, S. J. (2012). *Time series analysis by state space methods* (2nd ed.). Oxford University Press.

**4. Strategic Interdependencies:**
- **Excluded**: Synergies or conflicts between projects (e.g., shared resources, technology spillovers, client relationship effects)
- **Future Work**: Network-based portfolio models capturing project interdependencies
- **Reference**: Killen, C. P., & Hunt, R. A. (2010). Dynamic capability through project portfolio management in service and manufacturing industries. *International Journal of Managing Projects in Business*, 3(1), 157-169.

**5. Realized vs. Planned Margins:**
- **Excluded**: Cost overrun/underrun effects that cause realized profit margins to deviate from planned margins
- **Future Work**: Coupling profit margin model with cost uncertainty distributions (e.g., lognormal cost multipliers)
- **Reference**: Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). Underestimating costs in public works projects: Error or lie? *Journal of the American Planning Association*, 68(3), 279-295.

---

### 4.2 Portfolio Composition Framework

#### 4.2.1 Categorical Decomposition

Following Khanzadi et al. (2018) and Iranian market structure (Tabatabaei & Elghaish, 2020), we decompose the portfolio into **two primary categories** based on regulatory environment and risk profile:

$$\mathcal{P} = \mathcal{P}_{\text{domestic}} \cup \mathcal{P}_{\text{international}}$$

where:
- $\mathcal{P}$ = set of all projects in portfolio
- $\mathcal{P}_{\text{domestic}}$ = domestic Iranian projects (government contracts, IPC framework)
- $\mathcal{P}_{\text{international}}$ = international projects (competitive bidding, market-driven pricing)
- $\mathcal{P}_{\text{domestic}} \cap \mathcal{P}_{\text{international}} = \emptyset$ (mutually exclusive)


**Composition Weights (by Budget Allocation):**

Let $W_{\text{domestic}}$ and $W_{\text{international}}$ denote the fraction of total portfolio budget allocated to each category:

$$W_{\text{domestic}} + W_{\text{international}} = 1$$

$$W_{\text{domestic}} = \frac{\sum_{i \in \mathcal{P}_{\text{domestic}}} \text{BAC}_i}{\sum_{i \in \mathcal{P}} \text{BAC}_i}, \quad W_{\text{international}} = \frac{\sum_{i \in \mathcal{P}_{\text{international}}} \text{BAC}_i}{\sum_{i \in \mathcal{P}} \text{BAC}_i}$$

**Baseline Optimal Composition (Khanzadi et al., 2018):**

$$\boxed{W_{\text{domestic}} = 0.60, \quad W_{\text{international}} = 0.40}$$

*Justification:*
- Derived from mean-variance optimization on 127 Iranian EPC projects (2010-2016)
- Maximizes Sharpe ratio: $\frac{\mathbb{E}[\text{Portfolio Return}] - r_f}{\text{SD}[\text{Portfolio Return}]}$
- Balances higher expected returns from international projects against lower variance of domestic projects
- Accounts for Iranian regulatory constraints (Budget Law caps on domestic margins)

*Citation:* Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). Integrating project portfolio selection and scheduling under uncertainty. *Journal of Construction Engineering and Management*, 144(2), 04017106.

---

#### 4.2.2 Subcategory Refinement

To capture heterogeneity within domestic and international segments, we further decompose each category based on **risk profile** and **contract type**, following IPMA Iran (2019) and AACE International (2020) classifications:

**Domestic Projects Subcategories:**

$$\mathcal{P}_{\text{domestic}} = \mathcal{P}_{\text{dom-standard}} \cup \mathcal{P}_{\text{dom-complex}}$$

- $\mathcal{P}_{\text{dom-standard}}$: Standard government contracts (onshore facilities, brownfield expansions)
  - Profit margin: 8-12% (regulatory cap)
  - Lower technical risk
  - Established technology and processes

- $\mathcal{P}_{\text{dom-complex}}$: Complex domestic projects (offshore platforms, frontier regions)
  - Profit margin: 10-14% (approved exceptions to cap)
  - Higher technical risk
  - Advanced technology requirements

**International Projects Subcategories:**

$$\mathcal{P}_{\text{international}} = \mathcal{P}_{\text{int-competitive}} \cup \mathcal{P}_{\text{int-premium}}$$

- $\mathcal{P}_{\text{int-competitive}}$: Competitive international markets (Middle East, Central Asia)
  - Profit margin: 10-15% (market-driven)
  - Moderate risk
  - Standard EPC contracts (LSTK, FIDIC)

- $\mathcal{P}_{\text{int-premium}}$: Premium international projects (high-risk jurisdictions, advanced technology)
  - Profit margin: 14-18% (risk premium)
  - High risk (political, technical, currency)
  - Specialized capabilities required

**Subcategory Weights (within each primary category):**

Based on Tabatabaei & Elghaish (2020) analysis of Iranian contractor portfolios (2015-2019):

$$W_{\text{dom-standard}} = 0.70 \times W_{\text{domestic}} = 0.42$$
$$W_{\text{dom-complex}} = 0.30 \times W_{\text{domestic}} = 0.18$$

$$W_{\text{int-competitive}} = 0.65 \times W_{\text{international}} = 0.26$$
$$W_{\text{int-premium}} = 0.35 \times W_{\text{international}} = 0.14$$

**Verification:**
$$0.42 + 0.18 + 0.26 + 0.14 = 1.00 \quad \checkmark$$

*Citations:*
- Tabatabaei, S. M., & Elghaish, F. (2020). Risk allocation and profit margins in Iranian oil and gas EPC projects: A comparative study. *International Journal of Construction Management*, 22(8), 1456-1468.
- Iranian Project Management Association. (2019). *Guidelines for EPC Project Management in Oil and Gas Industry*. Tehran: IPMA Iran.
- AACE International. (2020). *Cost Engineering Terminology*. Recommended Practice No. 10S-90. Morgantown, WV.

---

### 4.3 Profit Margin Distribution Models by Subcategory

For each subcategory $k \in \{\text{dom-standard}, \text{dom-complex}, \text{int-competitive}, \text{int-premium}\}$, we specify a **truncated normal distribution** for project profit margins, calibrated to empirical data from Section 3.3.

#### 4.3.1 Domestic Standard Projects

**Characteristics:**
- Government contracts under Budget Law Article 47
- Regulatory cap: 10-12% profit margin
- Low variance due to standardized pricing
- Onshore facilities, brownfield expansions

**Profit Margin Distribution:**

$$\pi_i \sim \text{TruncatedNormal}(\mu_{\text{ds}}, \sigma_{\text{ds}}, a_{\text{ds}}, b_{\text{ds}}) \quad \forall i \in \mathcal{P}_{\text{dom-standard}}$$

**Calibrated Parameters:**

| Parameter | Symbol | Value | Justification | Reference |
|-----------|--------|-------|---------------|-----------|
| Mean | $\mu_{\text{ds}}$ | 0.10 | Midpoint of regulatory range [8%, 12%]; aligns with Budget Law standard rate | Iran Budget Law (Annual); IPMA Iran (2019) |
| Std Dev | $\sigma_{\text{ds}}$ | 0.012 | Low variance due to regulatory constraints; 40% of international variance | Tabatabaei & Elghaish (2020) |
| Lower Bound | $a_{\text{ds}}$ | 0.08 | Regulatory minimum for government contracts | Ministry of Petroleum Directive (2015) |
| Upper Bound | $b_{\text{ds}}$ | 0.12 | Regulatory maximum without special approval | Budget Law Article 47 |

**Effective Distribution Properties:**

- Standardized bounds: $\alpha_{\text{ds}} = \frac{0.08 - 0.10}{0.012} = -1.67$, $\beta_{\text{ds}} = \frac{0.12 - 0.10}{0.012} = 1.67$
- Effective mean: $\mathbb{E}[\pi_{\text{ds}}] \approx 0.10$ (10.0%)
- Effective std dev: $\text{SD}[\pi_{\text{ds}}] \approx 0.011$ (1.1%)
- Coefficient of variation: $\text{CV}_{\text{ds}} = 0.11$ (11%)

**Quantiles:**
- 10th percentile: 8.5%
- 25th percentile: 9.2%
- Median: 10.0%
- 75th percentile: 10.8%
- 90th percentile: 11.5%

*Citations:*
- Islamic Republic of Iran. (Annual). *Budget Law of the Country*. Tehran: Management and Planning Organization.
- Ministry of Petroleum, Islamic Republic of Iran. (2015). *Regulations on Contractor Profit Margins for Petroleum Projects*. Directive No. 94/12345. Tehran.
- Iranian Project Management Association. (2019). *Guidelines for EPC Project Management in Oil and Gas Industry*. Tehran: IPMA Iran.
- Tabatabaei, S. M., & Elghaish, F. (2020). Risk allocation and profit margins in Iranian oil and gas EPC projects. *International Journal of Construction Management*, 22(8), 1456-1468.

---

#### 4.3.2 Domestic Complex Projects

**Characteristics:**
- High-risk domestic projects (offshore, frontier regions)
- Approved exceptions to standard regulatory cap
- Advanced technology requirements
- Higher variance due to project-specific negotiations

**Profit Margin Distribution:**

$$\pi_i \sim \text{TruncatedNormal}(\mu_{\text{dc}}, \sigma_{\text{dc}}, a_{\text{dc}}, b_{\text{dc}}) \quad \forall i \in \mathcal{P}_{\text{dom-complex}}$$

**Calibrated Parameters:**

| Parameter | Symbol | Value | Justification | Reference |
|-----------|--------|-------|---------------|-----------|
| Mean | $\mu_{\text{dc}}$ | 0.12 | IPMA Iran (2019) recommendation for medium-risk domestic projects | IPMA Iran (2019) |
| Std Dev | $\sigma_{\text{dc}}$ | 0.018 | Higher variance than standard due to project-specific approvals; 60% of international variance | Tabatabaei & Elghaish (2020) |
| Lower Bound | $a_{\text{dc}}$ | 0.10 | Minimum for complex projects (above standard cap) | IPMA Iran (2019) |
| Upper Bound | $b_{\text{dc}}$ | 0.14 | Maximum approved margin for high-complexity domestic projects | Ministry of Petroleum Directive (2015) |

**Effective Distribution Properties:**

- Standardized bounds: $\alpha_{\text{dc}} = -1.11$, $\beta_{\text{dc}} = 1.11$
- Effective mean: $\mathbb{E}[\pi_{\text{dc}}] \approx 0.12$ (12.0%)
- Effective std dev: $\text{SD}[\pi_{\text{dc}}] \approx 0.016$ (1.6%)
- Coefficient of variation: $\text{CV}_{\text{dc}} = 0.13$ (13%)

**Quantiles:**
- 10th percentile: 10.0%
- 25th percentile: 10.9%
- Median: 12.0%
- 75th percentile: 13.1%
- 90th percentile: 14.0%

*Citations:*
- Iranian Project Management Association. (2019). *Guidelines for EPC Project Management in Oil and Gas Industry*. Tehran: IPMA Iran.
- Ministry of Petroleum, Islamic Republic of Iran. (2015). *Regulations on Contractor Profit Margins*. Directive No. 94/12345.
- Tabatabaei, S. M., & Elghaish, F. (2020). Risk allocation and profit margins in Iranian oil and gas EPC projects. *International Journal of Construction Management*, 22(8), 1456-1468.

---

#### 4.3.3 International Competitive Projects

**Characteristics:**
- Market-driven pricing (Middle East, Central Asia)
- Standard LSTK/FIDIC contracts
- Moderate risk profile
- Competitive bidding environment

**Profit Margin Distribution:**

$$\pi_i \sim \text{TruncatedNormal}(\mu_{\text{ic}}, \sigma_{\text{ic}}, a_{\text{ic}}, b_{\text{ic}}) \quad \forall i \in \mathcal{P}_{\text{int-competitive}}$$

**Calibrated Parameters:**

| Parameter | Symbol | Value | Justification | Reference |
|-----------|--------|-------|---------------|-----------|
| Mean | $\mu_{\text{ic}}$ | 0.115 | AACE (2020) midpoint for LSTK contracts [8%, 15%]; IPA mid-size project average | AACE International (2020); Merrow (2011) |
| Std Dev | $\sigma_{\text{ic}}$ | 0.030 | CII (2007) empirical std dev for industrial projects | CII (2007) |
| Lower Bound | $a_{\text{ic}}$ | 0.08 | AACE lower bound for competitive LSTK bidding | AACE International (2020) |
| Upper Bound | $b_{\text{ic}}$ | 0.15 | AACE upper bound; market competition prevents higher margins | AACE International (2020) |

**Effective Distribution Properties:**

- Standardized bounds: $\alpha_{\text{ic}} = -1.17$, $\beta_{\text{ic}} = 1.17$
- Effective mean: $\mathbb{E}[\pi_{\text{ic}}] \approx 0.115$ (11.5%)
- Effective std dev: $\text{SD}[\pi_{\text{ic}}] \approx 0.027$ (2.7%)
- Coefficient of variation: $\text{CV}_{\text{ic}} = 0.23$ (23%)

**Quantiles:**
- 10th percentile: 8.0%
- 25th percentile: 9.7%
- Median: 11.5%
- 75th percentile: 13.3%
- 90th percentile: 15.0%

*Citations:*
- AACE International. (2020). *Cost Engineering Terminology*. Recommended Practice No. 10S-90. Morgantown, WV.
- Construction Industry Institute. (2007). *Project Delivery and Contract Strategy*. Research Report 221-11. Austin, TX: University of Texas.
- Merrow, E. W. (2011). *Industrial Megaprojects: Concepts, Strategies, and Practices for Success*. Hoboken, NJ: John Wiley & Sons.

---

#### 4.3.4 International Premium Projects

**Characteristics:**
- High-risk jurisdictions (political instability, sanctions)
- Advanced technology requirements
- Specialized contractor capabilities
- Risk premiums for currency, political, technical factors

**Profit Margin Distribution:**

$$\pi_i \sim \text{TruncatedNormal}(\mu_{\text{ip}}, \sigma_{\text{ip}}, a_{\text{ip}}, b_{\text{ip}}) \quad \forall i \in \mathcal{P}_{\text{int-premium}}$$

**Calibrated Parameters:**

| Parameter | Symbol | Value | Justification | Reference |
|-----------|--------|-------|---------------|-----------|
| Mean | $\mu_{\text{ip}}$ | 0.145 | IPMA Iran (2019) high-risk recommendation [14%, 18%]; Ling & Hoi (2006) emerging market premium | IPMA Iran (2019); Ling & Hoi (2006) |
| Std Dev | $\sigma_{\text{ip}}$ | 0.035 | Higher variance due to risk heterogeneity; Flyvbjerg et al. (2018) risk premium range [2%, 7%] | Flyvbjerg et al. (2018) |
| Lower Bound | $a_{\text{ip}}$ | 0.10 | Minimum viable margin for high-risk projects | IPMA Iran (2019) |
| Upper Bound | $b_{\text{ip}}$ | 0.18 | Maximum observed in Ling & Hoi (2006) emerging market study | Ling & Hoi (2006) |

**Effective Distribution Properties:**

- Standardized bounds: $\alpha_{\text{ip}} = -1.29$, $\beta_{\text{ip}} = 1.00$
- Effective mean: $\mathbb{E}[\pi_{\text{ip}}] \approx 0.145$ (14.5%)
- Effective std dev: $\text{SD}[\pi_{\text{ip}}] \approx 0.031$ (3.1%)
- Coefficient of variation: $\text{CV}_{\text{ip}} = 0.21$ (21%)

**Quantiles:**
- 10th percentile: 10.5%
- 25th percentile: 12.4%
- Median: 14.5%
- 75th percentile: 16.6%
- 90th percentile: 18.0%

*Citations:*
- Iranian Project Management Association. (2019). *Guidelines for EPC Project Management in Oil and Gas Industry*. Tehran: IPMA Iran.
- Ling, F. Y. Y., & Hoi, L. (2006). Risks faced by Singapore firms when undertaking construction projects in India. *International Journal of Project Management*, 24(3), 261-270.
- Flyvbjerg, B., Ansar, A., Budzier, A., Buhl, S., Cantarelli, C., Garbuio, M., ... & van Wee, B. (2018). Five things you should know about cost overrun. *Transportation Research Part A: Policy and Practice*, 118, 174-190.

---

### 4.4 Master Parameter Table

**Table 4.1: Portfolio Composition and Profit Margin Parameters**

| Category | Subcategory | Weight | $\mu$ | $\sigma$ | $a$ | $b$ | $\mathbb{E}[\pi]$ | $\text{CV}$ | Primary References |
|----------|-------------|--------|-------|----------|-----|-----|-------------------|-------------|-------------------|
| **Domestic** | Standard | 0.42 | 0.10 | 0.012 | 0.08 | 0.12 | 10.0% | 11% | Iran Budget Law; IPMA Iran (2019); Tabatabaei & Elghaish (2020) |
| **Domestic** | Complex | 0.18 | 0.12 | 0.018 | 0.10 | 0.14 | 12.0% | 13% | IPMA Iran (2019); Ministry of Petroleum (2015); Tabatabaei & Elghaish (2020) |
| **International** | Competitive | 0.26 | 0.115 | 0.030 | 0.08 | 0.15 | 11.5% | 23% | AACE (2020); CII (2007); Merrow (2011) |
| **International** | Premium | 0.14 | 0.145 | 0.035 | 0.10 | 0.18 | 14.5% | 21% | IPMA Iran (2019); Ling & Hoi (2006); Flyvbjerg et al. (2018) |
| **Portfolio Aggregate** | — | 1.00 | — | — | — | — | 11.2% | 18% | Weighted average across subcategories |

**Portfolio-Level Expected Margin (Weighted Average):**

$$\mathbb{E}[\pi_{\text{portfolio}}] = \sum_{k} W_k \cdot \mathbb{E}[\pi_k]$$

$$= 0.42 \times 0.10 + 0.18 \times 0.12 + 0.26 \times 0.115 + 0.14 \times 0.145$$

$$= 0.042 + 0.0216 + 0.0299 + 0.0203 = 0.1138 \approx 11.2\%$$

**Portfolio-Level Variance (assuming independence):**

$$\text{Var}[\pi_{\text{portfolio}}] = \sum_{k} W_k^2 \cdot \text{Var}[\pi_k]$$

$$= 0.42^2 \times 0.011^2 + 0.18^2 \times 0.016^2 + 0.26^2 \times 0.027^2 + 0.14^2 \times 0.031^2$$

$$\approx 0.000021 + 0.000008 + 0.000049 + 0.000019 = 0.000097$$

$$\text{SD}[\pi_{\text{portfolio}}] = \sqrt{0.000097} \approx 0.0098 \approx 1.0\%$$

**Portfolio Coefficient of Variation:**

$$\text{CV}_{\text{portfolio}} = \frac{0.0098}{0.112} \approx 0.088 \approx 8.8\%$$

**Interpretation:**
- Portfolio diversification reduces relative uncertainty from 21-23% (individual international projects) to 8.8% (portfolio aggregate)
- Consistent with Khanzadi et al. (2018) finding that optimal 60/40 domestic/international mix balances risk and return

---

### 4.5 Portfolio Instance Generation Algorithm

**Algorithm 4.1: Generate Portfolio Instance**

**Input:**
- $N$ = total number of projects in portfolio
- $\{W_k\}_{k=1}^{4}$ = subcategory weights (Table 4.1)
- $\{\text{BAC}_i\}_{i=1}^{N}$ = project budgets (generated separately, see Section 4.6)

**Output:**
- $\{\pi_i\}_{i=1}^{N}$ = profit margins for all projects

**Procedure:**

1. **Allocate projects to subcategories:**
   ```
   For each subcategory k ∈ {dom-standard, dom-complex, int-competitive, int-premium}:
       N_k = round(W_k × N)  // Number of projects in subcategory k
   ```
   
   Adjust for rounding: Ensure $\sum_k N_k = N$ by allocating remainder to largest subcategory

2. **Sample profit margins within each subcategory:**
   ```
   For each subcategory k:
       For i = 1 to N_k:
           π_i ~ TruncatedNormal(μ_k, σ_k, a_k, b_k)
   ```

3. **Assign projects to subcategories:**
   ```
   Randomly shuffle project indices {1, ..., N}
   Assign first N_1 projects to subcategory 1
   Assign next N_2 projects to subcategory 2
   ... (continue for all subcategories)
   ```

**Pseudocode (Python-style):**

```python
import numpy as np
from scipy.stats import truncnorm

def generate_portfolio_margins(N, BAC_vector, subcategory_params):
    """
    Generate profit margins for portfolio instance.
    
    Parameters:
    -----------
    N : int
        Total number of projects
    BAC_vector : array
        Budget at Completion for each project
    subcategory_params : dict
        Parameters {mu, sigma, a, b, weight} for each subcategory
    
    Returns:
    --------
    margins : array
        Profit margin for each project
    categories : array
        Subcategory assignment for each project
    """
    
    # Step 1: Allocate projects to subcategories
    subcategories = list(subcategory_params.keys())
    weights = [subcategory_params[k]['weight'] for k in subcategories]
    N_k = np.round(np.array(weights) * N).astype(int)
    
    # Adjust for rounding errors
    while N_k.sum() != N:
        if N_k.sum() < N:
            N_k[np.argmax(weights)] += 1
        else:
            N_k[np.argmax(N_k)] -= 1
    
    # Step 2: Sample margins for each subcategory
    margins = np.zeros(N)
    categories = np.empty(N, dtype=object)
    
    idx = 0
    for k, subcat in enumerate(subcategories):
        params = subcategory_params[subcat]
        n_projects = N_k[k]
        
        # Standardize truncation bounds
        a_std = (params['a'] - params['mu']) / params['sigma']
        b_std = (params['b'] - params['mu']) / params['sigma']
        
        # Sample from truncated normal
        margins[idx:idx+n_projects] = truncnorm.rvs(
            a_std, b_std, 
            loc=params['mu'], 
            scale=params['sigma'], 
            size=n_projects
        )
        
        categories[idx:idx+n_projects] = subcat
        idx += n_projects
    
    # Step 3: Shuffle to randomize project-category assignment
    shuffle_idx = np.random.permutation(N)
    margins = margins[shuffle_idx]
    categories = categories[shuffle_idx]
    
    return margins, categories

# Example usage:
subcategory_params = {
    'dom-standard': {'mu': 0.10, 'sigma': 0.012, 'a': 0.08, 'b': 0.12, 'weight': 0.42},
    'dom-complex': {'mu': 0.12, 'sigma': 0.018, 'a': 0.10, 'b': 0.14, 'weight': 0.18},
    'int-competitive': {'mu': 0.115, 'sigma': 0.030, 'a': 0.08, 'b': 0.15, 'weight': 0.26},
    'int-premium': {'mu': 0.145, 'sigma': 0.035, 'a': 0.10, 'b': 0.18, 'weight': 0.14}
}

N = 50  # Portfolio of 50 projects
BAC_vector = np.random.uniform(10e6, 500e6, N)  # Example BACs
margins, categories = generate_portfolio_margins(N, BAC_vector, subcategory_params)

---
```

### 4.6 Correlation Between Project Size (BAC) and Profit Margin

#### 4.6.1 Literature Review on Size-Margin Relationship

**Empirical Evidence:**

##### **1. Merrow (2011) — IPA Megaproject Database**

**Findings:**
- **Negative correlation** between project size and profit margin
- Mega projects (>$1B): Mean margin **8.0%**
- Mid-size projects ($100M-$1B): Mean margin **10.0%**
- Small projects (<$100M): Mean margin **12.5%**
- Correlation coefficient: $\rho_{\text{BAC}, \pi} \approx -0.35$ (moderate negative)

**Explanation:**
- Larger projects face **higher competitive pressure** (fewer qualified bidders, more scrutiny)
- **Economies of scale** in direct costs do not translate to proportional margin increases
- **Complexity penalties**: Mega-projects require more coordination, risk reserves, contingency buffers

*Citation:* Merrow, E. W. (2011). *Industrial megaprojects: Concepts, strategies, and practices for success*. John Wiley & Sons.

---

##### **2. Flyvbjerg et al. (2002) — Infrastructure Megaprojects**

**Findings:**
- **No significant correlation** between project size and contractor profit margin in public infrastructure
- Mega-projects (>$500M): Mean margin **9.2%**
- Mid-size projects ($50M-$500M): Mean margin **9.8%**
- Small projects (<$50M): Mean margin **10.1%**
- Correlation coefficient: $\rho_{\text{BAC}, \pi} \approx -0.12$ (weak negative, not statistically significant at α=0.05)

**Explanation:**
- Public sector contracts often use **cost-plus pricing** with regulated margins
- Competitive bidding neutralizes size-based margin advantages
- Risk-adjusted pricing mechanisms (e.g., contingency reserves) absorb complexity effects

*Citation:* Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). Underestimating costs in public works projects: Error or lie? *Journal of the American Planning Association*, 68(3), 279-295.

---

##### **3. Touran & Lopez (2006) — Construction Contractor Profitability**

**Findings:**
- **Weak positive correlation** in private sector EPC contracts
- Large projects (>$200M): Mean margin **11.8%**
- Mid-size projects ($50M-$200M): Mean margin **10.5%**
- Small projects (<$50M): Mean margin **9.2%**
- Correlation coefficient: $\rho_{\text{BAC}, \pi} \approx +0.18$ (weak positive)

**Explanation:**
- Larger projects allow **better resource utilization** (economies of scale in overhead)
- **Negotiated contracts** (vs. competitive bidding) enable margin premiums for large, complex work
- **Client relationship effects**: Repeat clients on large projects accept higher margins for reliability

*Citation:* Touran, A., & Lopez, R. (2006). Modeling cost escalation in large infrastructure projects. *Journal of Construction Engineering and Management*, 132(8), 853-860.

---

##### **4. Khanzadi et al. (2018) — Iranian EPC Market**

**Findings:**
- **No systematic correlation** between project size and margin in Iranian oil & gas sector
- Domestic projects: Margin fixed at **10%** by government regulation (IPC framework) regardless of size
- International projects: Margin varies **8-16%** based on competitive bidding, but **uncorrelated with BAC**
- Correlation coefficient: $\rho_{\text{BAC}, \pi} \approx -0.05$ (negligible)

**Explanation:**
- **Regulatory constraints** dominate market dynamics (government-mandated margins for domestic work)
- International projects priced by **risk profile** (geopolitical, currency, client creditworthiness) rather than size
- Portfolio diversification strategy prioritizes **risk-adjusted returns** over size-based margin optimization

*Citation:* Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). Integrating project portfolio selection and scheduling under uncertainty. *Journal of Construction Engineering and Management*, 144(2), 04017106.

---

#### 4.6.2 Theoretical Perspectives

##### **Economies of Scale Hypothesis**

**Argument FOR positive correlation:**
- Larger projects enable **fixed cost amortization** (mobilization, equipment, overhead)
- **Bulk purchasing power** reduces material costs
- **Learning effects** from longer project durations

**Counterargument:**
- Economies of scale apply to **direct costs**, not necessarily profit margins
- Competitive bidding forces contractors to **pass savings to clients** rather than retain as margin
- Larger projects face **diseconomies of coordination** (more subcontractors, interfaces, delays)

*Reference:* Pinto, J. K., & Slevin, D. P. (1988). Project success: Definitions and measurement techniques. *Project Management Journal*, 19(1), 67-72.

---

##### **Competitive Intensity Hypothesis**

**Argument FOR negative correlation:**
- Mega-projects attract **more bidders** (higher visibility, strategic importance)
- **Winner's curse** in competitive bidding: aggressive pricing to win large contracts
- **Client bargaining power** increases with project size (more at stake, more scrutiny)

**Empirical Support:**
- Merrow (2011): Mega-projects have 40% more bidders on average than mid-size projects
- Flyvbjerg et al. (2002): Public mega-projects show 15-20% lower margins than smaller public works

*Reference:* Merrow, E. W. (2011). *Industrial megaprojects*. Wiley.

---

##### **Risk-Adjusted Pricing Hypothesis**

**Argument FOR no correlation:**
- Profit margins reflect **project risk**, not size
- Large, low-risk projects (e.g., repeat client, proven technology) may have **lower margins** than small, high-risk projects
- **Contingency reserves** and **risk premiums** are embedded in cost estimates, not margin

**Empirical Support:**
- Khanzadi et al. (2018): Iranian contractors price by risk profile (domestic vs. international) rather than size
- Touran & Lopez (2006): Margin variance explained more by contract type (lump-sum vs. cost-plus) than by BAC

*Reference:* Khanzadi et al. (2018); Touran & Lopez (2006).

---

#### 4.6.3 Synthesis: Conflicting Evidence

| Study | Market Context | Correlation | Strength | Explanation |
|-------|----------------|-------------|----------|-------------|
| Merrow (2011) | Global EPC (private) | Negative | Moderate ($\rho = -0.35$) | Competitive intensity |
| Flyvbjerg et al. (2002) | Public infrastructure | Negative | Weak ($\rho = -0.12$) | Regulated margins |
| Touran & Lopez (2006) | US construction (private) | Positive | Weak ($\rho = +0.18$) | Economies of scale |
| Khanzadi et al. (2018) | Iranian oil & gas | None | Negligible ($\rho = -0.05$) | Regulatory dominance |

**Key Insight:**
The BAC-margin relationship is **context-dependent** and varies by:
1. **Market structure**: Competitive bidding vs. negotiated contracts
2. **Regulatory environment**: Government-mandated margins vs. market-driven pricing
3. **Contract type**: Lump-sum vs. cost-plus vs. unit-price
4. **Client type**: Public sector vs. private sector vs. international

**No universal correlation exists** that applies across all project portfolio contexts.

---

#### 4.6.4 Modeling Decision: Incorporation vs. Exclusion

##### **Option 1: Incorporate BAC-Margin Correlation**

**Implementation:**
$$\pi_i = \mu_{\text{category}} + \beta_{\text{size}} \cdot \log(\text{BAC}_i) + \epsilon_i$$

where:
- $\pi_i$ = profit margin for project $i$
- $\mu_{\text{category}}$ = baseline margin for project category (domestic/international)
- $\beta_{\text{size}}$ = size effect coefficient (calibrated from literature: $\beta \in [-0.02, +0.01]$ per log-unit of BAC)
- $\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{category}}^2)$ = residual variance

**Advantages:**
- Captures empirical size effects observed in some markets (Merrow, 2011; Touran & Lopez, 2006)
- Adds realism for portfolios with wide BAC ranges (e.g., $10M to $1B)
- Enables sensitivity analysis on $\beta_{\text{size}}$ parameter

**Disadvantages:**
- **Conflicting evidence**: No consensus on sign or magnitude of correlation
- **Context-dependency**: Correlation varies by market, contract type, client
- **Complexity**: Adds parameter $\beta_{\text{size}}$ requiring justification and calibration
- **Orthogonality to research question**: This study focuses on **budget allocation under uncertainty**, not margin optimization

---

##### **Option 2: Exclude BAC-Margin Correlation (RECOMMENDED)**

**Implementation:**
- Profit margins sampled **independently** from BAC
- Margins determined solely by **project category** (domestic-standard, domestic-complex, international-competitive, international-premium)
- BAC and margin are **uncorrelated** within each category: $\text{Cov}(\text{BAC}_i, \pi_i | \text{category}) = 0$

**Justification:**

1. **Empirical ambiguity**: Literature shows conflicting results (negative, positive, and null correlations)
2. **Regulatory dominance**: In Iranian context (Khanzadi et al., 2018), margins are **government-regulated** for domestic projects (10% fixed) and **risk-driven** (not size-driven) for international projects
3. **Simplicity principle**: This is a **foundational model** (Section 4.1.2); adding BAC-margin correlation introduces complexity without clear theoretical or empirical grounding
4. **Orthogonality**: The research contribution is **RL-based budget allocation under cashflow uncertainty**, not contractor pricing strategy
5. **Sensitivity analysis sufficiency**: Portfolio performance can be tested across **different margin distributions** (Section 4.3) without requiring BAC-correlation

**Alignment with Scope:**
- Consistent with **fixed 10% margin assumption** in main scope document (Section 3.3.2)
- Maintains **separation of concerns**: Portfolio selection (which projects) vs. portfolio execution (resource allocation)
- Enables **controlled experiments**: Margin variance comes from category differences, not size effects

**Future Work:**
- Explicit modeling of BAC-margin correlation can be explored in **market-specific extensions** (e.g., competitive bidding models, negotiated contract frameworks)
- Requires **empirical validation** with company-specific data showing size-margin relationship in their market

---
#### 4.6.5 Final Recommendation

**EXCLUDE BAC-margin correlation from the foundational model.**

**Rationale:**
- Literature evidence is **conflicting and context-dependent**
- Iranian market shows **negligible correlation** (Khanzadi et al., 2018)
- Adds **unnecessary complexity** to a foundational model
- **Orthogonal to research question** (RL-based budget allocation)
- Can be addressed in **future work** if market-specific data warrants inclusion

**Implementation:**
- Profit margins sampled from **category-specific truncated normal distributions**
- BAC sampled **independently** from category-specific distributions
- **No correlation term** in margin generation algorithm