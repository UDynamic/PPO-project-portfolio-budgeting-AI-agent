# Phase 3 — Semantic Chunk Construction (continued)

---

## **CHUNK 5 of 6**

### **Metadata**
- **Chunk ID:** `4.7_chunk_05`
- **Sections Covered:** 4.7.6.2
- **Primary Focus:** Mathematical Formulation of BAC-Margin Correlation via Gaussian Copula
- **Dependencies:** BAC distribution (Chunks 2–3), profit margin distribution (Section 4.5), correlation evidence (Chunk 4)
- **Forward References:** Complete parameter table and implementation notes (Chunk 6)

---

### **Content**

**[Contextual overlap from Chunk 4]:**  
Empirical evidence supports a weak to moderate negative correlation between project size and profit margin ($\rho \approx -0.15$ central estimate), with segment-specific variations. The modeling challenge is to preserve the marginal distributions of BAC and profit margin while imposing the desired correlation structure.

---

#### 4.7.6.2 Mathematical Formulation: Gaussian Copula Approach

To model the joint distribution of project BAC and profit margin while preserving their individual (marginal) distributions, we employ a **Gaussian copula** framework. This approach separates the marginal behavior of each variable from their dependence structure.

**Rationale for copula methodology**:

1. **Marginal preservation**: BAC follows truncated lognormal; profit margin follows truncated normal (Section 4.5). Direct bivariate modeling would require specifying a joint distribution family that may not accommodate both marginals.
2. **Flexible correlation**: Copulas allow specification of dependence independently of marginal distributions.
3. **Computational tractability**: Gaussian copula admits closed-form transformations and efficient sampling algorithms.
4. **Interpretability**: Correlation parameter $\rho$ has intuitive interpretation as linear correlation in transformed (Gaussian) space.

---

**Step 1: Marginal Distribution Specification**

For project $i$ in the portfolio:

**Project BAC** (segment-specific):
$$\ln(\text{BAC}_i) \sim \text{TruncatedNormal}(\mu_{\ln}^{s}, \sigma_{\ln}^{s}, a_{\ln}, b_{\ln})$$

where $s \in \{\text{upstream}, \text{downstream}\}$ denotes project segment.

**Parameters** (from Chunks 2–3):
- Upstream: $\mu_{\ln}^{\text{up}} = 5.63$, $\sigma_{\ln}^{\text{up}} = 0.80$
- Downstream: $\mu_{\ln}^{\text{down}} = 5.01$, $\sigma_{\ln}^{\text{down}} = 0.70$
- Truncation bounds (both segments): $a_{\ln} = \ln(50) \approx 3.91$, $b_{\ln} = \ln(2000) \approx 7.60$

**Profit Margin** (from Section 4.5):
$$\text{Margin}_i \sim \text{TruncatedNormal}(\mu_m, \sigma_m, a_m, b_m)$$

**Parameters**:
- $\mu_m = 0.055$ (5.5% mean margin)
- $\sigma_m = 0.025$ (2.5% standard deviation)
- $a_m = -0.05$ (lower bound: -5% loss)
- $b_m = 0.15$ (upper bound: 15% margin)

---

**Step 2: Transformation to Uniform Marginals**

Define the **probability integral transform** (PIT) for each variable:

$$U_{\text{BAC}} = F_{\text{BAC}}(\text{BAC}_i) = \Phi_{\text{trunc}}\left(\frac{\ln(\text{BAC}_i) - \mu_{\ln}^{s}}{\sigma_{\ln}^{s}}; a_{\ln}, b_{\ln}\right)$$

$$U_{\text{Margin}} = F_{\text{Margin}}(\text{Margin}_i) = \Phi_{\text{trunc}}\left(\frac{\text{Margin}_i - \mu_m}{\sigma_m}; a_m, b_m\right)$$

where $\Phi_{\text{trunc}}(\cdot; a, b)$ is the CDF of a standard normal truncated to $[a, b]$:

$$\Phi_{\text{trunc}}(z; a, b) = \frac{\Phi(z) - \Phi(a)}{\Phi(b) - \Phi(a)}$$

**Property**: By construction, $U_{\text{BAC}}, U_{\text{Margin}} \sim \text{Uniform}(0, 1)$ marginally.

---

**Step 3: Gaussian Copula Specification**

Transform the uniform marginals to standard normal variates:

$$Z_{\text{BAC}} = \Phi^{-1}(U_{\text{BAC}})$$
$$Z_{\text{Margin}} = \Phi^{-1}(U_{\text{Margin}})$$

Impose correlation structure in the Gaussian space:

$$\begin{pmatrix} Z_{\text{BAC}} \\ Z_{\text{Margin}} \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)$$

where $\rho$ is the **copula correlation parameter**.

**Relationship to linear correlation**: The copula parameter $\rho$ is **not** equal to the Pearson correlation between BAC and Margin in their original scales. The relationship is:

$$\rho_{\text{Pearson}}(\text{BAC}, \text{Margin}) \approx \rho \cdot \sqrt{\frac{\text{Var}(\text{BAC})}{\text{Var}(\text{BAC})}} \cdot \text{adjustment factor}$$

For truncated distributions, the adjustment factor depends on truncation severity. Empirical calibration (via simulation) yields:

$$\rho_{\text{copula}} \approx 1.15 \cdot \rho_{\text{Pearson}}$$

**Calibration**: To achieve target Pearson correlation $\rho_{\text{Pearson}} = -0.15$ (from literature), set:

$$\rho_{\text{copula}} = -0.15 \times 1.15 = -0.173 \approx -0.17$$

**Segment-specific copula correlations**:
- Upstream: $\rho_{\text{copula}}^{\text{up}} = -0.20 \times 1.15 = -0.23$
- Downstream: $\rho_{\text{copula}}^{\text{down}} = -0.05 \times 1.15 = -0.06$

---

**Step 4: Sampling Algorithm**

To generate a correlated pair $(\text{BAC}_i, \text{Margin}_i)$ for project $i$:

1. **Sample correlated Gaussian variates**:
   $$\begin{pmatrix} Z_{\text{BAC}} \\ Z_{\text{Margin}} \end{pmatrix} = \mathbf{L} \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \end{pmatrix}$$
   
   where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, 1)$ independently, and $\mathbf{L}$ is the Cholesky decomposition of the correlation matrix:
   
   $$\mathbf{R} = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix} = \mathbf{L} \mathbf{L}^T = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1 - \rho^2} \end{pmatrix} \begin{pmatrix} 1 & \rho \\ 0 & \sqrt{1 - \rho^2} \end{pmatrix}$$
   
   **Explicit form**:
   $$Z_{\text{BAC}} = \epsilon_1$$
   $$Z_{\text{Margin}} = \rho \cdot \epsilon_1 + \sqrt{1 - \rho^2} \cdot \epsilon_2$$

2. **Transform to uniform marginals**:
   $$U_{\text{BAC}} = \Phi(Z_{\text{BAC}})$$
   $$U_{\text{Margin}} = \Phi(Z_{\text{Margin}})$$

3. **Invert marginal CDFs to obtain original-scale values**:
   
   **For BAC**:
   $$\ln(\text{BAC}_i) = \mu_{\ln}^{s} + \sigma_{\ln}^{s} \cdot \Phi_{\text{trunc}}^{-1}(U_{\text{BAC}}; a_{\ln}, b_{\ln})$$
   $$\text{BAC}_i = \exp(\ln(\text{BAC}_i))$$
   
   **For Margin**:
   $$\text{Margin}_i = \mu_m + \sigma_m \cdot \Phi_{\text{trunc}}^{-1}(U_{\text{Margin}}; a_m, b_m)$$

where $\Phi_{\text{trunc}}^{-1}(u; a, b)$ is the inverse CDF (quantile function) of truncated standard normal:

$$\Phi_{\text{trunc}}^{-1}(u; a, b) = \Phi^{-1}\left(\Phi(a) + u \cdot [\Phi(b) - \Phi(a)]\right)$$

---

**Step 5: Verification and Validation**

**Marginal preservation check**: After sampling $n = 10{,}000$ pairs, verify:
- $\ln(\text{BAC}_i)$ histogram matches truncated normal with specified parameters
- $\text{Margin}_i$ histogram matches truncated normal with specified parameters
- **Test**: Kolmogorov-Smirnov test, $p > 0.05$ confirms distributional match

**Correlation verification**: Compute sample Pearson correlation:
$$\hat{\rho}_{\text{Pearson}} = \text{corr}(\text{BAC}_1, \ldots, \text{BAC}_n; \text{Margin}_1, \ldots, \text{Margin}_n)$$

**Expected result**: $\hat{\rho}_{\text{Pearson}} \approx -0.15 \pm 0.02$ (95% CI for $n = 10{,}000$)

**Tail dependence check**: Gaussian copula exhibits **zero tail dependence** (asymptotic independence in extremes). This implies:
- Probability of joint extreme events (very large BAC **and** very low margin) is lower than under perfect tail dependence
- **Implication**: Model may underestimate risk of catastrophic portfolio outcomes if true dependence exhibits tail clustering
- **Mitigation**: Sensitivity analysis with Student-t copula ($\nu = 5$ degrees of freedom) to assess tail dependence impact

---

**Step 6: Portfolio-Level Implementation**

For a portfolio of $N$ projects (sampled from Section 4.7.3):

1. **Determine segment allocation**: Assign each project $i$ to upstream or downstream based on contractor's segment mix (e.g., 60% upstream, 40% downstream)

2. **For each project $i = 1, \ldots, N$**:
   - Identify segment $s_i \in \{\text{upstream}, \text{downstream}\}$
   - Select correlation parameter: $\rho_i = \rho_{\text{copula}}^{s_i}$
   - Execute sampling algorithm (Steps 1–3 above) to generate $(\text{BAC}_i, \text{Margin}_i)$

3. **Compute portfolio metrics**:
   - Total portfolio BAC: $\text{BAC}_{\text{portfolio}} = \sum_{i=1}^{N} \text{BAC}_i$
   - Weighted average margin: $\overline{\text{Margin}} = \frac{\sum_{i=1}^{N} \text{BAC}_i \cdot \text{Margin}_i}{\sum_{i=1}^{N} \text{BAC}_i}$
   - Portfolio profit: $\text{Profit}_{\text{portfolio}} = \sum_{i=1}^{N} \text{BAC}_i \cdot \text{Margin}_i$

---

**Computational Complexity**: $O(N)$ per portfolio realization (linear in number of projects). For Monte Carlo simulation with $M = 10{,}000$ portfolio realizations and average $N = 16$ projects, total sampling operations: $M \times N = 160{,}000$ (computationally trivial on modern hardware).

---

### **End of Chunk 5**

**Next Chunk Preview**: Chunk 6 consolidates all model parameters into a comprehensive reference table, provides implementation notes for numerical stability, discusses model limitations, and presents validation results from calibration against industry benchmarks.

---

**Status**: ✅ Chunk 5 extracted  
**Proceed to Chunk 6 (final chunk)?**