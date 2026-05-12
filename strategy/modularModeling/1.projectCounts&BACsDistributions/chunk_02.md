# Phase 3 — Semantic Chunk Construction (continued)

---

## **CHUNK 2 of 6**

### **Metadata**
- **Chunk ID:** `4.7_chunk_02`
- **Sections Covered:** 4.7.4 → 4.7.5.1
- **Primary Focus:** BAC Distribution — Literature Review + Base Model
- **Dependencies:** Portfolio size model (Chunk 1)
- **Forward References:** Segment-specific refinement (Chunk 3), correlation modeling (Chunks 4–5)

---

### **Content**

### 4.7.4 Project BAC Distribution: Literature Review

The distribution of project Budget at Completion (BAC) values within an EPC contractor's portfolio is a fundamental driver of portfolio-level risk characteristics. Larger projects typically exhibit:
- Higher absolute risk exposure (larger potential losses)
- Different risk profiles (complexity, technical uncertainty, stakeholder dynamics)
- Disproportionate impact on portfolio outcomes (concentration risk)

Understanding the statistical distribution of project sizes is essential for realistic portfolio simulation.

#### 4.7.4.1 Empirical Evidence on Project Size Distributions

**Industry data sources**:

1. **ENR Top Projects Database** (2015–2023):
   - Analysis of 2,400+ EPC projects in oil & gas, petrochemical, and power sectors
   - Contract values range from $50M to $15B, with strong right skew
   - Median project size: $180M; mean project size: $420M (indicating positive skewness)
   - 75th percentile: $450M; 90th percentile: $1.2B

2. **Company-specific portfolio data** (anonymized client data, 2018–2024):
   - Portfolio of 87 completed projects from a mid-sized EPC contractor
   - BAC range: $60M–$2.8B
   - Log-transformed BAC values approximately normally distributed (Shapiro-Wilk test: $p = 0.18$)

3. **Academic studies on project size distributions**:
   - Research on construction project portfolios (Flyvbjerg et al., 2018) documents lognormal characteristics in project cost distributions across multiple sectors
   - Studies on megaproject economics (Merrow, 2011) note that project size distributions exhibit "heavy tails" with occasional very large projects

**Statistical characteristics**:
- **Right skewness**: Most projects are small-to-medium, with a long tail of large projects
- **Lognormality**: Log-transformed BAC values often approximate normal distribution
- **Bounded support**: Practical lower bounds (minimum viable project size) and upper bounds (contractor capacity limits)

**Synthesis**: The empirical evidence strongly supports a **lognormal distribution** as the base model for project BAC, with truncation to reflect realistic portfolio constraints.

---

### 4.7.5 Project BAC Distribution Model Formulation

#### 4.7.5.1 Base Model: Truncated Lognormal Distribution

We model individual project BAC using a **truncated lognormal distribution**:

$$\text{BAC} \sim \text{TruncatedLognormal}(\mu_{\ln}, \sigma_{\ln}, \text{BAC}_{\min}, \text{BAC}_{\max})$$

where:
- $\mu_{\ln}$ = mean of the underlying normal distribution (log-scale location parameter)
- $\sigma_{\ln}$ = standard deviation of the underlying normal distribution (log-scale scale parameter)
- $\text{BAC}_{\min}$ = minimum project size (lower truncation bound)
- $\text{BAC}_{\max}$ = maximum project size (upper truncation bound)

**Probability density function** (for $\text{BAC}_{\min} < x < \text{BAC}_{\max}$):

$$f(x) = \frac{\frac{1}{x \sigma_{\ln} \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu_{\ln})^2}{2\sigma_{\ln}^2}\right)}{\Phi\left(\frac{\ln(\text{BAC}_{\max}) - \mu_{\ln}}{\sigma_{\ln}}\right) - \Phi\left(\frac{\ln(\text{BAC}_{\min}) - \mu_{\ln}}{\sigma_{\ln}}\right)}$$

where $\Phi(\cdot)$ is the standard normal cumulative distribution function.

**Parameter calibration** (based on empirical data synthesis):

1. **Truncation bounds**:
   - $\text{BAC}_{\min} = \$75M$ (minimum project size for portfolio inclusion)
   - $\text{BAC}_{\max} = \$3,000M$ (maximum project size reflecting contractor capacity)

2. **Log-scale parameters** (calibrated to match empirical moments):
   - Target median BAC: $\$200M$
   - Target mean BAC: $\$450M$
   - Target coefficient of variation: $CV \approx 0.85$

   Solving for $\mu_{\ln}$ and $\sigma_{\ln}$:
   - $\mu_{\ln} = \ln(200) \approx 5.30$ (log of median)
   - $\sigma_{\ln} \approx 0.75$ (calibrated to match target CV after truncation)

**Resulting distribution characteristics**:
- Median BAC: $\$200M$
- Mean BAC: $\$445M$ (after truncation adjustment)
- Standard deviation: $\$380M$
- 25th percentile: $\$130M$
- 75th percentile: $\$550M$
- 90th percentile: $\$1,100M$

**Modeling rationale**:

1. **Empirical fit**: Lognormal distribution aligns with observed project size distributions in EPC industry
2. **Theoretical justification**: Multiplicative growth processes (project scope expansion, escalation) naturally generate lognormal distributions
3. **Truncation necessity**: Reflects realistic portfolio constraints (minimum viable project size, maximum capacity)
4. **Flexibility**: Two-parameter model allows calibration to different contractor profiles

**Limitations**:
- Assumes homogeneous distribution across all project types (refined in 4.7.5.2)
- Does not capture temporal trends (project size inflation, market cycles)
- Truncation introduces edge effects that may underrepresent tail risks

---

### **End of Chunk 2**

**Next Chunk Preview**: Chunk 3 presents the segment-specific refinement of the BAC distribution model, introducing separate distributional parameters for upstream vs. downstream projects to capture observed heterogeneity in project size profiles.

---

**Status**: ✅ Chunk 2 extracted  
**Proceed to Chunk 3?**