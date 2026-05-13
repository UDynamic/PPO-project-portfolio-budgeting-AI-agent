# Phase 3 — Semantic Chunk Construction (continued)

---

## **CHUNK 3 of 6**

### **Metadata**
- **Chunk ID:** `4.7_chunk_03`
- **Sections Covered:** 4.7.5.2
- **Primary Focus:** BAC Distribution — Segment-Specific Refinement
- **Dependencies:** Base lognormal model (Chunk 2), portfolio size model (Chunk 1)
- **Forward References:** Correlation modeling (Chunks 4–5), parameter table (Chunk 6)

---

### **Content**

#### 4.7.5.2 Segment-Specific Refinement

The base lognormal model (Section 4.7.5.1) assumes a homogeneous distribution of project sizes across all project types. However, empirical evidence suggests systematic differences in project size distributions between upstream (oil & gas extraction, LNG) and downstream (refining, petrochemical) segments.

**Empirical observations**:

1. **Upstream projects**:
   - Tend toward larger average sizes (median: $\$280M$, mean: $\$520M$)
   - Higher prevalence of megaprojects ($>\$1B$)
   - Driven by capital intensity of offshore platforms, LNG facilities, and remote field development

2. **Downstream projects**:
   - Smaller average sizes (median: $\$150M$, mean: $\$380M$)
   - More concentrated in mid-range ($\$100M–\$500M$)
   - Refinery upgrades and petrochemical expansions typically smaller than greenfield upstream facilities

**Refined model formulation**:

We introduce **segment-specific distributional parameters**:

$$\text{BAC}_{\text{upstream}} \sim \text{TruncatedLognormal}(\mu_{\ln}^{\text{up}}, \sigma_{\ln}^{\text{up}}, \text{BAC}_{\min}, \text{BAC}_{\max})$$

$$\text{BAC}_{\text{downstream}} \sim \text{TruncatedLognormal}(\mu_{\ln}^{\text{down}}, \sigma_{\ln}^{\text{down}}, \text{BAC}_{\min}, \text{BAC}_{\max})$$

**Calibrated parameters**:

| Segment | $\mu_{\ln}$ | $\sigma_{\ln}$ | Median BAC | Mean BAC | CV |
|---------|-------------|----------------|------------|----------|-----|
| Upstream | 5.63 | 0.80 | $\$280M$ | $\$520M$ | 0.90 |
| Downstream | 5.01 | 0.70 | $\$150M$ | $\$380M$ | 0.80 |

**Implementation in simulation**:

For each project $i$ in the portfolio:

1. **Determine segment**: Based on project type assignment (from Section 4.6)
   - If project type ∈ {Upstream Oil & Gas, LNG, Offshore} → Upstream segment
   - If project type ∈ {Refining, Petrochemical, Downstream Infrastructure} → Downstream segment

2. **Sample BAC**: Draw from corresponding truncated lognormal distribution
   $$\text{BAC}_i \sim \begin{cases} \text{TruncatedLognormal}(\mu_{\ln}^{\text{up}}, \sigma_{\ln}^{\text{up}}, 75, 3000) & \text{if upstream} \\ \text{TruncatedLognormal}(\mu_{\ln}^{\text{down}}, \sigma_{\ln}^{\text{down}}, 75, 3000) & \text{if downstream} \end{cases}$$

3. **Verify portfolio constraints**: Ensure total portfolio BAC is within realistic bounds
   - If $\sum_{i=1}^{N} \text{BAC}_i$ exceeds contractor capacity ($\$15B$ threshold), resample or rescale

**Modeling benefits**:

1. **Improved realism**: Captures observed heterogeneity in project size profiles across segments
2. **Risk differentiation**: Upstream-heavy portfolios naturally exhibit higher concentration risk
3. **Calibration flexibility**: Allows adjustment to contractor-specific segment focus

**Validation approach**:

- Compare simulated portfolio BAC distributions against historical data from 5 mid-sized EPC contractors (2018–2024)
- Kolmogorov-Smirnov test for distributional fit: $p > 0.10$ for all contractors
- Segment-specific median BAC within 15% of observed values

**Limitations**:

- Binary segmentation (upstream/downstream) may oversimplify within-segment heterogeneity
- Does not account for temporal trends (e.g., energy transition shifting project mix)
- Assumes independence between project type and project size (conditional on segment)

**Alternative modeling approaches considered**:

1. **Project-type-specific distributions**: Separate parameters for each of 6 project types
   - Rejected due to insufficient data for robust calibration (sample size limitations)

2. **Mixture models**: Weighted combination of multiple lognormal components
   - Rejected due to added complexity without substantial improvement in fit

3. **Empirical distribution resampling**: Bootstrap from historical project database
   - Rejected due to limited ability to extrapolate beyond observed range

**Conclusion**: The segment-specific refinement provides a pragmatic balance between model realism and parameter parsimony, supported by empirical evidence and validated against industry data.

---

### **End of Chunk 3**

**Next Chunk Preview**: Chunk 4 introduces the correlation modeling framework, beginning with a comprehensive literature review on empirical evidence for project-level correlations in EPC portfolios, including common cause factors, market-driven dependencies, and contractor-specific effects.

---

**Status**: ✅ Chunk 3 extracted  
**Proceed to Chunk 4?**