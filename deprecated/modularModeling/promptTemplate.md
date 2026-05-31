# Modeling Prompt Template

## quick version
ok now using the insights and the optimal portfolio composition according to this section below in the previously generated markdown documentation;

I want you to:
* read and study the introduced content
* generate a new markdown document named "portfolio composition"
* give the output in one markdown code block
* maintain markdown format and writing style


definition of goals:
**portfolio composition:** the distribution of different levels of profit margins for projects inside the portfolio instances.  
**Project profit margin model:** the model and distributions to generate project profit margin inside boundaries.


the desired output:
* first you define the scope by mentioning assumptions and exclusion (out of the work as natural future work according to base assumption)
    * The base assumption: delivery of the simplest model for the foundational framework contribution
    * > the assumption here would be that all portfolio selections (as it's another problem in another domain) in all instances are optimal (according to the referenced section with literature citation)
* I want a documentation of the portfolio instance generator, specifically the composition of the generated portfolios around the profit margin parameter.
    * according to the citation I expect of you to introduce explicit mathematical model with literature calibrated parameters for both the composition model and the profit boundary model for each level.
* all models and parameters are literature calibrated. meaning according to the literature introduced statistics and data.
    * all must be cited to the reference
    * the master parameter table including the name, symbol, value, reference etc. must be included.

### extra:
* study the decision of correlation of the project BACs and profit margines according to the literature.
* give incorporation or exclusion insights for this matter

### general guidelines for output:
* I want you to write the mathematical model in an OR/IE Q1 paper quality.
* preferably use higly valid and newest resources for citation as references
---

### references and material

#### previous output
last output

#### the section about optimal composition:

**Diversification Effect:**
- Individual project CV ≈ 0.30 (30% relative uncertainty)
- Portfolio of 10 equal-sized projects: $\text{CV}_{\text{portfolio}} \approx 0.095$ (9.5% relative uncertainty)
- Portfolio of 50 projects: $\text{CV}_{\text{portfolio}} \approx 0.042$ (4.2% relative uncertainty)

**Strategic Insight (Khanzadi et al., 2018):**
- Optimal portfolio mix: 60% domestic (lower margin, lower risk) + 40% international (higher margin, higher risk)
- Maximizes risk-adjusted return (Sharpe ratio)

*Citation:* Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). *Integrating project portfolio selection and scheduling under uncertainty*. Journal of Construction Engineering and Management, 144(2), 04017106.

---


## detailed + full context Instructions for Generating Modular Modeling Documents

---

### 1. Context & Purpose

This template guides the generation of **modular modeling documentation** for the Q1 research paper on "Reinforcement Learning for Dynamic Project Portfolio Budgeting under Cashflow Uncertainty: A Rolling Horizon Approach."

Each modular document focuses on a **single modeling dimension** (e.g., S-curves, profit margins, durations, uncertainty) with complete mathematical formulation, literature calibration, and implementation details.

**Foundational Principle:** Deliver the **simplest defensible model** for a foundational framework contribution. All complexity must be justified by empirical necessity or theoretical rigor.

---

### 2. Embedded Project Context

This section contains all the essential context from the project's core documents, embedded here so you can generate modular documents without needing local file access.

---

#### 2.1 Research Problem & Solution Approach

**Core Challenge:**
Project portfolio budgeting under cashflow uncertainty requires dynamic decision-making where:
- Future contractor performance is unpredictable
- Traditional optimization methods (MIP, Stochastic Programming) fail due to incomplete observability
- Sequential decisions must adapt to emerging information
- Projects have staggered start times and variable durations
- Portfolio horizons extend beyond fixed planning windows

**Proposed Solution:**
Train a Reinforcement Learning (RL) agent using **Rolling Horizon Control** with:
1. **Pre-training**: Literature and industry-wide project data
2. **Fine-tuning**: Company-specific historical performance data
3. **Rolling Horizon Framework**: Fixed lookahead window (H=12 periods) with receding control

**Key Advantages:**
- Adaptability: Learns from sequential interactions
- Partial Observability: Handles uncertainty without full probability distributions
- Dynamic Optimization: Adjusts decisions as new information emerges
- Transferability: Pre-trained model can be specialized for organizational contexts
- Scalability: Rolling horizon prevents state-space explosion

---

#### 2.2 Industry Context & Calibration

**Target Domain: EPC Oil & Gas Projects**
- Model calibrated for Engineering, Procurement, and Construction (EPC) contractors in oil & gas sector
- Project cashflow profiles reflect typical EPC spending patterns from empirical studies
- S-curve parameters derived from large-scale energy infrastructure projects
- Findings generalizable to other capital-intensive project portfolios

**Why Synthetic Data:**
- **Confidentiality Barrier**: Project cashflow timeseries are commercially sensitive; no company shares granular data
- **Scientific Validity**: Synthetic data with literature calibration is methodologically sound for OR/IE research
- **Controlled Experiments**: Enables fair algorithmic comparison across designed parameter spaces
- **Reproducibility**: Full specification is publishable and verifiable

**Why International (Non-Iranian) Calibration:**
- **Inflation Problem**: Iranian projects face 30-50% annual inflation, confounding real project management performance
- **Sanctions Effects**: Supply chain disruptions and procurement delays create systematic distortions
- **Generalizability**: International calibration ensures findings apply to stable economies

---

#### 2.3 Foundational Modeling Assumptions

**Simplest Model Principle:**
This is a brand new field in literature. The Q1 paper establishes foundational proof-of-concept. Extension of each assumption is explicitly recognized as future work.

**Key Assumptions:**

1. **Cashflow-Based Budgeting**: All decisions made using aggregated project cashflows (inflows/outflows); detailed resource-level requirements not modeled individually

2. **Aggregated Uncertainty**: Project performance uncertainty captured through aggregate SPI/CPI multipliers, not task-level or resource-level stochastic models

3. **Fixed Profit Margins**: Project profit margins determined at contract signing and remain constant throughout execution (no renegotiation)

4. **Deterministic S-Curves**: All projects follow planned spending S-curve (Beta CDF with α=2.5, β=2.0); deviations captured by performance uncertainty model

5. **Independent Projects**: No resource conflicts, technology spillovers, or strategic interdependencies between projects

6. **Rolling Horizon (H=12)**: Fixed 12-period lookahead window; agent re-plans at each timestep with updated information

7. **Optimal Portfolio Selection**: Generated portfolios reflect optimal project selection (60% domestic, 40% international per Khanzadi et al. 2018)

---

#### 2.4 Portfolio Composition Framework

**Optimal Mix (Khanzadi et al. 2018):**
- **60% Domestic Projects**: Lower profit margins (8-12%), lower variance, regulatory stability
- **40% International Projects**: Higher profit margins (12-16%), higher variance, market-driven pricing
- This composition maximizes risk-adjusted return (Sharpe ratio) for Iranian EPC contractors

**Diversification Effect:**
- Individual project CV ≈ 0.30 (30% relative uncertainty)
- Portfolio of 10 projects: CV_portfolio ≈ 0.095 (9.5%)
- Portfolio of 50 projects: CV_portfolio ≈ 0.042 (4.2%)

**Profit Margin Distributions:**

*Domestic Projects:*
- Distribution: Truncated Normal
- Mean (μ_domestic): 10%
- Std Dev (σ_domestic): 1.5%
- Bounds: [8%, 12%]
- Rationale: Government-regulated contracts with standardized pricing (IPC framework)

*International Projects:*
- Distribution: Truncated Normal
- Mean (μ_international): 14%
- Std Dev (σ_international): 2.0%
- Bounds: [12%, 16%]
- Rationale: Competitive bidding with market-driven pricing and higher risk premiums

**Key Modeling Decision:**
- **No BAC-Margin Correlation**: Profit margins sampled independently from project size (BAC)
- Rationale: Iranian market shows negligible correlation (Khanzadi et al. 2018); adds unnecessary complexity to foundational model

---

#### 2.5 Portfolio Size & BAC Distribution

**Portfolio Size (N):**
- Distribution: Discrete Uniform(8, 18)
- Baseline simulation value: N = 10
- Rationale: Optimal diversification zone per CII (2019); typical for Iranian Tier 1 EPC contractors

**Literature Benchmarks:**
- Tier 1 EPC contractors: 15-25 major projects (Merrow 2011)
- Tier 2 contractors: 8-15 projects
- Optimal for risk diversification: 10-20 projects (CII 2019)
- Iranian Tier 1 contractors: 8-15 projects (Khanzadi et al. 2018)

**Project BAC Distribution:**
- Distribution: Truncated Lognormal (segment-specific)
- Rationale: EPC project sizes are right-skewed; lognormal best fits empirical data (AACE 2020)

*Domestic Projects:*
- ln(BAC) ~ N(μ_ln = 4.8, σ_ln = 0.6)
- Median: ~$120M
- Bounds: [$50M, $500M]
- Smaller projects due to domestic market constraints

*International Projects:*
- ln(BAC) ~ N(μ_ln = 5.3, σ_ln = 0.8)
- Median: ~$200M
- Bounds: [$80M, $1.5B]
- Larger projects with higher complexity and risk

**Key Validation:**
- No single project should exceed 25-30% of total portfolio BAC (Flyvbjerg et al. 2018)
- Portfolio concentration risk: >40% in single project → 2.3× higher financial distress probability

---

#### 2.6 S-Curve Cashflow Model

**Mathematical Formulation:**
All projects follow a Beta CDF spending profile:

$$S(\tau) = I_{\tau}(\alpha, \beta) = \frac{\int_0^{\tau} x^{\alpha-1}(1-x)^{\beta-1} dx}{B(\alpha, \beta)}$$

where:
- τ = normalized project progress (0 to 1)
- α = 2.5 (shape parameter, controls front-loading)
- β = 2.0 (shape parameter, controls tail behavior)
- S(τ) = cumulative fraction of BAC spent by progress τ

**Period Cashflow:**
$$\Delta C_i(t) = \text{BAC}_i \times [S(\tau_t) - S(\tau_{t-1})]$$

where τ_t = (t - T_start) / D_i (normalized time within project duration)

**Calibration Validation:**
| Milestone | Model Prediction | Literature Benchmark | Source |
|-----------|------------------|---------------------|--------|
| Spend at τ=0.25 | 16.1% | 15-20% | Barraza & Bueno (2007) |
| Spend at τ=0.50 | 50.0% | 45-55% | Cioffi (2005) |
| Spend at τ=0.75 | 84.0% | 80-88% | Miskawi (1989) |
| Peak spending | τ=0.60 | 0.40-0.65 | Barraza & Bueno (2007) |

**Key Properties:**
- Slow start (engineering phase)
- Accelerating mid-phase (procurement + construction)
- Tapering toward completion (commissioning)
- Empirically validated for EPC oil & gas projects

**Uniform Parameterization Rationale:**
1. Empirical clustering: EPC projects in same sector exhibit similar spending profiles
2. State space tractability: Per-project parameters would add 2N dimensions
3. Data unavailability: Project-specific calibration requires granular historical data
4. Separation of concerns: Micro-level dynamics belong to operational layer
5. Sensitivity stability: Optimal RL policies stable across α∈[2.0,3.0], β∈[1.5,2.5]

---

#### 2.7 Key Literature References

**Portfolio Optimization & Composition:**
- Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). Integrating project portfolio selection and scheduling under uncertainty. *Journal of Construction Engineering and Management*, 144(2), 04017106.
- Archer, N. P., & Ghasemzadeh, F. (1999). An integrated framework for project portfolio selection. *International Journal of Project Management*, 17(4), 207-216.

**S-Curve Modeling:**
- Barraza, G. A., & Bueno, R. A. (2007). Probabilistic control of project performance using control limit curves. *Journal of Construction Engineering and Management*, 133(12), 957-965.
- Cioffi, D. F. (2005). A tool for managing projects: An analytic parameterization of the S-curve. *International Journal of Project Management*, 23(3), 215-222.
- Kenley, R., & Wilson, O. D. (1986). A construction project cash flow model: An idiographic approach. *Construction Management and Economics*, 4(3), 213-232.

**Project Size & Risk:**
- Merrow, E. W. (2011). *Industrial megaprojects: Concepts, strategies, and practices for success*. Wiley.
- Flyvbjerg, B., Ansar, A., Budzier, A., et al. (2018). Five things you should know about cost overrun. *Transportation Research Part A: Policy and Practice*, 118, 174-190.
- AACE International. (2020). *Cost estimate classification system*. AACE Recommended Practice 18R-97.

**Uncertainty & Performance:**
- Construction Industry Institute. (2019). *CII benchmarking and metrics report*. The University of Texas at Austin.
- Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). Underestimating costs in public works projects: Error or lie? *Journal of the American Planning Association*, 68(3), 279-295.

---

#### 2.8 Current Progress Status

**Completed Modules:**
1. ✅ **profitMarginsCompositions.md** - Portfolio composition (60/40 mix) and profit margin distributions by category
2. ✅ **projectCounts&BACsDistributions.md** - Portfolio size (N=8-18) and BAC distributions (truncated lognormal)
3. ✅ **projectSCurves.md** - S-curve cashflow model using Beta CDF (α=2.5, β=2.0)

**Planned Modules:**
4. ⬜ **projectDurations.md** - Duration distributions and start time staggering
5. ⬜ **projectRevenuePlans.md** - Revenue recognition and payment structures
6. ⬜ **projectsPerformances.md** - Contractor performance uncertainty (SPI/CPI)
7. ⬜ **projectsRevenues.md** - Payment delays and revenue uncertainty
8. ⬜ **portfolioTemporalStructure.md** - Temporal structure and seasonal effects
9. ⬜ **portfolioGenerator.md** - Complete instance generation algorithm
10. ⬜ **datasetValidation.md** - Validation framework and sanity checks
11. ⬜ **rewardFunction.md** - Reward function design and components
12. ⬜ **MDP.md** - MDP formulation (state, action, transition, reward)

---

### 3. Document Generation Instructions

**How to Use This Template:**
When generating a new modular document, use the embedded context in Section 2 above to inform your work. You now have all the necessary information about:
- Research problem and solution approach
- Industry context and why synthetic data is used
- Foundational assumptions and modeling principles
- Portfolio composition, size, and BAC distributions
- S-curve cashflow modeling
- Key literature references
- Current progress and what's already been completed

Use this context to maintain consistency with existing modules and ensure your new document aligns with the overall research framework.

---

#### 3.1 Document Structure (Mandatory Sections)

**Section X.1: Scope Definition**

**X.1.1 Foundational Assumptions**

**Format:**
```markdown
**Primary Assumption — [Concise Title]:**

[1-2 sentence statement of the core modeling assumption]

[Optional: Mathematical expression if applicable]

**Rationale:**

1. **[Justification 1]**: [Explanation with empirical/theoretical grounding]
2. **[Justification 2]**: [Explanation]
3. **[Justification 3]**: [Explanation]
...

*Citations:*
- [Author et al. (Year). Title. Journal, Volume(Issue), Pages.]
```

**Content Requirements:**
- State the **single most important assumption** for this modeling dimension
- Provide **3-5 numbered justifications** (empirical clustering, tractability, data limitations, separation of concerns, sensitivity stability)
- Include **at least 2 literature citations** supporting the assumption
- Connect to the **foundational simplicity principle** (scope Section 3.2)

**Example (from projectSCurves.md):**
> **Primary Assumption — Uniform Beta CDF Parameterization:**
> All projects follow the same Beta CDF with $\alpha = 2.5$, $\beta = 2.0$...
> **Rationale:**
> 1. **Empirical clustering**: EPC projects exhibit similar spending profiles (Barraza & Bueno, 2007)
> 2. **State space tractability**: Per-project parameters add $2N$ dimensions...

---

##### **X.1.2 Exclusions and Future Work**

**Format:**
```markdown
The following elements are **explicitly excluded** from the current model scope. Each represents a natural extension for follow-up research.

**1. [Exclusion Category]:**
- **Excluded**: [Specific modeling element 1]
- **Excluded**: [Specific modeling element 2]
- **Future Work**: [Concrete extension path with methodology]
- **Reference**: [Citation for future work methodology]

**2. [Exclusion Category]:**
...
```

**Content Requirements:**
- List **3-5 major exclusions** relevant to this modeling dimension
- For each exclusion:
  - State **what is excluded** (specific, concrete)
  - Explain **why it's excluded** (data, complexity, orthogonality to research question)
  - Propose **future work** (specific methodology, not vague "could be explored")
  - Cite **reference** for the proposed future methodology
- Maintain consistency with **Section 4 (Out of Scope Exclusions)** in scope.md

**Example:**
> **1. Project-Specific S-Curve Calibration:**
> - **Excluded**: Estimation of individual $(\alpha_i, \beta_i)$ from historical data
> - **Future Work**: Hierarchical Beta model with category-level priors
> - **Reference**: Gelman & Hill (2006). *Data analysis using regression...*

---

#### **Section X.2: Literature Review**

**Format:**
```markdown
##### X.2.1 [Thematic Grouping]

###### **1. [Author(s) (Year)] — [Paper Title or Key Contribution]**

**Findings:**
- [Key empirical result 1 with quantitative data]
- [Key empirical result 2]
- [Methodological insight]

*Citation:* [Full citation in APA/IEEE format]

---

###### **2. [Next Study]**
...

##### X.2.2 Comparative Assessment / Synthesis

[Optional: Table comparing models/approaches]

**Decision:** [Justify the chosen model/approach based on literature]
```

**Content Requirements:**
- Review **4-6 key studies** directly relevant to this modeling dimension
- For each study:
  - Extract **quantitative findings** (parameter ranges, empirical distributions, correlation coefficients)
  - Note **methodological contributions** (why this model/approach was chosen)
  - Provide **full citation** (author, year, title, journal, volume, pages)
- Prioritize **recent (post-2000) and high-impact** sources (Q1 journals, established textbooks)
- Include **comparative table** if multiple modeling approaches exist (e.g., Beta CDF vs. logistic vs. polynomial)
- End with **decision justification**: why the chosen model is superior

**Example Structure:**
> ###### **1. Cioffi (2005) — Analytic Parameterization via Beta Distribution**
> **Findings:**
> - Beta CDF outperforms polynomial models (lower AIC/BIC across 37 projects)
> - $\alpha/\beta$ ratio determines front-loading intensity
> *Citation:* Cioffi, D. F. (2005). A tool for managing projects...

---

#### **Section X.3: Mathematical Model**

**Format:**
```markdown
##### X.3.1 [Primary Model Component]

[Prose description of what the model represents]

$$[Primary equation with full notation]$$

where:
- $[symbol]$ = [definition with units]
- $[symbol]$ = [definition]
...

**Boundary conditions / Constraints:**
$$[Constraint equations]$$

---

##### X.3.2 [Derived Quantities / Properties]

[Additional formulations, analytical properties, special cases]

---

##### X.3.3 [Portfolio-Level Aggregation] (if applicable)

[How individual project models aggregate to portfolio level]
```

**Content Requirements:**
- Present **complete mathematical formulation** with all notation defined
- Use **consistent notation** with scope.md (e.g., $\text{BAC}_i$, $T_i^{\text{start}}$, $D_i$)
- Include **boundary conditions** and **constraints** explicitly
- Derive **analytical properties** (e.g., peak location, inflection points, cumulative values at milestones)
- Provide **physical interpretation** of parameters (what does $\alpha$ control? what does $\beta$ mean?)
- Show **portfolio-level aggregation** formula if relevant
- Use **LaTeX math** for all equations; inline for simple expressions, display mode for primary formulations

**Example:**
> $$S_i(\tau) = I_{\tau}(\alpha, \beta) = \frac{\int_0^{\tau} u^{\alpha-1}(1-u)^{\beta-1} du}{B(\alpha, \beta)}$$
> where:
> - $\tau = (t - T_i^{\text{start}})/D_i \in [0,1]$ = normalized project progress
> - $\alpha, \beta > 0$ = shape parameters

---

#### **Section X.4: Parameter Calibration**

##### **X.4.1 Master Parameter Table**

**Format:**
```markdown
| Parameter | Symbol | Value | Bounds | Source | Notes |
|-----------|--------|-------|--------|--------|-------|
| [Name] | $[symbol]$ | [value] | $[range]$ | [Citation] | [Context] |
```

**Content Requirements:**
- Include **all parameters** introduced in Section X.3
- Provide **point estimates** (baseline values) and **sensitivity ranges** (bounds)
- Cite **literature source** for each parameter value
- Add **notes** column for context (e.g., "Baseline for EPC oil & gas", "Derived from empirical fit")
- Ensure **units** are clear (if applicable)

##### **X.4.2 Sensitivity Ranges** (if applicable)

**Format:**
```markdown
| Scenario | [Param 1] | [Param 2] | [Derived Metric] | Profile Character |
|----------|-----------|-----------|------------------|-------------------|
| [Name] | [value] | [value] | [value] | [Description] |
```

**Content Requirements:**
- Define **3-5 sensitivity scenarios** spanning the empirically observed parameter range
- Include a **baseline scenario** (bolded)
- Show **derived metrics** (e.g., peak location, inflection point) for each scenario
- Describe **qualitative behavior** (e.g., "symmetric", "front-loaded", "back-loaded")

---

#### **Section X.5: Implementation**

##### **X.5.1 Algorithm Specification**

###### Format:

**Algorithm X.1: [Algorithm Name]**

**Input:**
- [parameter 1] = [description]
- [parameter 2] = [description]

**Output:**
- [output] = [description]

**Procedure:**

```python
import numpy as np
from scipy.stats import [distribution]

def algorithm_name(param1, param2, ...):
    """
    [Docstring with description, parameters, returns]
    """
    # Step 1: [Description]
    [code]
    
    # Step 2: [Description]
    [code]
    
    return result


# Example usage
if __name__ == "__main__":
    [example instantiation]
    [print key outputs]
```

**Content Requirements:**
- Provide **complete, runnable Python code** (not pseudocode)
- Use **scipy/numpy** for statistical distributions and numerical operations
- Include **docstrings** with parameter descriptions and return types
- Add **inline comments** explaining each algorithmic step
- Provide **example usage** with realistic parameter values
- Print **key validation metrics** (e.g., total spend, peak period, conservation check)
- Ensure code matches the mathematical formulation exactly

---

#### **Section X.6: Validation**

##### **X.6.1 Analytical Checks**

**Format:**
```markdown
| Check | Condition | Tolerance |
|-------|-----------|-----------|
| [Invariant name] | $[mathematical condition]$ | [numerical tolerance] |
```

**Content Requirements:**
- List **4-6 invariants** that must hold for any generated instance
- Common checks: conservation laws, non-negativity, boundary values, monotonicity, sum-to-one constraints
- Specify **numerical tolerances** (e.g., $< 0.01\%$ relative error, strict equality)

##### **X.6.2 Benchmark Validation Against Literature**

**Format:**
```markdown
| Milestone | Model Prediction | Literature Benchmark | Source |
|-----------|-----------------|---------------------|--------|
| [Description] | [value] | [range] | [Citation] |
```

**Content Requirements:**
- Compare **model predictions** to **reported empirical values** from literature
- Show that predictions **fall within reported ranges**
- Cite **specific studies** for each benchmark

##### **X.6.3 Portfolio-Level Validation** (if applicable)

**Content Requirements:**
- Describe **aggregate-level checks** (e.g., portfolio cashflow smoothness, concentration risk bounds)
- Reference **portfolio-level empirical benchmarks** if available

---

## Quality Standards

### Writing Style
- **Tone**: Formal, precise, technical (Q1 Operations Research / Industrial Engineering journal quality)
- **Voice**: Third person, passive constructions for methodology ("is modeled", "are calibrated")
- **Tense**: Present tense for model description, past tense for literature review
- **Clarity**: Define all notation before use; avoid ambiguous pronouns

### Mathematical Rigor
- All equations must be **dimensionally consistent**
- Use **standard notation** from the field (e.g., $\mathbb{E}[\cdot]$ for expectation, $\sim$ for "distributed as")
- Provide **closed-form expressions** where possible; numerical approximations only when necessary
- State **assumptions explicitly** (e.g., "assuming independence", "under stationarity")

### Literature Citations
- Prioritize **peer-reviewed journal articles** (Q1/Q2 journals in OR, IE, construction management)
- Include **seminal works** (e.g., Merrow 2011, Flyvbjerg et al. 2002) even if older
- Use **full citations** in text: Author(s), Year, Title, Journal, Volume(Issue), Pages
- Avoid **grey literature** (blog posts, unpublished reports) unless no alternative exists

### Code Quality
- Follow **PEP 8** style guidelines for Python
- Use **type hints** where helpful (e.g., `def func(x: np.ndarray) -> float:`)
- Ensure **reproducibility**: set random seeds, document scipy/numpy versions if critical
- Provide **unit tests** or validation checks within the example usage

---

## Context Persistence for Offline Use

To ensure **high-quality generation in new conversations** (offline context), each modular document must be **self-contained**:

### Embedded Context Requirements
1. **Restate the research problem** in Section X.1.1 (1-2 sentences linking to the overall RL portfolio budgeting framework)
2. **Define all notation locally** — do not assume reader has access to scope.md
3. **Include full citations** — do not reference "see scope.md Section 3.2"
4. **Provide complete parameter tables** — all values, bounds, sources in one place
5. **Self-contained validation** — benchmarks and checks fully specified

### Cross-Reference Protocol
- **Within-document references**: Use section numbers (e.g., "as defined in Section X.3.1")
- **Cross-document references**: Use full path and section (e.g., "consistent with scope.md Section 5.2")
- **Avoid implicit dependencies**: If a concept from another module is needed, briefly restate it

### Metadata Block (Optional, at document end)
```markdown
---

## Document Metadata

- **Module**: [e.g., projectsPortfolioModel]
- **Topic**: [e.g., S-Curve Cashflow Model]
- **Dependencies**: [List other modules this depends on]
- **Status**: [Draft / Under Review / Finalized]
- **Last Updated**: [Date]
- **Primary Sources**: scope.md (Sections X.Y), literatureCalibratedSyntheticData.md (Sections A.B)
```

---

## Example Invocation

**User Prompt:**
```
Write root/strategy/modularModeling/projectsPortfolioModel/projectDurations.md

Using:
- root/strategy/scope.md (Section 3.3 for duration assumptions)
- root/strategy/literatureCalibratedSyntheticData.md (Section 4.3 for Gamma distribution calibration)
- root/strategy/modularModeling/0. promptTemplate.md (this template)

Focus on:
- Duration distributions by project category (domestic vs. international)
- Start time staggering mechanisms
- Temporal alignment with portfolio horizon
- Literature: Merrow (2011), Flyvbjerg et al. (2002)
```

**Expected Output:**
A complete markdown document following the structure:
- X.1 Scope Definition (assumptions + exclusions)
- X.2 Literature Review (4-6 studies on project durations)
- X.3 Mathematical Model (Gamma distribution formulation, staggering algorithm)
- X.4 Parameter Calibration (shape/scale parameters, master table)
- X.5 Implementation (Python code for duration sampling)
- X.6 Validation (analytical checks, literature benchmarks)

---

## Final Checklist Before Submission

- [ ] All assumptions stated with 3-5 justifications and citations
- [ ] 3-5 exclusions listed with future work paths and references
- [ ] 4-6 literature studies reviewed with quantitative findings
- [ ] Complete mathematical model with all notation defined
- [ ] Master parameter table with values, bounds, sources
- [ ] Runnable Python implementation with example usage
- [ ] 4-6 analytical validation checks specified
- [ ] Benchmark comparison to literature values
- [ ] All citations in full format (Author, Year, Title, Journal, Pages)
- [ ] Document is self-contained (can be understood without external files)
