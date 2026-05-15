# Semantic Chunk Constructs

---

## **CHUNK 1 of 6**

### **Metadata**
- **Chunk ID:** `4.7_chunk_01`  
- **Sections Covered:** 4.7.1 → 4.7.3.1

<!-- ⚠️ These chunk IDs are **mapings to the previous deprecated files**:
$ \boxed{\text{Id pattern \quad : `pps\_01`}}$ -->

- **Primary Focus:** Introduction + Portfolio Size Model

<!-- ⚠️ each chunk focuses on one or two small introduction or summaries. -->

<!-- Previous, next links  -->
- **Dependencies:** None (entry point)
- **Forward References:** BAC distribution (4.7.4+), correlation modeling (4.7.6+)

---

### **Content**

## 4.7 Portfolio Size and Project BAC Distribution Model

### 4.7.1 Scope of This Section

This section develops the quantitative models for two foundational portfolio characteristics:

1. **Portfolio size** ($N$): The number of active projects in the contractor's portfolio at any given time
2. **Project Budget at Completion (BAC) distribution**: The statistical distribution governing individual project contract values

These models serve as input generators for the portfolio-level Monte Carlo simulation framework. The portfolio size model determines the dimensionality of the simulation (how many project instances to generate per portfolio realization), while the BAC distribution model governs the scale and heterogeneity of individual projects within each portfolio.

The modeling approach follows a consistent structure:
- **Literature review** of empirical evidence and industry practices
- **Model formulation** with explicit distributional assumptions
- **Parameter calibration** using available data sources
- **Integration** with the broader portfolio risk model

Subsequent sections (4.7.4–4.7.6) extend this framework to model the joint distribution of BAC and project margin, incorporating correlation structures observed in EPC contractor portfolios.

---

### 4.7.2 Portfolio Size: Literature Review

The number of simultaneously active projects in an EPC contractor's portfolio is a critical determinant of operational complexity, resource allocation efficiency, and aggregate risk exposure. Despite its importance, portfolio size has received limited attention in the project management literature, with most studies focusing on single-project or program-level analysis.

#### 4.7.2.1 Empirical Evidence on Portfolio Size

Three primary sources inform our understanding of typical portfolio sizes for large EPC contractors:

**Industry benchmarking studies** (ENR, 2018–2023):
- Analysis of Top 400 Contractors reveals median active project counts ranging from 12 to 35 for firms in the $500M–$5B annual revenue range
- Portfolio size exhibits positive correlation with firm size (Spearman $\rho \approx 0.6$), but with high variance within revenue bands
- Sector specialization affects portfolio composition: process plant contractors typically manage 8–15 large projects, while infrastructure contractors may handle 20–40 smaller projects

**Academic studies on contractor capacity**:
- Empirical research on contractor portfolio management (Choi & Russell, 2005) suggests optimal portfolio sizes of 10–25 projects for firms with $1B–$3B revenue, balancing diversification benefits against coordination costs
- Studies on resource-constrained project scheduling (Browning & Yassine, 2010) indicate that portfolio sizes exceeding 30 projects often lead to resource contention and schedule delays

**Company annual reports and investor presentations** (2020–2024):
- Disclosed project backlogs for publicly traded EPC contractors (e.g., Fluor, KBR, Jacobs) indicate active portfolio sizes of 15–30 major projects (defined as contracts >$50M)
- Portfolio turnover rates (project completions per year) average 30–40% of portfolio size, implying mean project durations of 2.5–3.3 years

**Synthesis**: For a mid-to-large EPC contractor operating in the process industries sector with annual revenue of $1.5B–$3B, a reasonable portfolio size range is **8 to 25 active projects**, with a central tendency around 15–18 projects.

---

### 4.7.3 Portfolio Size Model Formulation

#### 4.7.3.1 Discrete Uniform Distribution

Given the limited granularity of available data and the absence of strong theoretical priors favoring specific distributional shapes, we adopt a **discrete uniform distribution** for portfolio size:

$$N \sim \text{DiscreteUniform}(N_{\min}, N_{\max})$$

where:
- $N$ = number of active projects in the portfolio
- $N_{\min}$ = minimum portfolio size (lower bound)
- $N_{\max}$ = maximum portfolio size (upper bound)

**Probability mass function**:

$$P(N = k) = \frac{1}{N_{\max} - N_{\min} + 1}, \quad k \in \{N_{\min}, N_{\min}+1, \ldots, N_{\max}\}$$

**Calibrated parameters** (based on literature synthesis in 4.7.2.1):
- $N_{\min} = 8$ (lower bound reflecting focused portfolio strategy)
- $N_{\max} = 25$ (upper bound reflecting capacity constraints)
- Expected portfolio size: $E[N] = \frac{N_{\min} + N_{\max}}{2} = 16.5$ projects

**Modeling rationale**:

1. **Parsimony**: The uniform distribution requires only two parameters and avoids overfitting to limited data
2. **Conservatism**: Equal probability across the range reflects epistemic uncertainty about the true portfolio size distribution
3. **Bounded support**: The discrete uniform naturally enforces realistic lower and upper bounds on portfolio size
4. **Computational efficiency**: Sampling from discrete uniform distributions is trivial in Monte Carlo simulation

**Limitations and future refinements**:
- The uniform assumption ignores potential clustering around typical portfolio sizes (e.g., mode at 15–18 projects)
- Does not account for temporal dynamics (portfolio size may vary over business cycles)
- Could be refined with triangular or beta-binomial distributions if more granular data becomes available

**Integration with simulation framework**: In each Monte Carlo iteration, a portfolio size $N$ is sampled from this distribution, determining the number of project-level BAC and margin realizations to generate for that portfolio instance.

---

### **End of Chunk 1**

**Next Chunk Preview**: Chunk 2 introduces the BAC distribution model, reviewing literature on project size distributions and formulating the base truncated lognormal model for individual project contract values.

---

**Status**: ✅ Chunk 1 extracted  
**Proceed to Chunk 2?**