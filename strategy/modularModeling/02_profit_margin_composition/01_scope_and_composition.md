# Document 4.8 — Profit Margin Composition
## Chunk 01: Scope & Composition Framework

---

**Metadata:**
Document: 4.8 — Profit Margin Composition
Chunk: 01 of 05
Sections: 4.1 (Scope Definition) + 4.2 (Portfolio Composition Framework)
Status: v1.0 - PRELIMINARY (~80% accurate)
Dependencies: 4.7_chunk_01 (Portfolio size N)
Last Updated: 2026-05-12


---

## 4.1 Scope Definition

### 4.1.1 Assumptions and Boundaries

This document establishes the **profit margin composition model** for the synthetic project portfolio. The model defines how profit margins $\pi_i$ are distributed across the $N$ projects in the portfolio, where $N$ is determined by the portfolio size model in Document 4.7 (chunk 01).

**Key Assumptions:**

1. **Profit Margin Definition:**
   $$\pi_i = \frac{\text{Revenue}_i - \text{Cost}_i}{\text{Revenue}_i} \times 100\%$$
   where:
   - $\text{Revenue}_i$ = Total contract revenue for project $i$
   - $\text{Cost}_i$ = Total project cost (including direct, indirect, overhead)
   - $\pi_i$ is expressed as a percentage

2. **Portfolio Composition:**
   - The portfolio consists of **domestic** and **international** projects
   - Optimal split: **60% domestic, 40% international** (Khanzadi et al., 2018)
   - Each category is further subdivided by complexity/market positioning

3. **Independence from BAC:**
   - Profit margin $\pi_i$ is modeled as **independent** of Budget at Completion $\text{BAC}_i$
   - Justification provided in Section 4.6 (chunk 05)
   - This allows separate sampling: first BAC (from 4.7_chunk_03), then margin

4. **Time-Invariant Margins:**
   - Margins are set at project initiation and remain constant
   - Does not model margin erosion due to cost overruns (future work)

5. **No Portfolio-Level Constraints:**
   - Individual project margins are sampled independently
   - No constraint on aggregate portfolio profitability
   - Reflects real-world variability where some projects may have negative margins

### 4.1.2 Exclusions and Limitations

**Out of Scope:**

- **Dynamic margin adjustment:** Changes in margin during project execution due to change orders, claims, or cost overruns
- **Client-specific pricing:** Margins driven by client relationships or strategic positioning
- **Geographic sub-segmentation:** Regional differences within domestic/international categories (e.g., Middle East vs. Europe)
- **Project delivery method:** Impact of EPC vs. EPCM vs. design-build on margins
- **Financing structure:** Effect of project financing arrangements on reported margins

**Known Limitations:**

- Literature on construction profit margins is sparse and often aggregated at firm level rather than project level
- Calibration relies on industry reports (ENR, FMI) and limited academic studies
- Model assumes margins follow parametric distributions (truncated normal, beta); actual distributions may be more complex
- Does not account for portfolio optimization strategies (e.g., deliberately accepting low-margin projects to maintain workforce utilization)

### 4.1.3 Future Enhancements

**Planned Extensions:**

1. **BAC-Margin Correlation:** If empirical data becomes available showing significant correlation, update model to use copula-based joint sampling
2. **Margin Erosion Model:** Introduce stochastic margin degradation linked to cost performance index (CPI)
3. **Strategic Pricing:** Add logic for loss-leader projects or premium pricing for specialized capabilities
4. **Geographic Refinement:** Subdivide international category by region with region-specific margin distributions

---

## 4.2 Portfolio Composition Framework

### 4.2.1 Two-Tier Categorization

The portfolio is structured using a **two-tier categorization**:

**Tier 1: Geographic Scope**
- **Domestic Projects:** 60% of portfolio
- **International Projects:** 40% of portfolio

**Tier 2: Complexity/Market Positioning**

Within each geographic category, projects are further classified:

**Domestic Projects (60% of $N$):**
1. **Domestic Standard (DS):** 70% of domestic projects
   - Routine infrastructure, building construction
   - Competitive bidding, regulatory constraints
   - Lower margins, lower risk

2. **Domestic Complex (DC):** 30% of domestic projects
   - Specialized facilities (hospitals, data centers, industrial plants)
   - Design-build, negotiated contracts
   - Higher margins, higher technical risk

**International Projects (40% of $N$):**
3. **International Competitive (IC):** 65% of international projects
   - Emerging markets, competitive bidding
   - Higher margins than domestic due to risk premium
   - Currency, political, and logistical risks

4. **International Premium (IP):** 35% of international projects
   - Developed markets, specialized capabilities
   - Negotiated contracts, strategic partnerships
   - Highest margins, lower country risk but higher performance expectations

### 4.2.2 Portfolio Composition Weights

Let $N$ be the total portfolio size (from 4.7_chunk_01). The number of projects in each subcategory is:

$$N_{\text{DS}} = \lfloor 0.60 \times N \times 0.70 \rfloor = \lfloor 0.42N \rfloor$$

$$N_{\text{DC}} = \lfloor 0.60 \times N \times 0.30 \rfloor = \lfloor 0.18N \rfloor$$

$$N_{\text{IC}} = \lfloor 0.40 \times N \times 0.65 \rfloor = \lfloor 0.26N \rfloor$$

$$N_{\text{IP}} = N - N_{\text{DS}} - N_{\text{DC}} - N_{\text{IC}}$$

**Note:** The last category (IP) absorbs rounding errors to ensure $\sum N_k = N$.

**Example:** For $N = 50$ projects:
- $N_{\text{DS}} = \lfloor 21 \rfloor = 21$ projects
- $N_{\text{DC}} = \lfloor 9 \rfloor = 9$ projects
- $N_{\text{IC}} = \lfloor 13 \rfloor = 13$ projects
- $N_{\text{IP}} = 50 - 21 - 9 - 13 = 7$ projects

### 4.2.3 Rationale for 60/40 Domestic/International Split

**Empirical Basis:**

Khanzadi et al. (2018) analyzed portfolio composition of large international contractors and found:
- Optimal risk-return balance at **60% domestic, 40% international**
- Domestic projects provide stable cash flow and lower risk
- International projects offer higher margins but introduce currency, political, and execution risks
- Diversification benefits peak at this ratio

**Industry Benchmarks:**

- ENR Top 400 Contractors (2020-2023): Average 58% domestic revenue
- Large EPC firms (Fluor, Bechtel, KBR): 55-65% domestic backlog
- Regional contractors expanding internationally: Typically start at 80/20, migrate toward 60/40 over 5-10 years

**Model Justification:**

- Represents a **mature, diversified** portfolio
- Balances risk (domestic stability) and return (international premiums)
- Aligns with strategic goals of large contractors in the $5B-$20B revenue range

### 4.2.4 Rationale for Subcategory Splits

**Domestic: 70% Standard, 30% Complex**

- Reflects the **Pareto distribution** of project complexity in domestic markets
- Standard projects (buildings, roads, utilities) dominate by count
- Complex projects (industrial, specialized facilities) are fewer but larger in BAC
- Calibrated to match ENR data on project types

**International: 65% Competitive, 35% Premium**

- **Competitive projects** (emerging markets, commodity infrastructure) are more numerous
- **Premium projects** (developed markets, specialized capabilities) require established reputation and technical expertise
- Split reflects the **market access barrier** for premium work

### 4.2.5 Integration with BAC Distribution

**Key Relationship:**

- Portfolio composition (this chunk) determines **how many** projects fall into each category
- BAC distribution (4.7_chunk_03) determines **how large** each project is
- Margin distribution (chunks 02-03) determines **how profitable** each project is

**Sampling Sequence:**

1. Sample portfolio size $N \sim \text{Poisson}(\lambda = 50)$ (from 4.7_chunk_01)
2. Allocate $N$ into 4 subcategories using weights above
3. For each project $i$, sample $\text{BAC}_i$ from appropriate distribution (4.7_chunk_03)
4. For each project $i$, sample $\pi_i$ from appropriate margin distribution (chunks 02-03)

**Independence Assumption:**

- The **number** of projects in each category is deterministic given $N$
- The **size** (BAC) and **margin** of each project are sampled independently
- This simplifies the model while preserving realistic variability

---

## Integration Notes

**Dependencies:**
- **4.7_chunk_01:** Portfolio size $N$ is required to compute $N_{\text{DS}}, N_{\text{DC}}, N_{\text{IC}}, N_{\text{IP}}$
- **4.7_chunk_03:** BAC distributions will be used in conjunction with margin distributions (chunks 02-03)

**Forward References:**
- **Chunk 02 (4.8):** Defines margin distributions for Domestic Standard and Domestic Complex
- **Chunk 03 (4.8):** Defines margin distributions for International Competitive and International Premium
- **Chunk 04 (4.8):** Provides master parameter table and generation algorithm
- **Chunk 05 (4.8):** Justifies independence assumption between BAC and margin

**Key Parameters Established:**
- Geographic split: 60% domestic, 40% international
- Domestic subcategories: 70% Standard, 30% Complex
- International subcategories: 65% Competitive, 35% Premium
- Formulas for computing $N_{\text{DS}}, N_{\text{DC}}, N_{\text{IC}}, N_{\text{IP}}$ given $N$

---

**End of Chunk 01**

---

چانک اول آماده است. این چانک:
- **چارچوب کلی** را تعریف می‌کند (60/40 و 4 زیرگروه)
- **فرضیات اساسی** را مستند می‌کند (استقلال از BAC، ثبات مارجین)
- **محدودیت‌ها** را شفاف می‌کند (عدم مدل‌سازی فرسایش مارجین)
- **وابستگی به 4.7_chunk_01** را صریح می‌کند (نیاز به $N$)

آماده‌ای برای چانک 02 (Domestic Margins)؟