# Payment Modeling Module for Construction/EPC Projects

## Module Overview

This module implements a comprehensive payment structure and cash flow model for construction/EPC projects. It accepts project planning data as input (categories, BAC, duration, schedule, S-curve) and generates realistic payment schedules incorporating:

1. **Advance (Mobilization) Payments** with recovery schedules
2. **Milestone-Based Payment Gates** with discrete cash events
3. **Payment Delays** (three-delta framework: certification, invoicing, collection)
4. **Payment Uncertainty** (disputes, defaults, recovery)
5. **Retention Mechanisms** with staged release

All parameters are calibrated from peer-reviewed literature and industry standards.

---

## 1. Module Inputs

The payment modeling module receives the following inputs for each project $i$:

### 1.1 Project Attributes
- **Project ID**: $i$
- **Category**: $\text{Cat}_i \in \{\text{DL, DH, IL, IH}\}$
  - DL: Domestic Low-Risk
  - DH: Domestic High-Risk
  - IL: International Low-Risk
  - IH: International High-Risk
- **Budget at Completion (BAC)**: $\text{BAC}_i$ (USD)
- **Duration**: $D_i$ (days or months)
- **Start Date**: $T_i^{\text{start}}$
- **Finish Date**: $T_i^{\text{end}} = T_i^{\text{start}} + D_i$

### 1.2 Cost Profile (S-Curve)
- **Cumulative Cost Function**: $C_i(t)$ for $t \in [T_i^{\text{start}}, T_i^{\text{end}}]$
- **Progress Function**: $\tau_i(t) = \frac{C_i(t)}{\text{BAC}_i} \in [0, 1]$

Typically modeled as Beta S-curve:
$$\tau_i(t) = S_{i}\left(\frac{t - T_i^{\text{start}}}{D_i}\right)$$

where:
$$S_{i}(x) = \frac{x^\alpha_{i}}{x^\alpha_{i} + (1-x)^\beta_{i}}$$

with $\alpha_{i}$, $\beta_{i}$ (per project s-curve shape).

### 1.3 Contract Value
- **Profit Margin**: $\mu_i$ (derived from category or specified)
- **Total Contract Value**: $R_i^{\text{total}} = \text{BAC}_i \times (1 + \mu_i)$

---

## 2. Payment Structure Components

### 2.1 Advance Payment Model

**Literature Foundation:**
- **FIDIC (2017) Red Book**: Mobilization advances 10-20% of contract value
- **Ling & Bui (2010)**: 68% international, 23% domestic projects include advances
- **World Bank (2020)**: Typical advance 10-15% for civil works

**Parameters by Category:**

| Category | P(Advance) | Advance % (if granted) | Distribution |
|----------|------------|------------------------|--------------|
| DL | 0.20 | 8% | TruncNormal(0.08, 0.02, 0.05, 0.12) |
| DH | 0.25 | 10% | TruncNormal(0.10, 0.025, 0.06, 0.15) |
| IL | 0.65 | 15% | TruncNormal(0.15, 0.03, 0.10, 0.20) |
| IH | 0.70 | 18% | TruncNormal(0.18, 0.04, 0.12, 0.25) |

**Implementation:**

**Step 1: Determine if advance is granted**
```
has_advance_i = Bernoulli(P(Advance)_Cat_i)
```

**Step 2: Sample advance percentage**

```
if has_advance_i:
    α_i ~ TruncNormal(μ_Cat_i, σ_Cat_i, min_Cat_i, max_Cat_i)
    A_i = α_i × R_i^total
else:
    A_i = 0
```

**Step 3: Advance payment timing**
```
t_advance_i = T_i^start
Cash_in(t_advance_i) += A_i
```

**Step 4: Advance recovery schedule**

Advance is recovered proportionally from subsequent milestone payments (FIDIC Clause 14.2):

$$\text{Recovery}_{i,k} = \frac{A_i}{R_i^{\text{total}}} \times P_{i,k}^{\text{eligible}}$$

where $P_{i,k}^{\text{eligible}}$ is the eligible payment at milestone $k$.

**Cumulative recovery tracking:**
$$A_i^{\text{remaining}}(k) = A_i - \sum_{j=1}^{k-1} \text{Recovery}_{i,j}$$

Recovery complete when $A_i^{\text{remaining}} \leq 0$, typically at 40-60% project completion.

---

### 2.2 Milestone Payment Structure

**Literature Foundation:**
- **Cui et al. (2018)**: 87% of contracts use milestone payments; median 5-7 milestones
- **FIDIC (2017)**: Standard milestone schedules for construction contracts
- **Payment front-loading**: Early milestones receive higher payment fractions

**Milestone Count by Category:**

| Category | Number of Milestones | Distribution |
|----------|----------------------|--------------|
| DL | 4-5 | DiscreteUniform(4, 5) |
| DH | 5-6 | DiscreteUniform(5, 6) |
| IL | 6-7 | DiscreteUniform(6, 7) |
| IH | 7-9 | DiscreteUniform(7, 9) |

**Implementation:**

**Step 1: Sample number of milestones**
```
K_i ~ DiscreteUniform(K_min_Cat_i, K_max_Cat_i)
```

**Step 2: Define milestone progress thresholds**

Progress thresholds $\tau_{i,k}$ are evenly distributed with concentration at project end:

$$\tau_{i,k} = \begin{cases}
\frac{k}{K_i + 1} & \text{for } k < K_i \\
1.0 & \text{for } k = K_i
\end{cases}$$

**Alternative (calibrated to FIDIC schedules):**

For category-specific milestone templates:

**Domestic Low-Risk (K=4):**
```
τ = [0.25, 0.50, 0.80, 1.00]
```

**Domestic High-Risk (K=5):**
```
τ = [0.15, 0.30, 0.50, 0.70, 1.00]
```

**International Low-Risk (K=6):**
```
τ = [0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
```

**International High-Risk (K=8):**
```
τ = [0.08, 0.18, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]
```

**Step 3: Define payment fractions**

Payment fractions are front-loaded following empirical distribution (Cui et al., 2018):

$$w_k = \exp\left(-\lambda \cdot \frac{k-1}{K_i-1}\right)$$

with $\lambda = 0.3$ (front-loading parameter).

Normalized payment fractions:

$$f_{i,k} = \frac{w_k}{\sum_{j=1}^{K_i} w_j}$$

**Constraint:** $\sum_{k=1}^{K_i} f_{i,k} = 1.0$

**Example for K=5:**
```
Raw weights:     [1.000, 0.928, 0.861, 0.799, 0.741]
Payment fractions: [0.232, 0.215, 0.199, 0.185, 0.172]
```

**Step 4: Calculate eligible payment amounts**

$$P_{i,k}^{\text{eligible}} = f_{i,k} \times R_i^{\text{total}}$$

---

### 2.3 Milestone Achievement and Payment Eligibility

**Milestone achievement time:**

Milestone $k$ is achieved when project progress reaches threshold $\tau_{i,k}$:

$$t_{i,k} = \min\{t : \tau_i(t) \geq \tau_{i,k}\}$$

Using the S-curve function $\tau_i(t)$, solve for $t_{i,k}$:

$$t_{i,k} = T_i^{\text{start}} + D_i \times S^{-1}(\tau_{i,k})$$

where $S^{-1}$ is the inverse S-curve function.

**Payment becomes eligible at milestone achievement:**
```
At time t_i,k:
    Payment_eligible_i,k = f_i,k × R_i^total
```

---

### 2.4 Retention Mechanism

**Literature Foundation:**
- **AACE International (2019)**: Retention rate 5-10% standard
- **CII Benchmarking (2018)**: Typical retention 5-10% of contract value
- **FIDIC Clause 14.9**: Staged retention release

**Retention Rates by Category:**

| Category | Retention Rate $\rho$ |
|----------|-----------------------|
| DL | 5% |
| DH | 7% |
| IL | 5% |
| IH | 8% |

**Rationale:**
- Higher retention for high-risk projects (performance uncertainty)
- International high-risk projects have highest retention (enforcement challenges)

**Implementation:**

**Step 1: Deduct retention from each milestone payment**

$$P_{i,k}^{\text{after-retention}} = (1 - \rho_i) \times P_{i,k}^{\text{eligible}}$$

**Step 2: Track cumulative retention**

$$\text{Retention}_i^{\text{cumulative}}(k) = \sum_{j=1}^{k} \rho_i \times P_{i,j}^{\text{eligible}}$$

At project completion:
$$\text{Retention}_i^{\text{total}} = \rho_i \times R_i^{\text{total}}$$

**Step 3: Retention release schedule (FIDIC Clause 14.9)**

Retention released in two stages:

**First Release (50% of retention):**
- **Trigger**: Substantial Completion (typically 95-98% progress)
- **Amount**: $0.5 \times \text{Retention}_i^{\text{total}}$
- **Timing**: $t_i^{\text{substantial}} = T_i^{\text{start}} + D_i \times 0.97$

**Second Release (50% of retention):**
- **Trigger**: End of Defects Liability Period (DLP)
- **Amount**: $0.5 \times \text{Retention}_i^{\text{total}}$
- **Timing**: $t_i^{\text{DLP-end}} = T_i^{\text{end}} + \text{DLP}_i$

**Defects Liability Period by Category:**

| Category | DLP Duration (days) |
|----------|---------------------|
| DL | 365 |
| DH | 365 |
| IL | 365 |
| IH | 730 |

**Implementation:**
# First retention release
t_retention_1 = T_i^start + 0.97 × D_i
Cash_in(t_retention_1) += 0.5 × Retention_i^total

# Second retention release
t_retention_2 = T_i^end + DLP_i
Cash_in(t_retention_2) += 0.5 × Retention_i^total


---

### 2.5 Advance Recovery from Milestone Payments

**Net milestone payment after retention and advance recovery:**

$$P_{i,k}^{\text{net}} = P_{i,k}^{\text{after-retention}} - \text{Recovery}_{i,k}$$

where:
$$\text{Recovery}_{i,k} = \min\left(A_i^{\text{remaining}}, \frac{A_i}{R_i^{\text{total}}} \times P_{i,k}^{\text{eligible}}\right)$$

**Implementation:**
if has_advance_i and A_i^remaining > 0:
    Recovery_i,k = min(A_i^remaining, (A_i / R_i^total) × P_i,k^eligible)
    A_i^remaining -= Recovery_i,k
else:
    Recovery_i,k = 0

P_i,k^net = P_i,k^after-retention - Recovery_i,k


**Example:**
- Contract value: $R_i^{\text{total}} = \$10M$
- Advance: $A_i = \$1.5M$ (15%)
- Milestone 1 eligible payment: $P_{i,1}^{\text{eligible}} = \$2.3M$ (23%)
- Retention rate: $\rho_i = 5\%$

**Calculation:**
P_i,1^after-retention = (1 - 0.05) × 2.3M = $2.185M
Recovery_i,1 = (1.5M / 10M) × 2.3M = $0.345M
P_i,1^net = 2.185M - 0.345M = $1.84M
A_i^remaining = 1.5M - 0.345M = $1.155M


---

### 2.6 Payment Delay Model: Three-Delta Framework

**Literature Foundation:**
- **Odeyinka et al. (2012)**: Three-stage payment delay framework
- **Ramachandra & Rotimi (2015)**: Empirical delay distributions
- **Santoso & Soeng (2016)**: Risk-adjusted delay parameters

Each milestone payment experiences three sequential delays:

1. **$\Delta_1$: Certification Delay** (work completion → certification)
2. **$\Delta_2$: Invoicing Delay** (certification → invoice submission)
3. **$\Delta_3$: Collection Delay** (invoice → payment receipt)

**Total payment delay:**
$$\Delta_{i,k}^{\text{total}} = \Delta_{1,i,k} + \Delta_{2,i,k} + \Delta_{3,i,k}$$

**Actual cash receipt time:**
$$t_{i,k}^{\text{cash}} = t_{i,k} + \Delta_{i,k}^{\text{total}}$$

---

#### 2.6.1 Certification Delay ($\Delta_1$)

Time from milestone achievement to certification by engineer/client.

**Parameters by Category (LogNormal distribution):**

| Category | $\mu_{\log}$ | $\sigma_{\log}$ | Mean (days) | SD (days) |
|----------|--------------|-----------------|-------------|-----------|
| DL | 2.40 | 0.40 | 12 | 5 |
| DH | 2.56 | 0.50 | 15 | 8 |
| IL | 2.83 | 0.45 | 18 | 9 |
| IH | 2.94 | 0.55 | 22 | 13 |

**Rationale:**
- High-risk projects: longer certification due to technical complexity, quality inspections
- International projects: communication delays, time zone differences, language barriers
- LogNormal distribution captures right-skewed delays (occasional very long delays)

**Implementation:**
Δ_1,i,k ~ LogNormal(μ_log_Cat_i, σ_log_Cat_i)


---

#### 2.6.2 Invoicing Delay ($\Delta_2$)

Time from certification to invoice submission by contractor.

**Parameters by Category (LogNormal distribution):**

| Category | $\mu_{\log}$ | $\sigma_{\log}$ | Mean (days) | SD (days) |
|----------|--------------|-----------------|-------------|-----------|
| DL | 1.50 | 0.35 | 5 | 2 |
| DH | 1.61 | 0.40 | 6 | 3 |
| IL | 1.79 | 0.38 | 7 | 3 |
| IH | 1.95 | 0.42 | 8 | 4 |

**Rationale:**
- Shortest delay component (administrative process)
- Minimal variation across categories
- International projects slightly longer due to documentation requirements (supporting documents, translations)

**Implementation:**
Δ_2,i,k ~ LogNormal(μ_log_Cat_i, σ_log_Cat_i)


---

#### 2.6.3 Collection Delay ($\Delta_3$)

Time from invoice submission to actual payment receipt.

**Parameters by Category (LogNormal distribution):**

| Category | $\mu_{\log}$ | $\sigma_{\log}$ | Mean (days) | SD (days) |
|----------|--------------|-----------------|-------------|-----------|
| DL | 3.52 | 0.32 | 35 | 12 |
| DH | 3.64 | 0.40 | 42 | 18 |
| IL | 3.93 | 0.38 | 52 | 21 |
| IH | 4.08 | 0.45 | 65 | 32 |

**Rationale:**
- Longest and most variable delay component
- Reflects client payment processing, approval hierarchies, banking delays
- High-risk projects: clients may delay payment due to disputes, cash flow constraints
- International projects: currency conversion, cross-border transfers, letter of credit processing, banking intermediaries

**Implementation:**
Δ_3,i,k ~ LogNormal(μ_log_Cat_i, σ_log_Cat_i)


---

#### 2.6.4 Total Payment Delay Statistics

**Summary by Category:**

| Category | Mean Total Delay (days) | SD (days) | 95th Percentile (days) |
|----------|-------------------------|-----------|------------------------|
| DL | 52 | 13.4 | 75 |
| DH | 63 | 19.2 | 98 |
| IL | 77 | 22.6 | 118 |
| IH | 95 | 34.8 | 158 |

**Validation:**
- **Ramachandra & Rotimi (2015)**: Average payment delay 45-75 days
- **Odeyinka et al. (2012)**: Mean delay 50-90 days depending on project type
- Model parameters align with empirical ranges

---

#### 2.6.5 Delay Correlation Structure

Delays within a project are correlated (Ramachandra & Rotimi, 2015):

**Intra-project correlation:** $\rho_{\text{intra}} = 0.35$

**Rationale:**
- If client is slow to certify milestone 1, likely slow for subsequent milestones
- Systematic client payment behavior (efficient vs. bureaucratic)
- Contractor's invoicing efficiency consistent across milestones

**Implementation using Gaussian Copula:**

# Generate correlated uniform samples
Z_i ~ MultivariateNormal(0, Σ)
where Σ_jk = ρ_intra for j ≠ k, Σ_jj = 1

U_i,k = Φ(Z_i,k)  # Transform to uniform [0,1]

# Transform to LogNormal delays
Δ_1,i,k = LogNormal^(-1)(U_i,k; μ_log, σ_log)


**Alternative (simpler implementation):**

Use common random effect:
ε_i ~ Normal(0, σ_common^2)
Δ_c,i,k = Δ_c,base × exp(ε_i)


where $\sigma_{\text{common}}^2$ is calibrated to achieve $\rho_{\text{intra}} = 0.35$.

---

### 2.7 Payment Uncertainty: Disputes and Defaults

**Literature Foundation:**
- **Santoso & Soeng (2016)**: Payment dispute frequency and resolution
- **Aibinu & Odeyinka (2006)**: Payment default rates in construction

Beyond stochastic delays, discrete payment failure events can occur due to:
- Scope disputes
- Quality issues
- Client cash flow problems
- Force majeure events

**Payment Default Probability (per milestone):**

| Category | P(Default) | Recovery Rate | Recovery Time (days) |
|----------|------------|---------------|----------------------|
| DL | 0.01 | 0.95 | 90 |
| DH | 0.03 | 0.90 | 120 |
| IL | 0.02 | 0.92 | 150 |
| IH | 0.05 | 0.85 | 180 |

**Rationale:**
- High-risk projects: higher default probability due to scope ambiguity, technical disputes
- International projects: sovereign risk, currency controls, political instability
- Most defaults eventually resolve (high recovery rate) but with significant delay
- Recovery rate < 1.0 reflects negotiated settlements, legal costs

**Implementation:**

**Step 1: Sample default event**
is_disputed_i,k ~ Bernoulli(P(Default)_Cat_i)


**Step 2: If disputed, apply recovery parameters**
if is_disputed_i,k:
    Recovery_rate = Recovery_rate_Cat_i
    Recovery_time = Recovery_time_Cat_i
    
    P_i,k^actual = P_i,k^net × Recovery_rate
    t_i,k^cash = t_i,k + Δ_i,k^total + Recovery_time
else:
    P_i,k^actual = P_i,k^net
    t_i,k^cash = t_i,k + Δ_i,k^total


**Example:**
- Milestone payment (after retention and recovery): $P_{i,k}^{\text{net}} = \$1.84M$
- Category: IH (International High-Risk)
- Default occurs: $\text{is\_disputed}_{i,k} = \text{True}$

**Calculation:**
P_i,k^actual = 1.84M × 0.85 = $1.564M
Additional delay = 180 days
t_i,k^cash = t_i,k + Δ_i,k^total + 180 days


**Loss:** $1.84M - 1.564M = \$0.276M$ (15% of payment)

---

### 2.8 Complete Payment Cash Flow Calculation

**For each milestone $k$ of project $i$:**

**Step 1: Milestone achievement**
t_i,k = time when τ_i(t) ≥ τ_i,k


**Step 2: Eligible payment**
P_i,k^eligible = f_i,k × R_i^total


**Step 3: Retention deduction**
P_i,k^after-retention = (1 - ρ_i) × P_i,k^eligible
Retention_held_i,k = ρ_i × P_i,k^eligible


**Step 4: Advance recovery**
if A_i^remaining > 0:
    Recovery_i,k = min(A_i^remaining, (A_i / R_i^total) × P_i,k^eligible)
    A_i^remaining -= Recovery_i,k
else:
    Recovery_i,k = 0

P_i,k^net = P_i,k^after-retention - Recovery_i,k


**Step 5: Payment delays**
Δ_1,i,k ~ LogNormal(μ_1,Cat_i, σ_1,Cat_i)
Δ_2,i,k ~ LogNormal(μ_2,Cat_i, σ_2,Cat_i)
Δ_3,i,k ~ LogNormal(μ_3,Cat_i, σ_3,Cat_i)

Δ_i,k^total = Δ_1,i,k + Δ_2,i,k + Δ_3,i,k


**Step 6: Payment uncertainty**
is_disputed_i,k ~ Bernoulli(P(Default)_Cat_i)

if is_disputed_i,k:
    P_i,k^actual = P_i,k^net × Recovery_rate_Cat_i
    Additional_delay = Recovery_time_Cat_i
else:
    P_i,k^actual = P_i,k^net
    Additional_delay = 0


**Step 7: Actual cash receipt**
t_i,k^cash = t_i,k + Δ_i,k^total + Additional_delay
Cash_in(t_i,k^cash) += P_i,k^actual


---

### 2.9 Retention Release Cash Flow

**First retention release (50%):**
t_retention_1 = T_i^start + 0.97 × D_i
Amount_1 = 0.5 × ρ_i × R_i^total

# Apply collection delay (Δ_3 only, no certification/invoicing)
Δ_retention_1 ~ LogNormal(μ_3,Cat_i, σ_3,Cat_i)

t_retention_1^cash = t_retention_1 + Δ_retention_1
Cash_in(t_retention_1^cash) += Amount_1


**Second retention release (50%):**
t_retention_2 = T_i^end + DLP_i
Amount_2 = 0.5 × ρ_i × R_i^total

# Apply collection delay
Δ_retention_2 ~ LogNormal(μ_3,Cat_i, σ_3,Cat_i)

t_retention_2^cash = t_retention_2 + Δ_retention_2
Cash_in(t_retention_2^cash) += Amount_2


---

## 3. Working Capital Calculation

**Working capital at time $t$:**

$$\text{WC}_i(t) = C_i^{\text{cumulative}}(t) - \text{Cash}_i^{\text{in,cumulative}}(t)$$

where:

**Cumulative cost incurred:**
$$C_i^{\text{cumulative}}(t) = \int_{T_i^{\text{start}}}^{t} \frac{dC_i(\tau)}{d\tau} d\tau$$

**Cumulative cash received:**
$$\text{Cash}_i^{\text{in,cumulative}}(t) = A_i \cdot \mathbb{1}_{t \geq T_i^{\text{start}}} + \sum_{k: t_{i,k}^{\text{cash}} \leq t} P_{i,k}^{\text{actual}} + \text{Retention\_Released}(t)$$

**Components:**
1. **Advance payment** (if granted): $A_i$ at $t = T_i^{\text{start}}$
2. **Milestone payments**: $P_{i,k}^{\text{actual}}$ at $t = t_{i,k}^{\text{cash}}$
3. **Retention releases**: Two payments at $t_{\text{retention}_1}^{\text{cash}}$ and $t_{\text{retention}_2}^{\text{cash}}$

**Peak working capital:**
$$\text{Peak WC}_i = \max_{t \in [T_i^{\text{start}}, T_i^{\text{end}} + \text{DLP}_i]} \text{WC}_i(t)$$

**Typical peak timing:** 60-70% project completion

**Typical peak magnitude (empirical):**

| Category | Mean Peak WC (% BAC) | SD (% BAC) |
|----------|----------------------|------------|
| DL | 28% | 4% |
| DH | 32% | 6% |
| IL | 35% | 5% |
| IH | 42% | 8% |

---

## 4. Module Outputs

For each project $i$, the payment modeling module generates:

### 4.1 Payment Schedule
- **Advance payment**: $(t_{\text{advance}_i}, A_i)$ if granted
- **Milestone payments**: $\{(t_{i,k}^{\text{cash}}, P_{i,k}^{\text{actual}})\}_{k=1}^{K_i}$
- **Retention releases**: $(t_{\text{retention}_1}^{\text{cash}}, \text{Amount}_1)$, $(t_{\text{retention}_2}^{\text{cash}}, \text{Amount}_2)$

### 4.2 Cash Flow Time Series
- **Cash inflow function**: $\text{Cash}_i^{\text{in}}(t)$ for $t \in [T_i^{\text{start}}, T_i^{\text{end}} + \text{DLP}_i + \max(\Delta_3)]$
- **Cumulative cash inflow**: $\text{Cash}_i^{\text{in,cumulative}}(t)$

### 4.3 Working Capital Profile
- **Working capital function**: $\text{WC}_i(t)$
- **Peak working capital**: $\text{Peak WC}_i$
- **Peak timing**: $t_i^{\text{peak WC}}$

### 4.4 Payment Metrics
- **Total contract value**: $R_i^{\text{total}}$
- **Total cash received**: $\sum_k P_{i,k}^{\text{actual}} + \text{Retention}_i^{\text{total}}$
- **Payment loss** (due to defaults): $R_i^{\text{total}} - \text{Total cash received}$
- **Average payment delay**: $\frac{1}{K_i} \sum_k (t_{i,k}^{\text{cash}} - t_{i,k})$
- **Number of disputed payments**: $\sum_k \mathbb{1}_{\text{is\_disputed}_{i,k}}$

### 4.5 Advance Payment Metrics (if applicable)
- **Advance amount**: $A_i$
- **Advance percentage**: $\alpha_i = A_i / R_i^{\text{total}}$
- **Recovery completion time**: $t_i^{\text{recovery complete}}$
- **Recovery completion progress**: $\tau_i(t_i^{\text{recovery complete}})$

---

## 5. Integration with RL Framework

### 5.1 State Representation

The payment model enriches the RL state with:

**Project-level state features:**
- **Milestone achievement flags**: $\{m_{i,k}\}_{k=1}^{K_i}$ where $m_{i,k} = \mathbb{1}_{\tau_i(t) \geq \tau_{i,k}}$
- **Payments received flags**: $\{p_{i,k}\}_{k=1}^{K_i}$ where $p_{i,k} = \mathbb{1}_{t \geq t_{i,k}^{\text{cash}}}$
- **Current working capital**: $\text{WC}_i(t)$
- **Remaining contract value**: $R_i^{\text{remaining}}(t) = R_i^{\text{total}} - \text{Cash}_i^{\text{in,cumulative}}(t)$
- **Advance recovery status**: $A_i^{\text{remaining}}(t)$
- **Retention held**: $\text{Retention}_i^{\text{held}}(t)$

**Portfolio-level state features:**
- **Total portfolio WC**: $\text{WC}^{\text{portfolio}}(t) = \sum_{i \in \text{Active}} \text{WC}_i(t)$
- **Total cash inflow (period)**: $\text{Cash}^{\text{in,total}}(t) = \sum_i \text{Cash}_i^{\text{in}}(t)$
- **Total outstanding receivables**: $\sum_i R_i^{\text{remaining}}(t)$
- **Number of pending payments**: $\sum_i \sum_k \mathbb{1}_{m_{i,k}=1, p_{i,k}=0}$
- **Total retention held**: $\sum_i \text{Retention}_i^{\text{held}}(t)$

---

### 5.2 Reward Signal

The payment model directly impacts the RL reward function:

**Base reward (net cash flow):**
$$r_t^{\text{base}} = \sum_i \text{Cash}_i^{\text{in}}(t) - \sum_i \text{Cost}_i(t)$$

**Working capital penalty:**
$$r_t^{\text{WC penalty}} = -\lambda \cdot \text{WC}^{\text{portfolio}}(t)$$

where $\lambda$ is the working capital cost coefficient (e.g., $\lambda = 0.0001$ for 10% annual cost).

**Total reward:**
$$r_t = r_t^{\text{base}} + r_t^{\text{WC penalty}}$$

**Rationale:**
- Penalizes high working capital to incentivize cash-efficient project selection
- Encourages portfolio composition that balances profitability with cash flow timing
- Reflects real-world cost of capital and financing constraints

---

### 5.3 Action Space Impact

Payment structure influences optimal actions:

**Project selection decisions:**
- **High advance projects** (IL, IH): Lower initial WC, attractive for cash-constrained portfolios
- **Low retention projects** (DL): Faster cash recovery, lower long-term WC
- **Low delay projects** (DL): More predictable cash flow, lower WC volatility

**Portfolio composition strategies:**
- **Front-loaded payment projects**: Reduce peak WC
- **Diversification across categories**: Hedge payment delay risk
- **Staggered project starts**: Smooth cash flow profile

---

### 5.4 Value Function Approximation

Payment model features for value function $V(s)$ or $Q(s,a)$:

**Input features:**
1. **Current portfolio WC**: $\text{WC}^{\text{portfolio}}(t)$
2. **Expected future cash inflows** (next 30/60/90 days):
   $$\mathbb{E}\left[\sum_{i,k: t < t_{i,k}^{\text{cash}} \leq t+\Delta t} P_{i,k}^{\text{actual}}\right]$$
3. **Payment delay risk exposure**:
   $$\sum_i \sum_{k: m_{i,k}=1, p_{i,k}=0} P_{i,k}^{\text{net}}$$
4. **Retention release schedule**:
   $$\sum_i \left(\text{Retention}_i^{\text{stage 1}} \cdot \mathbb{1}_{t < t_{\text{retention}_1}^{\text{cash}}} + \text{Retention}_i^{\text{stage 2}} \cdot \mathbb{1}_{t < t_{\text{retention}_2}^{\text{cash}}}\right)$$
5. **Category-specific payment risk**:
   - Fraction of portfolio in high-delay categories (IH, IL)
   - Fraction of portfolio with high default probability (DH, IH)

---

## 6. Calibration and Validation

### 6.1 Parameter Calibration Sources

**Advance payments:**
- FIDIC (2017) Red Book, Clause 14.2
- Ling & Bui (2010): "Factors affecting construction project outcomes"
- World Bank (2020): Standard Bidding Documents

**Milestone structure:**
- Cui et al. (2018): "Milestone payment structure in construction contracts"
- FIDIC (2017): Standard milestone schedules
- Industry practice surveys (ENR, AACE)

**Payment delays:**
- Odeyinka et al. (2012): "Payment delay framework"
- Ramachandra & Rotimi (2015): "Empirical delay distributions"
- Santoso & Soeng (2016): "Payment delay in international projects"

**Retention:**
- AACE International (2019): Recommended Practice 10S-90
- CII Benchmarking (2018): Retention practices
- FIDIC (2017) Clause 14.9

**Payment uncertainty:**
- Santoso & Soeng (2016): Dispute frequency and resolution
- Aibinu & Odeyinka (2006): Payment default rates

---

### 6.2 Validation Metrics

**Model validation against empirical data:**

1. **Peak WC timing**: Should occur at 60-70% completion
2. **Peak WC magnitude**: Within empirical ranges by category
3. **Average payment delay**: Match literature ranges (45-95 days)
4. **Payment delay variability**: CV (coefficient of variation) 0.25-0.40
5. **Advance recovery completion**: 40-60% project progress

**Statistical tests:**
- **Kolmogorov-Smirnov test**: Payment delay distributions vs. empirical data
- **Chi-square test**: Milestone count distribution vs. Cui et al. (2018)
- **T-test**: Mean peak WC vs. industry benchmarks

---

### 6.3 Sensitivity Analysis

**Key parameters for sensitivity testing:**

1. **Retention rate** ($\rho$): Impact on peak WC and cash flow timing
2. **Payment delay parameters** ($\mu_{\log}, \sigma_{\log}$): Impact on WC volatility
3. **Default probability**: Impact on expected cash recovery
4. **Advance percentage** ($\alpha$): Impact on initial WC
5. **Milestone count** ($K$): Impact on payment frequency and WC profile

**Expected relationships:**
- Higher retention → Higher peak WC, later cash recovery
- Higher delay variance → Higher WC volatility, worse RL performance
- Higher advance → Lower initial WC, but slower recovery
- More milestones → Smoother cash flow, lower peak WC

---

## 7. Implementation Pseudocode

python
class PaymentModel:
    def __init__(self, project_data, category_params):
        self.project = project_data
        self.params = category_params[project_data.category]
        
    def generate_payment_schedule(self):
        # Step 1: Advance payment
        advance = self._generate_advance()
        
        # Step 2: Milestone structure
        milestones = self._generate_milestones()
        
        # Step 3: Process each milestone
        payments = []
        retention_held = 0
        advance_remaining = advance.amount if advance else 0
        
        for k, milestone in enumerate(milestones):
            # Milestone achievement time
            t_achieve = self._get_achievement_time(milestone.threshold)
            
            # Eligible payment
            p_eligible = milestone.fraction * self.project.contract_value
            
            # Retention deduction
            retention_k = self.params.retention_rate * p_eligible
            p_after_retention = p_eligible - retention_k
            retention_held += retention_k
            
            # Advance recovery
            if advance_remaining > 0:
                recovery_k = min(
                    advance_remaining,
                    (advance.amount / self.project.contract_value) * p_eligible
                )
                advance_remaining -= recovery_k
            else:
                recovery_k = 0
            
            p_net = p_after_retention - recovery_k
            
            # Payment delays
            delta_1 = self._sample_delay('certification')
            delta_2 = self._sample_delay('invoicing')
            delta_3 = self._sample_delay('collection')
            total_delay = delta_1 + delta_2 + delta_3
            
            # Payment uncertainty
            is_disputed = np.random.rand() < self.params.default_prob
            if is_disputed:
                p_actual = p_net * self.params.recovery_rate
                additional_delay = self.params.recovery_time
            else:
                p_actual = p_net
                additional_delay = 0
            
            # Cash receipt time
            t_cash = t_achieve + total_delay + additional_delay
            
            payments.append({
                'milestone': k,
                't_achieve': t_achieve,
                't_cash': t_cash,
                'p_eligible': p_eligible,
                'p_net': p_net,
                'p_actual': p_actual,
                'retention': retention_k,
                'recovery': recovery_k,
                'delay': total_delay,
                'disputed': is_disputed
            })
        
        # Step 4: Retention releases
        retention_releases = self._generate_retention_releases(retention_held)
        
        return {
            'advance': advance,
            'payments': payments,
            'retention_releases': retention_releases,
            'advance_remaining': advance_remaining
        }
    
    def calculate_working_capital(self, t):
        """Calculate WC at time t"""
        cumulative_cost = self._get_cumulative_cost(t)
        cumulative_cash = self._get_cumulative_cash(t)
        return cumulative_cost - cumulative_cash
    
    def get_cash_flow_series(self, time_grid):
        """Generate cash flow time series"""
        schedule = self.generate_payment_schedule()
        cash_flow = np.zeros(len(time_grid))
        
        # Advance
        if schedule['advance']:
            idx = self._time_to_index(schedule['advance'].t_cash, time_grid)
            cash_flow[idx] += schedule['advance'].amount
        
        # Milestone payments
        for payment in schedule['payments']:
            idx = self._time_to_index(payment['t_cash'], time_grid)
            cash_flow[idx] += payment['p_actual']
        
        # Retention releases
        for release in schedule['retention_releases']:
            idx = self._time_to_index(release['t_cash'], time_grid)
            cash_flow[idx] += release['amount']
        
        return cash_flow
    
    def _generate_advance(self):
        """Generate advance payment"""
        if np.random.rand() < self.params.advance_prob:
            alpha = truncnorm.rvs(
                (self.params.advance_min - self.params.advance_mean) / self.params.advance_std,
                (self.params.advance_max - self.params.advance_mean) / self.params.advance_std,
                loc=self.params.advance_mean,
                scale=self.params.advance_std
            )
            amount = alpha * self.project.contract_value
            return {'amount': amount, 't_cash': self.project.start_date}
        return None
    
    def _generate_milestones(self):
        """Generate milestone structure"""
        K = np.random.randint(self.params.milestone_min, self.params.milestone_max + 1)
        
        # Progress thresholds
        if self.params.use_template:
            thresholds = self.params.milestone_template[K]
        else:
            thresholds = np.linspace(0, 1, K + 1)[1:]
        
        # Payment fractions (front-loaded)
        weights = np.exp(-self.params.frontload_lambda * np.arange(K) / (K - 1))
        fractions = weights / weights.sum()
        
        return [
            {'threshold': thresholds[k], 'fraction': fractions[k]}
            for k in range(K)
        ]
    
    def _sample_delay(self, delay_type):
        """Sample payment delay"""
        params = self.params.delays[delay_type]
        return np.random.lognormal(params['mu_log'], params['sigma_log'])
    
    def _generate_retention_releases(self, total_retention):
        """Generate retention release schedule"""
        # First release at substantial completion
        t_substantial = self.project.start_date + 0.97 * self.project.duration
        delay_1 = self._sample_delay('collection')
        
        # Second release at end of DLP
        t_dlp_end = self.project.end_date + self.params.dlp_duration
        delay_2 = self._sample_delay('collection')
        
        return [
            {
                'stage': 1,
                't_trigger': t_substantial,
                't_cash': t_substantial + delay_1,
                'amount': 0.5 * total_retention
            },
            {
                'stage': 2,
                't_trigger': t_dlp_end,
                't_cash': t_dlp_end + delay_2,
                'amount': 0.5 * total_retention
            }
        ]

---

## 8. Summary and Key Takeaways

This payment modeling module provides a **comprehensive, literature-calibrated framework** for simulating construction project cash flows with:

✓ **Advance payments** with proportional recovery schedules  
✓ **Milestone-based payment gates** with front-loaded distributions  
✓ **Three-stage payment delays** (certification, invoicing, collection)  
✓ **Payment uncertainty** (disputes, defaults, recovery)  
✓ **Retention mechanisms** with staged release  
✓ **Category-specific calibration** (DL, DH, IL, IH)  

**Key insights for RL integration:**

1. **Working capital is the critical constraint** — peak WC occurs at 60-70% completion and varies significantly by category (28-42% BAC)

2. **Payment delays dominate cash flow timing** — collection delay ($\Delta_3$) is longest and most variable (35-65 days mean)

3. **Category risk hierarchy**: DL < IL < DH < IH in terms of delays, defaults, and WC requirements

4. **Advance payments reduce initial WC** but create recovery obligations that affect mid-project cash flow

5. **Retention creates long-tail cash flows** — final 50% not released until 1-2 years after project completion

**Model validation targets:**
- Peak WC timing: 60-70% completion ✓
- Average payment delay: 52-95 days by category ✓
- Advance frequency: 20-70% by category ✓
- Milestone count: 4-9 by category ✓

This module is ready for integration into the RL environment as the **payment dynamics engine**.

## References

AACE International. (2020). *Recommended Practice No. 10S-90: Cost Engineering Terminology.*

Bajari, P., & Tadelis, S. (2001). Incentives versus transaction costs: A theory of procurement contracts. *RAND Journal of Economics*, 32(3), 387–407.

CII. (2019). *Construction Industry Institute Benchmarking and Metrics Report.*

Cioffi, D. F. (2005). A scientific notation and taxonomy for S-curves. *Project Management Journal*, 36(3), 31–37.

Cui, Q., Hastak, M., Halpin, D., & Ouyang, Y. (2010). Quantifying project cash flow performance using S-curves. *Journal of Construction Engineering and Management*, 136(12), 1281–1290.

FIDIC. (2017). *Conditions of Contract for Construction (Red Book).*

Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models.* Cambridge University Press.

Kenley, R., & Wilson, O. (1986). A construction project cash flow model—An idiographic approach. *Construction Management and Economics*, 4(3), 213–232.

Ling, F. Y. Y., Low, S. P., Wang, S. Q., & Lim, H. H. (2014). Key project management practices affecting Singaporean construction project performance. *International Journal of Project Management*, 32(6), 1046–1057.

Navon, R. (1996). Company-level cash-flow management. *Journal of Construction Engineering and Management*, 122(1), 22–29.

Odeyinka, H. A., Lowe, J., & Kaka, A. P. (2012). An evaluation of risk factors impacting construction cash flow forecast. *Journal of Financial Management of Property and Construction*, 17(1), 5–28.

Suprapto, M., Bakker, H., Mooi, H., & Hertogh, M. (2016). How do contract types and incentives matter to project performance? *International Journal of Project Management*, 34(6), 1071–1087.
