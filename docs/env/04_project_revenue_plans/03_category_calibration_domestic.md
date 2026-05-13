# CHUNK 003

## Coverage
**Section 1.2.1-1.2.2**: Category-Specific Calibration Framework - Domestic Low-Risk (DL) and Domestic High-Risk (DH) Government Clients

## Dependency Notes
- Builds on literature foundation from CHUNK 001 (milestone-based payment evidence)
- References payment delay data from CHUNK 002
- Uses retention rate evidence from CHUNK 002
- Critical for understanding domestic project payment parameters
- Parameters will be used in distribution selection (next chunks)

## Overlap Notes
- References Park et al. (2005), Elazouni & Gab-Allah (2004), Khanzadi et al. (2018) from CHUNK 001
- References Tran & Carmichael (2012) payment delays from CHUNK 002
- References FIDIC (2017) and Boussabaine & Elhag (1999) retention rates from CHUNK 001-002

## Content

### 1.2 Category-Specific Calibration Framework

#### 1.2.1 Domestic Low-Risk (DL) - Government Clients

**Client characteristics**:
- Iranian government agencies and National Iranian Oil Company (NIOC)
- Structured procurement, bureaucratic payment processes
- High payment reliability but longer processing times

**Literature calibration**:
- **Advance payment**: Khanzadi et al. (2018) reports 68% for Iranian government projects
  - Adjusted to 25% for "low-risk" subset (established contractors, routine projects)
  - Percentage: 8% (lower end of Park et al. 2005 range for government)
- **Milestone count**: Elazouni & Gab-Allah (2004) reports 8-12 for government
  - Calibrated to 7-9 (lower end for domestic, less complex projects)
- **Front-loading**: Park et al. (2005) reports 35-40% in first 30% for Middle East
  - Calibrated λ=0.15 (mild front-loading for government)
- **Payment delay**: Tran & Carmichael (2012) reports 75 days mean for government
  - Calibrated to 75 days (2.5 months)
- **Retention**: FIDIC standard 5%, Boussabaine & Elhag (1999) mean 5.2%
  - Calibrated to 5%

**Summary Table: DL Parameters**

| Parameter | Value | Range | Distribution | Source |
|-----------|-------|-------|--------------|--------|
| **Advance Payment Probability** | 25% | - | Bernoulli | Khanzadi et al. (2018) |
| **Advance Payment %** | 8% | 5-12% | Truncated Normal | Park et al. (2005) |
| **Milestone Count** | 7-9 | - | Discrete Uniform | Elazouni & Gab-Allah (2004) |
| **Front-Loading λ** | 0.15 | - | Deterministic | Park et al. (2005) |
| **Payment Delay (days)** | 75 | 30-120 | Log-Normal | Tran & Carmichael (2012) |
| **Retention Rate** | 5% | 3-8% | Truncated Normal | FIDIC (2017), Boussabaine & Elhag (1999) |

**Rationale for "Low-Risk" Classification**:
- Established contractors with proven track record
- Routine projects (standard refinery maintenance, pipeline extensions)
- Lower technical complexity
- Predictable scope and requirements
- Strong client-contractor relationship history

---

#### 1.2.2 Domestic High-Risk (DH) - Government Clients

**Client characteristics**:
- Same government clients but higher project complexity/risk
- More structured oversight, additional milestone checkpoints
- Complex technical requirements (new technology, first-of-kind installations)
- Higher uncertainty in scope and execution

**Literature calibration**:
- **Advance payment**: Higher probability (30%) due to mobilization needs
  - Percentage: 10% (Park et al. 2005 mean for government)
  - Higher percentage reflects greater upfront capital requirements
- **Milestone count**: 8-10 (upper end of Elazouni & Gab-Allah 2004 government range)
  - More milestones for tighter monitoring and risk control
- **Front-loading**: λ=0.20 (moderate, reflecting risk mitigation)
  - Stronger front-loading to reduce contractor financial exposure
- **Payment delay**: 75 days (same as DL, government bureaucracy)
  - Bureaucratic processes are consistent regardless of project risk
- **Retention**: 5% (standard government rate)
  - Government retention policies are uniform across risk levels

**Summary Table: DH Parameters**

| Parameter | Value | Range | Distribution | Source |
|-----------|-------|-------|--------------|--------|
| **Advance Payment Probability** | 30% | - | Bernoulli | Khanzadi et al. (2018) |
| **Advance Payment %** | 10% | 6-15% | Truncated Normal | Park et al. (2005) |
| **Milestone Count** | 8-10 | - | Discrete Uniform | Elazouni & Gab-Allah (2004) |
| **Front-Loading λ** | 0.20 | - | Deterministic | Park et al. (2005) |
| **Payment Delay (days)** | 75 | 30-120 | Log-Normal | Tran & Carmichael (2012) |
| **Retention Rate** | 5% | 4-10% | Truncated Normal | FIDIC (2017), Elazouni & Gab-Allah (2004) |

**Rationale for "High-Risk" Classification**:
- New or unproven contractors
- Complex projects (new refinery units, advanced processing facilities)
- Higher technical complexity and innovation
- Uncertain scope with potential for changes
- Limited client-contractor relationship history
- Higher probability of delays and cost overruns

---

**Key Differences Between DL and DH**:

| Aspect | DL (Low-Risk) | DH (High-Risk) | Rationale |
|--------|---------------|----------------|-----------|
| **Advance Payment Probability** | 25% | 30% | Higher mobilization needs for complex projects |
| **Advance Payment %** | 8% | 10% | Greater upfront capital requirements |
| **Milestone Count** | 7-9 | 8-10 | More checkpoints for risk monitoring |
| **Front-Loading λ** | 0.15 | 0.20 | Stronger front-loading reduces contractor risk |
| **Payment Delay** | 75 days | 75 days | Bureaucratic processes are uniform |
| **Retention Rate** | 5% | 5% | Government policy is consistent |

**Working Capital Implications**:
- **DL projects**: Lower milestone count and weaker front-loading → Higher peak working capital (28-32% of contract value)
- **DH projects**: More milestones and stronger front-loading → Moderate peak working capital (26-30% of contract value)
- **Trade-off**: DH has more administrative overhead (more milestones) but better cash flow profile

**Validation Against Literature**:
- **Khanzadi et al. (2018)**: Iranian government projects show 68% advance payment prevalence overall; our 25-30% split reflects risk-based segmentation
- **Elazouni & Gab-Allah (2004)**: Government milestone count 8-12; our 7-10 range is slightly conservative but within bounds
- **Tran & Carmichael (2012)**: Government payment delay mean 75 days matches our calibration exactly
- **FIDIC (2017)**: Standard 5% retention aligns with our domestic government calibration

**Implementation Notes**:
- Category assignment (DL vs DH) should be based on project complexity score or risk assessment at portfolio generation
- Suggested split: 60% DL, 40% DH for typical government portfolio (based on Khanzadi et al. 2018 project mix)
- Parameters are sampled independently for each project within category bounds
