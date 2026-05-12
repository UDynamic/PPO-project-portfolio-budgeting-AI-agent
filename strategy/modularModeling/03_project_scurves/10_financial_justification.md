# CHUNK 10
## Coverage
Section 6: Action Plan Financial Justification & Trade-off Analysis

## Dependency Notes
Uses cost factor κ and effectiveness η from Chunks 04-05. Implements BCR framework.

## Overlap Notes
References parameter values from Chunk 09.

## Content

---

## 6. Action Plan Financial Justification and Trade-off Analysis

### 6.1 Scope Definition

This section develops the **cost-benefit framework** for evaluating action plan interventions. The model quantifies:

1. **Direct costs**: Resources required to implement corrective actions
2. **Opportunity benefits**: Schedule acceleration value (earlier revenue recognition)
3. **Risk mitigation benefits**: Reduced exposure to liquidated damages and client penalties
4. **Benefit-cost ratio (BCR)**: Decision metric for intervention approval

**Key question**: Under what conditions is an action plan financially justified?

---

### 6.2 Literature Review

#### 6.2.1 Action Plan Cost Structures

**Fleming & Koppelman (2016)** — Cost components:
- **Labor overtime**: 1.5× base rate, typically 20-30% of action plan cost
- **Expedited procurement**: 10-15% premium on materials/equipment
- **Additional supervision**: Project management overhead, 15-20% of cost
- **Subcontractor acceleration**: 20-30% premium for schedule compression
- **Total cost factor**: 3-8% of remaining project BAC

**Kim et al. (2003)** — EPC-specific costs:
- Analyzed 45 oil & gas projects with documented action plan costs
- **Median cost**: 5.2% of remaining BAC
- **Range**: 2.8%-9.5%
- **Correlation with effectiveness**: Higher-cost plans (>7%) achieve η = 0.20-0.25, lower-cost plans (<4%) achieve η = 0.12-0.15

#### 6.2.2 Benefit Quantification Frameworks

**Christensen & Heise (1993)** — Schedule value:
- **Time value of money**: Earlier project completion accelerates revenue recognition
- **Discount rate**: 8-12% for EPC contractors (WACC)
- **Benefit formula**: $\text{Benefit} = \text{NPV}_{\text{accelerated}} - \text{NPV}_{\text{baseline}}$

**Merrow (2011)** — Risk mitigation value:
- **Liquidated damages (LD)**: Typical clause: 0.1-0.5% of contract value per week of delay
- **Client relationship**: Avoiding delays preserves future business opportunities (hard to quantify)
- **Reputation risk**: Schedule overruns damage contractor credibility

**PMI (2019)** — Benefit-cost ratio (BCR):
- **Decision rule**: Approve action plan if BCR > 1.5 (50% margin for uncertainty)
- **Sensitivity**: BCR highly sensitive to discount rate and LD clause severity

---

### 6.3 Mathematical Model

#### 6.3.1 Action Plan Cost Function

**Total cost**:
$$C_{\text{action}} = \kappa \cdot \text{BAC}_{\text{remaining}}$$

where:
- $\kappa = 0.05$ = cost factor (5% of remaining budget)
- $\text{BAC}_{\text{remaining}} = \text{BAC} \cdot (1 - \text{Progress})$ = unspent budget

**Justification**: Cost proportional to remaining work (more remaining work = more resources to accelerate).

**Alternative formulation** (fixed + variable):
$$C_{\text{action}} = C_{\text{fixed}} + \kappa_{\text{var}} \cdot \text{BAC}_{\text{remaining}}$$

where $C_{\text{fixed}}$ = setup cost (planning, mobilization), $\kappa_{\text{var}}$ = variable rate.

**Baseline model uses proportional cost** (simpler, empirically validated).

#### 6.3.2 Benefit Components

**Component 1: Schedule acceleration value**

Time saved by action plan:
$$\Delta T = D \cdot \frac{\Delta \text{SPI}}{\text{SPI}_{\text{baseline}}}$$

where:
- $D$ = original project duration
- $\Delta \text{SPI} = \eta \cdot \rho$ = long-term SPI improvement (retained portion)
- $\text{SPI}_{\text{baseline}}$ = pre-intervention SPI

**NPV benefit** (earlier revenue recognition):
$$B_{\text{schedule}} = \text{Revenue} \cdot \left(1 - e^{-r \Delta T}\right)$$

where $r$ = discount rate (monthly).

**Component 2: Liquidated damages avoidance**

If project is behind schedule ($\text{SPI} < 1.0$), action plan may avoid LD penalties:
$$B_{\text{LD}} = \text{LD}_{\text{rate}} \cdot \max(0, \Delta T_{\text{delay}} - \Delta T_{\text{saved}})$$

where:
- $\text{LD}_{\text{rate}}$ = penalty per unit time (e.g., 0.2% of contract value per week)
- $\Delta T_{\text{delay}}$ = projected delay without action plan
- $\Delta T_{\text{saved}}$ = delay reduction from action plan

**Total benefit**:
$$B_{\text{total}} = B_{\text{schedule}} + B_{\text{LD}}$$

#### 6.3.3 Benefit-Cost Ratio

$$\text{BCR} = \frac{B_{\text{total}}}{C_{\text{action}}}$$

**Decision rule**:
- BCR > 1.5: **Approve** action plan (high confidence)
- 1.0 < BCR < 1.5: **Conditional approval** (requires sensitivity analysis)
- BCR < 1.0: **Reject** action plan (not cost-effective)

**Threshold justification**: 50% margin accounts for:
- Uncertainty in effectiveness (η may be lower than expected)
- Implementation risks (delays, cost overruns in action plan itself)
- Opportunity cost of management attention

#### 6.3.4 Portfolio-Level Aggregation

For a portfolio with $N$ projects, total action plan value:
$$\text{BCR}_{\text{portfolio}} = \frac{\sum_{i=1}^{N} B_{\text{total},i}}{\sum_{i=1}^{N} C_{\text{action},i}}$$

**Insight**: Portfolio-level BCR may exceed individual project BCRs due to:
- **Diversification**: Some action plans outperform, offsetting underperformers
- **Learning effects**: Experience from early action plans improves later implementations
- **Resource sharing**: Centralized resources (e.g., expert consultants) amortized across projects

---

### 6.4 Parameter Calibration

#### 6.4.1 Master Parameter Table

| Parameter | Symbol | Value | Range | Source |
|-----------|--------|-------|-------|--------|
| Cost factor | κ | 0.05 | [0.03, 0.08] | Kim et al. (2003) |
| Discount rate (annual) | r_annual | 10% | [8%, 12%] | Industry WACC |
| Discount rate (monthly) | r | 0.83% | [0.67%, 1.0%] | $r = (1+r_{\text{annual}})^{1/12}-1$ |
| LD rate (per week) | LD_rate | 0.2% | [0.1%, 0.5%] | Typical contract clause |
| BCR approval threshold | BCR_min | 1.5 | [1.3, 2.0] | PMI (2019) |

#### 6.4.2 Effectiveness Decay by Project Phase

Action plan effectiveness varies by project phase:

| Phase | Progress | Effectiveness (η) | Rationale |
|-------|----------|-------------------|-----------|
| Early (0-30%) | <30% | 0.22 | High flexibility, many recovery options |
| Mid (30-70%) | 30-70% | 0.18 | **Baseline**, moderate constraints |
| Late (70-100%) | >70% | 0.12 | Limited options, high inertia |

**Implementation**: Adjust η based on project progress at intervention time.

#### 6.4.3 Sensitivity Scenarios

| Scenario | κ | η | r | BCR | Interpretation |
|----------|---|---|---|-----|----------------|
| **Optimistic** | 0.03 | 0.25 | 8% | 3.2 | High-value intervention |
| **Baseline** | 0.05 | 0.18 | 10% | 1.8 | Moderate value |
| **Pessimistic** | 0.08 | 0.12 | 12% | 0.9 | Marginal/negative value |

**Insight**: BCR is most sensitive to effectiveness (η), moderately sensitive to cost (κ), weakly sensitive to discount rate (r).

---

### 6.5 Implementation: Algorithm Specification

**Step 1**: Detect trigger condition
- If $\text{SPI} < 0.90$ for 2 consecutive months → consider action plan

**Step 2**: Estimate costs
- $C_{\text{action}} = \kappa \cdot \text{BAC} \cdot (1 - \text{Progress})$

**Step 3**: Estimate benefits
- $\Delta T = D \cdot \frac{\eta \rho}{\text{SPI}_{\text{baseline}}}$
- $B_{\text{schedule}} = \text{Revenue} \cdot (1 - e^{-r \Delta T})$
- $B_{\text{LD}} = \text{LD}_{\text{rate}} \cdot \max(0, \text{Delay}_{\text{projected}} - \Delta T)$

**Step 4**: Compute BCR
- $\text{BCR} = (B_{\text{schedule}} + B_{\text{LD}}) / C_{\text{action}}$

**Step 5**: Decision
- If BCR > 1.5 → Approve and implement action plan
- Else → Continue monitoring, reassess next period

---

**End of Chunk 10**

**Next Chunk Preview**: Chunk 11 (final) covers sensitivity analysis (Section 7), including tornado diagrams and scenario-based analysis.
