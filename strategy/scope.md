# Scope of Work: Q1 Research Paper

## Paper Title
**Reinforcement Learning for Dynamic Project Portfolio Budgeting under Cashflow Uncertainty**

*Alternative:* Developing an RL Agent for Project Portfolio Budgeting Optimization under Cashflow Uncertainties

---

## 1. Research Problem

### 1.1 Core Challenge
Project portfolio budgeting under cashflow uncertainty requires dynamic decision-making in environments where:
- Future contractor performance is unpredictable
- Traditional optimization methods fail due to incomplete observability
- Sequential decisions must adapt to emerging information

### 1.2 Limitations of Existing Approaches
Classic Operations Research (OR) algorithms (MIP, Stochastic Programming) are inadequate because:
- They require complete observability of the environment
- They assume known probability distributions of future uncertainties
- They cannot adapt to real-time information updates
- They fail to handle the sequential nature of budget allocation decisions

### 1.3 Research Gap
As of 2026, no practical, validated models exist for portfolio budgeting optimization that:
- Handle cashflow uncertainty dynamically
- Adapt to company-specific historical patterns
- Provide sequential decision-making capabilities under partial observability

---

## 2. Proposed Solution

### 2.1 Methodology
Train a Reinforcement Learning (RL) agent using:
1. **Pre-training**: Literature and industry-wide project data
2. **Fine-tuning**: Company-specific historical performance data

### 2.2 Key Advantages of RL Approach
- **Adaptability**: Learns from sequential interactions rather than requiring complete upfront knowledge
- **Partial Observability**: Handles uncertainty without needing full probability distributions
- **Dynamic Optimization**: Adjusts decisions as new information emerges
- **Transferability**: Pre-trained model can be specialized for specific organizational contexts

---

## 3. Assumptions

### 3.1 Foundational Principle: Simplest Model
**Rationale**: This is a brand new field in the literature. The Q1 paper establishes foundational proof-of-concept.

**Strategy**: Extension of each assumption will be explicitly recognized as future work.

### 3.2 Project Structure Assumptions

#### Payment & Performance Method
- **All projects use Earned Value Management (EVM)** for performance measurement and payment
- Payments are tied to earned value milestones
- Contractor performance is measured through EVM metrics

**Justification**:
- EVM is industry-standard for project control
- Provides consistent performance measurement framework
- Enables quantifiable uncertainty modeling

### 3.3 Portfolio Structure Assumptions

#### Initial State (Timestep 0)
- **All projects at portfolio start (t=0) are brand new projects without history**
- Any project with prior history is assumed to have transitioned to a new contractor
- No legacy projects carry over into the portfolio timespan

**Justification**:
- Simplifies state representation for initial RL model
- Eliminates complexity of partial project histories
- Provides clean starting point for sequential decision-making

**Implication**:
- RL agent learns from scratch for each portfolio cycle
- Fine-tuning uses historical data from completed portfolios, not in-progress projects

#### Temporal Constraint
- **All project contracts finish no later than portfolio timespan (12 periods)**
- No projects extend beyond the 12-period horizon

**Justification**:
- Aligns with fixed time horizon assumption
- Eliminates need for modeling project continuation beyond portfolio boundary
- Consistent with annual budget cycle reset due to price/regulation adjustments

**Handling Edge Cases**:
- Projects that would naturally extend beyond 12 periods are either:
  - Not included in the portfolio, OR
  - Restructured/split to fit within the timespan

---

## 4. Exclusions

### 4.1 Out of Scope for Q1 Paper

**Multi-Period Project Histories**:
- Projects with partial completion at portfolio start
- Carryover projects from previous budget cycles
- Mid-contract contractor transitions with preserved history

**Disaggregated Uncertainty**:
- Separate modeling of human resource quality
- Explicit inflation modeling
- Supply chain disruption as independent variable
- Regulatory change as separate uncertainty source

**Multi-Objective Optimization**:
- Strategic value maximization
- Risk-adjusted portfolio selection
- Stakeholder preference modeling

**Advanced EVM Variations**:
- Multiple payment structures within same portfolio
- Hybrid payment models (EVM + milestone-based)
- Performance incentive mechanisms

**Extended Time Horizons**:
- Portfolios spanning multiple years
- Rolling horizon optimization
- Dynamic project addition/removal during portfolio execution

### 4.2 Justification for Exclusions
Each exclusion represents:
- Additional complexity that would obscure core contribution
- Natural extension for future work
- Opportunity for incremental research progression

---

## 5. Modeling Scope & Simplifications

### 5.1 Uncertainty Aggregation
**Decision**: Aggregate all factors affecting contractor performance and Earned Value into a single uncertainty parameter.

**Factors Collapsed**:
- Human resource availability and performance
- Inflation and economic conditions
- Supply chain disruptions
- Regulatory changes (within-year)
- Technical challenges

**Justification**:
- Establishes foundational model for proof-of-concept
- Reduces state-space complexity for initial validation
- Allows focus on RL architecture and learning dynamics

**Mitigation Strategy**:
- Frame as "aggregated contractor performance uncertainty"
- Conduct sensitivity analysis on this parameter
- Document as explicit limitation with future work directions

### 5.2 Temporal Boundaries

#### Fixed Time Horizon
- **Portfolio timespan**: 12 periods (symbolic representation)
- **Rationale**: Provides sufficient horizon for sequential decision-making while maintaining computational tractability

#### Project Handling Rules

**Projects Starting Before Portfolio Period**:
- Not applicable under current assumptions (all projects start at t=0)
- If encountered in future extensions: split at portfolio start date, treat remainder as new project

**Projects Extending Beyond 12 Periods**:
- Not included in portfolio under current assumptions
- Future work: model as new projects for next cycle due to annual parameter adjustments (prices, regulations)

**Boundary Assumption**:
- Year-end represents natural reset point due to:
  - Budget cycle boundaries
  - Regulatory updates
  - Price/cost adjustments
  - Contract renegotiations

---

## 6. Research Contribution

### 6.1 Primary Contribution
First validated demonstration of RL viability for project portfolio budgeting under cashflow uncertainty in an EVM-based environment.

### 6.2 Novelty Claims
- **Methodological**: Application of RL with transfer learning (pre-training + fine-tuning) to portfolio budgeting
- **Practical**: Framework for adapting general models to company-specific contexts
- **Theoretical**: Demonstration that sequential decision-making under partial observability outperforms static optimization
- **Domain**: First application to EVM-based portfolio budgeting with aggregated uncertainty

### 6.3 Positioning Strategy
- Emphasize RL's adaptability advantage over static optimization
- Focus on practical validation rather than claiming theoretical completeness
- Position simplifications as pragmatic choices for foundational work in a new field
- Clearly articulate assumptions as deliberate scope management

---

## 7. Paper Structure (Recommended)

### 7.1 Critical Sections

**Section 1: Problem Justification**
- Why stochastic programming fails for this problem class
- Requirements for complete observability vs. reality of uncertainty
- Gap between theoretical models and practical applicability
- Importance of EVM-based portfolio management

**Section 2: RL Advantage**
- Sequential decision-making under partial observability
- Learning from historical patterns without explicit probability distributions
- Adaptability to emerging information

**Section 3: Methodology**
- Pre-training on literature/industry data
- Fine-tuning process and its value proposition
- Generalization vs. specialization trade-off

**Section 4: Model Scope & Assumptions**
- Explicit documentation of all assumptions (project structure, portfolio structure)
- Justification for simplest model approach
- Clear articulation of exclusions
- Temporal boundary assumptions

**Section 5: Model Design**
- State representation (EVM metrics, budget status, time)
- Action space (budget allocation decisions)
- Reward function (cashflow optimization objectives)
- Aggregated uncertainty parameter modeling

**Section 6: Validation**
- Demonstration of RL viability
- Sensitivity analysis on uncertainty parameter
- Comparison with baseline approaches
- Robustness across different portfolio compositions

**Section 7: Limitations & Future Work**
- Disaggregating uncertainty into multiple parameters
- Incorporating projects with history at t=0
- Extending temporal horizon beyond 12 periods
- Multi-objective optimization (cost, risk, strategic value)
- Alternative payment structures beyond EVM

---

## 8. Literature Review Strategy

### 8.1 Coverage Requirements
- Comprehensive review of OR approaches to portfolio optimization
- EVM-based project management and performance measurement
- Existing applications of RL in project management (if any)
- Stochastic programming limitations in uncertain environments
- Transfer learning and fine-tuning methodologies

### 8.2 Positioning Statement
Support the claim: "No practical, validated models exist as of 2026 for dynamic portfolio budgeting under cashflow uncertainty using RL."

**Evidence Required**:
- Systematic review of portfolio optimization literature (2015-2026)
- Analysis of why existing approaches fail under uncertainty
- Documentation of gap between theoretical models and practical implementation
- Absence of RL applications to EVM-based portfolio budgeting

---

## 9. Validation Requirements

### 9.1 Minimum Viable Demonstration
- RL agent successfully learns budget allocation policy
- Performance improvement over baseline (random, heuristic, or static optimization)
- Stability across multiple training runs
- Convergence of learning process

### 9.2 Robustness Checks
- Sensitivity analysis on aggregated uncertainty parameter
- Performance across different portfolio compositions (varying project counts, sizes)
- Fine-tuning effectiveness with varying amounts of company data
- Generalization from pre-training to fine-tuning

### 9.3 Practical Viability
- Computational feasibility for real-world portfolio sizes
- Interpretability of learned policies
- Transferability of pre-trained model
- Alignment with EVM best practices

---

## 10. Assumption Management Strategy

### 10.1 Documentation Approach
**In Paper**:
- Dedicate subsection to "Model Assumptions and Scope"
- Present each assumption with clear justification
- Link assumptions to "simplest model" strategy
- Frame as deliberate choices for foundational research

### 10.2 Defense Strategy
**Anticipated Criticism**: "Too many simplifying assumptions"

**Response Framework**:
1. This is a brand new field—foundational models require clear scope
2. Each assumption is relaxable in future work (provide roadmap)
3. Even with simplifications, the model demonstrates RL viability
4. Complexity can be added incrementally once foundation is validated

### 10.3 Future Work Roadmap
Present clear progression:
- **Phase 1 (Q1)**: Simplest model with aggregated uncertainty, clean portfolio start
- **Phase 2**: Disaggregate uncertainty into 2-3 key factors
- **Phase 3**: Incorporate projects with history at t=0
- **Phase 4**: Extend time horizon and multi-period optimization
- **Phase 5**: Multi-objective optimization and alternative payment structures

---

## 11. Timeline & Milestones

### Q1 Deliverable
Complete research paper demonstrating:
1. Clear problem formulation and gap identification
2. RL methodology with pre-training + fine-tuning framework
3. Explicit assumptions and exclusions documentation
4. Initial validation results
5. Documented scope and limitations
6. Future research directions with clear progression path

### Success Criteria
- Crystal-clear contribution statement
- Well-validated demonstration of RL viability
- Defensible simplifications with clear justification
- Comprehensive literature review supporting novelty claims
- Transparent assumption documentation

---

## 12. Risk Mitigation

### 12.1 Novelty Challenge
**Risk**: Reviewers find existing work that addresses similar problems.

**Mitigation**:
- Conduct exhaustive literature review
- Position contribution carefully (practical validation vs. theoretical novelty)
- Emphasize unique combination: RL + transfer learning + EVM-based portfolio budgeting + aggregated uncertainty

### 12.2 Simplification Criticism
**Risk**: Reviewers question aggregated uncertainty parameter or "all new projects" assumption.

**Mitigation**:
- Frame as explicit modeling choice for foundational work in new field
- Provide sensitivity analysis
- Document clear path for future disaggregation
- Show that even simplified model provides value

### 12.3 Assumption Overload
**Risk**: Too many assumptions weaken contribution.

**Mitigation**:
- Present assumptions as deliberate scope management
- Show that each assumption is independently relaxable
- Provide future work roadmap demonstrating progression
- Emphasize that foundational research requires clear boundaries

### 12.4 Validation Concerns
**Risk**: Insufficient demonstration of practical viability.

**Mitigation**:
- Use realistic portfolio scenarios consistent with assumptions
- Compare against meaningful baselines
- Show robustness across multiple conditions within scope
- Demonstrate learning convergence and stability

---

## 13. Key Messages

1. **Problem**: Existing OR methods fail under cashflow uncertainty due to observability requirements.
2. **Solution**: RL provides adaptive, sequential decision-making under partial observability.
3. **Innovation**: Transfer learning (pre-training + fine-tuning) enables practical deployment for EVM-based portfolios.
4. **Scope**: Foundational model with pragmatic simplifications for proof-of-concept in a brand new field.
5. **Assumptions**: Deliberate choices to establish simplest viable model; each assumption is relaxable in future work.
6. **Contribution**: First validated demonstration of RL viability for this problem class.
7. **Progression**: Clear roadmap from foundational model to comprehensive solution.

---

## 14. Documentation Checklist

### Before Submission
- [ ] All assumptions explicitly stated and justified
- [ ] All exclusions clearly documented
- [ ] Future work roadmap provided for each assumption/exclusion
- [ ] Sensitivity analysis conducted on aggregated uncertainty parameter
- [ ] Baseline comparisons completed
- [ ] Literature review confirms novelty claim
- [ ] EVM framework properly integrated
- [ ] Validation demonstrates RL viability within stated scope
- [ ] Limitations section is transparent and comprehensive
- [ ] Contribution statement is precise and defensible

---

## Document Version
Version 2.0 | Date: 1405/01/31 (2026/04/20)

**Changelog**:
- Added comprehensive Assumptions section (3.0)
- Added Exclusions section (4.0)
- Integrated EVM-based payment structure
- Added Assumption Management Strategy (10.0)
- Expanded risk mitigation for assumption-related concerns
- Added documentation checklist
