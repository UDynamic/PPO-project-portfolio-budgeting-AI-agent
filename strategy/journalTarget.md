# Target Journal Strategy for Q1 Publication in OR/IE (2026)

---

## Journal Selection

| Journal | Impact Factor | Accept Probability | Notes |
|---------|--------------|-------------------|-------|
| European Journal of Operational Research (EJOR) | ~6.4 | 25-30% | **Primary target** |
| Computers & Operations Research (COR) | ~4.6 | 30-35% | Best backup |
| Omega | ~6.7 | 30-35% | Strong applied OR fit |
| Management Science | ~5.5 | 10-15% | High-risk/high-reward |
| IEEE Transactions on Engineering Management | ~5.0 | 28-32% | Good for managerial angle |

**Recommended path:** EJOR → COR → Omega

---

## The Big 3 That Actually Move the Needle

### 1. Novelty Framing
Frame your contribution precisely:

> "We propose the first RL framework that jointly handles budget allocation +
> project interdependencies + uncertainty in a unified policy — outperforming
> both MIP baselines and human decision-makers by X%"

- Vague contributions = desk rejection in 48 hours
- Be specific in abstract and introduction

### 2. Baseline Comparison

| Baseline Type | Example |
|--------------|---------|
| Classical OR | MIP / Stochastic Programming |
| Heuristic | Greedy, Genetic Algorithm |
| Simple RL | DQN or A2C (if you use PPO) |
| Human benchmark | Human-vs-agent comparison |

Beat all four → reviewers have very little to attack.

### 3. Writing Quality

- **Abstract:** full story in 200 words — problem, method, result, implication
- **Introduction:** broad problem → gap → solution → contributions → structure
- **Conclusion:** summary + limitations + future work

---

## Medium Impact Items

- Sensitivity analysis across parameter settings
- Computational complexity discussion
- Real-world applicability paragraph (even for synthetic data)
- Code on GitHub for reproducibility

---

## Small but Visible Signals

- Cite recent (2023–2025) papers from your target journal
- Use target journal citation style from day one
- Clean figures — no pixelated images, consistent fonts
- Table captions must be self-contained

---

## Cover Letter (Most People Skip This)

Write 3 sentences addressing:

1. What is the gap in literature?
2. What exactly did you do?
3. Why does this journal's audience care?

---

## Realistic Acceptance Probability

| Condition | EJOR | COR |
|-----------|------|-----|
| Baseline only | ~15% | ~20% |
| + Strong baselines + clear novelty | ~35-40% | ~45-50% |

---

## Checklist Before Submission

- [ ] Contribution statement is specific and falsifiable
- [ ] At least 4 baseline types included
- [ ] Sensitivity analysis present
- [ ] Abstract tells the full story independently
- [ ] Recent papers from target journal cited
- [ ] Figures are high resolution and consistent
- [ ] Code repository linked
- [ ] Cover letter answers the 3 key questions

---
# Targetting ML contribution
AI Prompt Instructions: Revising Thesis Scope for ML Contribution

## Context
You are revising a thesis scope document to strongly position it as a Machine Learning (ML) contribution for academic publication in OR and ML journals. The work applies Deep Reinforcement Learning with rolling horizon control to dynamic project portfolio budgeting under cashflow uncertainty, using Earned Value Management (EVM) as the performance measurement framework.

## Core ML Contributions to Emphasize
- Novel application of RL with rolling horizon control
- Transfer learning protocol (pre-training + fine-tuning)
- Masking/padding mechanism for variable-length project lifecycles within fixed episodes
- Deployable system with democratization potential

---

## Step 1: Reframe Title & Problem Statement

### Objective
Shift focus from domain-specific application to general ML problem with broader applicability.

### Instructions for Title Revision

**Current Title:**
"Reinforcement Learning for Dynamic Project Portfolio Budgeting under Cashflow Uncertainty: A Rolling Horizon Approach"

**Revision Instructions:**
- Replace "Reinforcement Learning" with "Deep Reinforcement Learning" or "Machine Learning"
- Add ML-centric keywords: "sequential decision-making," "model-free learning," "transfer learning"
- Reframe "budgeting under cashflow uncertainty" as a general sequential resource allocation problem
- Emphasize ML novelty: rolling horizon + transfer learning + masking mechanism

**Target Title Format:**
"Deep Reinforcement Learning with Transfer Learning for Sequential Resource Allocation under Uncertainty: A Rolling Horizon Framework"

### Instructions for Section 1.1 (Core Challenge)

**Revision Instructions:**
- Lead with the ML problem: "Sequential decision-making under partial observability with variable-length episodes"
- Then connect to the application domain (portfolio budgeting) as a validation case
- Emphasize ML challenges:
  - Unknown transition dynamics P(s'|s,a)
  - High-dimensional state space
  - Sparse rewards
  - Variable-length episodes within fixed decision horizons

**Example Opening:**
"Sequential resource allocation under uncertainty represents a fundamental challenge in machine learning, particularly when episode lengths vary and transition dynamics are unknown. This work addresses this challenge in the context of..."

### Instructions for Section 1.2 (Limitations of Existing Approaches)

**Revision Instructions:**
- Soften critique of Operations Research (OR) methods
- Instead of: "OR methods are inadequate"
- Use: "Traditional optimization approaches require known transition probabilities P(s'|s,a), which are unavailable in domains with contractor performance uncertainty and stochastic payment delays"
- Frame RL as complementary to OR, not superior
- Emphasize: "Model-free RL addresses this gap by learning optimal policies directly from experience"

### Instructions for Section 1.3 (Research Gap)

**Revision Instructions:**
- Replace domain-specific gap statements like "no practical validated models exist for EPC portfolio budgeting"
- With ML-focused gap: "No prior work has applied model-free RL with transfer learning to sequential resource allocation problems with variable-length episodes"
- Emphasize technical novelty:
  - "First application of fixed-horizon receding-control RL with variable-length masking"
  - "Novel transfer learning protocol for portfolio optimization"
  - "State representation design for partial observability under seasonal uncertainty"

---

## Step 2: Reframe Methodology as ML Architecture

### Objective
Highlight ML architecture, learning protocols, and technical innovations over domain-specific implementation details.

### Instructions for Section 2 (Proposed Solution) Restructuring

**Current Framing (likely):**
"Train an RL agent using Rolling Horizon Control with pre-training and fine-tuning"

**New Framing:**
Lead with ML architecture components:

1. **Model-free Deep RL Framework**
   - Fixed-horizon episodes (H=12 months)
   - Receding horizon control for indefinite planning
   - Policy network architecture (specify: MLP, attention, etc.)

2. **Variable-Length Masking Mechanism**
   - Handles dynamic project lifecycles within fixed episodes
   - Padding and masking protocol for inactive projects
   - Ensures valid action spaces at each timestep

3. **Two-Stage Transfer Learning Protocol**
   - Pre-training on synthetic/industry-wide data
   - Fine-tuning on company-specific historical data
   - Quantified performance gain: 8-12%

### Instructions for New Subsection: "2.4 Technical ML Contributions"

**Add the following content:**

#### 2.4.1 Masking and Padding Mechanism
- **Problem:** Projects have variable lifecycles (6-36 months) but RL requires fixed episode length
- **Solution:** Dynamic masking that zeros out actions for inactive projects
- **Implementation:** 
  - State padding for projects not yet started or already completed
  - Action masking to prevent invalid budget allocations
  - Reward masking to exclude inactive projects from objective
- **Include:** Pseudocode or architecture diagram

#### 2.4.2 Transfer Learning Protocol
- **Pre-training Phase:**
  - Dataset: Synthetic portfolios + industry benchmarks
  - Objective: Learn general S-curve dynamics and seasonal patterns
  - Training: 10,000-50,000 episodes
- **Fine-tuning Phase:**
  - Dataset: Company-specific historical projects (50-100 records)
  - Objective: Adapt to organization-specific constraints and risk preferences
  - Training: 1,000-5,000 episodes
- **Performance Gain:** 8-12% improvement over pre-trained baseline

#### 2.4.3 State Representation Design
- **Partial Observability Handling:**
  - Liquidity pressure indicator: $L(t) = \frac{\text{Committed} - \text{Available}}{\text{Total Budget}}$
  - Seasonal uncertainty: $\sigma(t)$ = rolling standard deviation of payment delays
  - EVM metrics: SPI, CPI, EAC for each project
  - Budget status: allocated, spent, remaining
  - Temporal features: month-of-year, project age

#### 2.4.4 Reward Shaping
- **Multi-objective formulation:**
  $$r_t = \sum_{i} \Delta EV_i(t) - \lambda_b \cdot \text{BudgetOveruse}(t) - \lambda_d \cdot \text{Delay}(t)$$
- **Components:**
  - Earned Value progress (primary objective)
  - Budget compliance penalty (hard constraint relaxation)
  - Delay penalty (schedule performance)
- **Hyperparameter tuning:** $\lambda_b$, $\lambda_d$ learned via grid search

### Instructions for MDP Formalization

**Add formal MDP specification:**

- **State Space:** $\mathcal{S} = \mathbb{R}^{n \times d}$ where $n$ = max projects, $d$ = feature dimension
  - $s_t = [\text{EVM metrics}, \text{budget status}, L(t), \sigma(t), \text{seasonal indicators}]$
  
- **Action Space:** $\mathcal{A} = \mathbb{R}^n$ (budget allocation vector, subject to masking)
  - Constraints: $\sum_i a_i \leq B(t)$, $a_i \geq 0$
  
- **Reward Function:** $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ (as defined above)

- **Transition Dynamics:** $P(s_{t+1}|s_t, a_t)$ **unknown** → justifies model-free RL approach

- **Discount Factor:** $\gamma = 0.95$ (12-month horizon)

---

## Step 3: Soften Domain-Specific Assumptions

### Objective
Present assumptions as experimental setup choices for tractability, not inherent domain limitations.

### Instructions for Section 3 Revision

**Current Title (likely):**
"Section 3: Assumptions"

**New Title:**
"Section 3: Experimental Setup and Modeling Choices"

**Add Preamble:**
"To validate the ML framework, we instantiate it in the domain of EPC project portfolio management. The following modeling choices reflect this domain's characteristics but do not limit the generality of the RL architecture. Each choice can be relaxed in future work."

### Instructions for Reframing Each Assumption

**Pattern to Follow:**
- Replace: "We assume..."
- With: "For this instantiation, we model..."
- Add: "This choice can be relaxed by [future ML extension]"

**Example Revisions:**

#### Assumption: Industry Context (EPC Oil & Gas)
**OLD:** "This work focuses on EPC Oil & Gas projects."
**NEW:** "We validate the framework on EPC Oil & Gas portfolio management, a domain characterized by high capital intensity, long project lifecycles (12-36 months), and stochastic payment delays. The RL architecture is domain-agnostic and applicable to any sequential resource allocation problem."

#### Assumption: Cashflow-Based Budgeting
**OLD:** "Budgeting decisions use only aggregated project cashflows, not resource-level details."
**NEW:** "We model budgeting at the cashflow level rather than resource level, consistent with EPC industry practice where portfolio managers allocate funds to projects without micromanaging resource assignments. The RL framework is agnostic to this choice—the state representation can accommodate resource-level features if available."

#### Assumption: Uniform S-Curve Shape
**OLD:** "All projects follow the same S-curve shape."
**NEW:** "We use a parametric S-curve model with category-specific parameters (α, β) to represent project progress dynamics. The RL agent learns to adapt to these patterns through experience. Future work could replace the parametric model with a learned dynamics model (e.g., using recurrent networks)."

#### Assumption: IID Payment Delays
**OLD:** "Payment delays are independent and identically distributed."
**NEW:** "We model payment delays as IID random variables for tractability. This assumption can be relaxed by incorporating temporal dependencies using LSTM or Transformer architectures in the policy network to capture autocorrelation in payment patterns."

#### Assumption: Known Project Pipeline
**OLD:** "The set of projects and their start dates are known in advance."
**NEW:** "We assume a known project pipeline for the planning horizon, reflecting typical EPC contracting practices where projects are committed 6-12 months in advance. The framework can be extended to handle stochastic project arrivals using online RL methods."

---

## Step 4: Add ML Evaluation Metrics Section

### Objective
Define ML-specific metrics and baselines to demonstrate rigor and reproducibility.

### Instructions: Add New Section "2.5 Evaluation Protocol"

**Insert after Section 2.4 (Technical ML Contributions)**

#### 2.5.1 Performance Metrics

**Learning Efficiency:**
- Sample efficiency: learning curves (episodes vs. cumulative reward)
- Convergence rate: episodes required to reach 95% of asymptotic performance
- Wall-clock training time on standard hardware

**Transfer Learning Effectiveness:**
- Transfer learning gain: $\frac{\text{Performance}_{\text{fine-tuned}} - \text{Performance}_{\text{pre-trained}}}{\text{Performance}_{\text{pre-trained}}} \times 100\%$
- Data efficiency: performance vs. fine-tuning dataset size
- Catastrophic forgetting: performance retention on pre-training distribution

**Generalization:**
- Performance on held-out test portfolios (unseen project combinations)
- Cross-validation across different time periods
- Performance under distribution shift (different delay distributions, seasonal patterns)

**Robustness:**
- Sensitivity to hyperparameters ($\lambda_b$, $\lambda_d$, learning