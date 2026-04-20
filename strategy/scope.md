# Scope of Work: Q1 Research Paper
## Reinforcement Learning for Dynamic Project Portfolio Budgeting under Cashflow Uncertainty: A Rolling Horizon Approach

---

## 1. Research Problem

### 1.1 Core Challenge
Project portfolio budgeting under cashflow uncertainty requires dynamic decision-making in environments where:
- Future contractor performance is unpredictable
- Traditional optimization methods fail due to incomplete observability
- Sequential decisions must adapt to emerging information
- Projects have staggered start times and variable durations
- Portfolio horizons extend beyond fixed planning windows

### 1.2 Limitations of Existing Approaches
Classic Operations Research (OR) algorithms (MIP, Stochastic Programming) are inadequate because:
- They require complete observability of the environment
- They assume known probability distributions of future uncertainties
- They cannot adapt to real-time information updates
- They fail to handle the sequential nature of budget allocation decisions
- Scenario tree formulations become computationally intractable for variable-horizon portfolios

### 1.3 Research Gap
As of 2026, no practical, validated models exist for portfolio budgeting optimization that:
- Handle cashflow uncertainty dynamically
- Adapt to company-specific historical patterns
- Provide sequential decision-making capabilities under partial observability
- Accommodate variable-length portfolios with staggered project lifecycles
- Balance computational tractability with real-world complexity

---

## 2. Proposed Solution

$$P(s_{t+1}|s_t, a_t) \text{ is unknown due to contractor performance uncertainty}$$

Traditional methods require:
- Known transition probabilities → **unavailable**
- Complete scenario enumeration → **intractable**
- Static optimization → **incompatible with sequential decisions**

RL provides:
- Model-free learning from experience
- Adaptive policy under uncertainty
- Scalable to real-time decision-making

### 2.1 Methodology
Train a Reinforcement Learning (RL) agent using **Rolling Horizon Control** with:
1. **Pre-training**: Literature and industry-wide project data
2. **Fine-tuning**: Company-specific historical performance data
3. **Rolling Horizon Framework**: Fixed lookahead window (H=12 periods) with receding control

### 2.2 Rolling Horizon Architecture

#### Core Concept
At each timestep $t$, the system:
1. Observes current portfolio state (active projects, budget status, performance trends)
2. Plans budget allocation for next $H=12$ periods
3. Executes only the allocation for period $t$
4. Advances to $t+1$ and re-plans with updated information

#### Key Properties
- **Fixed Episode Length**: RL trains on 12-period episodes regardless of total portfolio duration
- **Variable Portfolio Horizons**: Naturally handles portfolios spanning 12, 24, 36+ periods through sequential re-planning
- **Dynamic Project Management**: Accommodates projects starting/finishing at different times through masking and padding mechanisms
- **Computational Tractability**: Maintains feasible state-space size for both RL and OR baselines

#### Handling Variable Project Lifecycles
The rolling horizon framework uses **masking and padding** to handle:
- Projects that start after the current timestep (masked until start date)
- Projects that complete before the horizon end (masked after completion)
- Projects with partial completion at portfolio initialization
- Variable project durations and staggered entry/exit times

This mechanism allows the model to handle arbitrary project configurations without requiring all projects to be new at $t=0$.

### 2.3 Key Advantages of RL Approach
- **Adaptability**: Learns from sequential interactions rather than requiring complete upfront knowledge
- **Partial Observability**: Handles uncertainty without needing full probability distributions
- **Dynamic Optimization**: Adjusts decisions as new information emerges through rolling re-planning
- **Transferability**: Pre-trained model can be specialized for specific organizational contexts
- **Scalability**: Rolling horizon prevents state-space explosion for long-duration portfolios
- **Real-World Alignment**: Mimics actual planning behavior (annual budgets with periodic updates)

---

## 3. Assumptions

### 3.1 Foundational Principle: Simplest Model
**Rationale**: This is a brand new field in the literature. The Q1 paper establishes foundational proof-of-concept.

**Strategy**: Extension of each assumption will be explicitly recognized as future work.

### 3.2 Project Structure Assumptions

#### Cashflow‑Based Budgeting Assumption
- **All budgeting decisions are made solely using aggregated project cashflows (inflows and outflows)**
- Detailed resource‑level requirements (labor, materials, equipment) are **not modeled individually**
- Project execution schedules and resource demands are **compressed into a contractual S‑curve spending profile**
- Portfolio optimization operates on these S‑curves rather than task‑level or resource‑level data

**Justification**:
- S‑curve budgeting is the industry standard for high‑level financial planning
- Avoids unnecessary granularity inconsistent with Q1 scope
- Ensures tractable state and action spaces for RL models
- Maintains focus on strategic financial decisions rather than operational scheduling

#### Payment & Performance Method
- **All projects use Earned Value Management (EVM)** for performance measurement and payment
- Payments are tied to earned value milestones
- Contractor performance is measured through EVM metrics

**Justification**:
- EVM is industry-standard for project control
- Provides consistent performance measurement framework
- Enables quantifiable uncertainty modeling

#### Aggregate Performance Uncertainty
- **All project uncertainties are modeled as a single aggregated performance risk factor**
- Human resource variability, inflation effects, supply chain disruptions, and regulatory impacts are **not treated separately**
- Uncertainty enters the model through a unified performance noise term affecting project progress and cost
- No decomposition of uncertainty sources; only portfolio-level aggregate volatility is modeled

**Justification**:
- Simplifies stochastic modeling and avoids multi-factor calibration
- Keeps the environment stationary and tractable for RL/optimization
- Enables consistent uncertainty treatment across heterogeneous projects
- Prevents overfitting to specific external factors that are outside Q1 scope

### 3.3 Portfolio Structure Assumptions

#### Temporal Structure: Rolling Horizon Framework

- **Planning Horizon**: Fixed $H = 12$ periods (months) lookahead window  
- **Portfolio Duration**: Variable, determined by $[t_{\text{start}}, t_{\text{end}}]$ where:
  - $t_{\text{start}}$ = January of the year containing $\min(\text{project start times})$
  - $t_{\text{end}}$ = $\max(\text{project finish times})$
- **Project Lifecycles**: Heterogeneous start dates and durations  
- **Re-planning Frequency**: Monthly rolling re-optimization

**Justification**:

Anchoring $t_{\text{start}}$ to the calendar year start of the earliest project ensures explicit capture of **year-end productivity degradation** encoded in the seasonal variance function $\sigma(t)$. Key rationale:

- **Material Year-End Effects**: Contractor performance drops during November–January significantly impact portfolio outcomes. Aligning the temporal origin to January ensures seasonal shocks occur at consistent, interpretable month indices across all episodes.

- **Model Stability**: Mid-year temporal origins introduce shifted or aliased seasonal variance, complicating agent learning. A January anchor stabilizes the annual cycle and enables reliable pattern recognition.

- **Budget Cycle Alignment**: Most organizations operate on January–December fiscal cycles; this alignment improves realism and interpretability.

- **Stochastic Consistency**: Seasonal uncertainties (e.g., $\sigma_{\text{year-end}}$) synchronize cleanly with month indices, eliminating discontinuities or edge cases in the variance structure.

- **Agent Generalization**: Predictable recurrence of high-uncertainty months at fixed positions within episodes enables robust learning of seasonal dynamics.

This design balances realism and tractability without requiring explicit holiday calendars or variable season boundaries.

**Handling Variable Horizons**:

- Portfolio spanning 26 months: Solve 26 sequential 12-month rolling planning problems  
- Projects starting at $t = 7$: Enter planning window when current time $t_{\text{current}} + H \geq 7$  
- Projects finishing at $t = 19$: Exit planning window after EV target achieved  
- **Project Masking**: Projects not yet started ($t_{\text{current}} < t_{\text{project start}}$) or already completed are masked from the action space and state representation, ensuring the agent only allocates resources to active or imminent projects  
- **Termination**: Portfolio concludes when all projects complete or budget exhausts

#### Myopia Mitigation
- **Terminal Value Function**: Estimates value of projects extending beyond current planning horizon
- **Prevents Short-Termism**: Ensures decisions account for long-term project value
- **Implementation**: Learned as part of RL value function or approximated using remaining BAC

---

## 4. Out of Scope Exclusions

For the foundational paper, several modeling elements are intentionally excluded to preserve tractability and maintain a clear focus on the core contribution: **the rolling‑horizon reinforcement learning framework for portfolio budgeting decisions.** 

Detailed operational mechanisms—such as resource‑level scheduling, disaggregated uncertainty sources, multi‑objective optimization, heterogeneous contract structures, explicit holiday calendars, and year‑end parameter discontinuities are abstracted away in favor of a simplified, portfolio high level representation.

Instead, **many of these effects are absorbed into the seasonal variance component of the aggregated uncertainty parameter.** This allows the RL agent to learn conservative behavior during higher‑uncertainty periods while avoiding the need for explicit modeling of each underlying driver.

Each exclusion reflects a deliberate methodological choice. Introducing these elements would significantly increase model dimensionality, calibration requirements, and environmental non‑stationarity, potentially obscuring the primary contribution of the study. 

At the same time, these components represent **natural extensions of the framework (future work)** and provide clear opportunities for incremental research progression once the foundational RL formulation and experimental results are validated.


### Resource‑Level Budgeting & Micro‑Scheduling
- No modeling of task‑level schedules, activity networks, or WBS structures  
- No representation of resource calendars, skill categories, or utilization profiles  
- No simulation of material lead times or supply‑chain task sequences  
- No optimization of labor/equipment/material allocation within projects  

**Justification**:  
- These require operational project scheduling (CPM/PERT), outside strategic budgeting  
- Adds high dimensionality and non-stationarity incompatible with portfolio-level RL  
- Focus remains on financial and timing decisions, not execution logistics  


### Disaggregated Uncertainty
- No separate modeling of human resource variability  
- No explicit inflation modeling by category or market segment  
- No independent treatment of supply-chain disruption  
- No separate uncertainty source for regulatory or policy changes  

**Justification**:  
- Q1 models rely on aggregated uncertainty for tractability  
- Avoids multi-factor calibration against external economic datasets  
- Maintains a stationary stochastic environment needed for stable RL training  


### Multi‑Objective Optimization
- No explicit modeling of strategic value maximization  
- No risk-adjusted utility or weighted preference functions  
- No stakeholder-specific objective formulations  

**Justification**:  
- Q1 problem focuses on single-objective financial optimization  
- Multi-objective frameworks introduce Pareto front complexity  
- Requires stakeholder preference modeling outside the scope of automated RL  


### Advanced EVM Variations
- No mixed or hybrid payment schemes (EVM + milestones)  
- No portfolio with multiple payment logics simultaneously  
- No incentive/penalty mechanisms tied to performance metrics  

**Justification**:  
- Single EVM model ensures uniform performance and payment measurement  
- Simplifies reward computation and state transitions  
- Avoids heterogeneous contract structures that complicate the RL environment  


### Explicit Holiday Calendars
- No country-specific holiday calendars or productivity shutdown periods  
- No modeling of seasonal workforce productivity  
- No date-specific scheduling impacts  

**Justification**:  
- Calendar-driven productivity patterns introduce fine-grained temporal non-stationarity  
- Requires region-specific datasets outside Q1 scope  
- Q1 uses continuous time approximations, not calendar simulation  


### Year‑End Parameter Discontinuities
- No modeling of price index resets at new-year boundaries  
- No explicit regulatory update cycles at year-end  
- No contract renegotiation events tied to fiscal closure  
- No fiscal-year budget resets or rollover constraints  

**Justification**:  
- Year-end effects introduce deterministic discontinuities that break stationarity  
- Requires modeling economic policy cycles and real fiscal data calibration  
- Necessitates separating deterministic vs. stochastic drivers  
- Complicates reward design by adding discontinuous incentive structures  

---

## 5. Modeling Scope & Simplifications

### 5.1 Aggregated Contractor Performance Uncertainty with Seasonal Variation

To model contractor behavior in a tractable yet realistic manner, all sources of performance variability are consolidated into a single stochastic uncertainty parameter. This parameter governs deviations between planned progress and realized Earned Value (EV). Seasonal productivity effects, including year‑end holiday slowdowns, are incorporated through a time‑dependent variance structure.

#### 5.1.1 Unified Stochastic Representation

For each project \( i \) and time period \( t \), realized Earned Value is modeled as a normally distributed random variable:

\[
EV_i(t) \sim \mathcal{N}\big(\mu_i(t),\, \sigma(t)^2 \big)
\]

where:

- \( \mu_i(t) \) is the planned progress for project \( i \),
- \( \sigma(t) \) captures the aggregated performance uncertainty.

The unified uncertainty parameter \( \sigma(t) \) represents the combined effects of:

- workforce availability and performance fluctuations  
- inflation and macroeconomic conditions  
- supply chain instability  
- short-term regulatory shifts  
- technical and engineering challenges  
- **seasonal productivity variation**

These factors are not modeled individually; instead, their joint effect is absorbed into the stochastic variance term.

#### 5.1.2 Seasonal Variance Structure

Contractor productivity typically declines around the year‑end period (November, December, January). Instead of encoding explicit holiday calendars, this effect is introduced by allowing the uncertainty parameter to vary seasonally.

Let the model operate in monthly time steps with \( t \bmod 12 \) denoting the month of the year (0 = December, 1 = January, ..., 11 = November). The variance is defined piecewise:

\[
\sigma(t) =
\begin{cases}
\sigma_{\text{baseline}}, 
& t \bmod 12 \in \{2,3,\dots,10\} \\[6pt]
\sigma_{\text{year-end}}, 
& t \bmod 12 \in \{11,0,1\}
\end{cases}
\]

with:

\[
\sigma_{\text{year-end}} \approx 1.5 \,\sigma_{\text{baseline}}
\]

indicating a 50% increase in performance variance during the high‑uncertainty season. This elevates the risk of underperformance near the year‑end and encourages the RL agent to adopt more conservative allocation strategies during these periods.

#### 5.1.3 Modeling Rationale and Validation

This formulation is intentionally designed as an **aggregated contractor performance uncertainty model with seasonal variation**. It provides:

- a compact, low-dimensional representation suitable for RL training  
- sufficient realism to induce seasonally-aware agent behavior  
- a controlled structure that avoids explicit holiday calendars and discontinuities  

Validation and sensitivity analysis will evaluate how different choices of \( \sigma_{\text{baseline}} \) and \( \sigma_{\text{year-end}} \) affect system behavior. This approach is documented as a modeling simplification, with future extensions enabling explicit modeling of calendar effects or disaggregated uncertainty sources.


### 5.2 Rolling Horizon Framework Details

#### Planning Window Structure
At each timestep $t$:
- **Observation Window**: Current portfolio state
- **Planning Horizon**: $[t, t+H]$ where $H=12$ periods
- **Decision**: Budget allocation for period $t$ only
- **Lookahead**: Plans for $[t+1, t+H]$ to inform current decision but does not commit

#### State Representation
For each project $i$ at timestep $t$:

**Project-level state** (only for unmasked projects):
- Progress ratio: $p_i = \frac{EV_i(t)}{BAC_i}$
- Remaining duration: $r_i = \text{finish}_i - t$
- Performance trend: $\text{SPI}_i$ (moving average over last 3 periods)
- Active status: $\alpha_i \in \{0,1\}$
- Time to start: $s_i = \max(0, \text{start}_i - t)$

**Portfolio-level state**:
- Remaining budget: $B_{\text{rem}}(t)$
- Current period: $t$
- Periods to portfolio end: $t_{\text{end}} - t$
- Active project count: $n_{\text{active}}(t)$
- Seasonal indicator: $(t - t_{\text{start}}) \mod 12$ (captures year-end effects relative to portfolio origin)

**Project Masking**:
- Projects with $t < \text{start}_i$ (not yet started) or $EV_i(t) = BAC_i$ (completed) are masked from state and action space
- Masked projects contribute zero to state vectors and receive zero allocation

#### Action Space
- Budget allocation vector for current period: $\mathbf{b}(t) = [b_1(t), b_2(t), ..., b_n(t)]$
- Constraint: $\sum_i b_i(t) \leq B_{\text{available}}(t)$
- Only unmasked, active projects ($\alpha_i(t) = 1$ and $\text{start}_i \leq t < \text{finish}_i$) receive non-zero allocations
- Minimum payment constraints: $b_i(t) \geq b_{\min,i}$ if $\alpha_i(t) = 1$

#### Episode Termination
An episode ends when:
1. $t = t_{\text{end}}$ (all projects completed or reached latest finish time), OR
2. Budget exhausted ($B_{\text{rem}}(t) = 0$), OR
3. All projects masked (none active or imminent within horizon)

#### Reward Function

**Immediate reward** at timestep $t$:
$$r(t) = \sum_i [EV_i(t) - EV_i(t-1)] - \text{penalty}_{\text{budget overrun}} - \text{penalty}_{\text{project delay}}$$

**Terminal reward** (at episode end):
$$r_{\text{terminal}} = \sum_i EV_i(t_{\text{end}}) + \text{terminal\_value}(\text{incomplete projects})$$

where $\text{terminal\_value}$ estimates remaining project value for any unfinished work beyond $t_{\text{end}}$

---

#### Handling Edge-Case Portfolio Scenarios

The framework is designed to handle diverse portfolio configurations through its masking and temporal anchoring mechanisms:

**Scenario 1: Projects start after portfolio origin**  
*Example: Portfolio spans January–December, but all projects start in March*
- **Handling**: Projects are masked for $t \in [\text{Jan}, \text{Feb}]$
- **Behavior**: Agent observes empty action space; no allocations made
- **Impact**: Episode progresses normally; seasonal indicator still tracks from January origin

**Scenario 2: Temporal gaps between projects**  
*Example: Project A finishes in April, Project B starts in July*
- **Handling**: Both projects masked during $t \in [\text{May}, \text{Jun}]$
- **Behavior**: Agent makes no decisions during gap months
- **Impact**: Budget preserved; episode continues until $t_{\text{end}}$

**Scenario 3: Planning starts mid-project**  
*Example: Planning begins in June for projects that started in February*
- **Handling**: Historical data provided as initial state at $t_{\text{planning\_start}}$
  - Client supplies: $EV_i(t)$, $AC_i(t)$, $\text{SPI}_i(t)$, $\text{CPI}_i(t)$ for all $t < t_{\text{planning\_start}}$
  - These metrics become the **initial conditions** for agent decision-making
- **Behavior**: 
  - Agent receives actual performance history (Feb–May) as part of state at $t = \text{June}$
  - Makes first allocation decision at $t_{\text{planning\_start}}$ using real historical context
  - No simulation or inference of missing months required
- **Implementation**: Introduce parameter $t_{\text{planning\_start}} \geq t_{\text{start}}$
  - For $t < t_{\text{planning\_start}}$: historical data only (no agent actions)
  - For $t \geq t_{\text{planning\_start}}$: agent actively allocates budget
- **Impact**: Framework supports both fresh portfolio starts ($t_{\text{planning\_start}} = t_{\text{start}}$) and mid-project entry with known history

**Design Rationale**:
- **Temporal origin** ($t_{\text{start}}$) remains anchored to January for seasonal consistency and model generalization
- **Planning start** ($t_{\text{planning\_start}}$) decouples decision-making from temporal origin, enabling flexible deployment
- **Masking** ensures agent never acts on unavailable or completed projects, maintaining action space validity across all scenarios


---

## 6. Research Contribution

### 6.1 Primary Contribution
First validated demonstration of RL viability for project portfolio budgeting under cashflow uncertainty in an EVM-based environment using rolling horizon control for variable-length portfolios.

### 6.2 Novelty Claims
- **Methodological**: Application of RL with rolling horizon framework to handle variable-duration portfolios
- **Architectural**: Transfer learning (pre-training + fine-tuning) for portfolio budgeting with fixed-horizon episodes
- **Technical**: Masking and padding mechanisms for handling arbitrary project lifecycle configurations
- **Practical**: Framework for adapting general models to company-specific contexts while maintaining computational tractability
- **Theoretical**: Demonstration that sequential decision-making under partial observability with rolling re-planning outperforms static optimization
- **Domain**: First application to EVM-based portfolio budgeting with aggregated seasonal uncertainty

### 6.3 Positioning Strategy
- Emphasize rolling horizon as bridge between theoretical elegance and practical implementation
- Focus on RL's ability to learn seasonal patterns (including holiday effects) without explicit calendar modeling
- Highlight computational tractability advantage over scenario tree approaches
- Position simplifications as pragmatic choices for foundational work in a new field
- Clearly articulate assumptions as deliberate scope management

---

## 7. Paper Structure (Recommended)

### 7.1 Critical Sections

**Section 1: Problem Justification**
- Why stochastic programming fails for variable-horizon portfolios
- Computational intractability of scenario trees for long-duration portfolios
- Requirements for complete observability vs. reality of uncertainty
- Gap between theoretical models and practical applicability
- Importance of EVM-based portfolio management

**Section 2: Rolling Horizon Framework**
- Motivation: balancing planning depth with computational tractability
- Architecture: fixed lookahead window with receding control
- Handling variable portfolio durations through sequential re-planning
- Masking and padding mechanisms for arbitrary project configurations
- Terminal value function for myopia mitigation

**Section 3: RL Advantage**
- Sequential decision-making under partial observability
- Learning from historical patterns without explicit probability distributions
- Adaptability to emerging information through rolling re-planning
- Learning seasonal patterns (including holiday effects) from aggregated uncertainty

**Section 4: Methodology**
- Pre-training on literature/industry data
- Fine-tuning process and its value proposition
- Generalization vs. specialization trade-off
- Rolling horizon episode structure for RL training

**Section 5: Model Scope & Assumptions**
- Explicit documentation of all assumptions (project structure, portfolio structure)
- Rolling horizon framework details
- Justification for simplest model approach
- Clear articulation of exclusions
- Aggregated uncertainty with seasonal variation

**Section 6: Model Design**
- State representation (EVM metrics, budget status, time, seasonal indicators)
- Action space (budget allocation decisions for current period)
- Reward function (cashflow optimization with terminal value)
- Aggregated uncertainty parameter with time-varying variance
- Terminal value function design
- Masking and padding implementation

**Section 7: Validation**
- Demonstration of RL viability across variable-horizon portfolios
- Sensitivity analysis on uncertainty parameter and seasonal variance
- Comparison with OR baseline (rolling horizon MIP)
- Robustness across different portfolio compositions
- Evidence of learned seasonal adaptation

**Section 8: Limitations & Future Work**
- Disaggregating uncertainty into multiple parameters
- Explicit holiday calendar modeling for region-specific deployments
- Year-end parameter discontinuities (prices, regulations, fiscal resets)
- Multi-objective optimization (cost, risk, strategic value)
- Alternative payment structures beyond EVM
- Adaptive horizon length selection

---

## 8. Literature Review Strategy

### 8.1 Coverage Requirements
- Comprehensive review of OR approaches to portfolio optimization
- Rolling horizon control in operations research and control theory
- EVM-based project management and performance measurement
- Existing applications of RL in project management (if any)
- Stochastic programming limitations in uncertain environments
- Transfer learning and fine-tuning methodologies
- Seasonal pattern learning in RL

### 8.2 Positioning Statement
Support the claim: "No practical, validated models exist as of 2026 for dynamic portfolio budgeting under cashflow uncertainty using RL with rolling horizon control for variable-duration portfolios."

**Evidence Required**:
- Systematic review of portfolio optimization literature (2015-2026)
- Analysis of why existing approaches fail under uncertainty with variable horizons
- Documentation of gap between theoretical models and practical implementation
- Absence of RL applications to EVM-based portfolio budgeting with rolling horizon
- Review of rolling horizon applications in other domains (manufacturing, logistics)

---

## 9. Validation Requirements

### 9.1 Minimum Viable Demonstration
- RL agent successfully learns budget allocation policy in rolling horizon setting
- Performance improvement over baseline (rolling horizon MIP, greedy heuristic)
- Stability across multiple training runs
- Convergence of learning process
- Effective handling of portfolios with different total durations (12, 24, 36 periods)
- Correct handling of variable project lifecycles through masking/padding

### 9.2 Robustness Checks
- Sensitivity analysis on aggregated uncertainty parameter
- Sensitivity analysis on seasonal variance ratio ($\sigma_{\text{year-end}} / \sigma_{\text{baseline}}$)
- Performance across different portfolio compositions (varying project counts, sizes, start times)
- Fine-tuning effectiveness with varying amounts of company data
- Generalization from pre-training to fine-tuning
- Terminal value function accuracy assessment

### 9.3 Rolling Horizon Specific Validation
- Comparison of RL vs. MIP in rolling horizon setting (both use same framework)
- Demonstration that RL learns to anticipate seasonal productivity drops
- Analysis of decision consistency across re-planning cycles
- Myopia assessment: do decisions near horizon boundary degrade?
- Computational time comparison: RL inference vs. MIP solve time per timestep

### 9.4 Practical Viability
- Computational feasibility for real-world portfolio sizes
- Interpretability of learned policies
- Transferability of pre-trained model
- Alignment with EVM best practices
- Real-time decision-making capability (inference speed)

---

## 10. Assumption Management Strategy

### 10.1 Documentation Approach
**In Paper**:
- Dedicate subsection to "Model Assumptions and Scope"
- Present rolling horizon framework as architectural choice with clear justification
- Present each assumption with clear justification
- Link assumptions to "simplest model" strategy
- Frame as deliberate choices for foundational research

### 10.2 Defense Strategy
**Anticipated Criticism 1**: "Rolling horizon introduces myopia"

**Response Framework**:
1. Terminal value function explicitly addresses this concern
2. Validation demonstrates decision quality near horizon boundaries
3. Real-world planning actually operates this way (annual budgets with updates)
4. Computational tractability enables practical deployment vs. theoretical optimality

**Anticipated Criticism 2**: "Aggregating holiday effects into uncertainty is imprecise"

**Response Framework**:
1. Consistent with aggregated uncertainty assumption for foundational model
2. Seasonal variance pattern captures holiday impact without calendar complexity
3. RL demonstrates ability to learn seasonal adaptation
4. Explicit holiday calendars are natural extension for future work
5. Validation shows effective handling of year-end productivity variations

**Anticipated Criticism 3**: "Too many simplifying assumptions"

**Response Framework**:
1. This is a brand new field—foundational models require clear scope
2. Each assumption is relaxable in future work (provide roadmap)
3. Even with simplifications, the model demonstrates RL viability
4. Rolling horizon framework provides practical implementation path
5. Complexity can be added incrementally once foundation is validated

**Anticipated Criticism 4**: "Year-end parameter changes are ignored"

**Response Framework**:
1. Explicit year-end discontinuities require economic policy modeling beyond scope
2. Effects absorbed into seasonal uncertainty variance for foundational model
3. Separating deterministic policy changes from stochastic uncertainty requires additional theoretical framework
4. Natural extension once core RL+rolling horizon methodology is validated
5. Current approach allows RL to learn conservative behavior during transition periods

### 10.3 Future Work Roadmap
Present clear progression:
- **Phase 1 (Q1)**: Rolling horizon RL with aggregated seasonal uncertainty, masking/padding for variable lifecycles
- **Phase 2**: Disaggregate uncertainty into 2-3 key factors (labor, supply chain, economic)
- **Phase 3**: Explicit holiday calendar modeling for region-specific deployments
- **Phase 4**: Year-end parameter discontinuities (price indices, fiscal resets, regulatory updates)
- **Phase 5**: Multi-objective optimization and alternative payment structures
- **Phase 6**: Adaptive horizon length selection based on portfolio characteristics

---

## 11. Timeline & Milestones

### Q1 Deliverable
Complete research paper demonstrating:
1. Clear problem formulation and gap identification
2. Rolling horizon RL methodology with pre-training + fine-tuning framework
3. Explicit assumptions and exclusions documentation
4. Validation results across variable-horizon portfolios
5. Documented scope and limitations
6. Future research directions with clear progression path

### Success Criteria
- Crystal-clear contribution statement emphasizing rolling horizon innovation
- Well-validated demonstration of RL viability in rolling horizon setting
- Defensible simplifications with clear justification
- Comprehensive literature review supporting novelty claims
- Transparent assumption documentation
- Evidence of seasonal pattern learning without explicit calendar modeling
- Demonstration of masking/padding effectiveness for variable project lifecycles

---

## 12. Risk Mitigation

### 12.1 Novelty Challenge
**Risk**: Reviewers find existing work that addresses similar problems.

**Mitigation**:
- Conduct exhaustive literature review including rolling horizon applications
- Position contribution carefully (practical validation vs. theoretical novelty)
- Emphasize unique combination: RL + rolling horizon + transfer learning + EVM-based portfolio budgeting + aggregated seasonal uncertainty

### 12.2 Simplification Criticism
**Risk**: Reviewers question aggregated uncertainty parameter or holiday effect absorption.

**Mitigation**:
- Frame as explicit modeling choice for foundational work in new field
- Provide sensitivity analysis on seasonal variance
- Demonstrate RL learns seasonal patterns effectively
- Document clear path for future disaggregation and explicit calendar modeling
- Show that even simplified model provides value

### 12.3 Rolling Horizon Myopia Concern
**Risk**: Reviewers question short-sightedness of fixed horizon.

**Mitigation**:
- Terminal value function design and validation
- Demonstrate decision quality near horizon boundaries
- Compare with full-horizon optimization on small test cases
- Emphasize computational tractability vs. theoretical optimality trade-off
- Show real-world alignment with actual planning practices

### 12.4 Assumption Overload
**Risk**: Too many assumptions weaken contribution.

**Mitigation**:
- Present assumptions as deliberate scope management
- Show that each assumption is independently relaxable
- Provide future work roadmap demonstrating progression
- Emphasize that foundational research requires clear boundaries
- Highlight rolling horizon as practical implementation enabler

### 12.5 Validation Concerns
**Risk**: Insufficient demonstration of practical viability.

**Mitigation**:
- Use realistic portfolio scenarios with variable durations
- Compare against meaningful baselines (rolling horizon MIP, greedy heuristics)
- Show robustness across multiple conditions within scope
- Demonstrate learning convergence and stability
- Validate seasonal adaptation behavior
- Measure computational performance for real-time decision-making

### 12.6 Year-End Parameter Exclusion
**Risk**: Reviewers consider year-end effects too important to exclude.

**Mitigation**:
- Emphasize absorption into seasonal variance component
- Show RL learns conservative behavior during transition periods
- Argue that explicit modeling requires economic policy framework beyond foundational scope
- Position as high-priority future work with clear implementation path
- Demonstrate that current model still captures practical value

---
## 13. Key Messages

### 1. Problem Statement
Existing OR methods (MIP, Stochastic Programming) fail for dynamic project portfolio budgeting under cashflow uncertainty because:
- Traditional portfolio optimization assumes a **fixed planning horizon**, while real-world project portfolios evolve over time with **staggered project lifecycles**
- Complete observability requirement incompatible with emerging uncertainty
- Computational intractability of scenario trees for long-duration portfolios
- Static optimization paradigm cannot handle sequential decision-making under partial observability

### 2. Core Innovation
**Sequential Re-Planning with Fixed Horizon ($H=12$)**

Rolling Horizon RL framework that:
- Mimics real-world planning cycles (monthly re-planning)
- Handles variable portfolio durations through receding control
- Maintains fixed-episode training architecture
- Uses masking/padding for dynamic project lifecycles
- Mitigates myopia via terminal value function

### 3. Methodological Contribution
First application of RL with rolling horizon control to EVM-based portfolio budgeting, featuring:
- Transfer learning (pre-training on literature → fine-tuning on company data)
- Aggregated seasonal uncertainty with time-varying variance: $\sigma(t) = \sigma_0 \times (1 + \lambda S(t))$
- Computationally tractable framework bridging theory and practice
- Demonstration that sequential decision-making under uncertainty outperforms static optimization

### 4. Key Design Decisions

**✅ Included:**
- Rolling horizon control with 12-period lookahead
- EVM-based performance measurement
- Aggregated contractor uncertainty with seasonal variation ($\lambda S(t)$)
- Variable project lifecycles (staggered starts, different durations)
- Transfer learning architecture

**❌ Excluded (with justification):**
- **Disaggregated uncertainty factors**: Simplest Model principle for Q1 foundation (Phase 2 future work)
- **Explicit holiday calendars**: Effects absorbed into seasonal variance $\sigma(t)$ (Phase 3 future work)
- **Year-end parameter discontinuities**: Requires economic policy modeling; effects captured via $\lambda S(t)$ (Phase 4 future work)
- **Multi-objective optimization**: Single objective (EV maximization) for foundational proof-of-concept (Phase 5 future work)

### 5. Why RL is Justified (Not Overkill)
$$P(s_{t+1}|s_t, a_t) \text{ is unknown due to contractor performance uncertainty}$$

Traditional methods require:
- Known transition probabilities → **unavailable**
- Complete scenario enumeration → **intractable**
- Static optimization → **incompatible with sequential decisions**

RL provides:
- Model-free learning from experience
- Adaptive policy under uncertainty
- Scalable to real-time decision-making

### 6. State Representation Enhancement
Current state includes project-level EVM metrics. **Critical addition for Q1 paper:**

**Liquidity Pressure:**
$$L(t) = \frac{B_{\text{remaining}}(t)}{B_{\text{total}}}$$

This captures **proximity to budget exhaustion**, which critically affects allocation decisions.

### 7. Reward Function Formulation
$$r(t) = \sum_i \Delta EV_i(t) - \lambda_b \cdot \text{BudgetOveruse}(t) - \lambda_d \cdot \text{Delay}(t)$$

Where:
- $\Delta EV_i(t)$: Earned Value gained by project $i$
- $\lambda_b$: Penalty coefficient for budget inefficiency
- $\lambda_d$: Penalty coefficient for schedule delays

This ensures RL learns **capital efficiency**, not just EV maximization.

### 8. Seasonal Uncertainty Formalization
Replace informal notation with:

$$\sigma(t) = \sigma_0 \times (1 + \lambda S(t))$$

Where:
- $\sigma_0$: Baseline contractor performance variance
- $\lambda$: Seasonal volatility intensity
- $S(t) \in \{0, 1\}$: Seasonal indicator (1 for months 11, 12, 1)

**Benefits:**
- Formal mathematical representation
- Enables sensitivity analysis on $\lambda$
- Avoids non-stationary environment complexity

### 9. Validation Strategy (Critical for Q1 Acceptance)

**Baseline 1: Rolling Horizon MIP**
- Same $H=12$ horizon
- Perfect information within horizon
- Demonstrates RL performance under uncertainty vs. OR under certainty

**Baseline 2: Greedy Allocation**
- Allocate to highest SPI deficit: $\text{argmax}_i (1 - \text{SPI}_i)$
- Simple heuristic benchmark
- Shows value of learned policy

**Metrics:**
- Total EV delivered
- Budget utilization efficiency
- Schedule adherence
- Computational time

### 10. Defense Against Common Reviewer Objections

**Objection 1:** "Why not just use MIP with rolling horizon?"
**Response:** MIP requires known $P(s_{t+1}|s_t, a_t)$; RL learns from uncertain transitions.

**Objection 2:** "Model too simple (aggregated uncertainty)."
**Response:** Simplest Model principle for foundational Q1 paper; disaggregation is Phase 2 future work with clear roadmap.

**Objection 3:** "Fixed horizon arbitrary."
**Response:** $H=12$ matches industry practice (annual planning cycles); sensitivity analysis on $H$ included.

**Objection 4:** "Year-end effects oversimplified."
**Response:** Explicit parameter resets require economic policy modeling (out of scope); seasonal variance $\lambda S(t)$ captures operational effects.

### 11. Contribution Positioning

**To OR Community:**
RL provides practical solution where traditional methods become intractable under uncertainty.

**To RL Community:**
First successful application to EVM-based portfolio budgeting with rolling horizon architecture.

**To Practitioners:**
Computationally feasible framework mimicking real-world monthly planning cycles.

**To Reviewers:**
Clear scope, defensible assumptions, validated methodology, transparent limitations, concrete future work roadmap.

### 12. Critical Success Factors for Q1 Publication

✅ **Strong baselines** (Rolling MIP + Greedy)  
✅ **Formal mathematical notation** ($\sigma(t)$, $L(t)$, reward function)  
✅ **Sensitivity analysis** (on $\lambda$, $H$, portfolio composition)  
✅ **Computational feasibility** demonstration  
✅ **Clear contribution** statement in Introduction  
✅ **Transparent limitations** with future work roadmap  

### 13. Next Phase: RL Architecture Selection

For implementation, recommend evaluating:
- **PPO** (stable, widely validated)
- **SAC** (continuous action spaces)
- **Transformer-based RL** (multi-project attention mechanism)

Architecture choice can strengthen contribution if justified by problem structure.
