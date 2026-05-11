# Perfect Foresight Baseline Evaluation for RL Portfolio Optimization with Stress Testing and Failure Mode Analysis

## Executive Summary

This document establishes a rigorous evaluation framework for Reinforcement Learning (RL)-based portfolio optimization in project management contexts. We propose a **Perfect Foresight (PF)** baseline as a theoretical upper bound to measure the optimality gap of learned policies. The framework addresses the challenge of **endogenous stochasticity**—where intervention outcomes depend on agent actions—through a hybrid pre-sampling approach that ensures fair comparison across algorithms while preserving the causal structure of the decision problem.

Beyond standard performance evaluation, we introduce a **Stress Testing and Failure Mode Analysis** module designed to validate agent robustness under extreme conditions. This includes scenarios such as contractor collapse, intervention resistance, and sunk cost traps. These stress tests serve dual purposes: (1) demonstrating that the agent can identify severely underperforming projects, terminate them, and reallocate resources optimally, and (2) providing training scenarios and guidelines for real-world deployment.

We explicitly scope out interdependencies between projects, mid-episode budget shocks, and contract price volatility as natural extensions for future work, with clear methodological pathways grounded in the operations research and project management literature.

---

## 1. Introduction

### 1.1 Motivation

Portfolio optimization in project management involves sequential decision-making under uncertainty, where an agent must allocate limited resources (budget, interventions) across multiple projects to maximize cumulative net present value (NPV). Traditional approaches such as greedy heuristics and Model Predictive Control (MPC) often fail to account for long-term dependencies and stochastic dynamics (Gutjahr & Reiter, 2010; Caramanis & Iancu, 2021).

Reinforcement Learning offers a promising alternative by learning policies directly from interaction with the environment (Sutton & Barto, 2018). However, evaluating RL performance in this domain is non-trivial due to:

1. **Endogenous stochasticity**: Intervention outcomes (e.g., Schedule Performance Index recovery) depend on agent actions, creating a feedback loop between policy and environment dynamics.
2. **Lack of theoretical benchmarks**: Without a known optimal policy, it is difficult to assess whether an RL agent has converged to a near-optimal solution or is trapped in a local optimum.
3. **Robustness under stress**: Real-world portfolios face extreme events (contractor failures, intervention resistance) that may not be adequately represented in standard training distributions.

### 1.2 Contributions

This document makes the following contributions:

1. **Perfect Foresight Baseline**: We formalize PF as a theoretical upper bound that assumes complete knowledge of future stochastic realizations. PF provides a reference point for measuring the optimality gap of RL, MPC, and heuristic policies.

2. **Hybrid Pre-Sampling Framework**: We propose a methodology that pre-samples exogenous stochasticity (base contractor performance, recovery rates) while computing endogenous dynamics (performance gaps, intervention triggers) live during evaluation. This ensures fairness across algorithms with different policies.

3. **Stress Testing Module**: We introduce a suite of adversarial scenarios (contractor collapse, intervention resistance, sunk cost traps) designed to test agent robustness and decision-making under extreme conditions. These scenarios are grounded in project management literature and serve as both validation tools and training augmentation.

4. **Scoping and Future Work**: We explicitly exclude interdependencies, budget shocks, and price volatility from the current scope, providing clear methodological pathways for future extensions based on network optimization (Elmaghraby, 1977), real options theory (Trigeorgis, 1996), and stochastic programming (Birge & Louveaux, 2011).

---

## 2. Problem Formulation

### 2.1 Portfolio Environment

We model the portfolio optimization problem as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **State space** $\mathcal{S}$: Each state $s_t$ contains information about all projects at time $t$, including:
  - Schedule Performance Index (SPI)
  - Cost Performance Index (CPI)
  - Percent complete
  - Remaining budget
  - Intervention history
  
- **Action space** $\mathcal{A}$: At each timestep, the agent selects:
  - Budget allocation $b_i \in [0, B_{\text{max}}]$ for each project $i$
  - Intervention decision $\delta_i \in \{0, 1\}$ (trigger or not)
  - Termination decision $\tau_i \in \{0, 1\}$ (continue or terminate)

- **Transition dynamics** $\mathcal{P}(s_{t+1} | s_t, a_t)$: Project states evolve according to:
  - **Exogenous factors**: Base contractor SPI $\text{SPI}_{\text{base}} \sim \mathcal{N}(\mu_{\text{SPI}}, \sigma_{\text{SPI}})$
  - **Endogenous factors**: Performance gap $\Delta_t = \text{SPI}_{\text{target}} - \text{SPI}_t$ triggers interventions if $\Delta_t > \theta$
  - **Intervention effects**: If intervention is triggered, SPI recovers by $\text{recovery\_rate} \sim \mathcal{N}(\mu_r, \sigma_r)$

- **Reward function** $\mathcal{R}(s_t, a_t)$: Immediate reward is the discounted cash flow at time $t$:
  $$
  r_t = \sum_{i=1}^{N} \left( \text{Revenue}_i(t) - \text{Cost}_i(t) \right) \cdot \frac{1}{(1 + \rho)^t}
  $$
  where $\rho$ is the discount rate.

- **Discount factor** $\gamma$: Typically set to 1 for finite-horizon episodic tasks.

### 2.2 Objective

The agent's goal is to learn a policy $\pi^*$ that maximizes expected cumulative NPV:
$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} r_t \right]
$$
where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ is a trajectory sampled under policy $\pi$.

---

## 3. Perfect Foresight as a Theoretical Upper Bound

### 3.1 Definition

**Perfect Foresight (PF)** is a hypothetical policy that has access to the complete realization of all stochastic variables before making any decisions. Formally, PF observes:
- All future base SPI values $\{\text{SPI}_{\text{base}}^i(t)\}_{t=0}^{T}$ for each project $i$
- All future recovery rates $\{r_i(t)\}_{t=0}^{T}$ for each intervention
- All future exogenous shocks (if any)

Given this information, PF solves a deterministic optimization problem to compute the optimal action sequence $\{a_0^*, a_1^*, \ldots, a_T^*\}$ that maximizes total NPV.

### 3.2 Role in Evaluation

PF serves three critical functions:

1. **Upper Bound**: PF provides the maximum achievable NPV for a given episode. Any policy operating under uncertainty (RL, MPC, heuristics) must achieve NPV $\leq$ NPV$_{\text{PF}}$.

2. **Optimality Gap Metric**: The optimality gap quantifies how far a policy is from the theoretical optimum:
   $$
   \text{Gap}(\pi) = \frac{\text{NPV}_{\text{PF}} - \text{NPV}_{\pi}}{\text{NPV}_{\text{PF}}} \times 100\%
   $$
   A gap of 5% indicates the policy achieves 95% of the PF baseline.

3. **Sanity Check**: If an RL agent achieves NPV $>$ NPV$_{\text{PF}}$, it indicates a bug in either the PF implementation or the evaluation framework (e.g., information leakage, inconsistent environment dynamics).

### 3.3 Limitations

PF is **not a realistic policy**. It assumes:
- Complete knowledge of future stochastic realizations (violates causality)
- No computational constraints (can solve arbitrarily complex optimization problems)
- No model uncertainty (knows the true transition dynamics)

Despite these limitations, PF is a standard tool in operations research for benchmarking heuristics and approximate algorithms (Powell, 2011; Bertsekas, 2019). It is analogous to the "clairvoyant algorithm" in online optimization (Borodin & El-Yaniv, 1998).

---

## 4. The Challenge of Endogenous Stochasticity

### 4.1 Problem Statement

In many RL benchmarks (e.g., Atari, MuJoCo), stochasticity is **exogenous**: random events occur independently of agent actions. In such cases, one can pre-sample an entire episode's random seed and evaluate multiple policies on the same trajectory (Colas et al., 2018).

However, in portfolio optimization, stochasticity is **endogenous**: intervention outcomes depend on whether the agent chooses to intervene. Specifically:

- **Exogenous**: Base contractor SPI is sampled at the start of each timestep, independent of agent actions.
- **Endogenous**: If the agent triggers an intervention, a recovery rate is sampled and applied. If the agent does not intervene, no recovery occurs.

This creates a **causal dependency**: the sequence of random variables depends on the policy. Naive pre-sampling (e.g., sampling all recovery rates upfront) is invalid because:
1. It assumes interventions occur at predetermined times, which may not match the agent's policy.
2. It leads to unfair comparisons: an RL agent that intervenes at $t=5$ may face a different recovery rate than an MPC agent that intervenes at $t=7$, even though they are supposedly evaluated on the "same" episode.

### 4.2 Literature Context

This issue is well-known in the simulation-based optimization literature:

- **Common Random Numbers (CRN)**: A variance reduction technique that uses the same random seed across different policies to enable fair comparison (Law & Kelton, 2000). However, CRN assumes exogenous stochasticity.
- **Endogenous Uncertainty**: Arises in problems where decisions affect the information structure (e.g., clinical trials, adaptive experimentation). Standard CRN does not apply (Chick et al., 2015).
- **Hybrid Approaches**: Some works propose splitting randomness into exogenous and endogenous components, pre-sampling only the former (Glasserman, 2004).

Our framework adopts this hybrid approach, tailored to the portfolio optimization domain.

---

## 5. Proposed Solution: Hybrid Pre-Sampling

### 5.1 Key Idea

We decompose stochasticity into two categories:

1. **Exogenous Stochasticity**: Random variables that are independent of agent actions. These are pre-sampled once per episode and shared across all algorithms.
   - Base contractor SPI: $\text{SPI}_{\text{base}}^i(t) \sim \mathcal{N}(\mu_{\text{SPI}}, \sigma_{\text{SPI}})$
   - Recovery rate distribution parameters: $\mu_r, \sigma_r$ (but not the realized recovery rates)

2. **Endogenous Dynamics**: Random variables that depend on agent actions. These are sampled **live** during evaluation, triggered by the agent's policy.
   - Performance gap: $\Delta_t = \text{SPI}_{\text{target}} - \text{SPI}_t$ (depends on past actions)
   - Intervention trigger: $\mathbb{I}(\Delta_t > \theta)$ (depends on agent's intervention decision)
   - Recovery rate realization: $r_t \sim \mathcal{N}(\mu_r, \sigma_r)$ (sampled only if intervention is triggered)

### 5.2 Fairness Guarantee

By pre-sampling exogenous factors, we ensure that all algorithms (RL, MPC, PF, heuristics) face the **same baseline conditions**:
- Same contractor performance trends
- Same underlying difficulty of each project

By computing endogenous dynamics live, we ensure that each algorithm's **policy influences outcomes** in a causally consistent manner:
- An aggressive intervention policy will trigger more recovery events
- A conservative policy will face more performance degradation

This hybrid approach satisfies the **fairness criterion**: two policies are compared on the same episode if and only if they face the same exogenous conditions, while their distinct actions lead to different endogenous outcomes.

### 5.3 Formal Procedure

**Episode Generation:**
```
function generate_episode(env, seed):
    rng = RandomGenerator(seed)
    episode_data = {}
    
    for t in 0 to T:
        for project i in portfolio:
            # Pre-sample exogenous factors
            episode_data[i][t]['base_spi'] = rng.normal(μ_SPI, σ_SPI)
            episode_data[i][t]['recovery_rate_params'] = (μ_r, σ_r)
    
    return episode_data

**Hybrid Evaluation Environment:**

class HybridEvaluationEnv:
    def __init__(self, base_env, episode_data):
        self.base_env = base_env
        self.episode_data = episode_data
        self.t = 0
    
    def step(self, action):
        # Use pre-sampled exogenous data
        for i, project in enumerate(self.base_env.projects):
            project.base_spi = self.episode_data[i][self.t]['base_spi']
        
        # Compute endogenous dynamics live
        for i, project in enumerate(self.base_env.projects):
            if action.intervene[i]:
                μ_r, σ_r = self.episode_data[i][self.t]['recovery_rate_params']
                recovery = self.base_env.rng.normal(μ_r, σ_r)
                project.spi += recovery
        
        # Standard environment step
        next_state, reward, done, info = self.base_env.step(action)
        self.t += 1
        return next_state, reward, done, info
```

---

## 6. Perfect Foresight Implementation

### 6.1 MILP Formulation (Simplified)

A full MILP formulation for PF is complex due to conditional logic (intervention triggers, termination decisions). A simplified version assumes:
- Interventions can be triggered at any time (no threshold-based triggering)
- Termination decisions are binary variables
- Budget constraints are linear

**Decision Variables:**
- $x_{it} \in [0, 1]$: Fraction of project $i$ completed at time $t$
- $b_{it} \geq 0$: Budget allocated to project $i$ at time $t$
- $\delta_{it} \in \{0, 1\}$: Intervention triggered for project $i$ at time $t$
- $\tau_{it} \in \{0, 1\}$: Project $i$ is terminated at time $t$

**Objective:**
$$
\max \sum_{i=1}^{N} \sum_{t=0}^{T} \frac{\text{Revenue}_{it} - \text{Cost}_{it}}{(1 + \rho)^t}
$$

**Constraints:**
- Progress dynamics: $x_{i,t+1} = x_{it} + \text{SPI}_{it} \cdot \frac{b_{it}}{\text{BAC}_i}$
- SPI evolution: $\text{SPI}_{it} = \text{SPI}_{\text{base},it} + \delta_{it} \cdot r_{it}$ (where $r_{it}$ is pre-sampled)
- Budget limit: $\sum_{i=1}^{N} b_{it} \leq B_{\text{total}}$
- Termination logic: If $\tau_{it} = 1$, then $b_{i,t'} = 0$ for all $t' > t$

**Challenges:**
- Conditional constraints (e.g., "if $\Delta_t > \theta$, then allow $\delta_{it} = 1$") require Big-M formulations, which can be numerically unstable.
- Nonlinear reward functions (e.g., NPV with complex cash flow structures) may require piecewise linearization.

### 6.2 Simulation-Based Approach (Recommended)

Given the complexity of MILP, we recommend a **simulation-based PF** that uses the pre-sampled episode data:

```python
function solve_perfect_foresight_simulation(env, episode_data):
    # Initialize with pre-sampled data
    env_pf = HybridEvaluationEnv(env, episode_data)
    
    # Greedy forward simulation with perfect knowledge
    state = env_pf.reset()
    total_reward = 0
    
    for t in 0 to T:
        # Compute optimal action given future knowledge
        action = greedy_action_with_foresight(state, episode_data, t)
        state, reward, done, info = env_pf.step(action)
        total_reward += reward
    
    return total_reward

function greedy_action_with_foresight(state, episode_data, t):
    # For each project, compute expected NPV if continued
    npv_estimates = []
    for i, project in enumerate(state.projects):
        future_spi = [episode_data[i][t']['base_spi'] for t in range(t, T)]
        npv = compute_npv_with_known_spi(project, future_spi)
        npv_estimates.append(npv)
    
    # Allocate budget to projects with highest NPV
    action = allocate_budget_greedy(npv_estimates, state.budget)
    return action
```
This approach is computationally tractable and provides a reasonable upper bound, though it may not be globally optimal (since it uses a greedy heuristic with perfect information rather than solving a full dynamic program).

### 6.3 Validation

To validate the PF implementation:
1. **Monotonicity Check**: NPV$_{\text{PF}}$ should be $\geq$ NPV$_{\text{RL}}$ for all episodes.
2. **Sensitivity Analysis**: Perturb episode data (e.g., increase base SPI) and verify that NPV$_{\text{PF}}$ increases.
3. **Comparison with Hindsight Optimal**: For small problems, solve the full dynamic program with hindsight and compare with PF.

---

## 7. Stress Testing and Failure Mode Analysis

### 7.1 Motivation

Standard evaluation on randomly sampled episodes may not reveal critical failure modes. Real-world portfolios face **tail events** that are rare but consequential:
- Contractor bankruptcies leading to sudden performance collapse
- Interventions that fail to improve performance (or make it worse)
- Projects that become unviable mid-execution due to sunk cost fallacies

**Stress testing** involves deliberately constructing adversarial scenarios to test agent robustness. This serves three purposes:

1. **Validation**: Verify that the agent can identify and terminate bad projects, rather than blindly continuing them.
2. **Training Augmentation**: Use stress scenarios as additional training data to improve policy robustness (analogous to adversarial training in supervised learning; Goodfellow et al., 2014).
3. **Deployment Guidelines**: Provide operators with documented failure modes and expected agent behavior, building trust in the system.

### 7.2 Scenario Design Principles

We design stress scenarios based on:
- **Project management literature**: Common failure modes documented in case studies (Flyvbjerg et al., 2003; Merrow, 2011)
- **Extreme value theory**: Tail events that are 2-3 standard deviations from the mean (Embrechts et al., 1997)
- **Adversarial construction**: Scenarios that exploit potential weaknesses in RL policies (e.g., sunk cost bias, overconfidence in interventions)

Each scenario specifies:
- **Trigger condition**: When the stress event occurs (e.g., $t=8$, or when project is 50% complete)
- **Magnitude**: How severe the deviation is (e.g., SPI drops from 0.9 to 0.4)
- **Expected agent behavior**: What a rational policy should do (e.g., terminate within 3 months)
- **Baseline comparison**: How naive heuristics (e.g., "never terminate") perform

### 7.3 Scenario 1: Contractor Collapse

**Description:**  
A contractor experiences sudden operational failure (bankruptcy, loss of key personnel, supply chain disruption), causing SPI to drop precipitously.

**Parameterization:**
- **Trigger time**: $t = 8$ (mid-project)
- **SPI drop**: $\text{SPI}_{\text{base}}$ drops from $\mathcal{N}(0.9, 0.1)$ to $\mathcal{N}(0.4, 0.05)$
- **Duration**: Permanent (no recovery)

**Expected Agent Behavior:**
1. **Detection**: Agent should detect the anomaly within 1-2 timesteps (via SPI monitoring).
2. **Intervention attempt**: Agent may trigger an intervention to test if recovery is possible.
3. **Termination decision**: If intervention fails (recovery rate $\approx 0$), agent should terminate the project within 3 timesteps to minimize losses.
4. **Reallocation**: Freed budget should be reallocated to healthier projects with positive NPV.

**Baseline Comparison:**
- **Greedy NPV**: Continues the project indefinitely (sunk cost fallacy), leading to large losses.
- **MPC**: May detect the issue but has limited horizon, leading to delayed termination.
- **RL (expected)**: Should terminate quickly if trained with sufficient exploration.

**Empirical Metrics:**
- Time to termination: $t_{\text{term}} - t_{\text{trigger}}$
- NPV loss: $\text{NPV}_{\text{baseline}} - \text{NPV}_{\text{agent}}$
- Reallocation efficiency: NPV of projects that received reallocated budget

**Literature Context:**  
Contractor failure is a well-documented risk in megaprojects (Flyvbjerg et al., 2003). The optimal response is early termination and reallocation, as continuing a failing project leads to "throwing good money after bad" (Staw, 1976).

---

### 7.4 Scenario 2: Intervention Resistance

**Description:**  
A project has structural issues (poor design, misaligned incentives, technical infeasibility) that make interventions ineffective or counterproductive.

**Parameterization:**
- **Trigger condition**: Performance gap $\Delta_t > \theta$ (intervention is triggered)
- **Recovery rate**: $r_t \sim \mathcal{N}(-0.05, 0.02)$ (interventions make performance worse)
- **Duration**: Persistent across all interventions

**Expected Agent Behavior:**
1. **Initial intervention**: Agent triggers intervention based on performance gap.
2. **Observation**: Agent observes that SPI decreases after intervention.
3. **Learning**: Agent should learn to avoid further interventions on this project.
4. **Termination**: If project NPV becomes negative, agent should terminate.

**Baseline Comparison:**
- **Rule-based intervention**: Continues to intervene whenever $\Delta_t > \theta$, worsening performance.
- **MPC**: May reduce intervention frequency but lacks long-term memory to avoid the project entirely.
- **RL (expected)**: Should learn to stop intervening after 1-2 failed attempts.

**Empirical Metrics:**
- Number of interventions: Should decrease over time as agent learns.
- NPV trajectory: Should stabilize or improve after agent stops intervening.
- Termination rate: Higher for RL compared to baselines.

**Literature Context:**  
Intervention resistance is common in projects with misaligned stakeholder incentives (Merrow, 2011). The "intervention paradox" (where corrective actions worsen outcomes) has been studied in organizational behavior (Argyris, 1990). Optimal policy is to recognize futility early and cut losses.

---

### 7.5 Scenario 3: Sunk Cost Trap

**Description:**  
A project is 80% complete but has experienced severe delays (6 months behind schedule). Continuing the project requires significant additional investment, but the remaining NPV is marginal.

**Parameterization:**
- **Initial state**: $x_0 = 0.8$ (80% complete), $\text{SPI} = 0.6$ (severe delays)
- **Remaining budget required**: $1.5 \times$ original estimate
- **Remaining NPV**: $\text{NPV}_{\text{remaining}} = 0.1 \times \text{NPV}_{\text{original}}$
- **Opportunity cost**: Alternative projects have NPV $= 0.5 \times \text{NPV}_{\text{original}}$

**Expected Agent Behavior:**
1. **NPV re-evaluation**: Agent should compute remaining NPV (not total NPV including sunk costs).
2. **Opportunity cost comparison**: Compare remaining NPV with NPV of alternative projects.
3. **Termination decision**: If $\text{NPV}_{\text{remaining}} < \text{NPV}_{\text{alternative}}$, terminate and reallocate.

**Baseline Comparison:**
- **Completion-driven heuristic**: Always completes projects that are >50% done, regardless of NPV.
- **Greedy NPV**: May continue due to high total NPV (including sunk costs).
- **RL (expected)**: Should terminate if trained to ignore sunk costs.

**Empirical Metrics:**
- Termination rate: Percentage of episodes where agent terminates the project.
- NPV gain: $\text{NPV}_{\text{RL}} - \text{NPV}_{\text{baseline}}$ (should be positive if agent reallocates).
- Decision time: How quickly agent makes the termination decision.

**Literature Context:**  
The sunk cost fallacy is a well-known cognitive bias in decision-making (Arkes & Blumer, 1985). In project management, it leads to "escalation of commitment" (Staw, 1976). Rational decision-making requires ignoring sunk costs and focusing on marginal NPV (Brealey et al., 2020). This scenario tests whether the RL agent has learned this principle.

---

### 7.6 Excluded Scenarios and Future Work

We explicitly exclude the following scenarios from the current scope, with clear pathways for future research:

#### 7.6.1 Cascading Delays (Interdependencies)

**Description:**  
Project $A$ is a prerequisite for project $B$. Delays in $A$ block progress on $B$, creating a cascading effect.

**Why Excluded:**  
Modeling interdependencies requires a **network representation** of the portfolio (e.g., directed acyclic graph of dependencies). This introduces:
- **State space explosion**: Each project's state depends on the states of its predecessors.
- **Coordination complexity**: Optimal policy must balance local project performance with global network effects.
- **Temporal constraints**: Precedence relationships impose hard constraints on action feasibility.

**Future Work:**  
This is a natural extension that connects to the **project scheduling literature** (Elmaghraby, 1977; Herroelen & Leus, 2005). Relevant methodologies include:
- **Critical Path Method (CPM)**: Identify bottleneck projects and prioritize them
- **Resource-Constrained Project Scheduling (RCPSP)**: Model dependencies and shared resources jointly
- **Graph Neural Networks (GNNs)**: Learn policies over project dependency graphs (Bengio et al., 2021)
- **Multi-agent RL**: Treat projects as interacting agents in a networked environment

A future implementation would augment the state representation with an adjacency matrix $A \in \{0,1\}^{N \times N}$ and include precedence constraints in the transition dynamics.

#### 7.6.2 Budget Shock Scenarios

**Description:**  
The portfolio experiences an exogenous budget cut mid-episode (e.g., 30-50% reduction due to macroeconomic conditions).

**Why Excluded:**  
The current environment assumes:
- The initial portfolio capital is assigned at $t=0$
- Liquidity flexibility is handled through a dynamic interest-bearing credit mechanism for temporary negative cash flow
- Budget evolution is endogenous to project performance and financing constraints, not external shocks

Introducing exogenous budget shocks would fundamentally alter the financial structure of the environment and require:
- Dynamic capital market modeling
- Credit repricing mechanisms
- Stochastic liquidity constraints
- Bankruptcy and refinancing logic

**Future Work:**  
This extension aligns with:
- **Stochastic cash flow optimization** (Birge & Louveaux, 2011)
- **Corporate liquidity management** (Almeida et al., 2004)
- **Robust optimization under financial uncertainty** (Ben-Tal et al., 2009)

Future implementations may formulate the portfolio as a multi-stage stochastic program with endogenous financing and dynamic credit limits.

#### 7.6.3 Market Price and Revenue Volatility

**Description:**  
Project revenues decline due to market condition changes (e.g., commodity price collapse, demand shock).

**Why Excluded:**  
The current environment assumes:
- Contract values and revenue schedules remain fixed as planned
- Portfolio uncertainty arises from execution dynamics rather than market repricing
- Revenue streams are deterministic conditional on project completion

This assumption isolates operational decision-making from macroeconomic uncertainty, allowing clearer attribution of agent performance to project management capabilities.

**Future Work:**  
Future extensions may incorporate:
- **Real options valuation** (Trigeorgis, 1996)
- **Commodity-linked project valuation**
- **Stochastic demand models**
- **Dynamic pricing and contract renegotiation**

Methodologically, this would transform the environment into a joint operational-financial optimization problem where project continuation depends on both execution feasibility and evolving market conditions.

---

## 8. Evaluation Framework

### 8.1 Benchmark Algorithms

We recommend evaluating the following benchmark set:

| Algorithm | Description | Purpose |
|---|---|---|
| Greedy NPV | Allocates budget to highest immediate NPV projects | Simple heuristic baseline |
| Greedy BCR | Uses Benefit-Cost Ratio prioritization | Financially interpretable heuristic |
| MPC | Finite-horizon optimization with receding horizon | Strong OR baseline |
| RL Agent | Learned policy (e.g., PPO, SAC, DQN) | Main method |
| Perfect Foresight | Hindsight optimal benchmark | Theoretical upper bound |

### 8.2 Evaluation Loop

```py
function evaluate_all_algorithms(env, algorithms, num_episodes):
    results = {}
    
    for episode_id in range(num_episodes):
        # Generate shared episode data
        episode_data = generate_episode(env, seed=episode_id)
        
        # Evaluate each algorithm
        for algo_name, algo in algorithms.items():
            env_eval = HybridEvaluationEnv(env, episode_data)
            
            if algo_name == 'perfect_foresight':
                total_reward = solve_perfect_foresight_simulation(
                    env_eval,
                    episode_data
                )
            else:
                total_reward = run_policy(
                    algo,
                    env_eval
                )
            
            results[algo_name].append(total_reward)
    
    # Compute optimality gaps
    pf_rewards = results['perfect_foresight']
    
    for algo_name in algorithms:
        if algo_name != 'perfect_foresight':
            gaps = []
            for i in range(num_episodes):
                gap = (
                    pf_rewards[i] - results[algo_name][i]
                ) / pf_rewards[i]
                gaps.append(gap)
            
            results[f'{algo_name}_gap'] = gaps
    
    return results
```

---

## 9. Statistical Reporting

### 9.1 Core Metrics

For each algorithm, report:

- Mean NPV
- Standard deviation
- Median NPV
- Worst-case percentile (e.g., 5th percentile)
- Optimality gap relative to PF
- Termination frequency
- Intervention frequency
- Portfolio completion rate

### 9.2 Confidence Intervals

For all reported metrics:
$$
\text{CI}_{95\%} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}
$$
where:
- $\bar{x}$ is the sample mean
- $s$ is the sample standard deviation
- $n$ is the number of episodes

### 9.3 Hypothesis Testing

Recommended tests:
- Welch's t-test for mean NPV comparison
- Mann-Whitney U test for non-normal distributions
- Kolmogorov-Smirnov test for distributional comparison
- Bootstrap confidence intervals for robustness

### 9.4 Stress Scenario Metrics

Additional metrics for stress tests:
- Time-to-termination
- False continuation rate
- Budget reallocation efficiency
- Recovery attempt efficiency
- Sunk cost susceptibility index

---

## 10. Visualization Targets

### 10.1 Performance Comparison

- NPV distribution across algorithms
- PF optimality gap histogram
- Episode-level reward trajectories

### 10.2 Stress Scenario Analysis

- SPI evolution under stress conditions
- Termination timelines
- Budget reallocation heatmaps
- Intervention frequency over time

### 10.3 Portfolio Dynamics

- Cash flow trajectories
- Project completion distributions
- Budget allocation patterns
- Portfolio composition evolution

### 10.4 Robustness Visualization

- Performance degradation under increasing stress severity
- Sensitivity curves for SPI collapse magnitude
- Intervention effectiveness heatmaps

---

## 11. Minimal Python Reference Implementation

### 11.1 Episode Generation

```python
def generate_episode(env, seed):
    rng = np.random.default_rng(seed)

    episode_data = {
        "base_spi": [],
        "recovery_params": []
    }

    for t in range(env.horizon):
        timestep_spi = []
        timestep_recovery = []

        for project in env.projects:
            base_spi = rng.normal(
                project.spi_mean,
                project.spi_std
            )

            timestep_spi.append(base_spi)

            timestep_recovery.append({
                "mu": project.recovery_mean,
                "sigma": project.recovery_std
            })

        episode_data["base_spi"].append(timestep_spi)
        episode_data["recovery_params"].append(
            timestep_recovery
        )

    return episode_data
```

### 11.2 Hybrid Evaluation Environment

```python
class HybridEvaluationEnv:

    def __init__(self, env, episode_data):
        self.env = copy.deepcopy(env)
        self.episode_data = episode_data
        self.t = 0

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, action):

        # Apply exogenous data
        for i, project in enumerate(self.env.projects):
            project.base_spi = (
                self.episode_data["base_spi"][self.t][i]
            )

        # Apply endogenous intervention effects
        for i, intervene in enumerate(action["intervene"]):

            if intervene:

                params = (
                    self.episode_data["recovery_params"][self.t][i]
                )

                recovery = np.random.normal(
                    params["mu"],
                    params["sigma"]
                )

                self.env.projects[i].spi += recovery

        next_state, reward, done, info = (
            self.env.step(action)
        )

        self.t += 1

        return next_state, reward, done, info
```

### 11.3 Stress Scenario Injection

```python
def inject_contractor_collapse(
    episode_data,
    project_id,
    collapse_time
):

    for t in range(collapse_time, len(episode_data["base_spi"])):

        episode_data["base_spi"][t][project_id] = (
            np.random.normal(0.4, 0.05)
        )

    return episode_data


def inject_intervention_resistance(
    episode_data,
    project_id
):

    for t in range(len(episode_data["recovery_params"])):

        episode_data["recovery_params"][t][project_id] = {
            "mu": -0.05,
            "sigma": 0.02
        }

    return episode_data


def inject_sunk_cost_trap(
    env,
    project_id
):

    project = env.projects[project_id]

    project.percent_complete = 0.8
    project.spi = 0.6
    project.remaining_cost_multiplier = 1.5

    return env
```

---

## 12. Implementation Considerations

### 12.1 Reproducibility

- Fix all random seeds
- Log episode generation seeds separately
- Store pre-sampled episode data
- Version-control environment configurations

### 12.2 Fair Comparison

To ensure fairness:
- Use identical episode sets across algorithms
- Prevent information leakage from PF to RL
- Separate training and evaluation environments
- Use deterministic evaluation policies

### 12.3 Computational Budget

PF evaluation can be expensive. Recommended strategies:
- Parallelize episode evaluation
- Use reduced horizon for PF approximation
- Cache repeated computations
- Use heuristic PF for large-scale portfolios

### 12.4 Training vs Evaluation Separation

Stress scenarios should be:
- Partially included during training for robustness
- Fully included during evaluation for validation
- Stratified by severity level

This prevents overfitting to specific failure modes.

---

## 13. Limitations

### 13.1 Perfect Foresight Unrealism

PF assumes complete future knowledge and therefore cannot be deployed operationally. Its purpose is purely evaluative.

### 13.2 Approximate PF Solutions

Simulation-based PF is not globally optimal and may underestimate the true upper bound.

### 13.3 No Project Interdependencies

Projects are modeled independently except through shared budget constraints.

### 13.4 Fixed Contract Valuation

Revenue streams remain deterministic and fixed over time.

### 13.5 Simplified Intervention Dynamics

Interventions affect SPI through simplified stochastic recovery models rather than detailed operational mechanisms.

---

## 14. Recommended Experimental Protocol

### Training
- Train RL agents on standard stochastic environments
- Introduce moderate stress scenarios progressively
- Use curriculum learning for robustness

### Evaluation
- Evaluate on unseen seeds
- Include both normal and adversarial scenarios
- Report PF optimality gaps

### Ablation Studies
- Without stress training
- Without termination actions
- Without intervention capability
- Different reward formulations

### Robustness Analysis
- Vary stress severity
- Vary intervention effectiveness
- Vary portfolio size
- Vary credit limits

---

## 15. Conclusion

This document presented a rigorous evaluation framework for RL-based portfolio optimization grounded in operations research principles and reinforcement learning methodology.

The proposed contributions include:

1. A Perfect Foresight benchmark for measuring optimality gaps
2. A hybrid pre-sampling framework that resolves endogenous stochasticity while preserving evaluation fairness
3. A stress testing module that validates agent robustness under severe project failure conditions
4. A clearly scoped research agenda for future extensions involving interdependencies, financial shocks, and market uncertainty

The introduced stress scenarios—contractor collapse, intervention resistance, and sunk cost traps—are particularly important because they evaluate whether the agent has learned economically rational behavior rather than merely exploiting statistical regularities. An effective portfolio agent must not only identify profitable opportunities but also recognize when projects become irrecoverable and reallocate resources accordingly.

From an OR/IE perspective, this framework bridges:
- Stochastic project portfolio optimization
- Sequential decision-making under uncertainty
- Simulation-based benchmarking
- Robust policy evaluation

The resulting methodology provides a reproducible, theoretically grounded, and practically meaningful benchmark suite for future research in RL-driven portfolio management systems.

---

# References

Almeida, H., Campello, M., & Weisbach, M. S. (2004). The cash flow sensitivity of cash. Journal of Finance, 59(4), 1777–1804.

Argyris, C. (1990). Overcoming Organizational Defenses. Allyn and Bacon.

Arkes, H. R., & Blumer, C. (1985). The psychology of sunk cost. Organizational Behavior and Human Decision Processes, 35(1), 124–140.

Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). Robust Optimization. Princeton University Press.

Bengio, Y., Lodi, A., & Prouvost, A. (2021). Machine learning for combinatorial optimization: A methodological tour d’horizon. European Journal of Operational Research, 290(2), 405–421.

Bertsekas, D. P. (2019). Reinforcement Learning and Optimal Control. Athena Scientific.

Birge, J. R., & Louveaux, F. (2011). Introduction to Stochastic Programming. Springer.

Borodin, A., & El-Yaniv, R. (1998). Online Computation and Competitive Analysis. Cambridge University Press.

Brealey, R. A., Myers, S. C., & Allen, F. (2020). Principles of Corporate Finance. McGraw-Hill.

Caramanis, C., & Iancu, D. A. (2021). Dynamic optimization under uncertainty. Foundations and Trends in Optimization.

Chick, S. E., et al. (2015). Simulation optimization under endogenous uncertainty. Proceedings of the Winter Simulation Conference.

Colas, C., et al. (2018). How many random seeds? Statistical power analysis in deep reinforcement learning experiments.

Elmaghraby, S. E. (1977). Activity Networks: Project Planning and Control by Network Models. Wiley.

Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). Modelling Extremal Events. Springer.

Flyvbjerg, B., Bruzelius, N., & Rothengatter, W. (2003). Megaprojects and Risk. Cambridge University Press.

Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.

Goodfellow, I., et al. (2014). Explaining and harnessing adversarial examples.

Gutjahr, W. J., & Reiter, P. (2010). Bi-objective project portfolio selection and staff assignment under uncertainty. European Journal of Operational Research, 207(3), 1711–1721.

Herroelen, W., & Leus, R. (2005). Project scheduling under uncertainty. European Journal of Operational Research, 165(2), 289–306.

Law, A. M., & Kelton, W. D. (2000). Simulation Modeling and Analysis. McGraw-Hill.

Merrow, E. W. (2011). Industrial Megaprojects. Wiley.

Powell, W. B. (2011). Approximate Dynamic Programming. Wiley.

Staw, B. M. (1976). Knee-deep in the big muddy: A study of escalating commitment. Organizational Behavior and Human Performance, 16(1), 27–44.

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.

Trigeorgis, L. (1996). Real Options. MIT Press.
