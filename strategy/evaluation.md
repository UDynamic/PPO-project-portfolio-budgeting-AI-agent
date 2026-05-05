# Perfect Foresight Baseline Evaluation for RL Portfolio Optimization

## 1. Introduction

### 1.1 What is Perfect Foresight?

**Perfect Foresight** is a theoretical upper bound baseline that solves the entire portfolio optimization problem at time $t=0$ with **complete knowledge of all future stochastic events**. It represents the best possible performance achievable if an oracle could predict:

- All contractor base performance realizations (SPI)
- All recovery rates during interventions
- All project delays and cost overruns
- All budget availability shocks

In the context of **RL-based portfolio optimization**, Perfect Foresight serves as:

1. **Theoretical Upper Bound**: No algorithm (RL or OR) can outperform it
2. **Optimality Gap Metric**: Measures how close the RL agent gets to optimal performance
3. **Sanity Check**: If RL outperforms Perfect Foresight, there's a bug in the evaluation

### 1.2 Why Perfect Foresight Matters

In traditional OR benchmarking, you compare against:
- **Heuristics** (Greedy NPV, BCR) → Easy to beat
- **MPC** (Model Predictive Control) → Realistic competitor
- **Perfect Foresight** → Aspirational target

The **optimality gap** is defined as:

$$
\text{Optimality Gap} = \frac{\text{NPV}_{\text{Perfect Foresight}} - \text{NPV}_{\text{RL}}}{\text{NPV}_{\text{Perfect Foresight}} - \text{NPV}_{\text{Greedy}}} \times 100\%
$$

- **Gap = 0%**: RL is optimal
- **Gap = 50%**: RL captures half the value between Greedy and optimal
- **Gap = 100%**: RL performs no better than Greedy

---

## 2. The Challenge: Endogenous Stochasticity

### 2.1 The Problem

In a typical RL environment, you can pre-sample all stochastic events and create a **deterministic episode** for fair comparison:

```python
# Naive approach (DOES NOT WORK for this problem)
episode = {
    'spi': {project: [0.95, 0.88, 0.92, ...] for project in projects},
    'recovery_rates': {project: [0.65, 0.58, ...] for project in projects}
}
```
**Why this fails:**

In your EPC portfolio problem, **management interventions are action-dependent**:

```
Agent allocates budget → Performance gap computed → Intervention triggered (if gap > 10%)
                                                    ↓
                                            Recovery applied → Modified SPI
```

**The causal chain:**
- Agent's budget allocation at $t=2$ determines performance gap at $t=3$
- Gap determines whether intervention triggers
- Intervention modifies future SPI values
- Future states depend on past actions

**You cannot pre-sample intervention outcomes** because they don't exist until the agent acts.

### 2.2 Example: Why Naive Pre-Sampling Breaks

```python
# Pre-sampled episode
episode = {
    'base_spi_A': [0.95, 0.88, 0.92, 0.85],
    'intervention_triggered_A': [False, False, True, False]  # ❌ WRONG!
}

# Problem: Intervention trigger depends on agent's allocation!
# If RL allocates 100k to A at t=1, gap might be 8% → No intervention
# If RL allocates 50k to A at t=1, gap might be 15% → Intervention triggered
```
The intervention trigger is **endogenous** — it's determined by the agent's policy, not by nature.

---

## 3. Solution: Hybrid Pre-Sampling with Live Intervention Logic

### 3.1 Key Insight

Separate stochasticity into two layers:

| Component | Type | Handling |
|-----------|------|----------|
| **Base contractor SPI** | Exogenous | Pre-sample for entire episode |
| **Recovery rates** | Exogenous | Pre-sample (conditional on intervention) |
| **Performance gap** | Endogenous | Compute live based on agent's actions |
| **Intervention trigger** | Endogenous | Compute live (gap > 10% rule) |
| **Intervention effects** | Endogenous | Apply live using pre-sampled recovery rates |

### 3.2 The Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│ Episode Generation (Pre-Sampling)                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Sample base_spi[project][t] ~ N(μ, σ) for all t         │
│ 2. Sample recovery_rate[project][intervention_id] ~ U(0.5,0.7) │
│ 3. Store as deterministic episode seed                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Evaluation (Live Computation)                               │
├─────────────────────────────────────────────────────────────┤
│ At each timestep t:                                         │
│   1. Agent selects action (budget allocation)               │
│   2. Compute performance gap (depends on allocation)        │
│   3. Check intervention trigger (gap > 10%?)                │
│   4. If triggered: Apply pre-sampled recovery_rate          │
│   5. Compute realized SPI and next state                    │
└─────────────────────────────────────────────────────────────┘
```
### 3.3 Fairness Guarantee

**All algorithms (RL, MPC, Perfect Foresight) face:**
- ✅ Same base contractor performance (pre-sampled SPI)
- ✅ Same recovery rates (pre-sampled, applied when intervention triggers)
- ✅ Same intervention mechanics (deterministic >10% rule)

**What differs:**
- ❌ Their budget allocation policies
- ❌ Which interventions they trigger (and when)
- ❌ The resulting portfolio NPV

**This is exactly what we want to measure** — policy quality, not luck.

---

## 4. Step-by-Step Implementation Guide

### Step 1: Episode Generation

Create a deterministic episode by pre-sampling all exogenous stochasticity:

```python
def generate_episode(seed, projects, horizon):
    """
    Pre-sample all exogenous stochastic events for a single episode.
    
    Returns:
        episode: dict with pre-sampled base_spi and recovery_rates
    """
    np.random.seed(seed)
    
    episode = {
        'seed': seed,
        'base_spi': {},
        'recovery_rates': {}
    }
    
    for project in projects:
        # Pre-sample base contractor performance
        μ = project.expected_spi
        σ = project.spi_std
        episode['base_spi'][project.id] = [
            np.clip(np.random.normal(μ, σ), 0.5, 1.5)
            for t in range(horizon)
        ]
        
        # Pre-sample recovery rates (for potential interventions)
        # Generate enough for worst case (one intervention per month)
        episode['recovery_rates'][project.id] = [
            np.random.uniform(0.5, 0.7)
            for _ in range(horizon)
        ]
    
    return episode
```
### Step 2: Hybrid Environment Wrapper

Wrap your environment to use pre-sampled values during evaluation:

```python
class HybridEvaluationEnv:
    def __init__(self, base_env, episode=None):
        """
        Wrapper for deterministic evaluation.
        
        Args:
            base_env: Original stochastic environment
            episode: Pre-sampled episode (None for training mode)
        """
        self.env = base_env
        self.episode = episode
        self.deterministic = (episode is not None)
        
        # Track intervention state
        self.intervention_active = {}
        self.intervention_counter = {}  # Counts interventions per project
        self.intervention_timer = {}
    
    def reset(self):
        state = self.env.reset()
        self.intervention_active = {p: False for p in self.env.projects}
        self.intervention_counter = {p: 0 for p in self.env.projects}
        self.intervention_timer = {p: 0 for p in self.env.projects}
        return state
    
    def step(self, action):
        # 1. Allocate budget (agent's action)
        self.env.allocate_budget(action)
        
        # 2. Compute realized SPI with intervention logic
        for project in self.env.active_projects:
            # Get base SPI (pre-sampled or live)
            if self.deterministic:
                base_spi = self.episode['base_spi'][project.id][self.env.t]
            else:
                base_spi = np.random.normal(project.expected_spi, project.spi_std)
            
            # Compute performance gap (endogenous)
            planned = project.get_planned_progress(self.env.t)
            actual = project.get_actual_progress(self.env.t)
            gap = (planned - actual) / planned if planned > 0 else 0
            
            # Check intervention trigger (endogenous)
            if gap > 0.10 and not self.intervention_active[project.id]:
                # Trigger intervention
                self.intervention_active[project.id] = True
                self.intervention_timer[project.id] = 0
                print(f"[t={self.env.t}] Intervention triggered for {project.name}")
            
            # Apply intervention effect
            if self.intervention_active[project.id]:
                timer = self.intervention_timer[project.id]
                intervention_idx = self.intervention_counter[project.id]
                
                if timer < 3:  # Active recovery (3 months)
                    if self.deterministic:
                        recovery = self.episode['recovery_rates'][project.id][intervention_idx]
                    else:
                        recovery = np.random.uniform(0.5, 0.7)
                    
                    realized_spi = base_spi * (1 + recovery)
                    self.intervention_timer[project.id] += 1
                else:  # Decay phase
                    months_since = timer - 3
                    decay = np.exp(-0.2 * months_since)
                    
                    if self.deterministic:
                        recovery = self.episode['recovery_rates'][project.id][intervention_idx] * decay
                    else:
                        recovery = np.random.uniform(0.5, 0.7) * decay
                    
                    realized_spi = base_spi * (1 + recovery)
                    self.intervention_timer[project.id] += 1
                    
                    if recovery < 0.05:  # End intervention
                        self.intervention_active[project.id] = False
                        self.intervention_counter[project.id] += 1
            else:
                realized_spi = base_spi
            
            # Update project with realized SPI
            earned_value = action[project.id] * realized_spi
            project.update_progress(earned_value)
        
        # 3. Compute next state and reward
        next_state = self.env.get_state()
        reward = self.env.compute_reward()
        done = self.env.is_done()
        info = self.env.get_info()
        
        self.env.t += 1
        return next_state, reward, done, info
```
### Step 3: Perfect Foresight MILP Formulation

Formulate the optimization problem with full knowledge of pre-sampled stochasticity:

```python
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

def solve_perfect_foresight(env, episode):
    """
    Solve portfolio optimization with perfect knowledge of future.
    
    Args:
        env: Environment instance
        episode: Pre-sampled episode with base_spi and recovery_rates
    
    Returns:
        optimal_actions: dict mapping (project, t) to budget allocation
        optimal_npv: Maximum achievable NPV
    """
    projects = env.projects
    horizon = env.horizon
    budget_per_period = env.budget_per_period
    
    # Create optimization model
    model = LpProblem("PerfectForesight_Portfolio", LpMaximize)
    
    # Decision variables
    x = {}  # x[p,t] = budget allocated to project p at time t
    y = {}  # y[p,t] = binary, whether project p is active at time t
    intervention = {}  # intervention[p,t] = binary, intervention triggered
    
    for p in projects:
        for t in range(horizon):
            x[p.id, t] = LpVariable(f"x_{p.id}_{t}", lowBound=0, upBound=p.max_monthly_budget)
            y[p.id, t] = LpVariable(f"y_{p.id}_{t}", cat=LpBinary)
            intervention[p.id, t] = LpVariable(f"int_{p.id}_{t}", cat=LpBinary)
    
    # Objective: Maximize total NPV
    # NPV = Σ (Revenue_p × Completion_p) - Σ (Cost_p,t)
    discount_factor = 1 / (1 + env.discount_rate)
    
    objective = lpSum([
        p.revenue * y[p.id, horizon-1] * (discount_factor ** horizon)  # Revenue at completion
        - lpSum([x[p.id, t] * (discount_factor ** t) for t in range(horizon)])  # Costs
        for p in projects
    ])
    model += objective
    
    # Constraints
    
    # 1. Budget constraint per period
    for t in range(horizon):
        model += lpSum([x[p.id, t] for p in projects]) <= budget_per_period
    
    # 2. Project completion constraint
    # Σ (x[p,t] × SPI[p,t]) >= Total_Work[p]
    for p in projects:
        # Use pre-sampled SPI with intervention logic
        total_earned_value = lpSum([
            x[p.id, t] * episode['base_spi'][p.id][t] * (1 + 0.6 * intervention[p.id, t])
            for t in range(horizon)
        ])
        model += total_earned_value >= p.total_work * y[p.id, horizon-1]
    
    # 3. Intervention trigger constraint (linearized)
    # If gap > 10%, intervention can be triggered
    for p in projects:
        for t in range(1, horizon):
            planned_progress = p.get_planned_progress(t)
            # Actual progress = Σ earned_value up to t
            actual_progress = lpSum([
                x[p.id, s] * episode['base_spi'][p.id][s] * (1 + 0.6 * intervention[p.id, s])
                for s in range(t)
            ])
            gap = (planned_progress - actual_progress) / planned_progress
            
            # If gap > 0.1, allow intervention[p,t] = 1
            # This is a simplification; full MILP would need big-M constraints
            model += intervention[p.id, t] <= 1  # Can trigger at most once per period
    
    # 4. Project activation constraints
    for p in projects:
        for t in range(1, horizon):
            model += y[p.id, t] >= y[p.id, t-1]  # Once active, stays active
    
    # Solve
    model.solve()
    
    # Extract solution
    optimal_actions = {}
    for p in projects:
        for t in range(horizon):
            optimal_actions[p.id, t] = x[p.id, t].varValue
    
    optimal_npv = model.objective.value()
    
    return optimal_actions, optimal_npv
```
**Note:** The above MILP is a **simplified formulation**. A full implementation would require:
- Big-M constraints for intervention trigger logic
- Intervention duration tracking (3-month active + decay)
- Working capital constraints
- Project dependency constraints

For a **minimal viable baseline**, you can use a **simulation-based approach** instead:

```python
def solve_perfect_foresight_simulation(env, episode, num_iterations=1000):
    """
    Approximate Perfect Foresight using Monte Carlo Tree Search or Genetic Algorithm.
    
    This is computationally cheaper than full MILP for large problems.
    """
    best_npv = -np.inf
    best_actions = None
    
    for _ in range(num_iterations):
        # Generate random policy
        actions = generate_random_policy(env, episode)
        
        # Simulate with pre-sampled episode
        eval_env = HybridEvaluationEnv(env, episode)
        state = eval_env.reset()
        total_npv = 0
        
        for t in range(env.horizon):
            action = actions[t]
            state, reward, done, info = eval_env.step(action)
            total_npv += reward
            if done:
                break
        
        if total_npv > best_npv:
            best_npv = total_npv
            best_actions = actions
    
    return best_actions, best_npv
```
### Step 4: Evaluation Loop

Run all algorithms on the same set of pre-sampled episodes:

```python
def evaluate_all_algorithms(env, num_episodes=100):
    """
    Evaluate RL agent, MPC, Greedy, and Perfect Foresight on same episodes.
    """
    results = {
        'RL': [],
        'MPC': [],
        'Greedy': [],
        'PerfectForesight': []
    }
    
    for episode_id in range(num_episodes):
        print(f"\n=== Episode {episode_id+1}/{num_episodes} ===")
        
        # Generate deterministic episode
        episode = generate_episode(
            seed=episode_id,
            projects=env.projects,
            horizon=env.horizon
        )
        
        # 1. Perfect Foresight (upper bound)
        _, npv_pf = solve_perfect_foresight_simulation(env, episode)
        results['PerfectForesight'].append(npv_pf)
        print(f"Perfect Foresight NPV: ${npv_pf:,.0f}")
        
        # 2. RL Agent
        eval_env_rl = HybridEvaluationEnv(env, episode)
        npv_rl = run_rl_agent(eval_env_rl)
        results['RL'].append(npv_rl)
        print(f"RL Agent NPV: ${npv_rl:,.0f}")
        
        # 3. MPC
        eval_env_mpc = HybridEvaluationEnv(env, episode)
        npv_mpc = run_mpc(eval_env_mpc)
        results['MPC'].append(npv_mpc)
        print(f"MPC NPV: ${npv_mpc:,.0f}")
        
        # 4. Greedy Heuristic
        eval_env_greedy = HybridEvaluationEnv(env, episode)
        npv_greedy = run_greedy(eval_env_greedy)
        results['Greedy'].append(npv_greedy)
        print(f"Greedy NPV: ${npv_greedy:,.0f}")
        
        # Compute optimality gap
        gap = (npv_pf - npv_rl) / (npv_pf - npv_greedy) * 100
        print(f"RL Optimality Gap: {gap:.1f}%")
    
    return results
```
---

## 5. Minimal Python Implementation

Below is a **complete, runnable script** for Perfect Foresight evaluation:

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================================
# 1. PROJECT AND ENVIRONMENT DEFINITIONS
# ============================================================================

@dataclass
class Project:
    id: str
    name: str
    total_work: float  # Total work in $M
    revenue: float  # Revenue upon completion in $M
    expected_spi: float  # Expected Schedule Performance Index
    spi_std: float  # SPI standard deviation
    max_monthly_budget: float  # Max budget per month in $M
    
    def __post_init__(self):
        self.actual_progress = 0.0
        self.planned_progress_rate = self.total_work / 24  # Assume 24-month baseline

class PortfolioEnv:
    def __init__(self, projects: List[Project], horizon: int, budget_per_period: float):
        self.projects = projects
        self.horizon = horizon
        self.budget_per_period = budget_per_period
        self.discount_rate = 0.08 / 12  # Monthly discount rate
        self.t = 0
        
    def reset(self):
        self.t = 0
        for p in self.projects:
            p.actual_progress = 0.0
        return self.get_state()
    
    def get_state(self):
        # Simple state: [progress_p1, progress_p2, ..., remaining_budget, time]
        state = [p.actual_progress / p.total_work for p in self.projects]
        state.append(self.budget_per_period)
        state.append(self.t / self.horizon)
        return np.array(state)
    
    def allocate_budget(self, action: Dict[str, float]):
        # Action is dict: {project_id: budget_allocated}
        total_allocated = sum(action.values())
        assert total_allocated <= self.budget_per_period, "Budget exceeded"
    
    def compute_reward(self):
        # Reward = NPV of completed projects - costs
        discount = 1 / (1 + self.discount_rate) ** self.t
        
        completed_revenue = sum([
            p.revenue * discount if p.actual_progress >= p.total_work else 0
            for p in self.projects
        ])
        
        return completed_revenue
    
    def is_done(self):
        return self.t >= self.horizon or all(p.actual_progress >= p.total_work for p in self.projects)
    
    def get_info(self):
        return {
            'completed_projects': [p.id for p in self.projects if p.actual_progress >= p.total_work],
            'total_spent': sum([p.actual_progress for p in self.projects])
        }

# ============================================================================
# 2. EPISODE GENERATION
# ============================================================================

def generate_episode(seed: int, projects: List[Project], horizon: int) -> Dict:
    """Pre-sample all exogenous stochastic events."""
    np.random.seed(seed)
    
    episode = {
        'seed': seed,
        'base_spi': {},
        'recovery_rates': {}
    }
    
    for project in projects:
        # Pre-sample base contractor performance
        episode['base_spi'][project.id] = [
            np.clip(np.random.normal(project.expected_spi, project.spi_std), 0.5, 1.5)
            for t in range(horizon)
        ]
        
        # Pre-sample recovery rates (for potential interventions)
        episode['recovery_rates'][project.id] = [
            np.random.uniform(0.5, 0.7)
            for _ in range(horizon)
        ]
    
    return episode

# ============================================================================
# 3. HYBRID EVALUATION ENVIRONMENT
# ============================================================================

class HybridEvaluationEnv:
    def __init__(self, base_env: PortfolioEnv, episode: Dict = None):
        self.env = base_env
        self.episode = episode
        self.deterministic = (episode is not None)
        
        # Intervention tracking
        self.intervention_active = {}
        self.intervention_counter = {}
        self.intervention_timer = {}
        self.intervention_history = []
    
    def reset(self):
        state = self.env.reset()
        self.intervention_active = {p.id: False for p in self.env.projects}
        self.intervention_counter = {p.id: 0 for p in self.env.projects}
        self.intervention_timer = {p.id: 0 for p in self.env.projects}
        self.intervention_history = []
        return state
    
    def step(self, action: Dict[str, float]):
        # 1. Allocate budget
        self.env.allocate_budget(action)
        
        # 2. Compute realized SPI with intervention logic
        for project in self.env.projects:
            # Get base SPI
            if self.deterministic:
                base_spi = self.episode['base_spi'][project.id][self.env.t]
            else:
                base_spi = np.random.normal(project.expected_spi, project.spi_std)
                base_spi = np.clip(base_spi, 0.5, 1.5)
            
            # Compute performance gap
            planned = project.planned_progress_rate * (self.env.t + 1)
            actual = project.actual_progress
            gap = (planned - actual) / planned if planned > 0 else 0
            
            # Check intervention trigger
            if gap > 0.10 and not self.intervention_active[project.id]:
                self.intervention_active[project.id] = True
                self.intervention_timer[project.id] = 0
                self.intervention_history.append({
                    'project': project.id,
                    'time': self.env.t,
                    'gap': gap
                })
            
            # Apply intervention effect
            if self.intervention_active[project.id]:
                timer = self.intervention_timer[project.id]
                intervention_idx = self.intervention_counter[project.id]
                
```
# Perfect Foresight Baseline Evaluation for RL Portfolio Optimization

## 1. Introduction

### 1.1 What is Perfect Foresight?

Perfect Foresight is a **theoretical upper-bound benchmark** used to evaluate decision policies in stochastic dynamic optimization problems.  
In portfolio allocation problems solved with Reinforcement Learning (RL), it represents an **oracle planner** that has **complete knowledge of all future realizations of uncertainty** before making any decisions.

In this project, Perfect Foresight assumes advance knowledge of:

- All future contractor base performance values (SPI)
- Recovery rates if an intervention occurs
- All stochastic variations affecting project execution

The planner therefore solves the **entire horizon optimization problem at time $t=0$**, producing the best possible allocation policy under the realized scenario.

Because no real algorithm can know the future, the Perfect Foresight solution is **not a practical policy**. Its role is strictly evaluative.

### 1.2 Role in RL Benchmarking

Perfect Foresight serves three critical purposes:

- **Upper Bound Benchmark**  
  No realistic algorithm (RL or OR-based) should outperform it.

- **Optimality Gap Measurement**  
  It quantifies how close the RL policy approaches the theoretical optimum.

- **Sanity Check for Evaluation**  
  If an algorithm exceeds Perfect Foresight performance, the evaluation pipeline likely contains an error.

A common evaluation metric is the **normalized optimality gap**:

$$
Gap_{RL} =
\frac{NPV_{PF} - NPV_{RL}}
{NPV_{PF} - NPV_{Greedy}}
\times 100
$$

Where:

- $NPV_{PF}$ = Perfect Foresight value  
- $NPV_{RL}$ = RL agent value  
- $NPV_{Greedy}$ = simple heuristic baseline

Interpretation:

- 0% → RL reaches optimal performance  
- 50% → RL captures half the available improvement over heuristics  
- 100% → RL performs no better than greedy baseline

---

# 2. The Core Evaluation Challenge

## 2.1 Why Deterministic Episodes Are Needed

To fairly compare RL against optimization baselines (MPC, heuristics, Perfect Foresight), all algorithms must be evaluated under **identical environmental conditions**.

Without control of randomness:

- RL may experience an easier scenario
- MPC may experience worse contractor performance
- comparisons become statistically biased

The standard solution is to create **deterministic test episodes**.

This is done by **pre‑sampling all stochastic variables before evaluation**.

Example:

```
Episode seed = 42

base_spi_project_A = [0.94, 0.88, 0.91, ...]
base_spi_project_B = [0.89, 0.83, 0.87, ...]

Each algorithm then runs against the same sequence.

---

## 2.2 Why Naive Pre‑Sampling Fails in This Problem

The portfolio environment includes **management intervention logic**.

An intervention is triggered when:


Performance gap > 10%

But the **performance gap depends on the agent’s previous decisions**.

Causal chain:


Budget allocation
→ project progress
→ performance gap
→ intervention trigger
→ recovery effect
→ future SPI dynamics

Therefore intervention outcomes cannot be pre-sampled independently.

Example:


Time 2:
RL allocates large budget → gap = 7% → no intervention

MPC allocates small budget → gap = 18% → intervention triggered

Even under identical contractor performance, the intervention dynamics diverge because the **actions differ**.

This creates **endogenous stochasticity**.

---

# 3. Hybrid Solution: Controlled Stochastic Evaluation

## 3.1 Separation of Stochastic Sources

To resolve the issue we divide randomness into two categories.

### Exogenous Stochasticity

Independent of agent actions.

Examples:

- contractor base SPI
- recovery rate distribution

These **can be pre‑sampled**.

### Endogenous Dynamics

Dependent on agent decisions.

Examples:

- performance gap
- intervention trigger
- intervention timing
- progress trajectory

These **must be computed during simulation**.

---

## 3.2 Hybrid Evaluation Principle

Evaluation proceeds using two layers.

Layer 1 — Pre-sampled randomness:

- base contractor SPI
- recovery coefficients

Layer 2 — Runtime dynamics:

- budget allocation
- project progress
- intervention trigger
- recovery application

Diagram:


Pre‑sampled episode
     │
     ├─ base SPI paths
     └─ recovery coefficients

Evaluation simulation
     │
     ├─ agent selects allocation
     ├─ compute project progress
     ├─ check intervention trigger
     └─ apply recovery using pre-sampled parameters

All algorithms therefore experience **identical external uncertainty** while maintaining **policy-dependent outcomes**.

---

# 4. Minimal Evaluation Workflow

The evaluation pipeline consists of five steps.

---

## Step 1 — Generate Deterministic Episodes

Each episode pre‑samples contractor behavior.

Example structure:


episode = {
  base_spi : {project → [values]},
  recovery_rates : {project → [values]}
}

Each seed corresponds to a reproducible scenario.

---

## Step 2 — Wrap the Environment

A hybrid environment wrapper must:

- read pre‑sampled values
- override random sampling
- maintain intervention logic

Training uses the original stochastic environment.

Evaluation uses the deterministic wrapper.

---

## Step 3 — Compute Perfect Foresight Baseline

Two possible implementations:

### Exact MILP

Advantages

- mathematically optimal
- deterministic

Disadvantages

- difficult to model nonlinear intervention dynamics
- computationally expensive

### Simulation-based Search (Practical)

Approaches include:

- genetic algorithms
- Monte Carlo tree search
- random search with pruning

This provides a **strong approximate upper bound**.

---

## Step 4 — Evaluate Algorithms

For each episode:


run Perfect Foresight
run RL agent
run MPC baseline
run heuristic baseline

Record:

- portfolio NPV
- intervention count
- project completion times

---

## Step 5 — Aggregate Statistics

Compute across all episodes:

- mean NPV
- variance
- optimality gap
- intervention frequency

---

# 5. Minimal Python Implementation

Below is a simplified reference structure.


import numpy as np

def generate_episode(seed, projects, horizon):

    rng = np.random.default_rng(seed)

    episode = {
        "base_spi": {},
        "recovery_rates": {}
    }

    for p in projects:

        episode["base_spi"][p] = rng.normal(
            loc=1.0,
            scale=0.1,
            size=horizon
        )

        episode["recovery_rates"][p] = rng.uniform(
            0.5,
            0.7,
            size=horizon
        )

    return episode

Hybrid environment example:


class HybridEnv:

    def __init__(self, episode):

        self.episode = episode
        self.t = 0

        self.intervention_active = {}
        self.intervention_timer = {}

    def get_base_spi(self, project):

        return self.episode["base_spi"][project][self.t]

    def step(self, action):

        spi = self.get_base_spi(project)

        gap = compute_gap(project)

        if gap > 0.10:
            trigger_intervention(project)

        spi = apply_intervention_effect(project, spi)

        update_project_progress(project, spi)

        self.t += 1

Minimal Perfect Foresight search:


def perfect_foresight(env, episode, iterations=500):

    best_value = -1e9
    best_policy = None

    for i in range(iterations):

        policy = random_policy()

        value = simulate_policy(env, policy, episode)

        if value > best_value:
            best_value = value
            best_policy = policy

    return best_policy, best_value

---

# 6. Visualization and Reporting Targets

Evaluation should produce visual outputs supporting analysis.

Recommended plots:

### Portfolio Value Comparison


Bar Chart

RL
MPC
Greedy
Perfect Foresight

---

### Optimality Gap Distribution

Histogram showing:


Gap_RL across episodes

---

### Intervention Frequency

Compare how often each algorithm triggers interventions.


Average interventions per project

---

### Project Completion Profiles

Plot completion times across algorithms.

Example:


Completion Month vs Algorithm

---

### Budget Allocation Patterns

Visualize policy behavior.

Possible plot:


Time vs Budget Allocation per Project

This reveals structural policy differences.

---

# 7. Statistical Reporting

Results should include:

Mean performance


mean NPV
std deviation
confidence interval

Statistical tests:

- paired t-test
- Wilcoxon signed-rank test

These determine whether RL improvement over MPC is statistically significant.

---

# 8. Key Implementation Considerations

## Reproducibility

Every episode must be associated with a deterministic seed.


episode_seed = evaluation_id

This guarantees repeatability.

---

## Computational Budget

Perfect Foresight search can become expensive.

Recommended practice:


episodes = 100 – 1000
search iterations = 200 – 1000

Balance accuracy with runtime.

---

## Fair Comparison

Ensure that:

- identical episode seeds are used
- environment resets completely
- state initialization is identical

---

## Training vs Evaluation Separation

Training environment:


fully stochastic

Evaluation environment:


deterministic wrapper

Mixing the two invalidates results.

---

# 9. Limitations of Perfect Foresight

Despite its usefulness, Perfect Foresight has limitations.

### Unrealistic Information Assumption

Real planners do not know future contractor performance.

### Computational Complexity

Exact optimization becomes difficult for:

- long horizons
- large project portfolios
- nonlinear intervention effects

### Not a Deployable Policy

Perfect Foresight is only an **evaluation reference**, not an operational strategy.

---

# 10. Recommended Benchmark Set

A strong experimental evaluation should include:


1. Greedy NPV heuristic
2. Greedy BCR heuristic
3. Model Predictive Control (rolling MILP)
4. RL Agent
5. Perfect Foresight upper bound

This produces a **complete performance ladder**:


Greedy  <  MPC  <  RL  <  Perfect Foresight

The goal is demonstrating that RL closes a meaningful portion of the gap toward the theoretical optimum.

---

# 11. Final Remarks

Perfect Foresight benchmarking provides a **rigorous evaluation framework** for RL-based portfolio management.

By carefully separating:

- exogenous uncertainty
- action-dependent dynamics

it is possible to construct deterministic evaluation scenarios that ensure:

- fair algorithm comparison
- reproducibility
- interpretable optimality gaps

When implemented correctly, this methodology provides strong empirical evidence regarding the **strategic value of reinforcement learning for capital allocation under uncertainty**.
