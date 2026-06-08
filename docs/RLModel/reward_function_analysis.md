# Reward Function Analysis for Portfolio Budgeting RL Agent

## Your Objectives (Priority Order)

1. **Primary**: Project completion (learn to budget for completion)
2. **Secondary**: Efficiency in allocations and working with money

## Current Reward Function (from docs)

```
r_t = r_t^base + r_t^WC_penalty

where:
- r_t^base = Σ Cash_in(t) - Σ Cost(t)  [Net cash flow]
- r_t^WC_penalty = -λ * WC_portfolio(t)  [Working capital penalty]
```

## Problem Analysis

### Issue 1: Cash Flow ≠ Project Completion

**Current reward focuses on cash flow**, not completion:

- Agent gets rewarded for cash inflows (payments received)
- Agent gets penalized for costs (spending)
- **Missing**: Direct reward for completing projects

**Consequence**: Agent might learn to:

- Prefer projects with high advance payments (quick cash)
- Avoid spending on projects near completion (cost without immediate cash)
- Abandon low-margin projects even if nearly complete

### Issue 2: No Terminal Value for Incomplete Projects

- Episode ends after H=12 periods
- Projects may still be in progress
- No reward for partial completion → agent has no incentive to start long-duration projects

### Issue 3: Sparse Reward Signal

- Cash inflows only occur at milestones (discrete events)
- Long periods with no positive reward
- Makes learning difficult (credit assignment problem)

## Literature Review: RL Reward Design Principles

### 1. Reward Shaping (Ng et al., 1999)

**Principle**: Add potential-based shaping to guide learning without changing optimal policy

```
F(s, s') = γ * Φ(s') - Φ(s)
```

where Φ(s) is a potential function (e.g., total project progress)

### 2. Multi-Objective RL (Roijers et al., 2013)

**Principle**: Decompose reward into multiple objectives with weights

```
r_t = w1*r_completion + w2*r_efficiency + w3*r_constraints
```

### 3. Sparse Reward Solutions (Andrychowicz et al., 2017)

**Approaches**:

- Hindsight Experience Replay (HER)
- Curriculum learning (start with easy portfolios)
- Dense intermediate rewards (progress-based)

### 4. Portfolio Optimization RL (Jiang et al., 2017)

**Financial portfolio management lessons**:

- Reward based on portfolio value change (not individual assets)
- Risk-adjusted returns (Sharpe ratio)
- Transaction costs as penalties

## Proposed Reward Function Designs

### Option 1: Completion-Focused with Progress Rewards

```python
r_t = r_completion + r_progress + r_efficiency + r_constraints

where:
# Primary objective: Project completion
r_completion = Σ_i [completed_i(t) * BAC_i * (1 + margin_i)]
  # Large reward when project completes
  # Scaled by project value and profitability

# Dense intermediate signal: Progress rewards
r_progress = Σ_i [ΔEV_i(t) * margin_i]
  # Reward for earned value progress
  # Weighted by profit margin (prioritize profitable projects)

# Secondary objective: Efficiency
r_efficiency = -λ_wc * WC_portfolio(t) - λ_delay * Σ_i delay_penalty_i(t)
  # Penalize high working capital
  # Penalize schedule delays (SPI < 1.0)

# Constraints: Budget feasibility
r_constraints = -μ * max(0, allocation_t - budget_t)^2
  # Quadratic penalty for budget violations
```

**Pros**:

- Direct reward for completion (primary objective)
- Dense progress signal (helps learning)
- Efficiency captured in secondary terms
- Clear priority hierarchy

**Cons**:

- Multiple hyperparameters to tune (λ_wc, λ_delay, μ)
- May need careful weight balancing

### Option 2: NPV-Based with Completion Bonus

```python
r_t = r_npv + r_completion_bonus + r_wc_penalty

where:
# Base reward: NPV of cash flows
r_npv = (Cash_in(t) - Cost(t)) / (1 + discount_rate)^t

# Completion bonus (large terminal reward)
r_completion_bonus = Σ_i [completed_i(t) * BAC_i * margin_i * β]
  # β > 1 (e.g., β=2) to strongly incentivize completion

# Working capital penalty
r_wc_penalty = -λ * WC_portfolio(t)
```

**Pros**:

- Economically grounded (NPV maximization)
- Completion bonus ensures projects finish
- Simpler (fewer hyperparameters)

**Cons**:

- Sparse completion signal (only at project end)
- May struggle with long-duration projects

### Option 3: Milestone-Based Reward Shaping

```python
r_t = r_milestone + r_progress + r_efficiency

where:
# Milestone rewards (shaped by potential function)
r_milestone = Σ_i [milestone_achieved_i(t) * payment_i(t) * (1 + margin_i)]
  # Reward when milestones achieved (not just when paid)
  # Aligns with project progress, not just cash timing

# Progress shaping (potential-based)
Φ(s) = Σ_i [EV_i(t) / BAC_i] * BAC_i * margin_i
r_progress = γ * Φ(s_t+1) - Φ(s_t)
  # Potential function = total portfolio progress weighted by value

# Efficiency penalty
r_efficiency = -λ_wc * WC_portfolio(t) - λ_spi * Σ_i max(0, 0.85 - SPI_i(t))
  # Penalize high WC and poor schedule performance
```

**Pros**:

- Milestone rewards provide intermediate signals
- Potential-based shaping is theoretically sound
- Encourages steady progress

**Cons**:

- More complex implementation
- Requires careful milestone definition

### Option 4: Hierarchical Objectives (Recommended)

```python
# Stage 1: Completion-focused reward
r_completion = Σ_i [
    completed_i(t) * value_i +           # Terminal reward for completion
    ΔEV_i(t) * margin_i +                # Progress reward
    milestone_i(t) * bonus_i             # Milestone bonuses
]

# Stage 2: Efficiency modifiers (only after completion incentive)
r_efficiency = -λ_wc * WC_portfolio(t) - λ_delay * Σ_i delay_cost_i(t)

# Stage 3: Hard constraints (large penalties)
r_constraints = -M * [
    max(0, allocation_t - budget_t) +    # Budget overrun
    Σ_i bankruptcy_i(t)                  # Project failures
]

# Total reward
r_t = r_completion + α * r_efficiency + r_constraints

where:
- value_i = BAC_i * (1 + margin_i)  # Project value at completion
- α ∈ [0.1, 0.3]  # Weight for efficiency (secondary objective)
- M >> 1  # Large penalty for constraint violations
```

**Pros**:

- Clear priority: completion first, efficiency second
- Dense signal from progress and milestones
- Hard constraints enforced via large penalties
- Tunable trade-off via α parameter

**Cons**:

- Requires defining milestone bonuses
- Need to calibrate α for desired behavior

## Recommended Approach

### Phase 1: Start Simple (Baseline)

```python
r_t = Σ_i [ΔEV_i(t) * (1 + margin_i)] - λ * WC_portfolio(t)
```

- Reward progress weighted by profitability
- Penalize working capital
- **Test if this learns completion behavior**

### Phase 2: Add Completion Bonus (If needed)

```python
r_t = Σ_i [ΔEV_i(t) * (1 + margin_i)] + 
      β * Σ_i [completed_i(t) * BAC_i * margin_i] - 
      λ * WC_portfolio(t)
```

- Add large completion bonus (β=2 or 3)
- **Test if completion rate improves**

### Phase 3: Add Efficiency Terms (Fine-tuning)

```python
r_t = Σ_i [ΔEV_i(t) * (1 + margin_i)] + 
      β * Σ_i [completed_i(t) * BAC_i * margin_i] - 
      λ_wc * WC_portfolio(t) - 
      λ_delay * Σ_i max(0, 1.0 - SPI_i(t))
```

- Add schedule delay penalty
- **Test if efficiency improves without hurting completion**

## Key Design Decisions

### 1. Should we reward cash flow or project progress?

**Recommendation**: **Project progress (EV)**, not cash flow

- Cash flow timing is stochastic (payment delays)
- EV reflects actual work completed
- Aligns with your primary objective (completion)

### 2. How to handle incomplete projects at episode end?

**Options**:
A. **Terminal value**: Estimate remaining value of incomplete projects
B. **Continuation value**: Use value function V(s_terminal) as terminal reward
C. **Ignore**: Only reward completed projects (may bias against long projects)

**Recommendation**: **Option A (Terminal value)**

```python
terminal_reward = Σ_i [
    completed_i * BAC_i * (1 + margin_i) +  # Full value if complete
    (1 - completed_i) * EV_i * margin_i      # Partial value if incomplete
]
```

### 3. How to balance completion vs efficiency?

**Recommendation**: **Weighted sum with tunable α**

```python
r_t = r_completion + α * r_efficiency
```

- Start with α=0.1 (90% completion, 10% efficiency)
- Increase α gradually if completion rate is satisfactory
- Use sensitivity analysis to find optimal α

### 4. How to handle budget constraints?

**Options**:
A. **Soft penalty**: -μ * max(0, violation)^2
B. **Hard constraint**: Reject invalid actions (action masking)
C. **Hybrid**: Mask + small penalty for near-violations

**Recommendation**: **Option C (Hybrid)**

- Use action masking to prevent violations
- Add small penalty for allocations near budget limit
- Encourages conservative budgeting

## Implementation Roadmap

### Step 1: Implement baseline reward

```python
def compute_reward(state, action, next_state):
    # Progress reward
    delta_ev = sum(next_state.ev[i] - state.ev[i] for i in active_projects)
    progress_reward = sum(delta_ev[i] * (1 + state.margin[i]) for i in active_projects)
    
    # Working capital penalty
    wc_penalty = -lambda_wc * next_state.wc_portfolio
    
    return progress_reward + wc_penalty
```

### Step 2: Add completion tracking

```python
def compute_reward(state, action, next_state):
    # Progress reward
    progress_reward = sum(
        (next_state.ev[i] - state.ev[i]) * (1 + state.margin[i]) 
        for i in active_projects
    )
    
    # Completion bonus
    completion_bonus = sum(
        beta * state.bac[i] * state.margin[i]
        for i in newly_completed_projects(state, next_state)
    )
    
    # Working capital penalty
    wc_penalty = -lambda_wc * next_state.wc_portfolio
    
    return progress_reward + completion_bonus + wc_penalty
```

### Step 3: Add terminal value

```python
def compute_terminal_reward(state):
    terminal_value = 0
    for i in all_projects:
        if state.completed[i]:
            # Full value for completed projects
            terminal_value += state.bac[i] * (1 + state.margin[i])
        else:
            # Partial value for incomplete projects
            completion_ratio = state.ev[i] / state.bac[i]
            terminal_value += completion_ratio * state.bac[i] * state.margin[i]
    
    return terminal_value
```

## Hyperparameter Recommendations

Based on typical RL portfolio management:

```python
# Completion incentive
beta = 2.0  # Completion bonus multiplier (2x project value)

# Efficiency penalties
lambda_wc = 0.0001  # Working capital cost (10% annual)
lambda_delay = 0.01  # Schedule delay penalty

# Constraint penalties
mu = 1000.0  # Budget violation penalty (large)

# Multi-objective weight
alpha = 0.1  # 90% completion, 10% efficiency
```

## Validation Metrics

Track these metrics to validate reward design:

1. **Completion rate**: % of projects completed by episode end
2. **Average completion time**: Mean time to complete projects
3. **Portfolio NPV**: Total value realized
4. **Working capital efficiency**: Peak WC / Portfolio value
5. **Budget utilization**: % of available budget used
6. **Schedule performance**: Mean SPI across portfolio

**Target**: 

- Completion rate > 85%
- Portfolio NPV > 90% of perfect foresight
- Peak WC < 35% of portfolio value

