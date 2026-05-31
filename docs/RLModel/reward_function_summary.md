# Reward Function Analysis - Executive Summary

## Problem Identified

Your current reward function:
```
r_t = [Cash_in(t) - Cost(t)] - λ * WC_portfolio(t)
```

**Does NOT directly incentivize project completion**. It rewards cash flow timing, not project completion.

---

## Critical Issues

### 1. Cash Flow ≠ Completion
- Agent gets rewarded for receiving payments (cash inflow)
- Agent gets penalized for spending (costs)
- **No direct reward for finishing projects**

**Bad behavior this encourages:**
- Start many projects to collect advance payments
- Avoid spending on nearly-complete projects
- Abandon low-margin projects even if 90% done

### 2. No Terminal Value
- Episodes end after 12 months
- Projects at 50% completion get zero reward
- Agent won't start long-duration projects (24-36 months)

### 3. Sparse Rewards
- Cash only arrives at milestones (every 3-4 months)
- Long periods with no positive feedback
- Makes learning very slow

---

## Recommended Solution

### Hierarchical Multi-Objective Reward Function

```python
r_t = r_completion + α * r_efficiency + r_constraints

where:

# PRIMARY: Project completion (90% of reward)
r_completion = Σ_i [
    β * completed_i(t) * BAC_i * (1 + margin_i) +  # Big bonus when project completes
    ΔEV_i(t) * (1 + margin_i)                       # Small reward for progress
]

# SECONDARY: Efficiency (10% of reward)
r_efficiency = -λ_wc * WC_portfolio(t) - λ_delay * Σ_i max(0, 1.0 - SPI_i(t))

# CONSTRAINTS: Budget violations (large penalty)
r_constraints = -M * max(0, allocation_t - budget_t)^2
```

### Hyperparameters

```python
β = 2.0          # Completion bonus = 2x project value
α = 0.1          # Efficiency weight = 10% of total
λ_wc = 0.0001    # Working capital cost (10% annual)
λ_delay = 0.01   # Schedule delay penalty
M = 1000.0       # Budget violation penalty (very large)
```

---

## Why This Works

### 1. Direct Completion Incentive
- **Completion bonus**: Agent gets 2x project value when project finishes
- Much larger than any intermediate cash flow reward
- Strongly incentivizes finishing projects

### 2. Dense Learning Signal
- **Progress reward**: Agent gets small reward for every % of progress
- Weighted by profit margin (prioritize profitable projects)
- Helps agent learn which actions lead to completion

### 3. Clear Priority Hierarchy
- Completion = 90% of reward (α = 0.1)
- Efficiency = 10% of reward
- Agent learns to complete first, optimize second

### 4. Terminal Value
```python
def compute_terminal_reward(state):
    terminal_value = 0
    for i in all_projects:
        if completed[i]:
            terminal_value += BAC_i * (1 + margin_i)  # Full value
        else:
            terminal_value += (EV_i / BAC_i) * BAC_i * margin_i  # Partial value
    return terminal_value
```
- Incomplete projects still get credit for partial completion
- Encourages starting long-duration projects

---

## Implementation Roadmap

### Phase 1: Start Simple (Baseline)
```python
r_t = Σ_i [ΔEV_i(t) * (1 + margin_i)] - λ_wc * WC_portfolio(t)
```
- Test if progress-based reward learns completion
- Train 100K episodes
- **Target**: Completion rate >70%

### Phase 2: Add Completion Bonus
```python
r_t = Σ_i [ΔEV_i(t) * (1 + margin_i)] + 
      β * Σ_i [completed_i(t) * BAC_i * margin_i] - 
      λ_wc * WC_portfolio(t)
```
- Add large bonus for completing projects
- Train 100K episodes
- **Target**: Completion rate >85%

### Phase 3: Add Efficiency Terms
```python
r_t = r_completion + α * r_efficiency
```
- Add working capital and schedule delay penalties
- Train 100K episodes
- **Target**: Maintain >85% completion, improve WC efficiency

### Phase 4: Add Terminal Value
```python
terminal_reward = Σ_i [completion_ratio_i * value_i]
```
- Give credit for incomplete projects at episode end
- Encourages long-duration projects

---

## Expected Outcomes

| Metric | Target | Current (estimated) |
|--------|--------|---------------------|
| Completion rate | >85% | ~50-60% |
| Portfolio NPV | >90% of perfect foresight | ~70-75% |
| Peak working capital | <35% of portfolio value | ~40-45% |
| Training time | <100K episodes | Unknown |

---

## Key Insights from Literature

1. **Reward Shaping (Ng et al., 1999)**
   - Progress rewards = potential-based shaping
   - Theoretically guaranteed to preserve optimal policy

2. **Multi-Objective RL (Roijers et al., 2013)**
   - Use weighted sum with completion >> efficiency
   - Lexicographic ordering: optimize completion first

3. **Sparse Rewards (Andrychowicz et al., 2017)**
   - Dense intermediate rewards critical for learning
   - Hindsight Experience Replay can help

4. **Portfolio Optimization (Jiang et al., 2017)**
   - Reward portfolio value change, not individual assets
   - Terminal value for open positions essential

---

## Validation Metrics

Track these to validate reward design:

**Primary (Completion)**:
1. Completion rate (% projects finished)
2. Average completion time
3. Abandonment rate (% started but not finished)

**Secondary (Efficiency)**:
4. Portfolio NPV
5. Working capital efficiency
6. Budget utilization
7. Schedule performance (mean SPI)

**Learning**:
8. Sample efficiency (episodes to convergence)
9. Training stability (std dev of returns)
10. Convergence (reward plateau)

---

## Common Pitfalls to Avoid

### Pitfall 1: Agent "games" the reward
**Symptom**: Starts many projects, completes few
**Solution**: Make completion bonus >> sum of progress rewards (β=3.0)

### Pitfall 2: Ignores long projects
**Symptom**: Only selects projects <12 months
**Solution**: Add terminal value for incomplete projects

### Pitfall 3: Reward scale issues
**Symptom**: Training unstable or no learning
**Solution**: Normalize rewards to [-1, 1] range

### Pitfall 4: Conflicting objectives
**Symptom**: Agent oscillates between completion and efficiency
**Solution**: Use α << 1 (e.g., α=0.1) so completion dominates

---

## Next Steps

1. **Read full analysis**: `docs/RLModel/reward_function_analysis.md`
2. **Implement Phase 1**: Progress-based reward (baseline)
3. **Validate**: Train and measure completion rate
4. **Iterate**: Add completion bonus if needed (Phase 2)
5. **Optimize**: Add efficiency terms once completion is stable (Phase 3)

---

## Bottom Line

**Your current reward function optimizes for cash flow timing, not project completion.**

**Recommended fix**: Add completion bonus (β=2.0) and progress rewards weighted by margin. This directly incentivizes your primary objective (completion) while maintaining efficiency as a secondary concern.

**Expected improvement**: Completion rate from ~50-60% to >85%, Portfolio NPV from ~70-75% to >90% of perfect foresight.
