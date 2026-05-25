# Trajectory Strategy: Rollout Simulation vs. Rolling Horizon Planning

## Overview

This document clarifies two distinct strategies for generating forward-looking trajectories from a trained RL agent, compares their complexity, and justifies the design choice made in this work.

---

## 1. The Agent's Core Behavior

The trained PPO agent is a **reactive sequential decision-maker**. At each timestep $t$, it observes the current state $s_t$ and outputs a single budget allocation action:

$$a_t = \pi_\theta(s_t)$$

It does **not** output a multi-period plan. The agent implicitly accounts for future consequences through its learned value function $V_\theta(s_t)$, which was shaped during training via the reward signal.

---

## 2. Two Strategies for Forward-Looking Analysis

### Strategy A: Rollout-Based Trajectory Simulation (This Work)

After the agent commits to action $a_t$, we can simulate forward from the current state to generate a **forecast trajectory**. This is done by repeatedly applying the learned policy through the environment's stochastic dynamics model $\hat{P}$:

$$s_{t+k+1} \sim \hat{P}(\cdot \mid s_{t+k}, \pi_\theta(s_{t+k})), \quad k = 0, 1, \ldots, T-1$$

We run $N$ independent rollouts and average the results to obtain a statistically stable expected trajectory.

**Key parameters:**

| Parameter | Role | Typical Value |
|---|---|---|
| $N$ (rollouts) | Number of parallel simulations — reduces stochastic noise | 50–100 |
| $T$ (horizon) | Number of steps simulated forward — controls forecast length | 12 months |

> ⚠️ **Common Misconception:** Setting $N = 12$ because there are 12 months is **conceptually wrong**. $N$ and $T$ are independent parameters. $N$ controls statistical stability; $T$ controls forecast depth.

**What this produces:**

$$\text{Forecast} = \mathbb{E}\left[\sum_{k=0}^{T} (s_{t+k}, a_{t+k})\right] \approx \frac{1}{N} \sum_{i=1}^{N} \text{trajectory}_i$$

The agent only **commits** to $a_t$. The rest of the trajectory is a **prediction**, not a plan.

---

### Strategy B: Rolling Horizon Planning (MPC-Style)

In rolling horizon planning, at each timestep $t$ the agent explicitly solves an optimization problem over a future window $[t, t+H]$, selects the first action $a_t^*$, executes it, then re-solves at $t+1$:

$$a_t^* = \arg\max_{a_t, \ldots, a_{t+H}} \sum_{k=0}^{H} \gamma^k r(s_{t+k}, a_{t+k})$$

This is the core idea behind **Model Predictive Control (MPC)**.

---

## 3. Complexity Comparison

### 3.1 Reward Function Complexity

| Aspect | Rollout Simulation | Rolling Horizon Planner |
|---|---|---|
| **Design** | Single reward $r_t$ used during training | Reward must be decomposable and differentiable over the horizon $H$ |
| **Shaping** | Progress rewards, terminal bonuses, WC penalties | Must remain consistent across all $H$ steps; shaping errors compound |
| **Multi-objective** | Weighted sum at each step is sufficient | Weights must be carefully tuned to avoid myopic or greedy behavior over the window |
| **Sparse rewards** | Handled via potential-based shaping or terminal value | Sparse rewards over $H$ steps cause severe credit assignment problems |
| **Risk terms** | CVaR, variance added as auxiliary terms | Risk over the horizon requires distributional modeling at each step |

**Verdict:** The rollout simulator uses the **same reward function** designed for training — no additional complexity. The rolling horizon planner requires the reward to be **well-behaved across the entire planning window**, which significantly increases design burden.

---

### 3.2 Training Complexity

| Aspect | Rollout Simulation | Rolling Horizon Planner |
|---|---|---|
| **Algorithm** | Standard PPO — well-understood, stable | Requires differentiable world model or nested optimization |
| **Model requirement** | Dynamics model $\hat{P}$ used only at inference | Dynamics model must be differentiable and accurate enough for $H$-step lookahead |
| **Computational cost** | $O(N \cdot T)$ rollouts at inference only | $O(H \cdot \text{optimizer steps})$ at **every** decision step |
| **Training stability** | PPO with clipped surrogate objective is stable | Gradient propagation through $H$ steps causes vanishing/exploding gradients |
| **Hyperparameter sensitivity** | Moderate (PPO clip $\epsilon$, entropy $\beta$) | High — horizon $H$, discount $\gamma$, model accuracy all interact |
| **Generalization** | Policy generalizes via learned $\pi_\theta$ | Re-optimization at each step; no generalization across states |
| **Data efficiency** | Learns from experience replay | Requires accurate model; model errors accumulate over $H$ steps |

**Verdict:** Rolling horizon planning introduces a **nested optimization loop** at every decision step, requires a differentiable and accurate dynamics model, and suffers from compounding model errors. Standard PPO with rollout simulation avoids all of these issues.

---

### 3.3 Inference Complexity

| Aspect | Rollout Simulation | Rolling Horizon Planner |
|---|---|---|
| **Decision latency** | Single forward pass $\pi_\theta(s_t)$ — $O(1)$ | Solve optimization over $H$ steps — $O(H \cdot K)$ where $K$ = optimizer iterations |
| **Forecast generation** | $N$ parallel rollouts, embarrassingly parallelizable | Forecast is a byproduct of the optimization — not independently controllable |
| **Interpretability** | Forecast shows expected trajectory under current policy | Plan shows intended sequence — but may be invalidated by stochastic transitions |
| **Adaptability** | Policy adapts instantly to new $s_t$ | Must re-solve from scratch at each $t$ |

---

## 4. Why Rollout Simulation Was Chosen

Given the complexity analysis above, rollout-based trajectory simulation is the appropriate choice for this work for the following reasons:

1. **No additional training overhead.** The same PPO-trained policy $\pi_\theta$ is used directly. No differentiable world model is required.

2. **Separation of concerns.** The agent's decision $a_t$ and the forecast trajectory are cleanly separated. The agent commits only to $a_t$; the forecast is advisory.

3. **Scalable scenario analysis.** By varying the initial state $s_0$ and running $N$ rollouts with horizon $T$, we can perform rich scenario analysis across different portfolio configurations.

4. **Statistical control.** The analyst controls forecast quality independently via $N$ (noise reduction) and $T$ (forecast depth), without retraining the agent.

5. **Practical deployment.** A single neural network forward pass makes real-time deployment feasible. Rolling horizon planning would require an optimization solver running at every decision point.

---

## 5. Correct Parameter Interpretation

To generate a **12-month forward forecast** with statistical stability:
```python
def forecast(env, policy, s_t, horizon=12, n_rollouts=50):
"""
horizon   : how many steps forward to simulate (= 12 months)
n_rollouts: how many independent stochastic paths to average over
"""
trajectories = []
for _ in range(n_rollouts):
traj = simulate_forward(env, policy, s_t, steps=horizon)
trajectories.append(traj)
return aggregate(trajectories)  # mean, std, percentiles

```

- `horizon=12` → 12-month planning window
- `n_rollouts=50` → 50 stochastic samples for variance reduction

These are **independent parameters** and must not be conflated.

---

## 6. Summary Table

| Criterion | Rollout Simulation | Rolling Horizon (MPC) |
|---|---|---|
| Reward design complexity | Low — same as training | High — must be horizon-consistent |
| Training complexity | Low — standard PPO | High — nested optimization, differentiable model |
| Inference cost | $O(1)$ decision + $O(N \cdot T)$ forecast | $O(H \cdot K)$ per decision |
| Model requirement | Stochastic simulator sufficient | Differentiable, accurate dynamics model |
| Generalization | Yes — via learned policy | No — re-solves at each step |
| Scenario analysis | Natural — vary $s_0$, run rollouts | Expensive — re-optimize per scenario |
| Recommended for this work | ✅ Yes | ❌ Future work |

---

## 7. Conclusion

The rollout-based trajectory simulation strategy is both **theoretically sound** and **practically superior** for this portfolio budgeting problem. It leverages the full power of the trained PPO policy while providing interpretable, statistically stable forecasts at minimal computational cost. Rolling horizon planning, while powerful in deterministic or low-dimensional settings, introduces substantial complexity in reward design, training, and inference that is not justified at this stage of the work.

Rolling horizon planning and MPC-style extensions are identified as natural directions for **future work**, as outlined in the paper's future work section.
`