# <div align="center">PPO Budget Allocation Agent for Project Portfolios

<br/><br/><br/>
A **Deep Reinforcement Learning** (DRL) framework—built around **Proximal Policy Optimization (PPO)**—for **dynamic, uncertainty-aware budget allocation** across multi-project portfolios. 
The agent learns **continuous allocation policies** that adapt to **stochastic cash flows, risks, and shifting priorities** over discrete time horizons.

<br/><br/><br/>

## <div align="center"> Why this matters

Traditional portfolio budgeting often relies on **static, deterministic** assumptions and **one-shot** decisions. Real portfolios face **volatile cash flows**, **evolving risks**, and **competing objectives**. This project provides a **learning-based decision policy** that adapts over time, **maximizes value under uncertainty**, and **outperforms static baselines** in controlled simulations.

---
## <div align="center"> Problem Formulation (MDP)
!!!to be updated

- **State** \(s_t\): features per project and portfolio (progress, milestone needs, available liquidity, risk indices, cost variance, schedule slippage, etc.).
- **Action** \(a_t\): **continuous** budget allocation vector subject to liquidity and policy constraints.
- **Transition** \(p(s_{t+1}\,|\,s_t,a_t)\): stochastic dynamics over progress, costs, and revenues (parametric noise; beta/normal assumptions).
- **Reward** \(r_t\): portfolio value signal (e.g., risk-adjusted cash return, penalty for overruns/delays), aggregated over projects.
- **Objective**:
  \[
  \max_{\theta}\; J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\sum_{t=0}^{T}\gamma^t r_t\Big]
  \]
  with discount \(\gamma\in(0,1)\).

<div align="center"> **Advantage estimation** 
  
  (using rewards-to-go \(R_t\) and value function \(V_\phi\)):
\[
A_t = R_t - V_\phi(s_t)
\]

<div align="center"> **PPO (clipped) objective**:
  
\[
\mathcal{L}^{\text{CLIP}}(\theta) =
\mathbb{E}_t\Big[
\min\big(r_t(\theta)A_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t\big)
\Big],\quad
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
\]

---

## <div align="center"> Synthetic Environment & Data Generator<div/>

Because public, fully-labeled portfolio cash-flow datasets with transition details are **not available**, we implement a **reproducible synthetic generator** that doubles as the **training environment**:

- **Episode composition**: sample **5–15 projects** uniformly each episode.
- **Horizon**: discrete periods (e.g., **12–24**).
- **Seeds**: generate **1,000** random seeds; split **80% train / 20% eval**.
- **Uncertainty**: parametric noise for costs/revenues (e.g., normal), reliability/risks (e.g., beta).
- **Progress-linked payments**: cash needs and disbursements tied to milestones.
- **Features**: include risk of cost overrun, schedule delay, HR/resource risk, etc.

---

## <div align="center"> Capabilities<div/>

1. **Continuous allocation** under liquidity constraints across many projects.
2. **Adaptive** to non-stationary conditions (cost drift, contractor productivity shifts).
3. **Robust evaluation** vs. static baselines (LP/IP/heuristics) under identical seeds.
4. **Metrics**: budget efficiency, goal attainment, risk-adjusted return, regret vs. oracle, resilience under shocks.
5. **Scalable design**: single-agent core, multi-agent ready architecture.

---

## <div align="center"> Method (aligned with “My PPO Reference Steps”)<div/>

1. **Initialize Actor** (policy network).
2. **Initialize Critic** (value network).
3. **Collect Trajectories** in the synthetic portfolio environment.
4. **Compute Rewards-to-Go** (discounted).
5. **Compute Advantages** (returns – value).
6. **Optimize Critic** (value loss, e.g., MSE).
7. **Optimize Actor** (PPO clipped objective; gradient ascent via negated loss).
8. **Repeat Optimization** for several epochs per batch.
9. **Repeat Batches** until training budget is met.



---

To be planned:
* devloping next.js frontend 
* developing Django backend







