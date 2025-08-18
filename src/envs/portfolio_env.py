# envs/portfolio.py
"""
ProjectPortfolioEnv
-------------------

A custom Gymnasium environment for training PPO (Stable-Baselines3) to optimize
budget allocation across a portfolio of projects with stochastic costs and
cashflow milestones.

Key features
============
- Portfolio of 10–15 projects (configurable).
- Each project has multiple milestones with uncertain costs and uncertain cashflows
  (amount and timing/delay).
- At each step, the agent allocates budget weights across projects. Actual dollar
  allocations are derived from a softmax over the action vector and scaled by the
  step's spendable budget.
- Observations include per-project progress, expected costs/cashflows, pending
  payments, and simple historical performance features, plus portfolio-level
  features.
- Rewards are realized profit (cash-in minus cost-out) with an optional downside
  risk penalty.
- Compatible with Stable-Baselines3 PPO (Box observation, Box action).
- Includes helper functions to generate synthetic projects and milestone schemas.

Usage
=====
from envs.portfolio import ProjectPortfolioEnv
env = ProjectPortfolioEnv()  # or pass custom config
obs, info = env.reset(seed=42)

With SB3:
---------
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = ProjectPortfolioEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

Notes
=====
- The action is an unconstrained vector (shape: num_projects). Internally it is
  transformed with softmax to nonnegative weights summing to 1, then scaled by
  the maximum amount you allow to be spent in this step (a fraction of the
  current budget).
- Cash inflows can optionally be reinvested to the spendable budget (default True).

Author: you :-)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------
# Data structures & helpers
# ---------------------------

@dataclass
class Milestone:
    """A single project milestone with uncertain cost and payoff."""
    cost_mean: float
    cost_std: float
    payoff_mean: float
    payoff_std: float
    delay_mean_steps: float = 1.0  # mean of Poisson delay for payment realization

@dataclass
class Project:
    """Project state and parameters over multiple milestones."""
    milestones: List[Milestone]
    perf_std: float = 0.2  # stochastic efficiency of spend to progress (lognormal)
    idx: int = 0  # current milestone index
    progress_within: float = 0.0  # [0,1) progress inside current milestone
    pending_payments: List[Tuple[float, int]] = field(default_factory=list)  # (amount, steps_remaining)
    # Tracking for simple historical performance:
    # EMA of cost overrun ratio when a milestone completes: realized_cost / expected_cost - 1
    overrun_ema: float = 0.0
    overrun_beta: float = 0.9

    # Expected cost remaining cache (updated on the fly)
    expected_remaining_cost: float = 0.0

    # For accumulating realized cost spent toward the current milestone
    spent_current_ms: float = 0.0
    expected_cost_current_ms: float = 0.0

    def is_complete(self) -> bool:
        return self.idx >= len(self.milestones)

    def current_milestone(self) -> Optional[Milestone]:
        if self.is_complete():
            return None
        return self.milestones[self.idx]


# ---------------------------
# Synthetic generators
# ---------------------------

def generate_synthetic_projects(
    rng: np.random.Generator,
    num_projects: int = 12,
    milestone_count_range: Tuple[int, int] = (3, 6),
    cost_scale: Tuple[float, float] = (50_000.0, 200_000.0),
    payoff_scale: Tuple[float, float] = (80_000.0, 300_000.0),
    avg_delay_range: Tuple[float, float] = (0.5, 3.0),
    perf_std_range: Tuple[float, float] = (0.10, 0.35),
) -> List[Project]:
    """
    Create a list of synthetic Project objects with random milestone schemas.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    num_projects : int
        Number of projects to generate.
    milestone_count_range : (int, int)
        Inclusive range for milestone counts per project.
    cost_scale : (float, float)
        Range of mean cost per milestone (uniform).
    payoff_scale : (float, float)
        Range of mean payoff per milestone (uniform).
    avg_delay_range : (float, float)
        Range of mean payment delay (in steps); used as Poisson mean.
    perf_std_range : (float, float)
        Range of per-project efficiency noise (lognormal sigma).

    Returns
    -------
    List[Project]
    """
    projects: List[Project] = []
    for _ in range(num_projects):
        mcount = rng.integers(milestone_count_range[0], milestone_count_range[1] + 1)
        milestones = []
        # Create milestones that generally grow in cost & payoff as index increases
        for mi in range(mcount):
            c_mean = rng.uniform(*cost_scale) * (1.0 + 0.1 * mi)
            c_std = 0.25 * c_mean
            p_mean = rng.uniform(*payoff_scale) * (1.0 + 0.12 * mi)
            p_std = 0.30 * p_mean
            delay_mean = rng.uniform(*avg_delay_range)
            milestones.append(Milestone(
                cost_mean=c_mean,
                cost_std=c_std,
                payoff_mean=p_mean,
                payoff_std=p_std,
                delay_mean_steps=delay_mean,
            ))
        perf_std = rng.uniform(*perf_std_range)
        proj = Project(milestones=milestones, perf_std=perf_std)
        # initialize expected cost for the first milestone
        proj.expected_cost_current_ms = milestones[0].cost_mean if milestones else 0.0
        proj.expected_remaining_cost = sum(m.cost_mean for m in milestones)
        projects.append(proj)
    return projects


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0.0:
        return np.ones_like(x) / len(x)
    return e / s


def _clip_positive(x: float) -> float:
    return max(0.0, float(x))


# ---------------------------
# Environment
# ---------------------------

class ProjectPortfolioEnv(gym.Env):
    """
    Gymnasium environment for stochastic project portfolio budgeting.

    Observation
    -----------
    A single flattened Box vector containing per-project features followed by
    portfolio-level features. Shape = num_projects * P + K, where:

    Per-project features (P = 7):
      0: milestone_index / total_milestones
      1: progress_within_current_milestone in [0,1]
      2: expected_payoff_next (0 if complete)
      3: expected_remaining_cost (mean) for this project
      4: pending_payments_sum (face value not yet realized)
      5: overrun_ema (historical cost overrun metric)
      6: is_complete (0/1)

    Portfolio-level features (K = 3):
      - remaining_budget_fraction (current_budget / initial_budget)
      - time_remaining_fraction ((horizon - t) / horizon)
      - outstanding_payments_total (sum over all projects)

    Action
    ------
    Box(low=-inf, high=inf, shape=(num_projects,))
    Interpreted as unnormalized logits; internally transformed via softmax to
    nonnegative weights that sum to 1. The environment then spends up to
    `max_spend_frac_per_step * current_budget` distributed by those weights.

    Reward
    ------
    step_profit = realized_cash_in - realized_cost_out
    reward = step_profit - risk_aversion * max(0, -step_profit)

    Episode End
    -----------
    - When time step reaches horizon (truncated=True)
    - Or when all projects are complete (terminated=True)

    Compatibility
    -------------
    - Gymnasium API: reset(seed), step(action)
    - Stable-Baselines3 PPO compatible (Box obs/action). For Dict obs, use
      MultiInputPolicy; here we use a flat Box for simplicity.

    Parameters
    ----------
    num_projects : int, default 12
    horizon : int
        Number of decision steps in an episode.
    initial_budget : float
        Starting budget (cash) available to allocate.
    reinvest_cashflows : bool
        If True, realized cashflows increase the spendable budget.
    max_spend_frac_per_step : float in (0, 1]
        Maximum fraction of current budget that can be spent in a single step.
    risk_aversion : float >= 0
        Downside penalty weight.
    project_generator : callable | None
        If provided, called as project_generator(rng, num_projects) -> List[Project].
        Otherwise, synthetic projects are generated with defaults.

    render_mode : None or "human"
        If "human", render() prints a compact textual summary.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_projects: int = 12,
        horizon: int = 48,
        initial_budget: float = 2_000_000.0,
        reinvest_cashflows: bool = True,
        max_spend_frac_per_step: float = 0.25,
        risk_aversion: float = 0.25,
        project_generator=None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert 1 <= num_projects <= 64, "num_projects must be reasonable"
        assert 0 < max_spend_frac_per_step <= 1.0

        self.num_projects = num_projects
        self.horizon = horizon
        self.initial_budget = float(initial_budget)
        self.reinvest_cashflows = bool(reinvest_cashflows)
        self.max_spend_frac_per_step = float(max_spend_frac_per_step)
        self.risk_aversion = float(risk_aversion)
        self._custom_project_generator = project_generator
        self.render_mode = render_mode

        # Spaces
        # Observation: flattened per-project features + portfolio features
        self.per_project_features = 7
        self.portfolio_features = 3
        obs_dim = self.num_projects * self.per_project_features + self.portfolio_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: unconstrained logits -> softmax weights
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_projects,), dtype=np.float32
        )

        # Internal state
        self.projects: List[Project] = []
        self.t: int = 0
        self.budget: float = 0.0  # spendable
        self.initialized: bool = False
        self._last_step_profit: float = 0.0
        self._rng: np.random.Generator = np.random.default_rng()

    # -------------
    # Gym API
    # -------------

    def seed(self, seed: Optional[int] = None) -> None:
        """Deprecated in Gymnasium; use reset(seed). Provided for convenience."""
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Generate portfolio
        if self._custom_project_generator is not None:
            self.projects = self._custom_project_generator(self._rng, self.num_projects)
        else:
            self.projects = generate_synthetic_projects(self._rng, self.num_projects)

        # Initialize expected remaining costs for each project
        for p in self.projects:
            p.idx = 0
            p.progress_within = 0.0
            p.pending_payments.clear()
            p.overrun_ema = 0.0
            p.spent_current_ms = 0.0
            if p.current_milestone() is not None:
                p.expected_cost_current_ms = p.current_milestone().cost_mean
            else:
                p.expected_cost_current_ms = 0.0
            p.expected_remaining_cost = sum(m.cost_mean for m in p.milestones)

        self.t = 0
        self.budget = float(self.initial_budget)
        self._last_step_profit = 0.0
        self.initialized = True

        obs = self._get_obs()
        info = {"last_step_profit": self._last_step_profit}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.initialized, "Call reset() before step()."

        # Transform action -> allocation weights
        action = np.asarray(action, dtype=np.float32).reshape(self.num_projects)
        weights = _softmax(action)
        max_spend = self.max_spend_frac_per_step * self.budget
        allocations = weights * max_spend  # dollars to spend this step

        # Apply allocations: progress + costs realized immediately
        total_cost_out = float(np.sum(allocations))
        self.budget -= total_cost_out

        # Realized payments this step (from matured pending payments)
        realized_cash_in = 0.0

        # Progress each project
        for i, p in enumerate(self.projects):
            if p.is_complete():
                continue

            ms = p.current_milestone()
            assert ms is not None

            spend_i = float(allocations[i])
            # Stochastic efficiency: spend turns into progress with lognormal multiplier
            # efficiency ~ LogNormal(mu=-0.5*sigma^2, sigma=perf_std), mean approx 1.0
            sigma = max(1e-6, p.perf_std)
            mu = -0.5 * sigma * sigma
            efficiency = float(self._rng.lognormal(mean=mu, sigma=sigma))

            # Expected remaining cost is our progress denominator proxy
            denom = max(1e-6, p.expected_cost_current_ms - p.spent_current_ms)
            progress_gain = (spend_i / denom) * efficiency
            p.progress_within += progress_gain
            p.spent_current_ms += spend_i
            p.expected_remaining_cost = max(
                0.0,
                p.expected_remaining_cost - spend_i
            )

        # Check milestone completions and enqueue payments with random delays
        for p in self.projects:
            while (not p.is_complete()) and (p.progress_within >= 1.0):
                ms = p.current_milestone()
                assert ms is not None

                # Sample payoff (clipped to nonnegative)
                payoff = _clip_positive(
                    float(self._rng.normal(loc=ms.payoff_mean, scale=ms.payoff_std))
                )
                # Sample integer delay via Poisson with mean ms.delay_mean_steps
                lam = max(1e-6, ms.delay_mean_steps)
                delay = int(self._rng.poisson(lam=lam))
                # Add to pending payments (0 delay means immediate next loop maturity)
                p.pending_payments.append((payoff, delay))

                # Compute cost overrun realized for this milestone and update EMA
                realized_cost_ms = p.spent_current_ms
                expected_cost_ms = max(1e-6, p.expected_cost_current_ms)
                overrun = (realized_cost_ms / expected_cost_ms) - 1.0
                p.overrun_ema = p.overrun_beta * p.overrun_ema + (1.0 - p.overrun_beta) * overrun

                # Advance to next milestone
                p.idx += 1
                p.progress_within -= 1.0
                p.spent_current_ms = 0.0
                if not p.is_complete():
                    nxt = p.current_milestone()
                    p.expected_cost_current_ms = nxt.cost_mean
                else:
                    p.expected_cost_current_ms = 0.0

        # Age pending payments; realize those that mature
        for p in self.projects:
            if not p.pending_payments:
                continue
            new_queue: List[Tuple[float, int]] = []
            for amt, d in p.pending_payments:
                if d <= 0:
                    realized_cash_in += amt
                else:
                    new_queue.append((amt, d - 1))
            p.pending_payments = new_queue

        # Reinvest cashflows if enabled
        if self.reinvest_cashflows:
            self.budget += realized_cash_in

        # Reward (risk-adjusted)
        step_profit = realized_cash_in - total_cost_out
        downside = max(0.0, -step_profit)
        reward = step_profit - self.risk_aversion * downside
        self._last_step_profit = step_profit

        # Time update
        self.t += 1

        # Termination & truncation
        all_done = all(p.is_complete() for p in self.projects)
        terminated = bool(all_done)
        truncated = bool(self.t >= self.horizon)

        obs = self._get_obs()
        info = {
            "last_step_profit": step_profit,
            "realized_cash_in": realized_cash_in,
            "cost_out": total_cost_out,
            "budget": self.budget,
            "time_step": self.t,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # -------------
    # Rendering
    # -------------

    def render(self):
        if self.render_mode != "human":
            return
        completed = sum(1 for p in self.projects if p.is_complete())
        pending_total = sum(sum(amt for amt, _ in p.pending_payments) for p in self.projects)
        print(
            f"[t={self.t:02d}] budget={self.budget:,.0f}  "
            f"completed={completed}/{self.num_projects}  "
            f"pending=${pending_total:,.0f}  "
            f"last_step_profit={self._last_step_profit:,.0f}"
        )

    def close(self):
        pass

    # -------------
    # Observations
    # -------------

    def _get_obs(self) -> np.ndarray:
        per_proj = []
        outstanding_total = 0.0

        for p in self.projects:
            total_m = max(1, len(p.milestones))
            idx_norm = min(1.0, p.idx / total_m)
            prog = 0.0 if p.is_complete() else float(np.clip(p.progress_within, 0.0, 1.0))
            expected_pay_next = 0.0
            if not p.is_complete():
                ms = p.current_milestone()
                assert ms is not None
                expected_pay_next = ms.payoff_mean
            pending_sum = sum(amt for amt, _ in p.pending_payments)
            outstanding_total += pending_sum
            is_comp = 1.0 if p.is_complete() else 0.0

            per_proj.extend([
                idx_norm,
                prog,
                float(expected_pay_next),
                float(p.expected_remaining_cost),
                float(pending_sum),
                float(p.overrun_ema),
                float(is_comp),
            ])

        remaining_budget_fraction = float(self.budget / max(1e-6, self.initial_budget))
        time_remaining_fraction = float(max(0, self.horizon - self.t) / max(1, self.horizon))

        portfolio_feats = [
            remaining_budget_fraction,
            time_remaining_fraction,
            float(outstanding_total),
        ]

        obs = np.array(per_proj + portfolio_feats, dtype=np.float32)
        return obs

    # -------------
    # Utility
    # -------------

    @property
    def projects_completed(self) -> int:
        return sum(1 for p in self.projects if p.is_complete())

    def set_portfolio(self, projects: List[Project]) -> None:
        """
        Replace the current portfolio with user-specified projects.
        Call before reset() for custom setups.
        """
        self._custom_project_generator = lambda rng, n: projects
