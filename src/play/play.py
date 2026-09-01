# src/play/play.py
#
# Manual play loop for PortfolioBudgetingEnv.
#
# Design rules (from spec):
#   - No formatting logic here.  All print/format functions live in
#     src/env/render.py; this file only calls them.
#   - Input collection and env interaction only.
#   - Imports env via sys.path insertion so play.py can be run directly
#     from src/play/ without installing the package.
#
# Usage
# -----
#   cd <repo_root>
#   python src/play/play.py [--config single|dual] [--seed N] [--no-db]
#
# Controls (each period)
# ----------------------
#   Enter             → allocate budget equally across active projects
#   q / quit / exit   → quit immediately
#   Comma-separated floats (e.g. "30,20") → explicit allocation amounts;
#       values are normalised to fractions of budget before env.step()

from __future__ import annotations

import argparse
import sys
import os

# ── path setup ──────────────────────────────────────────────────────────────
# Resolve repo root (two levels up from this file: src/play/play.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)          # src/
_ENV_DIR  = os.path.join(_SRC_DIR, "env")       # src/env/
_DB_DIR   = os.path.join(_SRC_DIR, "db")        # src/db/
_CFGS_DIR = os.path.join(_ENV_DIR, "configs")   # src/env/configs/

for _d in (_SRC_DIR, _ENV_DIR, _DB_DIR, _CFGS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import render as rnd
from env import PortfolioBudgetingEnv


# ── argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual play loop — Portfolio Budgeting RL Env"
    )
    p.add_argument(
        "--config", choices=["single", "dual"], default="single",
        help="Which config to load (default: single)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for env.reset() (default: random)",
    )
    p.add_argument(
        "--no-db", action="store_true",
        help="Skip DB logging entirely (no database.db created/used)",
    )
    return p.parse_args()


# ── config loader ────────────────────────────────────────────────────────────

def _load_config(name: str) -> dict:
    if name == "single":
        from single_project import CONFIG
    else:
        from dual_project import CONFIG
    return CONFIG


# ── DB connection ─────────────────────────────────────────────────────────────

def _open_conn(cfg: dict, no_db: bool):
    """
    Return an open sqlite3.Connection or None.
    Swallows all errors — DB is side-effect only.
    """
    if no_db:
        return None
    try:
        from db_init import init_db
        conn = init_db()
        if cfg.get("config_id", "").startswith("CFG-SINGLE"):
            from single_project import seed
        else:
            from dual_project import seed
        seed(conn)
        return conn
    except Exception as exc:
        print(f"[play] DB unavailable ({exc}); continuing without logging.")
        return None


# ── action parsing ────────────────────────────────────────────────────────────

def _parse_action(raw: str, n: int, budget: float) -> np.ndarray | None:
    """
    Parse a comma-separated allocation string into a float32 action array.

    Values are treated as raw monetary amounts and converted to fractions
    of budget.  If their sum exceeds budget they are rescaled proportionally.
    Returns None on parse failure (caller will prompt again).
    """
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) != n:
        print(f"  [!] Expected {n} value(s), got {len(parts)}.")
        return None
    try:
        amounts = [float(p) for p in parts]
    except ValueError:
        print("  [!] Non-numeric input.")
        return None
    if any(a < 0 for a in amounts):
        print("  [!] Allocations must be non-negative.")
        return None

    total = sum(amounts)
    if total <= 0.0:
        return np.zeros(n, dtype=np.float32)

    if budget > 0.0:
        fractions = [a / budget for a in amounts]
    else:
        fractions = [0.0] * n

    frac_sum = sum(fractions)
    if frac_sum > 1.0 + 1e-9:
        fractions = [f / frac_sum for f in fractions]

    return np.array(fractions, dtype=np.float32)


def _default_action(n: int, active_mask: list[bool]) -> np.ndarray:
    """Equal split across active projects."""
    n_active = sum(active_mask)
    if n_active == 0:
        return np.zeros(n, dtype=np.float32)
    share = 1.0 / n_active
    return np.array(
        [share if active_mask[i] else 0.0 for i in range(n)],
        dtype=np.float32,
    )


# ── render helper ─────────────────────────────────────────────────────────────

def _render(env: PortfolioBudgetingEnv,
            proj_cf: list[dict],
            period_inflow: float,
            period_outflow: float,
            reward: float | None,
            cum_reward: float) -> None:
    """
    Single call-site for render.render() used both before first step
    and after each step.
    """
    rnd.render(
        t              = env.t,
        budget         = env.budget,
        horizon        = env.horizon,
        initial_budget = env.initial_budget,
        projects       = env.projects,
        proj_state     = env.proj_state,
        milestones     = env.milestones,
        milestone_state= env.milestone_state,
        episode_id     = env.episode_id,
        proj_cf        = proj_cf,
        period_inflow  = period_inflow,
        period_outflow = period_outflow,
        reward         = reward,
        cum_reward     = cum_reward,
    )


# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)
    conn = _open_conn(cfg, args.no_db)

    env = PortfolioBudgetingEnv(
        config      = cfg,
        render_mode = "ansi",
        conn        = conn,
        method      = "manual",
    )

    obs, info = env.reset(seed=args.seed)
    n          = len(env.projects)
    cum_reward = 0.0

    # Empty cashflow dicts for the pre-step render
    empty_cf = [
        {"advance": 0.0, "milestone_net": 0.0, "settlement": 0.0,
         "allocation": 0.0, "interest": 0.0, "treasury_draw": 0.0}
        for _ in range(n)
    ]

    # Print initial state (t=0, before first step)
    _render(env, empty_cf, 0.0, 0.0, reward=None, cum_reward=0.0)

    terminated = False
    truncated  = False

    while not (terminated or truncated):
        active_mask = [ps["status"] == "active" for ps in env.proj_state]
        active_ids  = [str(i) for i, a in enumerate(active_mask) if a]

        print(
            f"  Budget: {env.budget:,.2f}"
            f"   Active project(s): {', '.join(active_ids) if active_ids else 'none'}"
        )

        if n == 1:
            prompt = "  Allocate (amount or Enter for all-in, q to quit): "
        else:
            prompt = (
                f"  Allocate {n} values separated by commas "
                f"(Enter for equal split, q to quit): "
            )

        raw = input(prompt).strip()

        if raw.lower() in ("q", "quit", "exit"):
            print("  Quitting.")
            break

        if raw == "":
            action = _default_action(n, active_mask)
        else:
            action = _parse_action(raw, n, env.budget)
            if action is None:
                continue    # bad input — prompt again

        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

        # proj_cf from info is a list indexed by project position
        proj_cf        = info.get("cashflow",       empty_cf)
        period_inflow  = info.get("period_inflow",  0.0)
        period_outflow = info.get("period_outflow", 0.0)

        _render(env, proj_cf, period_inflow, period_outflow,
                reward=reward, cum_reward=cum_reward)

    # Episode end
    print()
    print("=" * 60)
    if terminated:
        print("  Episode terminated.")
    if truncated:
        print("  Episode truncated (horizon reached).")
    print(f"  Cumulative reward: {cum_reward:+.4f}")
    print("=" * 60)

    env.close()
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()