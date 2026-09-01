# src/play/play.py
#
# Manual play loop for PortfolioBudgetingEnv.
#
# Usage
# -----
#   cd <repo_root>
#   python src/play/play.py [--config single|dual] [--seed N] [--no-db]
#
# Controls
# --------
#   Enter                   → equal split across active projects
#   q / quit / exit         → quit
#   Comma-separated floats  → explicit amounts (normalised to budget fractions)

from __future__ import annotations

import argparse
import sys
import os

# ── path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
_ENV_DIR  = os.path.join(_SRC_DIR, "env")
_DB_DIR   = os.path.join(_SRC_DIR, "db")
_CFGS_DIR = os.path.join(_ENV_DIR, "configs")

for _d in (_SRC_DIR, _ENV_DIR, _DB_DIR, _CFGS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import render as rnd
from env import PortfolioBudgetingEnv


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual play — Portfolio Budgeting RL Env"
    )
    p.add_argument(
        "--config", default=None,
        help="Config name ('single' or 'dual') to skip the DB prompt.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-db", action="store_true",
                   help="Disable DB logging.")
    return p.parse_args()


# ── config selection ──────────────────────────────────────────────────────────

def _list_db_configs(conn) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT config_id, config_name, created_at "
            "FROM environment_config ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _prompt_config_selection(conn) -> str | None:
    configs = _list_db_configs(conn)
    if not configs:
        print("  [!] No configs found in database.")
        return None

    print()
    print("  Available configurations:")
    print(f"  {'#':>3}  {'Config ID':<40}  {'Name'}")
    print("  " + "─" * 70)
    for idx, c in enumerate(configs):
        print(f"  {idx:>3}  {c['config_id']:<40}  {c['config_name']}")
    print()

    while True:
        raw = input("  Select config number (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            sel = int(raw)
            if 0 <= sel < len(configs):
                return configs[sel]["config_id"]
            print(f"  [!] Enter a number between 0 and {len(configs)-1}.")
        except ValueError:
            print("  [!] Enter a valid number.")


def _load_config_by_id(config_id: str) -> dict | None:
    if "SINGLE" in config_id.upper():
        from single_project import CONFIG
        return CONFIG
    if "DUAL" in config_id.upper():
        from dual_project import CONFIG
        return CONFIG
    print(f"  [!] Cannot resolve config_id {config_id!r}.")
    return None


def _load_config_by_name(name: str) -> dict | None:
    n = name.strip().lower()
    if n == "single":
        from single_project import CONFIG
        return CONFIG
    if n == "dual":
        from dual_project import CONFIG
        return CONFIG
    print(f"  [!] Unknown config name {name!r}. Use 'single' or 'dual'.")
    return None


# ── DB connection ─────────────────────────────────────────────────────────────

def _open_conn(no_db: bool):
    if no_db:
        return None
    try:
        from db_init import init_db
        return init_db()
    except Exception as exc:
        print(f"  [play] DB unavailable ({exc}); continuing without logging.")
        return None


def _seed_config(conn, cfg: dict) -> None:
    if conn is None:
        return
    try:
        cid = cfg.get("config_id", "")
        if not cid:
            return
        if "SINGLE" in cid.upper():
            from single_project import seed
        else:
            from dual_project import seed
        seed(conn)
    except Exception:
        pass


# ── action helpers ────────────────────────────────────────────────────────────

def _parse_action(raw: str, n: int, budget: float) -> np.ndarray | None:
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

    fractions = [a / budget for a in amounts] if budget > 0 else [0.0] * n
    frac_sum  = sum(fractions)
    if frac_sum > 1.0 + 1e-9:
        fractions = [f / frac_sum for f in fractions]
    return np.array(fractions, dtype=np.float32)


def _default_action(n: int, active_mask: list[bool]) -> np.ndarray:
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
            cum_reward: float,
            net_cashflow: float = 0.0) -> None:
    rnd.render(
        t               = env.t,
        budget          = env.budget,
        horizon         = env.horizon,
        initial_budget  = env.initial_budget,
        projects        = env.projects,
        proj_state      = env.proj_state,
        milestones      = env.milestones,
        milestone_state = env.milestone_state,
        episode_id      = env.episode_id,
        net_cashflow    = net_cashflow,
        cum_reward      = cum_reward,
    )


# ── action prompt ─────────────────────────────────────────────────────────────

def _prompt(env: PortfolioBudgetingEnv) -> np.ndarray | None:
    n           = len(env.projects)
    active_mask = [ps["status"] == "active" for ps in env.proj_state]

    if n == 1:
        prompt = "  Allocate amount (Enter = all-in, q = quit): "
    else:
        prompt = (
            f"  Allocate {n} comma-separated amounts"
            f" (Enter = equal split, q = quit): "
        )

    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw == "":
            return _default_action(n, active_mask)
        action = _parse_action(raw, n, env.budget)
        if action is not None:
            return action


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    conn = _open_conn(args.no_db)

    # ── config selection ──────────────────────────────────────────────────────
    if args.config:
        cfg = _load_config_by_name(args.config)
        if cfg is None:
            sys.exit(1)
    elif conn is not None:
        config_id = _prompt_config_selection(conn)
        if config_id is None:
            print("  Exiting.")
            conn.close()
            sys.exit(0)
        cfg = _load_config_by_id(config_id)
        if cfg is None:
            sys.exit(1)
    else:
        print("  [play] No DB and no --config; defaulting to 'single'.")
        from single_project import CONFIG
        cfg = CONFIG

    _seed_config(conn, cfg)

    # ── build env ─────────────────────────────────────────────────────────────
    env = PortfolioBudgetingEnv(
        config      = cfg,
        render_mode = "ansi",
        conn        = conn,
        method      = "manual",
    )

    obs, info  = env.reset(seed=args.seed)
    cum_reward = 0.0

    # Initial render
    _render(env, cum_reward, net_cashflow=0.0)

    terminated = False
    truncated  = False

    # ── game loop ─────────────────────────────────────────────────────────────
    while not (terminated or truncated):
        action = _prompt(env)
        if action is None:
            print("  Quitting.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

        net_cashflow = info.get("period_inflow", 0.0) - info.get("period_outflow", 0.0)
        _render(env, cum_reward, net_cashflow=net_cashflow)

    # ── episode end ───────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    if terminated:
        print("  Episode terminated (all projects done).")
    if truncated:
        print("  Episode truncated (horizon reached).")
    print(f"  Cumulative reward: {cum_reward:+.4f}")
    print("═" * 60)

    env.close()
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()