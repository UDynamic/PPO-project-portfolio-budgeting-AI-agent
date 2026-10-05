# src/env/baseline_solvers.py
#
# Three deterministic baseline solvers for PortfolioBudgetingEnv.
# Each solver implements a single `act(env) -> np.ndarray` method that
# returns an action array (fractions of current budget) identical in
# shape to env.action_space.
#
# Solvers
# -------
#   CatchupTSolver      — allocate catchup_alloc_t     per active project
#   ReachTSolver        — allocate reach_plan_t        per active project
#   CatchupNextTSolver  — allocate catchup_alloc_next_t per active project
#
# Integration
# -----------
# Called from play.py via run_baselines(conn, run_dir, db_path, args).
# Each solver runs over every config in _CFGS_DIR sequentially, creates
# one episode per config, and logs under method = solver.METHOD_NAME.

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env import PortfolioBudgetingEnv



# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class _BaseSolver:
    METHOD_NAME: str = "baseline"

    def act(self, env: "PortfolioBudgetingEnv") -> np.ndarray:
        raise NotImplementedError

    def _amounts_to_action(self, env: "PortfolioBudgetingEnv",
                           amounts: list[float]) -> np.ndarray:
        """
        Convert raw monetary amounts → budget-fraction action vector.
        Inactive projects are zeroed. The result is clipped so fractions
        never exceed 1.0 (the env normalises further internally).
        """
        n      = len(env.projects)
        budget = env.budget if env.budget > 0 else 1.0

        fracs = np.zeros(n, dtype=np.float32)
        for i, (ps, amt) in enumerate(zip(env.proj_state, amounts)):
            if ps["status"] == "active":
                fracs[i] = max(0.0, float(amt)) / budget

        total = float(fracs.sum())
        if total > 1.0 + 1e-9:
            fracs /= total

        return fracs


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER 1 — Catchup-T
# ══════════════════════════════════════════════════════════════════════════════

class CatchupTSolver(_BaseSolver):
    """
    Allocate catchup_alloc_t to each active project.

    Brings the project to plan_t minus the delay tolerance cap —
    stays within breach threshold this period without over-allocating.
    """

    METHOD_NAME = "catchup_t"

    def act(self, env: "PortfolioBudgetingEnv") -> np.ndarray:
        amounts = [
            ps.get("catchup_alloc_t", 0.0)
            for ps in env.proj_state
        ]
        return self._amounts_to_action(env, amounts)


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER 2 — Reach-T
# ══════════════════════════════════════════════════════════════════════════════

class ReachTSolver(_BaseSolver):
    """
    Allocate reach_plan_t to each active project.

    Fully closes the gap to plan_t this period — no breach, no carry-over.
    """

    METHOD_NAME = "reach_t"

    def act(self, env: "PortfolioBudgetingEnv") -> np.ndarray:
        amounts = [
            ps.get("reach_plan_t", 0.0)
            for ps in env.proj_state
        ]
        return self._amounts_to_action(env, amounts)


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER 3 — Catchup-Next-T
# ══════════════════════════════════════════════════════════════════════════════

class CatchupNextTSolver(_BaseSolver):
    """
    Allocate catchup_alloc_next_t to each active project.

    Allocates enough now so that after this period's progress the project
    will be within tolerance of next period's plan — one step ahead of
    the minimum required to avoid a future breach.
    """

    METHOD_NAME = "catchup_next_t"

    def act(self, env: "PortfolioBudgetingEnv") -> np.ndarray:
        amounts = [
            ps.get("catchup_alloc_next_t", 0.0)
            for ps in env.proj_state
        ]
        return self._amounts_to_action(env, amounts)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

ALL_SOLVERS: list[type[_BaseSolver]] = [
    CatchupTSolver,
    ReachTSolver,
    CatchupNextTSolver,
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER  (called from play.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_baselines(
    conn,
    run_dir: str,
    db_path: str,
    args,           # argparse.Namespace from play.py
    _CFGS_DIR: str,
    _banner,
    _ok,
    _warn,
    _render,
    run_export,
) -> None:
    """
    Interactive entry-point called from play.py after DB is initialised.

    Flow
    ----
    1. Ask which solver(s) to run  (1 / 2 / 3 / all).
    2. Ask which config(s)         (filename or 'all').
    3. For each solver × config: reset → step loop → record → next.
    4. Export once at the end.
    """
    import importlib.util
    from env import PortfolioBudgetingEnv

    _banner("BASELINE SOLVERS")

    # ── solver selection ──────────────────────────────────────────────────────
    print()
    print("  Available solvers:")
    for idx, cls in enumerate(ALL_SOLVERS, start=1):
        print(f"    {idx}. {cls.METHOD_NAME}")
    print("    a. all")
    print()

    while True:
        raw = input("  Select solver (1/2/3/a, or q to cancel): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            print("  Cancelled.")
            return
        if raw in ("a", "all"):
            chosen_solvers = [cls() for cls in ALL_SOLVERS]
            break
        if raw.isdigit() and 1 <= int(raw) <= len(ALL_SOLVERS):
            chosen_solvers = [ALL_SOLVERS[int(raw) - 1]()]
            break
        print(f"  [!] Enter 1–{len(ALL_SOLVERS)} or 'a'.")

    # ── config selection ──────────────────────────────────────────────────────
    py_files = sorted(
        f for f in os.listdir(_CFGS_DIR)
        if f.endswith(".py") and not f.startswith("__")
    )
    if not py_files:
        _warn("No config files found", _CFGS_DIR)
        return

    print()
    print("  Available configs:")
    for idx, fname in enumerate(py_files, start=1):
        print(f"    {idx:>3}. {fname}")
    print("    a. all")
    print()

    while True:
        raw = input(
            "  Config filename (e.g. TC_1LMAF_P.py), index number, or 'a' for all: "
        ).strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("  Cancelled.")
            return
        if raw.lower() in ("a", "all"):
            chosen_files = py_files
            break
        # numeric index
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(py_files):
                chosen_files = [py_files[idx]]
                break
            print(f"  [!] Index out of range (1–{len(py_files)}).")
            continue
        # filename match (with or without .py)
        candidate = raw if raw.endswith(".py") else raw + ".py"
        if candidate in py_files:
            chosen_files = [candidate]
            break
        print(f"  [!] '{candidate}' not found. Try the exact filename or index.")

    # ── run ───────────────────────────────────────────────────────────────────
    total_episodes = len(chosen_solvers) * len(chosen_files)
    done_count     = 0

    print()
    print(f"  Running {len(chosen_solvers)} solver(s) × {len(chosen_files)} config(s)"
          f" = {total_episodes} episode(s).")
    print()

    for solver in chosen_solvers:
        _banner(f"SOLVER: {solver.METHOD_NAME}")

        for fname in chosen_files:
            # ── load config ───────────────────────────────────────────────────
            fpath       = os.path.join(_CFGS_DIR, fname)
            module_name = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, fpath)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cfg  = getattr(mod, "CONFIG", None)
                if cfg is None:
                    _warn(f"No CONFIG in {fname} — skipped")
                    continue
            except Exception as exc:
                _warn(f"Failed to load {fname}", str(exc))
                continue

            config_id = cfg.get("config_id", fname)
            print(f"  ▶  {solver.METHOD_NAME}  ×  {config_id}")

            # ── build env ─────────────────────────────────────────────────────
            try:
                env = PortfolioBudgetingEnv(
                    config      = cfg,
                    render_mode = "ansi",
                    conn        = conn,
                    method      = solver.METHOD_NAME,
                )
                obs, info  = env.reset(seed=args.seed)
                cum_reward = 0.0

                _ok(f"Episode ID", env.episode_id)
                _render(env, cum_reward, net_cashflow=0.0)
            except Exception as exc:
                _warn(f"Env init failed for {config_id}", str(exc))
                continue

            # ── episode loop ──────────────────────────────────────────────────
            terminated = False
            truncated  = False

            while not (terminated or truncated):
                try:
                    action = solver.act(env)
                    obs, reward, terminated, truncated, info = env.step(action)
                    cum_reward += reward
                    net_cashflow = info.get("period_inflow", 0.0) - info.get("period_outflow", 0.0)
                    _render(env, cum_reward, net_cashflow=net_cashflow)
                except Exception as exc:
                    _warn(f"Step failed (t={env.t})", str(exc))
                    break

            # ── episode summary ───────────────────────────────────────────────
            print()
            print("  " + "─" * 56)
            if terminated:
                _ok("Episode terminated (all projects done)")
            if truncated:
                _ok("Episode truncated (horizon reached)")
            print(f"  Cumulative reward : {cum_reward:+.4f}")
            print(f"  Episode ID        : {env.episode_id}")
            print("  " + "─" * 56)
            print()

            env.close()
            done_count += 1

        # ── per-solver completion ─────────────────────────────────────────────
        _ok(f"Solver '{solver.METHOD_NAME}' complete",
            f"{len(chosen_files)} config(s) processed")
        print()

    # ── export ────────────────────────────────────────────────────────────────
    if conn is not None and not args.no_export:
        conn.close()
        exports_dir = args.out or os.path.join(run_dir, "exports")
        run_export(
            db_path        = db_path,
            fmt            = args.fmt,
            out_dir        = exports_dir,
            tables_subset  = args.tables,
            episode_id     = None,
        )

    # ── final summary ─────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  ✓  All baseline runs complete.")
    print(f"     Solvers  : {', '.join(s.METHOD_NAME for s in chosen_solvers)}")
    print(f"     Configs  : {len(chosen_files)}")
    print(f"     Episodes : {done_count} / {total_episodes}")
    if run_dir:
        print(f"     Run dir  : {run_dir}")
    print("═" * 60)