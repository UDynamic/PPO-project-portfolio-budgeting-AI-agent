"""
tests/regression/add_baseline_fixtures.py

Runs the three deterministic baseline solvers over every config in
db/configs/ and writes the results as regression fixtures.

Solvers
-------
  greedy_catchup    — allocate catchup_alloc_t  per active project
  greedy_reach_plan — allocate reach_plan_t     per active project
  uniform           — allocate bac / planned_duration per active project

Pending periods are auto-stepped with zeros (no action consumed).
actions.json only records the active-period actions.

Usage
-----
Run from src/ or any directory:

    python tests/regression/add_baseline_fixtures.py

Options
-------
  --configs  TC-10SF-R TC-1LSF-R ...   only run these config IDs (default: all)
  --methods  greedy_catchup uniform ... only run these methods    (default: all)
  --overwrite                           overwrite existing fixtures without prompt
  --seed N                              RNG seed passed to env.reset() (default: 0)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
_REGRESSION_DIR = Path(__file__).parent          # src/tests/regression
_TESTS_DIR      = _REGRESSION_DIR.parent         # src/tests
_SRC_DIR        = _TESTS_DIR.parent              # src/
_CFGS_DIR       = _SRC_DIR / "db" / "configs"
_ENV_DIR        = _SRC_DIR / "env"
_GT_DIR         = _REGRESSION_DIR / "ground_truth"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ENV_DIR) not in sys.path:
    sys.path.append(str(_ENV_DIR))

from env.env import PortfolioBudgetingEnv  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# SOLVERS
# ══════════════════════════════════════════════════════════════════════════════

def _amounts_to_action(env: PortfolioBudgetingEnv,
                        amounts: list[float]) -> np.ndarray:
    """Convert raw monetary amounts → budget-fraction action vector."""
    budget = env.budget if env.budget > 0 else 1.0
    fracs  = np.zeros(len(env.projects), dtype=np.float32)
    for i, (ps, amt) in enumerate(zip(env.proj_state, amounts)):
        if ps["status"] == "active":
            fracs[i] = max(0.0, float(amt)) / budget
    total = float(fracs.sum())
    if total > 1.0 + 1e-9:
        fracs /= total
    return fracs


def act_greedy_catchup(env: PortfolioBudgetingEnv) -> np.ndarray:
    amounts = [ps.get("catchup_alloc_t", 0.0) for ps in env.proj_state]
    return _amounts_to_action(env, amounts)


def act_greedy_reach_plan(env: PortfolioBudgetingEnv) -> np.ndarray:
    amounts = [ps.get("reach_plan_t", 0.0) for ps in env.proj_state]
    return _amounts_to_action(env, amounts)


def act_uniform(env: PortfolioBudgetingEnv) -> np.ndarray:
    amounts = []
    for proj, ps in zip(env.projects, env.proj_state):
        if ps["status"] == "active":
            dur = max(1.0, float(proj.get("planned_duration", 1.0)))
            amounts.append(proj["bac"] / dur)
        else:
            amounts.append(0.0)
    return _amounts_to_action(env, amounts)


SOLVERS = {
    "greedy_catchup":    act_greedy_catchup,
    "greedy_reach_plan": act_greedy_reach_plan,
    "uniform":           act_uniform,
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_configs(only: list[str] | None) -> list[dict]:
    """Load all CONFIG dicts from db/configs/. Filter by config_id if given."""
    cfgs = []
    for fname in sorted(os.listdir(_CFGS_DIR)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = _CFGS_DIR / fname
        spec  = importlib.util.spec_from_file_location(fname[:-3], fpath)
        mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "CONFIG", None)
        if cfg is None:
            print(f"  [!] No CONFIG in {fname} — skipped")
            continue
        if only and cfg.get("config_id") not in only:
            continue
        cfgs.append(cfg)
    return cfgs


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER — returns (actions_json, export_tables)
# ══════════════════════════════════════════════════════════════════════════════

HARNESS_TABLES = [
    "portfolios",
    "projects_status",
    "projects_observation",
    "milestones_status",
]


def _run_episode(cfg: dict, method: str, seed: int) -> tuple[list, dict]:
    """
    Run one episode with the given solver.
    Returns:
        actions  — list of per-active-step allocations (monetary amounts)
        tables   — dict of table_name → list[dict] rows
    """
    act_fn = SOLVERS[method]

    env = PortfolioBudgetingEnv(
        config=cfg, render_mode=None, conn=None, method=method,
    )
    obs, info = env.reset(seed=seed)

    n  = len(env.projects)
    actions: list = []
    tables: dict[str, list] = {k: [] for k in HARNESS_TABLES}

    terminated = truncated = False
    while not (terminated or truncated):

        if info["n_active"] == 0:
            # Pending gap — auto-step, don't record action
            action = np.zeros(n, dtype=np.float32)
        else:
            action = act_fn(env)
            # Record as monetary amounts (budget * fraction), rounded to 4dp
            amounts = [round(float(action[i]) * env.budget, 4) for i in range(n)]
            actions.append(amounts if n > 1 else amounts[0])

        obs, reward, terminated, truncated, info = env.step(action)

        rt          = info["recorded_t"]
        proj_states = info["recorded_proj_state"]

        # portfolios
        tables["portfolios"].append({
            "t_episode":        rt,
            "net_cashflow":     info["recorded_net_cashflow"],
            "reward":           info["recorded_reward"],
            "budget_available": info["recorded_budget"],
        })

        # projects_status + projects_observation
        for proj, ps in zip(env.projects, proj_states):
            i = proj["i"]
            tables["projects_status"].append({
                "i":                     i,
                "t_episode":             rt,
                "status":                ps.get("status", ""),
                "progress_actual":       ps.get("progress_actual", 0.0),
                "allocation_action":     ps.get("allocation_action", 0.0),
                "inflow":                ps.get("inflow", 0.0),
                "outflow":               ps.get("outflow", 0.0),
            })
            tables["projects_observation"].append({
                "i":                     i,
                "t_episode":             rt,
                "catchup_alloc_t":       ps.get("catchup_alloc_t", 0.0),
                "catchup_alloc_next_t":  ps.get("catchup_alloc_next_t", 0.0),
                "reach_plan_t":          ps.get("reach_plan_t", 0.0),
                "reach_plan_next_t":     ps.get("reach_plan_next_t", 0.0),
                "target_progress_gap":   ps.get("target_progress_gap", 0.0),
                "target_timestep_gap":   ps.get("target_timestep_gap", 0),
                "target_required_alloc": ps.get("target_required_alloc", 0.0),
                "target_npv":            ps.get("target_npv", 0.0),
            })

        # milestones_status — only certified this period
        for proj, ms_list, ms_sl in zip(
                env.projects, env.milestones, env.milestone_state):
            for ms, ms_state in zip(ms_list, ms_sl):
                if ms_state["certified"] and ms_state["certified_t"] == rt:
                    ep_thresh = proj["planned_start"] + ms["timestep_threshold"]
                    delay = (ms_state["certified_t"] - ep_thresh
                             if ms_state["certified_t"] is not None else None)
                    tables["milestones_status"].append({
                        "i":                   proj["i"],
                        "j":                   ms["j"],
                        "certified_t":         ms_state["certified_t"],
                        "certification_delay": delay,
                        "net_payment":         ms_state["payment_released"],
                    })

    env.close()
    return actions, tables


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _write_fixture(config_id: str, method: str,
                   actions: list, tables: dict,
                   overwrite: bool) -> bool:
    """Write actions.json + CSVs into ground_truth/config_id/method/."""
    import csv

    dest = _GT_DIR / config_id / method

    if dest.exists() and not overwrite:
        ans = input(f"  Fixture {config_id}/{method} exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print(f"  Skipped {config_id}/{method}.")
            return False

    dest.mkdir(parents=True, exist_ok=True)

    # actions.json
    with open(dest / "actions.json", "w") as f:
        json.dump(actions, f, indent=4)

    # CSVs
    for table_name, rows in tables.items():
        if not rows:
            continue
        fpath = dest / f"{table_name}.csv"
        with open(fpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate regression fixtures for all baseline solvers."
    )
    parser.add_argument(
        "--configs", nargs="+", default=None, metavar="CONFIG_ID",
        help="Config IDs to run (default: all).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(SOLVERS),
        choices=list(SOLVERS), metavar="METHOD",
        help="Solver methods to run (default: all).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing fixtures without prompting.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for env.reset() (default: 0).",
    )
    args = parser.parse_args()

    cfgs = _load_configs(args.configs)
    if not cfgs:
        sys.exit("[!] No configs found.")

    methods = args.methods

    total   = len(cfgs) * len(methods)
    written = 0
    skipped = 0
    failed  = 0

    print(f"\n  {len(cfgs)} config(s) × {len(methods)} method(s) = {total} fixture(s)\n")

    for cfg in cfgs:
        config_id = cfg["config_id"]
        for method in methods:
            label = f"{config_id}__{method}"
            print(f"  ▶  {label} ... ", end="", flush=True)
            try:
                actions, tables = _run_episode(cfg, method, seed=args.seed)
                ok = _write_fixture(config_id, method, actions, tables,
                                    overwrite=args.overwrite)
                if ok:
                    written += 1
                    print("✓")
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"✗  {exc}")

    print()
    print("═" * 60)
    print(f"  written : {written}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")
    print("═" * 60)


if __name__ == "__main__":
    main()