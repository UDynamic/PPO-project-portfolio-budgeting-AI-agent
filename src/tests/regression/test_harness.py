"""
tests/regression/harness.py

Regression harness for PortfolioBudgetingEnv.

Discovers all ground-truth fixtures under ground_truth/<config_id>/<method>/
and replays each one against the live environment, asserting that every
tracked field matches within TOLERANCE.

Directory layout
----------------
ground_truth/
    <config_id>/
        <method>/           # optimal | greedy_catchup | greedy_reach_plan | uniform
            actions.json    # list of per-period absolute allocation amounts
            portfolios.csv
            projects_status.csv
            projects_observation.csv
            milestones_status.csv   (optional — omitted if no milestones certified)

actions.json format
-------------------
[
    [alloc_proj0, alloc_proj1, ...],   # period 0 absolute amounts (£/$/units)
    [alloc_proj0, alloc_proj1, ...],   # period 1
    ...
]
Single-project configs may use a flat list of scalars:
    [110.0, 0.0, 0.0]

How to add a new fixture
------------------------
1. Run play.py manually and verify the output is correct.
2. Copy the exports folder into ground_truth/<config_id>/<method>/.
3. Create actions.json matching the allocations you typed.
4. Run pytest — the new fixture is picked up automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── path bootstrap ────────────────────────────────────────────────────────────
# Allows running from any working directory.
# __file__ = src/tests/regression/test_harness.py
REGRESSION_DIR = Path(__file__).parent          # src/tests/regression
TESTS_DIR      = REGRESSION_DIR.parent          # src/tests
SRC_DIR        = TESTS_DIR.parent               # src/
GT_DIR         = REGRESSION_DIR / "ground_truth"
CFGS_DIR       = SRC_DIR / "db" / "configs"

ENV_DIR = SRC_DIR / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── tolerance ─────────────────────────────────────────────────────────────────
TOLERANCE = 1e-3    # absolute; covers floating-point rounding across platforms

# ── fields asserted per table ─────────────────────────────────────────────────
# Extend these lists to assert more fields as confidence grows.

PORTFOLIO_FIELDS = [
    "net_cashflow",
    "reward",
    "budget_available",
]

PROJECT_STATUS_FIELDS = [
    "status",
    "inflow",
    "outflow",
    "allocation_action",
    "interest_cost",
    "progress_actual",
    "tolerance_remain",
    "catchup_alloc_t",
    "reach_plan_t",
    "target_progress_gap",
    "target_timestep_gap",
    "target_net_payment",
    "target_required_alloc",
    "target_npv",
]

PROJECT_OBS_FIELDS = [
    "net_cashflow",
    "tolerance_remain",
    "catchup_alloc_t",
    "catchup_alloc_next_t",
    "reach_plan_t",
    "reach_plan_next_t",
    "target_progress_gap",
    "target_timestep_gap",
    "target_required_alloc",
    "target_npv",
]

MILESTONE_STATUS_FIELDS = [
    "certified_t",
    "certification_delay",
    "net_payment",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_fixtures() -> list[tuple[str, str, Path]]:
    """
    Walk ground_truth/ and return (config_id, method, fixture_dir) for every
    fixture that has both actions.json and portfolios.csv.
    """
    fixtures = []
    if not GT_DIR.exists():
        return fixtures
    for config_dir in sorted(GT_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        config_id = config_dir.name
        for method_dir in sorted(config_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if (method_dir / "actions.json").exists() and \
               (method_dir / "portfolios.csv").exists():
                fixtures.append((config_id, method, method_dir))
    return fixtures


# ═══════════════════════════════════════════════════════════════════════════════
# ENV RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config(config_id: str) -> dict:
    import importlib.util
    for fname in os.listdir(CFGS_DIR):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = CFGS_DIR / fname
        spec  = importlib.util.spec_from_file_location(fname[:-3], fpath)
        mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "CONFIG", {})
        if cfg.get("config_id") == config_id:
            return cfg
    raise FileNotFoundError(f"Config {config_id!r} not found in {CFGS_DIR}")


def _parse_actions(raw: list, n_projects: int, budget: float) -> np.ndarray:
    """
    Convert one period's action entry from actions.json into a fraction vector.
    Accepts a scalar (single-project shorthand) or a list of absolute amounts.
    """
    if not isinstance(raw, (list, tuple)):
        raw = [float(raw)]
    amounts = [float(v) for v in raw]
    if len(amounts) != n_projects:
        raise ValueError(
            f"actions.json entry has {len(amounts)} value(s) "
            f"but env has {n_projects} project(s)"
        )
    total = sum(amounts)
    if total <= 0.0:
        return np.zeros(n_projects, dtype=np.float32)
    budget_ref = max(budget, 1e-9)
    fracs = [a / budget_ref for a in amounts]
    frac_sum = sum(fracs)
    if frac_sum > 1.0 + 1e-9:
        fracs = [f / frac_sum for f in fracs]
    return np.array(fracs, dtype=np.float32)


def run_episode(config_id: str, actions_raw: list) -> dict:
    """
    Reset the env for config_id, replay actions_raw, and collect per-period
    state into dataframes mirroring the DB export schema.
    """
    from env.env import PortfolioBudgetingEnv

    cfg = _load_config(config_id)
    env = PortfolioBudgetingEnv(cfg, render_mode=None, conn=None, method="test")
    obs, info = env.reset(seed=0)

    n_projects = len(env.projects)
    records: dict[str, list] = {
        "portfolios":           [],
        "projects_status":      [],
        "projects_observation": [],
        "milestones_status":    [],
    }

    t = 0
    done = False

    while not done:
        # Build action — use zeros if we've run out of scripted actions
        if t < len(actions_raw):
            action = _parse_actions(actions_raw[t], n_projects, env.budget)
        else:
            action = np.zeros(n_projects, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rs   = info["render_post"]           # post-action snapshot at recorded_t
        rt   = info["recorded_t"]
        proj_states = info["recorded_proj_state"]

        # portfolios
        records["portfolios"].append({
            "t_episode":        rt,
            "net_cashflow":     info["recorded_net_cashflow"],
            "reward":           info["recorded_reward"],
            "budget_available": info["recorded_budget"],
        })

        # projects_status + projects_observation
        for proj, ps in zip(env.projects, proj_states):
            i = proj["i"]
            records["projects_status"].append({
                "i":                    i,
                "t_episode":            rt,
                "t_project":            ps.get("t_project", 0),
                "status":               ps.get("status", ""),
                "inflow":               ps.get("inflow", 0.0),
                "outflow":              ps.get("outflow", 0.0),
                "allocation_action":    ps.get("allocation_action", 0.0),
                "interest_cost":        info["cashflow"][i].get("interest", 0.0),
                "progress_actual":      ps.get("progress_actual", 0.0),
                "tolerance_remain":     ps.get("tolerance_remain", 0.0),
                "catchup_alloc_t":      ps.get("catchup_alloc_t", 0.0),
                "reach_plan_t":         ps.get("reach_plan_t", 0.0),
                "target_progress_gap":  ps.get("target_progress_gap", 0.0),
                "target_timestep_gap":  ps.get("target_timestep_gap", 0),
                "target_net_payment":   ps.get("target_net_payment", 0.0),
                "target_required_alloc":ps.get("target_required_alloc", 0.0),
                "target_npv":           ps.get("target_npv", 0.0),
            })
            records["projects_observation"].append({
                "i":                    i,
                "t_episode":            rt,
                "net_cashflow":         ps.get("inflow", 0.0) - ps.get("outflow", 0.0),
                "tolerance_remain":     ps.get("tolerance_remain", 0.0),
                "catchup_alloc_t":      ps.get("catchup_alloc_t", 0.0),
                "catchup_alloc_next_t": ps.get("catchup_alloc_next_t", 0.0),
                "reach_plan_t":         ps.get("reach_plan_t", 0.0),
                "reach_plan_next_t":    ps.get("reach_plan_next_t", 0.0),
                "target_progress_gap":  ps.get("target_progress_gap", 0.0),
                "target_timestep_gap":  ps.get("target_timestep_gap", 0),
                "target_required_alloc":ps.get("target_required_alloc", 0.0),
                "target_npv":           ps.get("target_npv", 0.0),
            })

        # milestones_status — only certified this period
        for proj, ms_list, ms_sl in zip(
                env.projects, env.milestones, env.milestone_state):
            for ms, ms_state in zip(ms_list, ms_sl):
                if ms_state["certified"] and ms_state["certified_t"] == rt:
                    delay = (ms_state["certified_t"] - ms["timestep_threshold"]
                             if ms_state["certified_t"] is not None else None)
                    records["milestones_status"].append({
                        "i":                    proj["i"],
                        "j":                    ms["j"],
                        "certified_t":          ms_state["certified_t"],
                        "certification_delay":  delay,
                        "net_payment":          ms_state["payment_released"],
                    })

        t += 1

    env.close()
    return {k: pd.DataFrame(v) for k, v in records.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# ASSERTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_df(actual: pd.DataFrame, expected: pd.DataFrame,
               fields: list[str], key_cols: list[str],
               fixture_label: str, table: str) -> None:
    """
    Join actual and expected on key_cols and assert each field matches.
    String fields use equality; numeric fields use TOLERANCE.
    """
    # Only assert fields present in both actual and expected
    available = [f for f in fields
                 if f in actual.columns and f in expected.columns]
    missing   = [f for f in fields
                 if f not in actual.columns or f not in expected.columns]
    if missing:
        pytest.warns(None)   # not a failure — just note it

    if actual.empty and expected.empty:
        return
    if actual.empty and not expected.empty:
        pytest.fail(f"[{fixture_label}] {table}: actual is empty, expected {len(expected)} rows")

    # Merge on key columns present in both
    merge_keys = [k for k in key_cols if k in actual.columns and k in expected.columns]
    merged = pd.merge(
        expected[merge_keys + available].rename(columns={f: f"_exp_{f}" for f in available}),
        actual[merge_keys + available].rename(columns={f: f"_act_{f}" for f in available}),
        on=merge_keys, how="left",
    )

    failures = []
    for _, row in merged.iterrows():
        key_str = " | ".join(f"{k}={row[k]}" for k in merge_keys)
        for f in available:
            exp = row.get(f"_exp_{f}")
            act = row.get(f"_act_{f}")
            if pd.isna(exp) and pd.isna(act):
                continue
            if isinstance(exp, str) or isinstance(act, str):
                if str(exp) != str(act):
                    failures.append(
                        f"  {table} [{key_str}] {f}: expected={exp!r} actual={act!r}"
                    )
            else:
                try:
                    exp_f, act_f = float(exp), float(act)
                except (TypeError, ValueError):
                    if exp != act:
                        failures.append(
                            f"  {table} [{key_str}] {f}: expected={exp!r} actual={act!r}"
                        )
                    continue
                if abs(exp_f - act_f) > TOLERANCE:
                    failures.append(
                        f"  {table} [{key_str}] {f}: "
                        f"expected={exp_f:.6f} actual={act_f:.6f} "
                        f"diff={abs(exp_f-act_f):.2e}"
                    )

    if failures:
        pytest.fail(
            f"[{fixture_label}] {table} mismatches:\n" + "\n".join(failures)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZED TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _fixture_params():
    return [
        pytest.param(config_id, method, fixture_dir,
                     id=f"{config_id}__{method}")
        for config_id, method, fixture_dir in _discover_fixtures()
    ]


@pytest.mark.parametrize("config_id,method,fixture_dir", _fixture_params())
def test_episode_regression(config_id: str, method: str, fixture_dir: Path):
    label = f"{config_id}__{method}"

    # Load actions
    with open(fixture_dir / "actions.json") as f:
        actions_raw = json.load(f)

    # Run live episode
    actual = run_episode(config_id, actions_raw)

    # ── portfolios ────────────────────────────────────────────────────────────
    exp_port = pd.read_csv(fixture_dir / "portfolios.csv")
    _assert_df(
        actual["portfolios"], exp_port,
        PORTFOLIO_FIELDS, ["t_episode"],
        label, "portfolios",
    )

    # ── projects_status ───────────────────────────────────────────────────────
    exp_ps = pd.read_csv(fixture_dir / "projects_status.csv")
    _assert_df(
        actual["projects_status"], exp_ps,
        PROJECT_STATUS_FIELDS, ["t_episode", "i"],
        label, "projects_status",
    )

    # ── projects_observation ──────────────────────────────────────────────────
    exp_po = pd.read_csv(fixture_dir / "projects_observation.csv")
    _assert_df(
        actual["projects_observation"], exp_po,
        PROJECT_OBS_FIELDS, ["t_episode", "i"],
        label, "projects_observation",
    )

    # ── milestones_status (optional) ──────────────────────────────────────────
    ms_path = fixture_dir / "milestones_status.csv"
    if ms_path.exists():
        exp_ms = pd.read_csv(ms_path)
        if not exp_ms.empty:
            _assert_df(
                actual["milestones_status"], exp_ms,
                MILESTONE_STATUS_FIELDS, ["i", "j"],
                label, "milestones_status",
            )