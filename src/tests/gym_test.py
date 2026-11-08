# src/tests/gym_test.py
#
# Gymnasium compatibility test for PortfolioBudgetingEnv.
#
# Covers:
#   1. Import & instantiation  (class loadable, all Gym attributes present)
#   2. Space declarations      (observation_space, action_space shapes/dtypes/bounds)
#   3. reset() API contract    (return types, info keys, t=0, budget>0)
#   4. step() API contract     (5-tuple types, obs within space, info keys)
#   5. Determinism             (same seed → identical obs; different seeds → different obs)
#   6. Full rollout            (random policy, terminates, obs shape stable)
#   7. Zero-action rollout     (all zeros every step)
#   8. Greedy rollout          (all budget → first active project)
#   9. Multi-episode stability (shape consistency across N episodes)
#  10. gymnasium.utils.env_checker  (official Gym checker, if available)
#
# Usage
# -----
#   python gym_test.py                        # uses built-in fallback config TC-3XMAIF-R
#   python gym_test.py --config NAME          # loads src/db/configs/NAME.json
#   python gym_test.py --episodes 10          # more multi-episode stress
#   python gym_test.py --cert path/out.png    # custom certificate path
#   python gym_test.py --no-cert              # skip certificate generation

from __future__ import annotations

import argparse
import datetime
import os
import sys
import traceback
from typing import Any

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
# File lives at src/tests/gym_test.py
#   _THIS_DIR  →  src/tests/
#   _SRC_DIR   →  src/
#   _ENV_DIR   →  src/env/      (where env.py and helper.py live)
#   _DB_DIR    →  src/db/
#   _CFGS_DIR  →  src/db/configs/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/tests/
_SRC_DIR  = os.path.dirname(_THIS_DIR)                   # src/
_ENV_DIR  = os.path.join(_SRC_DIR, "env")                # src/env/
_DB_DIR   = os.path.join(_SRC_DIR, "db")                 # src/db/
_CFGS_DIR = os.path.join(_DB_DIR,  "configs")            # src/db/configs/

for _d in (_SRC_DIR, _ENV_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── fallback config: TC-3XMAIF-R ─────────────────────────────────────────────
# Note: projects 1 and 2 have planned_start at t=6 and t=7, so at any seed
# the initial obs for those two projects is all-zeros (pending). The different-
# seed determinism check therefore uses _DIFF_SEED_CONFIG (TC-10MAIF-R) which
# has planned_start=0 and a uniform efficiency distribution whose draws vary
# with the RNG seed.
_FALLBACK_CONFIG = {
    # ── identity ──────────────────────────────────────────────────────────────
    "config_id"               : "TC-3XMAIF-R",
    "config_name"             : "TC-3XMAIF-R",

    # ── portfolio ─────────────────────────────────────────────────────────────
    "n_projects_dist"         : "fixed",
    "n_projects_p1"           : 3.0,
    "n_projects_p2"           : None,
    "n_projects_p3"           : None,
    "n_projects_p4"           : None,

    "budget_tightness_dist"   : "fixed",
    "budget_tightness_p1"     : 3.0,
    "budget_tightness_p2"     : None,
    "budget_tightness_p3"     : None,
    "budget_tightness_p4"     : None,

    "discount_dist"           : "fixed",
    "discount_p1"             : 0.95,
    "discount_p2"             : None,
    "discount_p3"             : None,
    "discount_p4"             : None,

    # ── project ───────────────────────────────────────────────────────────────
    "bac_dist"                : "custom_ordinal",
    "bac_p1"                  : [100, 200, 300],
    "bac_p2"                  : None,
    "bac_p3"                  : None,
    "bac_p4"                  : None,

    "profit_percent_dist"     : "fixed",
    "profit_percent_p1"       : 0.2,
    "profit_percent_p2"       : None,
    "profit_percent_p3"       : None,
    "profit_percent_p4"       : None,

    "planned_start_dist"      : "custom_ordinal",
    "planned_start_p1"        : [0, 6, 7],
    "planned_start_p2"        : None,
    "planned_start_p3"        : None,
    "planned_start_p4"        : None,

    "planned_duration_dist"   : "fixed",
    "planned_duration_p1"     : 3.0,
    "planned_duration_p2"     : None,
    "planned_duration_p3"     : None,
    "planned_duration_p4"     : None,

    "scurve_a_dist"           : "fixed",
    "scurve_a_p1"             : 1.5,
    "scurve_a_p2"             : None,
    "scurve_a_p3"             : None,
    "scurve_a_p4"             : None,

    "scurve_b_dist"           : "fixed",
    "scurve_b_p1"             : 3.0,
    "scurve_b_p2"             : None,
    "scurve_b_p3"             : None,
    "scurve_b_p4"             : None,

    "advance_percent_dist"    : "fixed",
    "advance_percent_p1"      : 0.2,
    "advance_percent_p2"      : None,
    "advance_percent_p3"      : None,
    "advance_percent_p4"      : None,

    "advance_trigger_dist"    : "fixed",
    "advance_trigger_p1"      : 0.0,
    "advance_trigger_p2"      : None,
    "advance_trigger_p3"      : None,
    "advance_trigger_p4"      : None,

    "advance_recovery_dist"   : "fixed",
    "advance_recovery_p1"     : 0.2,
    "advance_recovery_p2"     : None,
    "advance_recovery_p3"     : None,
    "advance_recovery_p4"     : None,

    "retention_rate_dist"     : "fixed",
    "retention_rate_p1"       : 0.05,
    "retention_rate_p2"       : None,
    "retention_rate_p3"       : None,
    "retention_rate_p4"       : None,

    "progress_delay_cap_dist" : "fixed",
    "progress_delay_cap_p1"   : 0.15,
    "progress_delay_cap_p2"   : None,
    "progress_delay_cap_p3"   : None,
    "progress_delay_cap_p4"   : None,

    "finish_delay_cap_dist"   : "fixed",
    "finish_delay_cap_p1"     : 2.0,
    "finish_delay_cap_p2"     : None,
    "finish_delay_cap_p3"     : None,
    "finish_delay_cap_p4"     : None,

    "termination_tolerance_dist": "fixed",
    "termination_tolerance_p1"  : 2.0,
    "termination_tolerance_p2"  : None,
    "termination_tolerance_p3"  : None,
    "termination_tolerance_p4"  : None,

    "efficiency_dist"         : "custom_ordinal",
    "efficiency_p1"           : [
        [0.91, 1.17, 0.83, 0.91, 1.17, 0.83],
        [1.09, 0.84, 1.18, 1.09, 0.84, 1.18],
        [0.97, 1.13, 0.86, 0.97, 1.13, 0.86],
    ],
    "efficiency_p2"           : None,
    "efficiency_p3"           : None,
    "efficiency_p4"           : None,

    # ── milestone ─────────────────────────────────────────────────────────────
    "n_milestones_dist"       : "fixed",
    "n_milestones_p1"         : 3.0,
    "n_milestones_p2"         : None,
    "n_milestones_p3"         : None,
    "n_milestones_p4"         : None,

    "progress_threshold_dist" : "custom_ordinal",
    "progress_threshold_p1"   : [[0.4, 0.97, 1.0], [0.4, 0.97, 1.0], [0.4, 0.97, 1.0]],
    "progress_threshold_p2"   : None,
    "progress_threshold_p3"   : None,
    "progress_threshold_p4"   : None,

    "payment_weight_dist"     : "custom_ordinal",
    "payment_weight_p1"       : [[0.3, 0.7, 1.0], [0.3, 0.7, 1.0], [0.3, 0.7, 1.0]],
    "payment_weight_p2"       : None,
    "payment_weight_p3"       : None,
    "payment_weight_p4"       : None,

    "timestep_threshold_dist" : "custom_ordinal",
    "timestep_threshold_p1"   : [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    "timestep_threshold_p2"   : None,
    "timestep_threshold_p3"   : None,
    "timestep_threshold_p4"   : None,

    # ── global ────────────────────────────────────────────────────────────────
    "annual_interest_rate"    : 0.12,
}


# ── TC-10MAIF-R — used only for the different-seed determinism check ──────────
# planned_start=0 → project active at reset → first step draws efficiency via
# rng.uniform(p1, p2), so different seeds produce different efficiency values
# and therefore different obs after one step.
_DIFF_SEED_CONFIG = {
    "config_id"               : "TC-10MAIF-R",
    "config_name"             : "TC-10MAIF-R",
    "n_projects_dist"         : "fixed",     "n_projects_p1"           : 1.0,
    "n_projects_p2"           : None,        "n_projects_p3"           : None,        "n_projects_p4"           : None,
    "budget_tightness_dist"   : "fixed",     "budget_tightness_p1"     : 3.0,
    "budget_tightness_p2"     : None,        "budget_tightness_p3"     : None,        "budget_tightness_p4"     : None,
    "discount_dist"           : "fixed",     "discount_p1"             : 0.95,
    "discount_p2"             : None,        "discount_p3"             : None,        "discount_p4"             : None,
    "bac_dist"                : "fixed",     "bac_p1"                  : 100.0,
    "bac_p2"                  : None,        "bac_p3"                  : None,        "bac_p4"                  : None,
    "profit_percent_dist"     : "fixed",     "profit_percent_p1"       : 0.2,
    "profit_percent_p2"       : None,        "profit_percent_p3"       : None,        "profit_percent_p4"       : None,
    "planned_start_dist"      : "fixed",     "planned_start_p1"        : 0.0,
    "planned_start_p2"        : None,        "planned_start_p3"        : None,        "planned_start_p4"        : None,
    "planned_duration_dist"   : "fixed",     "planned_duration_p1"     : 3.0,
    "planned_duration_p2"     : None,        "planned_duration_p3"     : None,        "planned_duration_p4"     : None,
    "scurve_a_dist"           : "fixed",     "scurve_a_p1"             : 1.5,
    "scurve_a_p2"             : None,        "scurve_a_p3"             : None,        "scurve_a_p4"             : None,
    "scurve_b_dist"           : "fixed",     "scurve_b_p1"             : 3.0,
    "scurve_b_p2"             : None,        "scurve_b_p3"             : None,        "scurve_b_p4"             : None,
    "advance_percent_dist"    : "fixed",     "advance_percent_p1"      : 0.2,
    "advance_percent_p2"      : None,        "advance_percent_p3"      : None,        "advance_percent_p4"      : None,
    "advance_trigger_dist"    : "fixed",     "advance_trigger_p1"      : 0.0,
    "advance_trigger_p2"      : None,        "advance_trigger_p3"      : None,        "advance_trigger_p4"      : None,
    "advance_recovery_dist"   : "fixed",     "advance_recovery_p1"     : 0.2,
    "advance_recovery_p2"     : None,        "advance_recovery_p3"     : None,        "advance_recovery_p4"     : None,
    "retention_rate_dist"     : "fixed",     "retention_rate_p1"       : 0.05,
    "retention_rate_p2"       : None,        "retention_rate_p3"       : None,        "retention_rate_p4"       : None,
    "progress_delay_cap_dist" : "fixed",     "progress_delay_cap_p1"   : 0.15,
    "progress_delay_cap_p2"   : None,        "progress_delay_cap_p3"   : None,        "progress_delay_cap_p4"   : None,
    "finish_delay_cap_dist"   : "fixed",     "finish_delay_cap_p1"     : 2.0,
    "finish_delay_cap_p2"     : None,        "finish_delay_cap_p3"     : None,        "finish_delay_cap_p4"     : None,
    "termination_tolerance_dist": "fixed",   "termination_tolerance_p1": 2.0,
    "termination_tolerance_p2": None,        "termination_tolerance_p3": None,        "termination_tolerance_p4": None,
    "efficiency_dist"         : "uniform",
    "efficiency_p1"           : 0.6,
    "efficiency_p2"           : 1.4,
    "efficiency_p3"           : None,        "efficiency_p4"           : None,
    "n_milestones_dist"       : "fixed",     "n_milestones_p1"         : 3.0,
    "n_milestones_p2"         : None,        "n_milestones_p3"         : None,        "n_milestones_p4"         : None,
    "progress_threshold_dist" : "custom_ordinal",
    "progress_threshold_p1"   : [0.4, 0.97, 1.0],
    "progress_threshold_p2"   : None,        "progress_threshold_p3"   : None,        "progress_threshold_p4"   : None,
    "payment_weight_dist"     : "custom_ordinal",
    "payment_weight_p1"       : [0.3, 0.7, 1.0],
    "payment_weight_p2"       : None,        "payment_weight_p3"       : None,        "payment_weight_p4"       : None,
    "timestep_threshold_dist" : "custom_ordinal",
    "timestep_threshold_p1"   : [1, 2, 3],
    "timestep_threshold_p2"   : None,        "timestep_threshold_p3"   : None,        "timestep_threshold_p4"   : None,
    "annual_interest_rate"    : 0.12,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = PASS if cond else FAIL
    _results.append((name, cond, detail))
    mark = "  ✓" if cond else "  ✗"
    line = f"{mark}  [{tag}]  {name}"
    if detail and not cond:
        line += f"\n         {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _load_env_class():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "env_module", os.path.join(_ENV_DIR, "env.py")
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PortfolioBudgetingEnv


def _load_config(name: str | None) -> dict:
    if name is None:
        return _FALLBACK_CONFIG
    try:
        import helper as h
        cfg_path = os.path.join(_CFGS_DIR, f"{name}.json")
        if not os.path.exists(cfg_path):
            print(f"  [!] Config '{name}' not found at {cfg_path}, using fallback.")
            return _FALLBACK_CONFIG
        import json
        with open(cfg_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Could not load config '{name}': {e}, using fallback.")
        return _FALLBACK_CONFIG


def _random_action(env) -> np.ndarray:
    return env.action_space.sample()


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _full_first_active_action(env) -> np.ndarray:
    """Put all budget on the first active project."""
    n   = env.action_space.shape[0]
    act = np.zeros(n, dtype=np.float32)
    for i, ps in enumerate(env.proj_state):
        if ps["status"] == "active":
            act[i] = 1.0
            break
    return act


def _rollout(env, policy_fn, max_steps: int = 500) -> dict:
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps        = 0
    obs_shapes   = [obs.shape]

    for _ in range(max_steps):
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1
        obs_shapes.append(obs.shape)
        if terminated or truncated:
            break

    return {
        "steps":        steps,
        "total_reward": total_reward,
        "terminated":   terminated,
        "truncated":    truncated,
        "obs_shapes":   obs_shapes,
        "last_obs":     obs,
        "last_info":    info,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_import(Env) -> None:
    section("1. Import & instantiation")
    check("PortfolioBudgetingEnv importable", Env is not None)

    try:
        env = Env(config=_FALLBACK_CONFIG, render_mode=None)
        check("Env instantiation (fallback config)", True)
    except Exception as e:
        check("Env instantiation (fallback config)", False, str(e))
        return

    check("has observation_space", hasattr(env, "observation_space"))
    check("has action_space",      hasattr(env, "action_space"))
    check("has reset",             callable(getattr(env, "reset", None)))
    check("has step",              callable(getattr(env, "step", None)))
    check("has close",             callable(getattr(env, "close", None)))
    check("has render",            callable(getattr(env, "render", None)))


def test_spaces(Env, cfg: dict) -> None:
    section("2. Space declarations")
    env = Env(config=cfg, render_mode=None)
    obs, _ = env.reset(seed=0)

    check("observation_space dtype is float32",
          env.observation_space.dtype == np.float32)
    check("action_space dtype is float32",
          env.action_space.dtype == np.float32)
    check("action_space low  == 0",
          np.all(env.action_space.low  == 0.0))
    check("action_space high == 1",
          np.all(env.action_space.high == 1.0))
    check("obs shape matches observation_space",
          obs.shape == env.observation_space.shape,
          f"obs.shape={obs.shape}  space.shape={env.observation_space.shape}")
    check("obs within observation_space bounds",
          env.observation_space.contains(obs),
          f"obs min={obs.min():.4f}  max={obs.max():.4f}")
    check("action_space shape == (n_projects,)",
          env.action_space.shape == (len(env.projects),),
          f"shape={env.action_space.shape}  n_projects={len(env.projects)}")


def test_reset(Env, cfg: dict) -> None:
    section("3. reset() API contract")
    env = Env(config=cfg, render_mode=None)

    obs, info = env.reset(seed=7)
    check("reset returns tuple of length 2", True)
    check("obs is ndarray",     isinstance(obs, np.ndarray))
    check("info is dict",       isinstance(info, dict))
    check("info has t_episode", "t_episode" in info)
    check("info has budget",    "budget"    in info)
    check("info has horizon",   "horizon"   in info)
    check("info has n_active",  "n_active"  in info)
    check("info has episode_header",  "episode_header"  in info)
    check("info has render_state",    "render_state"    in info)
    check("t_episode == 0 after reset", info["t_episode"] == 0,
          f"t_episode={info['t_episode']}")
    check("budget > 0 after reset",    info["budget"] > 0,
          f"budget={info['budget']}")

    # Second reset without seed
    obs2, info2 = env.reset()
    check("second reset works", True)
    check("obs shape stable across resets",
          obs.shape == obs2.shape,
          f"{obs.shape} vs {obs2.shape}")


def test_step(Env, cfg: dict) -> None:
    section("4. step() API contract")
    env = Env(config=cfg, render_mode=None)
    obs, info = env.reset(seed=13)

    action = _random_action(env)
    result = env.step(action)
    check("step returns 5-tuple", len(result) == 5)

    obs2, reward, terminated, truncated, info2 = result
    check("obs2 is ndarray",          isinstance(obs2, np.ndarray))
    check("reward is float",          isinstance(reward, float))
    check("terminated is bool",       isinstance(terminated, bool))
    check("truncated is bool",        isinstance(truncated, bool))
    check("info2 is dict",            isinstance(info2, dict))
    check("obs2 shape matches space",
          obs2.shape == env.observation_space.shape,
          f"obs2.shape={obs2.shape}  space={env.observation_space.shape}")
    check("obs2 within space",
          env.observation_space.contains(obs2),
          f"min={obs2.min():.4f}  max={obs2.max():.4f}")
    check("info2 has cum_reward",     "cum_reward"    in info2)
    check("info2 has render_post",    "render_post"   in info2)


def test_determinism(Env, cfg: dict) -> None:
    section("5. Determinism (same seed)")

    # Same-seed: works with any config — only checks that two resets with the
    # same seed produce bit-identical results regardless of what those obs are.
    env = Env(config=cfg, render_mode=None)
    obs_a, _ = env.reset(seed=99)
    obs_b, _ = env.reset(seed=99)
    check("same seed → identical first obs",
          np.allclose(obs_a, obs_b),
          f"max_diff={np.abs(obs_a - obs_b).max():.6f}")

    # Different-seed: step() returns a PRE-allocation obs (built before the
    # efficiency draw in the late phase), so the returned obs is seed-invariant
    # no matter what action is used.  The efficiency draw's footprint appears in
    # info["render_post"]["proj_state"], which records progress_actual after the
    # allocation is applied.  We compare progress_actual from render_post across
    # two seeds: different eta values → different progress_actual values.
    # TC-10MAIF-R: planned_start=0, uniform efficiency distribution.
    _small = np.array([0.1], dtype=np.float32)
    env2 = Env(config=_DIFF_SEED_CONFIG, render_mode=None)
    env2.reset(seed=99)
    _, _, _, _, info_c = env2.step(_small)
    env2.reset(seed=100)
    _, _, _, _, info_d = env2.step(_small)
    prog_c = info_c["render_post"]["proj_state"][0]["progress_actual"]
    prog_d = info_d["render_post"]["proj_state"][0]["progress_actual"]
    different = not np.isclose(prog_c, prog_d)
    check("different seeds → different obs (probabilistic)",
          different,
          f"progress_actual: seed99={prog_c:.6f}  seed100={prog_d:.6f}  "
          f"(config: {_DIFF_SEED_CONFIG['config_id']}, after 1 step)")


def test_full_rollout_random(Env, cfg: dict) -> None:
    section("6. Full rollout — random policy")
    env = Env(config=cfg, render_mode=None)
    try:
        r = _rollout(env, _random_action, max_steps=2000)
        check("rollout completes without exception", True)
        check("episode ends (terminated or truncated)",
              r["terminated"] or r["truncated"],
              f"steps={r['steps']}")
        check("all obs shapes identical",
              len(set(r["obs_shapes"])) == 1,
              str(set(r["obs_shapes"])))
        check("last obs within space",
              env.observation_space.contains(r["last_obs"]),
              f"min={r['last_obs'].min():.4f}  max={r['last_obs'].max():.4f}")
        print(f"         steps={r['steps']}  "
              f"total_reward={r['total_reward']:+.4f}  "
              f"terminated={r['terminated']}  truncated={r['truncated']}")
    except Exception as e:
        check("rollout completes without exception", False, traceback.format_exc())


def test_full_rollout_zero(Env, cfg: dict) -> None:
    section("7. Full rollout — zero actions")
    env = Env(config=cfg, render_mode=None)
    try:
        r = _rollout(env, _zero_action, max_steps=2000)
        check("zero-action rollout completes", True)
        check("episode ends (terminated or truncated)",
              r["terminated"] or r["truncated"],
              f"steps={r['steps']}")
        print(f"         steps={r['steps']}  "
              f"total_reward={r['total_reward']:+.4f}")
    except Exception as e:
        check("zero-action rollout completes", False, traceback.format_exc())


def test_full_rollout_greedy(Env, cfg: dict) -> None:
    section("8. Full rollout — greedy (all budget → first active)")
    env = Env(config=cfg, render_mode=None)
    try:
        r = _rollout(env, _full_first_active_action, max_steps=2000)
        check("greedy rollout completes", True)
        check("episode ends (terminated or truncated)",
              r["terminated"] or r["truncated"],
              f"steps={r['steps']}")
        print(f"         steps={r['steps']}  "
              f"total_reward={r['total_reward']:+.4f}")
    except Exception as e:
        check("greedy rollout completes", False, traceback.format_exc())


def test_multi_episode(Env, cfg: dict, n: int = 5) -> None:
    section(f"9. Multi-episode stability ({n} episodes)")
    env = Env(config=cfg, render_mode=None)
    obs_shapes = set()
    act_shapes = set()
    failed     = False

    for ep in range(n):
        try:
            obs, _ = env.reset(seed=ep)
            obs_shapes.add(obs.shape)
            act_shapes.add(env.action_space.shape)
            for _ in range(200):
                action = _random_action(env)
                obs, _, terminated, truncated, _ = env.step(action)
                obs_shapes.add(obs.shape)
                if terminated or truncated:
                    break
        except Exception as e:
            check(f"episode {ep} ok", False, traceback.format_exc())
            failed = True
            break

    if not failed:
        check("no exceptions across episodes", True)
        check("obs shape consistent across episodes",
              len(obs_shapes) == 1, str(obs_shapes))


def test_env_checker(Env, cfg: dict) -> None:
    section("10. gymnasium.utils.env_checker")
    try:
        from gymnasium.utils.env_checker import check_env
        env = Env(config=cfg, render_mode=None)
        try:
            check_env(env, warn=True, skip_render_check=True)
            check("check_env passed", True)
        except Exception as e:
            check("check_env passed", False, str(e))
    except ImportError:
        print(f"  [{SKIP}]  gymnasium.utils.env_checker not available — skipping")


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICATE FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_certificate(results: list[tuple[str, bool, str]],
                         config_id: str,
                         out_path: str) -> None:
    """
    Render a Gymnasium compatibility certificate as a portrait A4 PNG.

    Layout: 6-column table — two side-by-side check panels.
    Each panel: §  |  Check Name  |  Result
    Checks fill the left panel first, then the right panel, halving row count.
    Section header rows span the full table width.
    Designed for clean printing on white A4 paper.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
        from collections import OrderedDict
    except ImportError:
        print("  [!] matplotlib not available — skipping certificate generation.")
        return

    # ── palette (print-friendly, light background) ────────────────────────────
    C_PAGE        = "#FFFFFF"   # page / outer background
    C_HEADER_BG   = "#1A237E"   # deep indigo — title banner + col headers
    C_HEADER_FG   = "#FFFFFF"
    C_SEC_BG      = "#E8EAF6"   # lavender tint — section header rows
    C_SEC_FG      = "#1A237E"
    C_ROW_ODD     = "#FFFFFF"   # alternating row fill
    C_ROW_EVEN    = "#F5F5F5"
    C_GRID        = "#C5CAE9"   # column dividers & outer border
    C_PASS_BG     = "#E8F5E9"   # soft green cell background
    C_PASS_FG     = "#2E7D32"
    C_FAIL_BG     = "#FFEBEE"   # soft red cell background
    C_FAIL_FG     = "#C62828"
    C_BODY_FG     = "#212121"
    C_MUTED       = "#757575"
    C_ACCENT      = "#3949AB"   # indigo accent (logos, rule lines)
    C_BADGE_PASS  = "#2E7D32"
    C_BADGE_FAIL  = "#C62828"
    C_BADGE_FG    = "#FFFFFF"
    C_GYM_STRIPE  = "#3949AB"   # thin gymnasium-brand stripe

    SECTION_LABELS = {
        "1":  "Import & Instantiation",
        "2":  "Space Declarations",
        "3":  "reset() Contract",
        "4":  "step() Contract",
        "5":  "Determinism",
        "6":  "Random Policy Rollout",
        "7":  "Zero-Action Rollout",
        "8":  "Greedy Rollout",
        "9":  "Multi-Episode Stability",
        "10": "gymnasium.utils.env_checker",
    }

    # ── section membership map ────────────────────────────────────────────────
    _import_names = {"PortfolioBudgetingEnv importable",
                     "Env instantiation (fallback config)",
                     "has observation_space", "has action_space",
                     "has reset", "has step", "has close", "has render"}
    _space_names  = {"observation_space dtype is float32",
                     "action_space dtype is float32",
                     "action_space low  == 0", "action_space high == 1",
                     "obs shape matches observation_space",
                     "obs within observation_space bounds",
                     "action_space shape == (n_projects,)"}
    _reset_names  = {"reset returns tuple of length 2", "obs is ndarray",
                     "info is dict", "info has t_episode", "info has budget",
                     "info has horizon", "info has n_active",
                     "info has episode_header", "info has render_state",
                     "t_episode == 0 after reset", "budget > 0 after reset",
                     "second reset works", "obs shape stable across resets"}
    _step_names   = {"step returns 5-tuple", "obs2 is ndarray", "reward is float",
                     "terminated is bool", "truncated is bool", "info2 is dict",
                     "obs2 shape matches space", "obs2 within space",
                     "info2 has cum_reward", "info2 has render_post"}
    _det_names    = {"same seed → identical first obs",
                     "different seeds → different obs (probabilistic)"}
    _r_names      = {"rollout completes without exception",
                     "episode ends (terminated or truncated)",
                     "all obs shapes identical", "last obs within space"}
    _z_names      = {"zero-action rollout completes"}
    _g_names      = {"greedy rollout completes"}
    _m_names      = {"no exceptions across episodes",
                     "obs shape consistent across episodes"}
    _c_names      = {"check_env passed"}

    def _sec_of(name: str) -> str:
        if name in _import_names: return "1"
        if name in _space_names:  return "2"
        if name in _reset_names:  return "3"
        if name in _step_names:   return "4"
        if name in _det_names:    return "5"
        if name in _r_names:      return "6"
        if name in _z_names or "zero"   in name.lower(): return "7"
        if name in _g_names or "greedy" in name.lower(): return "8"
        if name in _m_names or "episode" in name.lower(): return "9"
        if name in _c_names:      return "10"
        return "?"

    sec_map: dict[str, list[tuple[str, bool]]] = OrderedDict()
    for name, ok, _ in results:
        sec_map.setdefault(_sec_of(name), []).append((name, ok))

    # ── flatten into table rows ───────────────────────────────────────────────
    # Each logical row is either a section-header or a check entry.
    # We build a flat list first, then split it into LEFT and RIGHT panels so
    # checks sit side-by-side (6 visible columns = 3 per panel).
    # Section headers always span the full width (they are not split).
    #
    # Flat item types:
    #   {"kind": "sec",   "key": str, "label": str, "pass": int, "total": int}
    #   {"kind": "check", "sec": str, "name": str,  "ok": bool}

    flat: list[dict] = []
    for sec_key in sorted(sec_map.keys(), key=lambda k: int(k) if k.isdigit() else 99):
        checks   = sec_map[sec_key]
        sec_pass = sum(1 for _, ok in checks if ok)
        flat.append({"kind": "sec", "key": sec_key,
                     "label": SECTION_LABELS.get(sec_key, f"Section {sec_key}"),
                     "pass": sec_pass, "total": len(checks)})
        for name, ok in checks:
            flat.append({"kind": "check", "sec": sec_key, "name": name, "ok": ok})

    # ── geometry ─────────────────────────────────────────────────────────────
    # Portrait A4, generous margins.
    A4_W, A4_H = 8.27, 11.69
    MARGIN      = 0.50
    GUTTER      = 0.18          # gap between left and right panel
    TABLE_W     = A4_W - 2 * MARGIN
    PANEL_W     = (TABLE_W - GUTTER) / 2

    # Per-panel column fractions: [§-num | check-name | result-pill]
    CF          = [0.11, 0.68, 0.21]
    CW          = [f * PANEL_W for f in CF]   # col widths within one panel

    ROW_H       = 0.200   # data row height (inches)
    SEC_H       = 0.240   # section header row height
    COL_HDR_H   = 0.285   # column-header band height
    PAD         = 0.07    # inner cell horizontal padding

    # Build the actual row sequence for the left panel:
    # section headers expand to full width and consume one row each.
    # check rows pair up left/right.
    # We interleave: for each section header → one full-width row;
    # then its checks go into pairs.
    # Collect (left_item, right_item | None) pairs for check rows,
    # and section-header rows separately.

    display_rows: list[dict] = []   # each entry: "sec" or "pair"
    i = 0
    while i < len(flat):
        item = flat[i]
        if item["kind"] == "sec":
            display_rows.append({"kind": "sec", "item": item})
            i += 1
        else:
            left  = item
            right = flat[i + 1] if (i + 1 < len(flat) and flat[i + 1]["kind"] == "check") else None
            display_rows.append({"kind": "pair", "left": left, "right": right})
            i += 2 if right else 1

    # Compute figure height from row count
    n_sec_rows   = sum(1 for r in display_rows if r["kind"] == "sec")
    n_pair_rows  = sum(1 for r in display_rows if r["kind"] == "pair")
    content_h    = n_sec_rows * SEC_H + n_pair_rows * ROW_H + COL_HDR_H
    TITLE_H      = 1.65
    FOOTER_H     = 1.35
    fig_h        = TITLE_H + content_h + FOOTER_H + 2 * MARGIN

    fig = plt.figure(figsize=(A4_W, fig_h), facecolor=C_PAGE)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor(C_PAGE)

    # ── helpers ───────────────────────────────────────────────────────────────
    def rect(x, y, w, h, fc, ec=None, lw=0.5, zorder=1):
        ax.add_patch(Rectangle((x, y - h), w, h,
                               facecolor=fc,
                               edgecolor=ec or fc,
                               linewidth=lw, zorder=zorder))

    def hline(x0, x1, y, lw=0.4, color=C_GRID, z=3):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=z)

    def vline(x, y0, y1, lw=0.4, color=C_GRID, z=3):
        ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=z)

    def pill(cx, cy, label, fc, tc, row_h):
        pw, ph = 0.50, row_h * 0.60
        p = FancyBboxPatch((cx - pw / 2, cy - ph / 2), pw, ph,
                           boxstyle="round,pad=0.02",
                           facecolor=fc, edgecolor=tc,
                           linewidth=0.6, zorder=4)
        ax.add_patch(p)
        ax.text(cx, cy, label,
                ha="center", va="center",
                fontsize=6, fontweight="bold",
                color=tc, fontfamily="monospace", zorder=5)

    # Panel left-edge x positions
    PX = [MARGIN, MARGIN + PANEL_W + GUTTER]   # left panel x, right panel x

    # Column x positions within each panel (absolute)
    def col_x(panel: int, col: int) -> float:
        return PX[panel] + sum(CW[:col])

    def col_cx(panel: int, col: int) -> float:
        return col_x(panel, col) + CW[col] / 2

    # ── brand stripe (top) ────────────────────────────────────────────────────
    STRIPE_H = 0.050
    rect(0, fig_h, A4_W, STRIPE_H, C_GYM_STRIPE)

    # ── title banner ──────────────────────────────────────────────────────────
    BANNER_H = 0.90
    banner_y = fig_h - STRIPE_H
    rect(0, banner_y, A4_W, BANNER_H, C_HEADER_BG)

    ax.text(MARGIN, banner_y - 0.13,
            "GYMNASIUM", ha="left", va="top",
            fontsize=7, color="#7986CB",
            fontfamily="monospace", fontweight="bold", fontstyle="italic")

    ax.text(A4_W / 2, banner_y - 0.16,
            "Compatibility Certificate",
            ha="center", va="top", fontsize=17,
            fontweight="bold", color=C_HEADER_FG)

    ax.text(A4_W / 2, banner_y - 0.46,
            "PortfolioBudgetingEnv",
            ha="center", va="top", fontsize=10.5, color="#9FA8DA")

    # rule with comfortable gap above meta line
    rule_y = banner_y - 0.70
    ax.plot([MARGIN, A4_W - MARGIN], [rule_y, rule_y],
            color="#3F51B5", lw=0.7)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    ax.text(A4_W / 2, rule_y - 0.07,
            f"Config: {config_id}    ·    Generated: {ts}",
            ha="center", va="top", fontsize=7,
            color="#7986CB", fontfamily="monospace")

    # ── table top / bottom boundaries ────────────────────────────────────────
    table_top    = banner_y - BANNER_H
    table_bottom = table_top - COL_HDR_H - content_h

    # Outer border for each panel
    for px in PX:
        ax.add_patch(FancyBboxPatch(
            (px, table_bottom), PANEL_W, table_top - table_bottom,
            boxstyle="square,pad=0",
            linewidth=1.0, edgecolor=C_GRID, facecolor="none", zorder=6))

    # ── column header band ────────────────────────────────────────────────────
    cy = table_top
    for px in PX:
        rect(px, cy, PANEL_W, COL_HDR_H, C_HEADER_BG)

    col_hdr_labels  = ["§", "Check", "Result"]
    col_hdr_aligns  = ["center", "left", "center"]
    for panel in range(2):
        for c, (lbl, aln) in enumerate(zip(col_hdr_labels, col_hdr_aligns)):
            if aln == "center":
                ax.text(col_cx(panel, c), cy - COL_HDR_H / 2, lbl,
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color=C_HEADER_FG,
                        fontfamily="monospace")
            else:
                ax.text(col_x(panel, c) + PAD, cy - COL_HDR_H / 2, lbl,
                        ha="left", va="center", fontsize=7.5,
                        fontweight="bold", color=C_HEADER_FG,
                        fontfamily="monospace")

    cy -= COL_HDR_H

    # ── draw rows ─────────────────────────────────────────────────────────────
    passed_total = sum(1 for _, ok, _ in results if ok)
    failed_total = len(results) - passed_total
    pair_idx     = 0

    def draw_check_cell(panel: int, cy: float, item: dict, row_h: float,
                        bg: str) -> None:
        """Draw a single check cell (3 columns) inside the given panel."""
        px = PX[panel]
        rect(px, cy, PANEL_W, row_h, bg, ec=C_GRID, lw=0.3)

        # § col: section key
        ax.text(col_cx(panel, 0), cy - row_h / 2,
                item["sec"],
                ha="center", va="center",
                fontsize=6, color=C_MUTED,
                fontfamily="monospace")

        # Check name col
        max_ch = 38
        disp = item["name"] if len(item["name"]) <= max_ch \
               else item["name"][:max_ch - 1] + "…"
        ax.text(col_x(panel, 1) + PAD, cy - row_h / 2,
                disp, ha="left", va="center",
                fontsize=6.2, color=C_BODY_FG)

        # Result pill
        ok = item["ok"]
        pill(col_cx(panel, 2), cy - row_h / 2,
             "PASS" if ok else "FAIL",
             C_PASS_BG if ok else C_FAIL_BG,
             C_PASS_FG if ok else C_FAIL_FG,
             row_h)

        # inner column dividers
        vline(col_x(panel, 1), cy - row_h, cy, lw=0.3)
        vline(col_x(panel, 2), cy - row_h, cy, lw=0.3)

    for row in display_rows:

        if row["kind"] == "sec":
            item = row["item"]
            # Full-width section header spanning both panels + gutter
            full_w = 2 * PANEL_W + GUTTER
            rect(MARGIN, cy, full_w, SEC_H, C_SEC_BG, ec=C_GRID, lw=0.4)
            rect(MARGIN, cy, 0.035, SEC_H, C_ACCENT, zorder=3)

            # § number (left)
            ax.text(MARGIN + 0.06 + PAD, cy - SEC_H / 2,
                    item["key"],
                    ha="left", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=C_SEC_FG, fontfamily="monospace")

            # section label
            ax.text(MARGIN + CW[0] + PAD + 0.035, cy - SEC_H / 2,
                    item["label"],
                    ha="left", va="center",
                    fontsize=8, fontweight="bold", color=C_SEC_FG)

            # pass ratio (right)
            ratio_color = C_PASS_FG if item["pass"] == item["total"] else C_FAIL_FG
            ax.text(MARGIN + full_w - PAD, cy - SEC_H / 2,
                    f"{item['pass']} / {item['total']}",
                    ha="right", va="center",
                    fontsize=7, fontweight="bold",
                    color=ratio_color, fontfamily="monospace")

            hline(MARGIN, MARGIN + full_w, cy - SEC_H, lw=0.5)
            cy -= SEC_H

        else:  # "pair"
            bg       = C_ROW_ODD if pair_idx % 2 == 0 else C_ROW_EVEN
            is_last  = (row is display_rows[-1])

            draw_check_cell(0, cy, row["left"],  ROW_H, bg)

            if row["right"] is not None:
                draw_check_cell(1, cy, row["right"], ROW_H, bg)
            else:
                # Empty right cell — fill with background only
                rect(PX[1], cy, PANEL_W, ROW_H, bg, ec=C_GRID, lw=0.3)

            # Draw the row separator only between rows, never after the last one
            if not is_last:
                hline(MARGIN, MARGIN + 2 * PANEL_W + GUTTER, cy - ROW_H, lw=0.3)
            cy -= ROW_H
            pair_idx += 1

    # Draw full-height vertical dividers for each panel
    for panel in range(2):
        px = PX[panel]
        vline(col_x(panel, 1), table_bottom, table_top, lw=0.4)
        vline(col_x(panel, 2), table_bottom, table_top, lw=0.4)

    # ── footer ────────────────────────────────────────────────────────────────
    all_pass  = failed_total == 0
    footer_y  = cy - 0.55   # breathing room below the table

    badge_label = "ALL CHECKS PASSED" if all_pass else f"{failed_total}  CHECK(S)  FAILED"
    badge_fc    = C_BADGE_PASS if all_pass else C_BADGE_FAIL
    bw, bh      = 2.70, 0.36
    bx          = A4_W / 2 - bw / 2
    ax.add_patch(FancyBboxPatch(
        (bx, footer_y - bh), bw, bh,
        boxstyle="round,pad=0.05",
        facecolor=badge_fc, edgecolor=badge_fc,
        linewidth=0, zorder=2))
    ax.text(A4_W / 2, footer_y - bh / 2,
            badge_label, ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=C_BADGE_FG, fontfamily="monospace", zorder=3)

    ax.text(A4_W / 2, footer_y - bh - 0.22,
            f"{passed_total} passed  ·  {failed_total} failed  ·  {len(results)} total checks",
            ha="center", va="top",
            fontsize=7.5, color=C_MUTED, fontfamily="monospace")

    ax.text(A4_W / 2, footer_y - bh - 0.46,
            "Verified against the Gymnasium API standard  ·  gymnasium.farama.org",
            ha="center", va="top",
            fontsize=7, color=C_MUTED, fontstyle="italic")

    # bottom brand stripe
    rect(0, STRIPE_H, A4_W, STRIPE_H, C_GYM_STRIPE)

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C_PAGE, edgecolor="none")
    plt.close(fig)
    print(f"\n  Certificate saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gymnasium compatibility tests for PortfolioBudgetingEnv")
    p.add_argument("--config",   default=None,  help="Config name (without .json)")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes for multi-episode test")
    p.add_argument("--cert",     default=None, metavar="PATH",
                   help="Save a compatibility certificate PNG to PATH "
                        "(default: gym_certificate.png next to this file)")
    p.add_argument("--no-cert",  action="store_true",
                   help="Skip certificate generation entirely")
    return p.parse_args()


def main() -> None:
    args  = _parse_args()

    print("\n" + "═" * 60)
    print("  PortfolioBudgetingEnv — Gymnasium Compatibility Tests")
    print("═" * 60)

    try:
        Env = _load_env_class()
    except Exception as e:
        print(f"\n  [FATAL] Could not load env.py:\n  {e}")
        traceback.print_exc()
        sys.exit(1)

    cfg = _load_config(args.config)
    print(f"\n  config : {cfg.get('config_id', 'fallback')}")
    print(f"  seed   : {args.seed}")

    test_import(Env)
    test_spaces(Env, cfg)
    test_reset(Env, cfg)
    test_step(Env, cfg)
    test_determinism(Env, cfg)
    test_full_rollout_random(Env, cfg)
    test_full_rollout_zero(Env, cfg)
    test_full_rollout_greedy(Env, cfg)
    test_multi_episode(Env, cfg, n=args.episodes)
    test_env_checker(Env, cfg)

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    failed = total - passed

    print("\n" + "═" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"   ({failed} FAILED)")
    else:
        print("   — all clear")
    print("═" * 60 + "\n")

    if not args.no_cert:
        cert_path = args.cert or os.path.join(_THIS_DIR, "gym_certificate.png")
        generate_certificate(_results, cfg.get("config_id", "unknown"), cert_path)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()