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
    Render a Gymnasium compatibility certificate as a PNG figure.
    Groups results by section, shows pass/fail per check, and prints
    an overall verdict badge at the bottom.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("  [!] matplotlib not available — skipping certificate generation.")
        return

    # ── colours ───────────────────────────────────────────────────────────────
    C_BG       = "#0f1117"
    C_PANEL    = "#1a1d27"
    C_BORDER   = "#2e3250"
    C_PASS     = "#3ddc84"
    C_FAIL     = "#ff5c5c"
    C_SKIP     = "#f0a500"
    C_HEAD     = "#e8eaf6"
    C_SUB      = "#9e9eb8"
    C_ACCENT   = "#5c6bc0"
    C_BADGE_OK = "#1b5e20"
    C_BADGE_FL = "#7f0000"

    SECTION_LABELS = {
        "1": "Import & Instantiation",
        "2": "Space Declarations",
        "3": "reset() Contract",
        "4": "step() Contract",
        "5": "Determinism",
        "6": "Random Policy Rollout",
        "7": "Zero-Action Rollout",
        "8": "Greedy Rollout",
        "9": "Multi-Episode Stability",
        "10": "gymnasium check_env",
    }

    # Group by section prefix (first token before first space)
    from collections import OrderedDict
    sections: dict[str, list[tuple[str, bool]]] = OrderedDict()
    for name, ok, _ in results:
        # Section key: first word (e.g. "step()" → use section number from _results order)
        # We rely on the fact that checks are appended in section order; group by label.
        key = name
        sections.setdefault("all", []).append((name, ok))

    # Re-group by section number using insertion order of section() calls.
    # We reconstruct sections from result names by detecting which section
    # each check was registered under — use a simple ordered approach.
    sec_map: dict[str, list[tuple[str, bool]]] = OrderedDict()
    current_sec = "?"
    # Map check names back to section titles using known prefixes
    _sec_membership: dict[str, str] = {}
    _import_names  = {"PortfolioBudgetingEnv importable",
                      "Env instantiation (fallback config)",
                      "has observation_space","has action_space",
                      "has reset","has step","has close","has render"}
    _space_names   = {"observation_space dtype is float32",
                      "action_space dtype is float32",
                      "action_space low  == 0","action_space high == 1",
                      "obs shape matches observation_space",
                      "obs within observation_space bounds",
                      "action_space shape == (n_projects,)"}
    _reset_names   = {"reset returns tuple of length 2","obs is ndarray",
                      "info is dict","info has t_episode","info has budget",
                      "info has horizon","info has n_active",
                      "info has episode_header","info has render_state",
                      "t_episode == 0 after reset","budget > 0 after reset",
                      "second reset works","obs shape stable across resets"}
    _step_names    = {"step returns 5-tuple","obs2 is ndarray","reward is float",
                      "terminated is bool","truncated is bool","info2 is dict",
                      "obs2 shape matches space","obs2 within space",
                      "info2 has cum_reward","info2 has render_post"}
    _det_names     = {"same seed → identical first obs",
                      "different seeds → different obs (probabilistic)"}
    _r_names       = {"rollout completes without exception",
                      "episode ends (terminated or truncated)",
                      "all obs shapes identical","last obs within space"}
    _z_names       = {"zero-action rollout completes"}
    _g_names       = {"greedy rollout completes"}
    _m_names       = {"no exceptions across episodes",
                      "obs shape consistent across episodes"}
    _c_names       = {"check_env passed"}

    def _sec_of(name: str) -> str:
        if name in _import_names:  return "1"
        if name in _space_names:   return "2"
        if name in _reset_names:   return "3"
        if name in _step_names:    return "4"
        if name in _det_names:     return "5"
        if name in _r_names:       return "6"
        if name in _z_names or "zero" in name.lower(): return "7"
        if name in _g_names or "greedy" in name.lower(): return "8"
        if name in _m_names or "episode" in name.lower(): return "9"
        if name in _c_names:       return "10"
        return "?"

    for name, ok, _ in results:
        s = _sec_of(name)
        sec_map.setdefault(s, []).append((name, ok))

    # ── layout ────────────────────────────────────────────────────────────────
    n_sections   = len(sec_map)
    row_height   = 0.28        # inches per check row
    sec_padding  = 0.55        # inches per section header
    total_checks = len(results)
    fig_h = 2.8 + n_sections * sec_padding + total_checks * row_height + 1.4
    fig_w = 9.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C_BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # ── outer border ──────────────────────────────────────────────────────────
    border = FancyBboxPatch((0.18, 0.18), fig_w - 0.36, fig_h - 0.36,
                             boxstyle="round,pad=0.05",
                             linewidth=2, edgecolor=C_ACCENT,
                             facecolor=C_PANEL)
    ax.add_patch(border)

    y = fig_h - 0.55

    # ── title block ───────────────────────────────────────────────────────────
    ax.text(fig_w / 2, y, "Gymnasium Compatibility Certificate",
            ha="center", va="top", fontsize=17, fontweight="bold",
            color=C_HEAD, fontfamily="monospace")
    y -= 0.42
    ax.text(fig_w / 2, y, "PortfolioBudgetingEnv",
            ha="center", va="top", fontsize=13, color=C_ACCENT,
            fontfamily="monospace")
    y -= 0.35
    ax.text(fig_w / 2, y, f"Config: {config_id}",
            ha="center", va="top", fontsize=9.5, color=C_SUB,
            fontfamily="monospace")
    y -= 0.25
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ax.text(fig_w / 2, y, f"Generated: {ts}",
            ha="center", va="top", fontsize=8.5, color=C_SUB,
            fontfamily="monospace")
    y -= 0.30

    # separator
    ax.plot([0.4, fig_w - 0.4], [y, y], color=C_BORDER, lw=1.2)
    y -= 0.28

    # ── section rows ──────────────────────────────────────────────────────────
    X_NUM   = 0.55
    X_ICON  = 1.10
    X_NAME  = 1.45
    X_RIGHT = fig_w - 0.55

    for sec_key in sorted(sec_map.keys(), key=lambda k: int(k) if k.isdigit() else 99):
        checks   = sec_map[sec_key]
        sec_pass = sum(1 for _, ok in checks if ok)
        sec_tot  = len(checks)
        sec_ok   = sec_pass == sec_tot
        label    = SECTION_LABELS.get(sec_key, f"Section {sec_key}")
        ratio    = f"{sec_pass}/{sec_tot}"
        s_color  = C_PASS if sec_ok else C_FAIL

        # Section header
        ax.text(X_NUM, y, f"§{sec_key}", ha="left", va="top",
                fontsize=9, color=C_ACCENT, fontfamily="monospace",
                fontweight="bold")
        ax.text(X_NAME, y, label, ha="left", va="top",
                fontsize=10, color=C_HEAD, fontfamily="monospace",
                fontweight="bold")
        ax.text(X_RIGHT, y, ratio, ha="right", va="top",
                fontsize=9, color=s_color, fontfamily="monospace",
                fontweight="bold")
        y -= 0.30

        # Individual checks
        for name, ok in checks:
            icon  = "✓" if ok else "✗"
            color = C_PASS if ok else C_FAIL
            # truncate long names
            disp = name if len(name) <= 52 else name[:49] + "…"
            ax.text(X_ICON, y, icon, ha="left", va="top",
                    fontsize=8.5, color=color, fontfamily="monospace")
            ax.text(X_NAME, y, disp, ha="left", va="top",
                    fontsize=8, color=C_SUB, fontfamily="monospace")
            y -= row_height

        y -= 0.10   # gap after section

    # separator
    ax.plot([0.4, fig_w - 0.4], [y, y], color=C_BORDER, lw=1.2)
    y -= 0.32

    # ── overall badge ─────────────────────────────────────────────────────────
    passed_total = sum(1 for _, ok, _ in results if ok)
    failed_total = len(results) - passed_total
    all_pass     = failed_total == 0

    badge_label  = "ALL CHECKS PASSED" if all_pass else f"{failed_total} CHECK(S) FAILED"
    badge_color  = C_BADGE_OK if all_pass else C_BADGE_FL
    badge_tc     = C_PASS     if all_pass else C_FAIL
    badge_w, badge_h = 3.6, 0.50
    badge_x = (fig_w - badge_w) / 2
    badge_y = y - badge_h

    badge_rect = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                                 boxstyle="round,pad=0.06",
                                 linewidth=2, edgecolor=badge_tc,
                                 facecolor=badge_color)
    ax.add_patch(badge_rect)
    ax.text(fig_w / 2, badge_y + badge_h / 2, badge_label,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=badge_tc, fontfamily="monospace")

    y = badge_y - 0.28
    ax.text(fig_w / 2, y,
            f"{passed_total} passed  ·  {failed_total} failed  ·  {len(results)} total",
            ha="center", va="top", fontsize=8.5, color=C_SUB,
            fontfamily="monospace")

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
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