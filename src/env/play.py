# src/env/play.py
#
# Minimal interactive play loop for PortfolioBudgetingEnv.
#
# All infrastructure (DB, config, export) lives in helper.py.
# This file owns only: argument parsing, the prompt loop, and top-level wiring.
#
# Contract with the env
# ─────────────────────
# env.reset()  → obs, info
#   info["episode_header"]   – dict: episode_id, n_projects, horizon, budget, run_dir
#   info["render_state"]     – pre-action state at t=0
#
# env.step(action) → obs, reward, terminated, truncated, info
#   info["render_post"]      – post-action state for the just-completed period T
#   info["render_pre"]       – pre-action state for the next period T+1 (absent when done)
#   info["recorded_reward"]  – per-step reward for period T
#   info["cum_reward"]       – running cumulative reward
#
# env.prompt_hint            – dict: n_projects, n_active
#
# Display contract
# ────────────────
# Each screen shows exactly ONE timestep.
# After step:  show render_post  (what just happened — reward, post-alloc obs)
# Before next prompt: show render_pre (what the agent sees before acting)
# These are two separate screens; the user presses Enter to advance between them.
#
# Usage
# -----
#   python src/env/play.py [--config NAME] [--seed N] [--no-db] [--no-export]
#   python src/env/play.py --export-only [--format csv|excel|parquet]
#
# Controls
# --------
#   Enter                  → equal split across active projects
#   comma-separated floats → explicit amounts (normalised to budget fractions)
#   q / quit / exit        → quit

from __future__ import annotations

import argparse
import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/env/
_SRC_DIR  = os.path.dirname(_THIS_DIR)                   # src/
_DB_DIR   = os.path.join(_SRC_DIR, "db")
_CFGS_DIR = os.path.join(_DB_DIR,  "configs")
_RUNS_DIR = os.path.join(_DB_DIR,  "runs_record")

for _d in (_SRC_DIR, _THIS_DIR, _DB_DIR, _CFGS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import helper as h

SCHEMA_PATH = os.path.join(_DB_DIR, "schema.sql")


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

_OBS_FIELDS = [
    ("tolerance_remain",     "catchup_alloc_t"),
    ("catchup_alloc_next_t", "target_progress_gap"),
    ("target_timestep_gap",  "target_required_alloc"),
    ("target_npv",           None),
]

_LW = 22   # label width
_VW = 12   # value width


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4f}"
    return str(v)


def _obs_row(ll, lv, rl="", rv="") -> str:
    left  = f"  {ll:<{_LW}} {lv:>{_VW}}"
    right = f"    {rl:<{_LW}} {rv:>{_VW}}" if rl else ""
    return left + right


def render_obs(t: int, horizon: int,
               projects: list[dict], proj_state: list[dict],
               step_reward: float, cum_reward: float,
               label: str = "",
               **_) -> None:
    """Clear screen and print one compact obs block per project."""
    print("\033[2J\033[H", end="", flush=True)

    # ── timestep header ───────────────────────────────────────────────────────
    tag = f"  [{label}]" if label else ""
    print(f"  t = {t} / {horizon}"
          f"    step_reward = {step_reward:+.4f}"
          f"    cum_reward = {cum_reward:+.4f}"
          + (f"    {label}" if label else ""))
    print()

    # ── one block per project (dynamic) ───────────────────────────────────────
    for proj, ps in zip(projects, proj_state):
        status = ps.get("status", "pending").upper()
        print(f"  ── Project {proj['i']}  [{status}] {'─' * 40}")
        for ll, rl in _OBS_FIELDS:
            lv = _fmt(ps.get(ll))
            rv = _fmt(ps.get(rl)) if rl else ""
            print(_obs_row(ll, lv, rl or "", rv))
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def _prompt(env) -> "np.ndarray | None":
    hint = env.prompt_hint
    n    = hint["n_projects"]
    prompt = (
        f"  Allocate {n} comma-separated amounts"
        f" (Enter = equal split, q = quit): "
        if n > 1
        else "  Allocate amount (Enter = all-in, q = quit): "
    )
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        action = env.parse_input(raw)
        if action is not None:
            return action


# ═══════════════════════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play loop — Portfolio Budgeting RL Env")
    p.add_argument("--config",    default=None)
    p.add_argument("--seed",      type=int, default=None)
    p.add_argument("--no-db",     action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--format",    choices=["csv", "excel", "parquet"],
                   default="csv", dest="fmt")
    p.add_argument("--out",       default=None)
    p.add_argument("--tables",    nargs="+", default=None, metavar="TABLE")
    p.add_argument("--episode",   default=None, metavar="EPISODE_ID")
    p.add_argument("--db",        default=None, metavar="DB_PATH")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── export-only ───────────────────────────────────────────────────────────
    if args.export_only:
        if args.db:
            db_path = args.db
        else:
            if not os.path.isdir(_RUNS_DIR):
                print("  [!] No runs folder found. Run a play session first.")
                sys.exit(1)
            runs = sorted(os.listdir(_RUNS_DIR))
            if not runs:
                print("  [!] No runs in", _RUNS_DIR)
                sys.exit(1)
            db_path = os.path.join(_RUNS_DIR, runs[-1], "database.db")
        h.play_export(db_path, fmt=args.fmt, out_dir=args.out,
                      tables_subset=args.tables, episode_id=args.episode)
        return

    # ── startup + DB ──────────────────────────────────────────────────────────
    run_dir = h.play_startup(args.no_db, _RUNS_DIR, SCHEMA_PATH)

    conn    = None
    db_path = None
    if run_dir is not None:
        conn = h.play_init_db(run_dir, SCHEMA_PATH)
        if conn is None:
            print("\n  [!] Falling back to --no-db mode.\n")
        else:
            db_path = os.path.join(run_dir, "database.db")
            h.play_seed_configs(conn, _CFGS_DIR)

    # ── config selection ──────────────────────────────────────────────────────
    if args.config:
        cfg = h.play_load_config_by_name(args.config, _CFGS_DIR)
        if cfg is None:
            if conn: conn.close()
            sys.exit(1)
    elif conn is not None:
        config_id = h.play_prompt_config(conn)
        if config_id is None:
            print("  Exiting.")
            conn.close()
            sys.exit(0)
        cfg = h.play_load_config_by_id(config_id, _CFGS_DIR)
        if cfg is None:
            conn.close()
            sys.exit(1)
    else:
        print("  No DB and no --config; cannot select a configuration.")
        sys.exit(1)

    # ── build env ─────────────────────────────────────────────────────────────
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "env_module", os.path.join(_THIS_DIR, "env.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PortfolioBudgetingEnv = _mod.PortfolioBudgetingEnv

    env = PortfolioBudgetingEnv(
        config=cfg, render_mode="ansi", conn=conn, method="manual",
    )

    obs, info = env.reset(seed=args.seed)

    hdr = info["episode_header"]
    print(f"\n  Episode : {hdr['episode_id']}")
    print(f"  Projects: {hdr['n_projects']}   horizon={hdr['horizon']}   "
          f"budget={hdr['budget']:,.2f}")
    if hdr.get("run_dir"):
        print(f"  Run dir : {hdr['run_dir']}")

    input("\n  Press Enter to start episode...")

    # ── game loop ─────────────────────────────────────────────────────────────
    # Flow per timestep:
    #   1. render pre-action obs  (what the agent sees before acting)
    #   2. prompt for action      (or auto-step if no active projects)
    #   3. env.step()
    #   4. render post-action obs (result: reward, updated obs)
    #   5. user presses Enter to continue to next timestep
    # ─────────────────────────────────────────────────────────────────────────
    import numpy as np
    terminated = truncated = False

    # Seed the pre-action state with t=0 from reset
    pre_rs = info["render_state"]

    while not (terminated or truncated):

        # ── 1. render pre-action state ────────────────────────────────────────
        render_obs(
            t=pre_rs["t"], horizon=pre_rs["horizon"],
            projects=pre_rs["projects"], proj_state=pre_rs["proj_state"],
            step_reward=0.0, cum_reward=pre_rs.get("cum_reward", 0.0),
            label="PRE",
        )

        # ── 2+3. prompt (or auto) + step ──────────────────────────────────────
        if info["n_active"] == 0:
            action = np.zeros(env.prompt_hint["n_projects"], dtype=np.float32)
            print(f"  [auto] no active projects at t={pre_rs['t']}, stepping with zeros.")
        else:
            action = _prompt(env)
            if action is None:
                print("  Quitting.")
                break

        obs, reward, terminated, truncated, info = env.step(action)

        # ── 4. render post-action state ───────────────────────────────────────
        post_rs = info["render_post"]
        render_obs(
            t=post_rs["t"], horizon=post_rs["horizon"],
            projects=post_rs["projects"], proj_state=post_rs["proj_state"],
            step_reward=info["recorded_reward"],
            cum_reward=info["cum_reward"],
            label="POST",
        )

        if terminated or truncated:
            break

        # ── 5. wait for Enter, then advance to next pre-action state ──────────
        input("  Press Enter for next period...")
        pre_rs = info["render_pre"]

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if terminated: print("  Episode terminated.")
    if truncated:  print("  Episode truncated (horizon reached).")
    print(f"  Cumulative reward : {info['cum_reward']:+.4f}")
    print(f"  Episode ID        : {hdr['episode_id']}")
    if run_dir:    print(f"  Run folder        : {run_dir}")
    print("═" * 60)

    env.close()

    # ── export ────────────────────────────────────────────────────────────────
    if conn is not None and not args.no_export and db_path is not None:
        conn.close()
        h.play_export(db_path, fmt=args.fmt,
                      out_dir=args.out or run_dir,
                      tables_subset=args.tables)
    else:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()