# src/play/play.py
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
#   info["render_state"]     – dict passed verbatim to h.render()
#
# env.step(action) → obs, reward, terminated, truncated, info
#   info["render_post"]      – render_state for the just-recorded timestep (T)
#   info["render_pre"]       – render_state for the next timestep (T+1), absent when done
#   info["cum_reward"]       – running cumulative reward maintained by the env
#   info["prompt_hint"]      – dict: n_projects, n_active  (for the input prompt)
#
# env.prompt_hint            – same dict, available before every step()
#
# Usage
# -----
#   python src/play/play.py [--config single|dual] [--seed N] [--no-db] [--no-export]
#   python src/play/play.py --export-only [--format csv|excel|parquet]
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/play/
_SRC_DIR  = os.path.dirname(_THIS_DIR)                   # src/
_ENV_DIR  = os.path.join(_SRC_DIR, "env")
_DB_DIR   = os.path.join(_SRC_DIR, "db")
_CFGS_DIR = os.path.join(_DB_DIR,  "configs")
_RUNS_DIR = os.path.join(_THIS_DIR, "runs")

for _d in (_SRC_DIR, _ENV_DIR, _DB_DIR, _CFGS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import helper as h   # env + play infrastructure

SCHEMA_PATH = os.path.join(_DB_DIR, "schema.sql")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def _prompt(env) -> "np.ndarray | None":
    """Ask the user for an allocation; return action array or None to quit."""
    hint = env.prompt_hint   # {"n_projects": int, "n_active": int}
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
        # Delegate all action construction to the env / helper
        action = env.parse_input(raw)
        if action is not None:
            return action


# ═══════════════════════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play loop — Portfolio Budgeting RL Env")
    p.add_argument("--config",  default=None,
                   help="Config name to skip the interactive prompt.")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--no-db",   action="store_true",
                   help="Disable DB logging.")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the post-episode CSV export.")
    p.add_argument("--export-only", action="store_true",
                   help="Skip play; export the most recent run's database.")
    p.add_argument("--format",  choices=["csv", "excel", "parquet"],
                   default="csv", dest="fmt")
    p.add_argument("--out",     default=None,
                   help="Export output folder override.")
    p.add_argument("--tables",  nargs="+", default=None, metavar="TABLE")
    p.add_argument("--episode", default=None, metavar="EPISODE_ID")
    p.add_argument("--db",      default=None, metavar="DB_PATH",
                   help="Explicit database path (used with --export-only).")
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
    from env import PortfolioBudgetingEnv

    env = PortfolioBudgetingEnv(
        config=cfg, render_mode="ansi", conn=conn, method="manual",
    )

    obs, info = env.reset(seed=args.seed)

    hdr = info["episode_header"]   # episode_id, n_projects, horizon, budget, run_dir
    print(f"\n  Episode : {hdr['episode_id']}")
    print(f"  Projects: {hdr['n_projects']}   horizon={hdr['horizon']}   "
          f"budget={hdr['budget']:,.2f}")
    if hdr.get("run_dir"):
        print(f"  Run dir : {hdr['run_dir']}")

    h.render(**info["render_state"])

    # ── game loop ─────────────────────────────────────────────────────────────
    import numpy as np
    terminated = truncated = False
    while not (terminated or truncated):

        if info["n_active"] == 0:
            # No active projects -- auto-step with zeros, no prompt.
            action = np.zeros(env.prompt_hint["n_projects"], dtype=np.float32)
            print(f"  [auto] t={info['t_episode']} -- no active projects, skipping.")
            obs, reward, terminated, truncated, info = env.step(action)
            h.render(**info["render_post"])
            if not (terminated or truncated):
                h.render(**info["render_pre"])
            continue

        action = _prompt(env)
        if action is None:
            print("  Quitting.")
            break

        obs, reward, terminated, truncated, info = env.step(action)

        # Show the just-recorded timestep (T, post-allocation)
        h.render(**info["render_post"])

        # Show the next timestep (T+1, pre-allocation) if the episode continues
        if not (terminated or truncated):
            h.render(**info["render_pre"])

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
                      out_dir=args.out or os.path.join(run_dir, "exports"),
                      tables_subset=args.tables)
    else:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()