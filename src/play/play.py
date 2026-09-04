# src\play\play.py
#
# Self-contained manual play loop for PortfolioBudgetingEnv.
# Replaces: db_init.py  export.py  (those files can be deleted)
#
# Layout produced per run
# -----------------------
#   src/play/runs/<timestamp>/
#       database.db
#       exports/
#           portfolios.csv
#           projects_status.csv
#           ...
#
# Usage
# -----
#   python src/play/play.py [--config single|dual] [--seed N] [--no-db] [--no-export]
#   python src/play/play.py --export-only [--format csv|excel|parquet] [--out path]
#   python src/play/play.py --export-only --episode <episode_id>
#
# Controls (during play)
# ----------------------
#   Enter                   → equal split across active projects
#   q / quit / exit         → quit
#   Comma-separated floats  → explicit amounts (normalised to budget fractions)

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
# play.py lives at  <repo>/src/play/play.py
#   _THIS_DIR  →  <repo>/src/play/
#   _SRC_DIR   →  <repo>/src/
#   _ENV_DIR   →  <repo>/src/env/
#   _DB_DIR    →  <repo>/src/db/
#   _CFGS_DIR  →  <repo>/src/db/configs/    ← configs live here
#   _RUNS_DIR  →  <repo>/src/play/runs/     ← one timestamped folder per run
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/play/
_SRC_DIR  = os.path.dirname(_THIS_DIR)                   # src/
_ENV_DIR  = os.path.join(_SRC_DIR, "env")
_DB_DIR   = os.path.join(_SRC_DIR, "db")
_CFGS_DIR = os.path.join(_DB_DIR,  "configs")            # src/db/configs/
_RUNS_DIR = os.path.join(_THIS_DIR, "runs")              # src/play/runs/

for _d in (_SRC_DIR, _ENV_DIR, _DB_DIR, _CFGS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# STATUS REPORTER
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str, width: int = 60) -> None:
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✓  {label}{suffix}")

def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ⚠  {label}{suffix}")

def _fail(label: str, detail: str = "") -> None:
    suffix = f"  — {detail}" if detail else ""
    print(f"  ✗  {label}{suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN FOLDER  (created once per invocation, paths derived from it)
# ══════════════════════════════════════════════════════════════════════════════

def _make_run_dir() -> str:
    """Create and return src/play/runs/<timestamp>/"""
    stamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(_RUNS_DIR, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# FIXED PATHS  (schema lives in src/db/; everything else goes into the run dir)
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_PATH = os.path.join(_DB_DIR, "schema.sql")

ALL_TABLES = [
    "environment_config",
    "projects_profile",
    "milestones_profile",
    "portfolios",
    "projects_status",
    "projects_observation",
    "portfolio_observation",
    "milestones_status",
    "training_log",
]

EXPECTED_TABLES = ALL_TABLES


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def _startup(no_db: bool) -> "str | None":
    """Create the run folder and report. Returns run_dir (or None if --no-db)."""
    _banner("STARTUP")

    if no_db:
        _warn("DB disabled", "--no-db flag set; skipping run folder")
        return None

    run_dir = _make_run_dir()
    _ok("Run folder created", run_dir)
    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_db(run_dir: str) -> "sqlite3.Connection | None":
    """Create database.db inside run_dir and apply schema."""
    if not os.path.exists(SCHEMA_PATH):
        _fail("schema.sql not found", SCHEMA_PATH)
        return None

    db_path = os.path.join(run_dir, "database.db")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
        _ok("Database created & schema applied", db_path)
        return conn
    except Exception as exc:
        _fail("Database init failed", str(exc))
        return None


def _verify_db(conn: "sqlite3.Connection") -> bool:
    """Print table verification. Returns True if all expected tables present."""
    rows  = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    found = {row["name"] for row in rows}

    all_ok = True
    for table in EXPECTED_TABLES:
        if table in found:
            _ok(f"table: {table}")
        else:
            _fail(f"table: {table}", "MISSING")
            all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG SEEDING
# ══════════════════════════════════════════════════════════════════════════════

def _seed_all_configs(conn: "sqlite3.Connection") -> None:
    """Seed every known config into the DB upfront so the selection prompt has rows."""
    _SEEDS = [
        ("single_project", "single"),
        ("dual_project",   "dual"),
    ]
    for module_name, label in _SEEDS:
        try:
            mod = __import__(module_name)
            mod.seed(conn)
            _ok(f"Config seeded", label)
        except ModuleNotFoundError:
            pass                          # config file doesn't exist — skip silently
        except Exception as exc:
            _warn(f"Seed failed for '{label}'", str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _load_table(conn: sqlite3.Connection,
                table: str,
                episode_id: "str | None") -> "pd.DataFrame":
    import pandas as pd
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if episode_id and "episode_id" in cols:
        return pd.read_sql_query(
            f"SELECT * FROM {table} WHERE episode_id = ?", conn, params=(episode_id,)
        )
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _export_csv(tables: dict, out_dir: str) -> None:
    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        _ok(f"{name:<30} → {os.path.basename(path)}", f"{len(df)} rows")


def _export_excel(tables: dict, out_dir: str) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        _fail("openpyxl missing", "pip install openpyxl")
        return

    import pandas as pd
    path = os.path.join(out_dir, "database_export.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            _ok(f"{name:<30} → sheet '{sheet}'", f"{len(df)} rows")
    print(f"\n  Saved: {path}")


def _export_parquet(tables: dict, out_dir: str) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        _fail("pyarrow missing", "pip install pyarrow")
        return

    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        _ok(f"{name:<30} → {os.path.basename(path)}", f"{len(df)} rows")


def run_export(db_path: str,
               fmt: str = "csv",
               out_dir: "str | None" = None,
               tables_subset: "list[str] | None" = None,
               episode_id: "str | None" = None) -> None:
    """Export tables from db_path into out_dir (defaults to exports/ beside the db)."""
    _banner("EXPORT")

    if not os.path.exists(db_path):
        _fail("Database not found", db_path)
        return

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(db_path), "exports")

    try:
        import pandas as pd
    except ImportError:
        _fail("pandas missing", "pip install pandas")
        return

    os.makedirs(out_dir, exist_ok=True)
    tables_to_export = tables_subset or ALL_TABLES

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing = [t for t in tables_to_export if t not in existing]
    if missing:
        _warn("Tables missing in DB", ", ".join(missing))
        tables_to_export = [t for t in tables_to_export if t in existing]

    if not tables_to_export:
        _fail("No tables to export")
        conn.close()
        return

    print(f"  Format  : {fmt}")
    print(f"  Output  : {out_dir}")
    if episode_id:
        print(f"  Filter  : episode_id = {episode_id}")
    print(f"  Tables  : {len(tables_to_export)}")
    print()

    data: dict = {}
    for table in tables_to_export:
        try:
            data[table] = _load_table(conn, table, episode_id)
        except Exception as exc:
            _warn(f"Failed to load {table}", str(exc))

    conn.close()

    if fmt == "csv":
        _export_csv(data, out_dir)
    elif fmt == "excel":
        _export_excel(data, out_dir)
    elif fmt == "parquet":
        _export_parquet(data, out_dir)

    print(f"\n  Done. Files written to: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _list_db_configs(conn: sqlite3.Connection) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT config_id, config_name, created_at "
            "FROM environment_config ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _prompt_config_selection(conn: sqlite3.Connection) -> "str | None":
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


def _load_config_by_id(config_id: str) -> "dict | None":
    if "SINGLE" in config_id.upper():
        from single_project import CONFIG
        return CONFIG
    if "DUAL" in config_id.upper():
        from dual_project import CONFIG
        return CONFIG
    print(f"  [!] Cannot resolve config_id {config_id!r}.")
    return None


def _load_config_by_name(name: str) -> "dict | None":
    n = name.strip().lower()
    if n == "single":
        from single_project import CONFIG
        return CONFIG
    if n == "dual":
        from dual_project import CONFIG
        return CONFIG
    print(f"  [!] Unknown config name {name!r}. Use 'single' or 'dual'.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_action(raw: str, n: int, budget: float) -> "np.ndarray | None":
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


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _render(env, cum_reward: float, net_cashflow: float = 0.0) -> None:
    import helper as rnd
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


# ══════════════════════════════════════════════════════════════════════════════
# ACTION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _prompt(env) -> "np.ndarray | None":
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


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified play loop — Portfolio Budgeting RL Env"
    )
    p.add_argument("--config", default=None,
                   help="Config name ('single' or 'dual') to skip the DB prompt.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-db", action="store_true",
                   help="Disable DB logging.")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the post-episode CSV export.")
    p.add_argument("--export-only", action="store_true",
                   help="Skip play; export the most recent run's database.")
    p.add_argument("--format", choices=["csv", "excel", "parquet"],
                   default="csv", dest="fmt",
                   help="Export format (default: csv).")
    p.add_argument("--out", default=None,
                   help="Export output folder override.")
    p.add_argument("--tables", nargs="+", default=None, metavar="TABLE",
                   help="Tables to export (default: all).")
    p.add_argument("--episode", default=None, metavar="EPISODE_ID",
                   help="Filter export to one episode_id.")
    p.add_argument("--db", default=None, metavar="DB_PATH",
                   help="Explicit database path (used with --export-only).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── export-only mode ──────────────────────────────────────────────────────
    if args.export_only:
        # Resolve which db to export: explicit --db, or latest run folder
        if args.db:
            db_path = args.db
        else:
            if not os.path.isdir(_RUNS_DIR):
                print("  [!] No runs folder found. Run a play session first.")
                sys.exit(1)
            runs = sorted(os.listdir(_RUNS_DIR))
            if not runs:
                print("  [!] No runs found in", _RUNS_DIR)
                sys.exit(1)
            db_path = os.path.join(_RUNS_DIR, runs[-1], "database.db")

        run_export(db_path=db_path, fmt=args.fmt,
                   out_dir=args.out, tables_subset=args.tables,
                   episode_id=args.episode)
        return

    # ── STARTUP — create timestamped run folder ────────────────────────────────
    run_dir = _startup(args.no_db)

    # ── DB INIT ───────────────────────────────────────────────────────────────
    conn: "sqlite3.Connection | None" = None
    db_path: "str | None" = None

    if run_dir is not None:
        _banner("DATABASE INIT")
        conn = _init_db(run_dir)
        if conn is None:
            print("\n  [!] Falling back to --no-db mode.\n")
        else:
            db_path = os.path.join(run_dir, "database.db")
            print()
            all_ok = _verify_db(conn)
            if not all_ok:
                print("\n  [!] Schema verification failed. Check schema.sql.")
            print()
            _seed_all_configs(conn)   # seed before prompt so rows exist to display

    # ── CONFIG SELECTION ──────────────────────────────────────────────────────
    _banner("CONFIG")

    if args.config:
        cfg = _load_config_by_name(args.config)
        if cfg is None:
            if conn:
                conn.close()
            sys.exit(1)
    elif conn is not None:
        config_id = _prompt_config_selection(conn)
        if config_id is None:
            print("  Exiting.")
            conn.close()
            sys.exit(0)
        cfg = _load_config_by_id(config_id)
        if cfg is None:
            conn.close()
            sys.exit(1)
    else:
        print("  No DB and no --config; defaulting to 'single'.")
        from single_project import CONFIG
        cfg = CONFIG

    # ── BUILD ENV ─────────────────────────────────────────────────────────────
    _banner("ENVIRONMENT")

    from env import PortfolioBudgetingEnv

    env = PortfolioBudgetingEnv(
        config      = cfg,
        render_mode = "ansi",
        conn        = conn,
        method      = "manual",
    )

    obs, info  = env.reset(seed=args.seed)
    cum_reward = 0.0

    n_projects = len(env.projects)
    _ok(f"Env ready", f"{n_projects} project(s), horizon={env.horizon}, budget={env.budget:,.2f}")
    _ok(f"Episode ID", env.episode_id)
    _ok(f"Run folder", run_dir or "none")
    print()

    # ── INITIAL RENDER ────────────────────────────────────────────────────────
    _render(env, cum_reward, net_cashflow=0.0)

    terminated = False
    truncated  = False

    # ── GAME LOOP ─────────────────────────────────────────────────────────────
    while not (terminated or truncated):
        action = _prompt(env)
        if action is None:
            print("  Quitting.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

        net_cashflow = info.get("period_inflow", 0.0) - info.get("period_outflow", 0.0)
        _render(env, cum_reward, net_cashflow=net_cashflow)

    # ── EPISODE SUMMARY ───────────────────────────────────────────────────────
    print()
    print("═" * 60)
    if terminated:
        print("  Episode terminated (all projects done).")
    if truncated:
        print("  Episode truncated (horizon reached).")
    print(f"  Cumulative reward : {cum_reward:+.4f}")
    print(f"  Episode ID        : {env.episode_id}")
    if run_dir:
        print(f"  Run folder        : {run_dir}")
    print("═" * 60)

    env.close()

    # ── POST-EPISODE EXPORT ───────────────────────────────────────────────────
    if conn is not None and not args.no_export:
        conn.close()
        conn = None
        exports_dir = args.out or os.path.join(run_dir, "exports")
        run_export(db_path=db_path, fmt=args.fmt,
                   out_dir=exports_dir, tables_subset=args.tables,
                   episode_id=None)
    else:
        if conn is not None:
            conn.close()
        if args.no_export:
            print("\n  [export skipped — --no-export flag set]")


if __name__ == "__main__":
    main()