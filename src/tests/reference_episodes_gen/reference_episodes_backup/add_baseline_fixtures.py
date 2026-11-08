"""
tests/solvers/add_baseline_fixtures.py

Runs the three deterministic baseline solvers over every config in
db/configs/ and writes the results as regression fixtures.

Rather than duplicating the episode loop, this script drives the solvers
through the same infrastructure used by play.py / baseline_solvers.py:

  1. Initialise DB + run dir (same as play.py startup).
  2. For each config × solver: run the solver via PortfolioBudgetingEnv,
     log to the DB, then export via h.play_export().
  3. Copy each exported file into the ground_truth tree:
       ground_truth/<config_id>/<method>/<table>.csv

The exported files are the canonical ground truth — no bespoke harvesting.

Usage
-----
Run from src/ or any directory:

    python tests/solvers/add_baseline_fixtures.py

Options
-------
  --configs  TC-10SF-R TC-1LSF-R ...   only run these config IDs (default: all)
  --methods  catchup_t reach_t ...     only run these methods    (default: all)
  --overwrite                           overwrite existing fixtures without prompt
  --seed N                              RNG seed passed to env.reset() (default: 0)
  --no-db                               disable DB logging (export step is skipped)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
_SOLVERS_DIR = Path(__file__).parent          # src/tests/solvers
_TESTS_DIR   = _SOLVERS_DIR.parent            # src/tests
_SRC_DIR     = _TESTS_DIR.parent              # src/
_CFGS_DIR    = _SRC_DIR / "db" / "configs"
_ENV_DIR     = _SRC_DIR / "env"
_DB_DIR      = _SRC_DIR / "db"
_RUNS_DIR    = _ENV_DIR / "runs"
_GT_DIR      = _TESTS_DIR / "regression" / "ground_truth"
_SCHEMA_PATH = str(_DB_DIR / "schema.sql")

# src/env must be on the path so that env.py's own `import helper as h` resolves.
# We do NOT add src/ because that would make Python treat src/env as a package
# named "env" and shadow the bare module names inside it.
for _d in (str(_ENV_DIR), str(_DB_DIR), str(_SOLVERS_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import helper as h                            # src/env/helper.py
from baseline_solvers import ALL_SOLVERS      # src/tests/solvers/baseline_solvers.py

# Load PortfolioBudgetingEnv directly from file to avoid package/module name clash
import importlib.util as _ilu
_env_spec = _ilu.spec_from_file_location("_env_module", _ENV_DIR / "env.py")
_env_mod  = _ilu.module_from_spec(_env_spec)
_env_spec.loader.exec_module(_env_mod)
PortfolioBudgetingEnv = _env_mod.PortfolioBudgetingEnv


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
# EPISODE RUNNER — uses the real env + solver, logs to DB, exports
# ══════════════════════════════════════════════════════════════════════════════

def _run_episode(cfg: dict, solver, conn, seed: int) -> None:
    """
    Run one episode using the solver's act() method.
    Logging is handled by the env itself (via conn).
    Pending periods are auto-stepped with zeros.
    """
    env = PortfolioBudgetingEnv(
        config=cfg,
        render_mode=None,
        conn=conn,
        method=solver.METHOD_NAME,
    )
    obs, info = env.reset(seed=seed)

    n = len(env.projects)
    terminated = truncated = False

    while not (terminated or truncated):
        if info["n_active"] == 0:
            action = np.zeros(n, dtype=np.float32)
        else:
            action = solver.act(env)

        obs, reward, terminated, truncated, info = env.step(action)

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURE WRITER — copies exports → ground_truth/<config_id>/<method>/
# ══════════════════════════════════════════════════════════════════════════════

def _copy_to_fixture(exports_dir: Path, config_id: str, method: str,
                     overwrite: bool) -> bool:
    """
    Copy every file from exports_dir into ground_truth/<config_id>/<method>/.
    Returns True if files were written, False if skipped.
    """
    dest = _GT_DIR / config_id / method

    if dest.exists() and any(dest.iterdir()) and not overwrite:
        ans = input(
            f"  Fixture {config_id}/{method} exists. Overwrite? [y/N] "
        ).strip().lower()
        if ans != "y":
            print(f"  Skipped {config_id}/{method}.")
            return False

    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_file in exports_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dest / src_file.name)
            copied += 1

    if copied == 0:
        print(f"  [!] No exported files found in {exports_dir}")
        return False

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
        "--methods", nargs="+", default=[s.METHOD_NAME for s in
                                         [cls() for cls in ALL_SOLVERS]],
        choices=[cls.METHOD_NAME for cls in ALL_SOLVERS],
        metavar="METHOD",
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
    parser.add_argument(
        "--no-db", action="store_true",
        help="Disable DB logging (export/fixture step is skipped).",
    )
    args = parser.parse_args()

    cfgs = _load_configs(args.configs)
    if not cfgs:
        sys.exit("[!] No configs found.")

    method_names = args.methods
    solver_map   = {cls.METHOD_NAME: cls() for cls in ALL_SOLVERS}
    solvers      = [solver_map[m] for m in method_names]

    total   = len(cfgs) * len(solvers)
    written = 0
    skipped = 0
    failed  = 0

    print(f"\n  {len(cfgs)} config(s) × {len(solvers)} solver(s) = {total} fixture(s)\n")

    # ── one run dir + DB for the whole batch ──────────────────────────────────
    run_dir = h.play_startup(args.no_db, str(_RUNS_DIR), _SCHEMA_PATH)
    conn    = None
    db_path = None

    if run_dir is not None:
        conn = h.play_init_db(run_dir, _SCHEMA_PATH)
        if conn is None:
            print("\n  [!] DB init failed — falling back to --no-db mode.\n")
        else:
            db_path = os.path.join(run_dir, "database.db")
            h.play_seed_configs(conn, str(_CFGS_DIR))

    # ── main loop ─────────────────────────────────────────────────────────────
    for cfg in cfgs:
        config_id = cfg.get("config_id", "unknown")

        for solver in solvers:
            label = f"{config_id}__{solver.METHOD_NAME}"
            print(f"  ▶  {label} ... ", end="", flush=True)

            try:
                # -- run episode (logs to DB via env internals) ----------------
                _run_episode(cfg, solver, conn, seed=args.seed)

                if conn is None or db_path is None:
                    # No DB — nothing to export.
                    print("(no-db, skipped export)")
                    skipped += 1
                    continue

                # -- export this episode's tables to a temp folder -------------
                exports_dir = Path(run_dir) / "exports" / config_id / solver.METHOD_NAME
                exports_dir.mkdir(parents=True, exist_ok=True)

                # episode_id mirrors env.py line 78:
                #   f"{config_id}__{self.method}__{self.run_id}"
                # run_id defaults to "" so the id is fully deterministic.
                episode_id = f"{config_id}__{solver.METHOD_NAME}__"
                h.play_export(
                    db_path,
                    fmt="csv",
                    out_dir=str(exports_dir),
                    tables_subset=None,
                    episode_id=episode_id,
                )

                # -- copy exports → ground_truth -------------------------------
                ok = _copy_to_fixture(exports_dir, config_id,
                                      solver.METHOD_NAME, overwrite=args.overwrite)
                if ok:
                    written += 1
                    print("✓")
                else:
                    skipped += 1

            except Exception as exc:
                failed += 1
                print(f"✗  {exc}")

    # ── cleanup ───────────────────────────────────────────────────────────────
    if conn is not None:
        conn.close()

    print()
    print("═" * 60)
    print(f"  written : {written}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")
    if run_dir:
        print(f"  run dir : {run_dir}")
    print("═" * 60)


if __name__ == "__main__":
    main()