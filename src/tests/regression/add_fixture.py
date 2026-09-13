"""
tests/regression/add_fixture.py

CLI helper: copies a verified play.py exports folder into the ground_truth
directory and writes actions.json.

Usage
-----
python tests/regression/add_fixture.py \\
    --exports "src/play/runs/2026-09-12_15-20-03/exports" \\
    --config  TC-10SF-R \\
    --method  optimal \\
    --actions "[110.0]"

--actions must be valid JSON covering ONLY the active (prompted) timesteps.
  Pending periods are auto-stepped with zeros and must NOT appear in actions.
  Single-project, single active period:   "[110.0]"
  Single-project, multi active period:    "[85.0, 7.65, 0.0]"
  Multi-project, multi active period:     "[[50.0, 30.0], [20.0, 10.0]]"

Copies only the tables the harness asserts:
    portfolios.csv
    projects_status.csv
    projects_observation.csv
    milestones_status.csv   (optional — skipped if absent)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HARNESS_TABLES = [
    "portfolios.csv",
    "projects_status.csv",
    "projects_observation.csv",
    "milestones_status.csv",    # optional
]

GT_DIR = Path(__file__).parent / "ground_truth"


def main():
    parser = argparse.ArgumentParser(description="Add a regression fixture.")
    parser.add_argument("--exports", required=True,
                        help="Path to the play.py exports folder.")
    parser.add_argument("--config",  required=True,
                        help="Config ID, e.g. TC-10SF-R.")
    parser.add_argument("--method",  required=True,
                        choices=["optimal", "greedy_catchup",
                                 "greedy_reach_plan", "uniform"],
                        help="Baseline method name.")
    parser.add_argument("--actions", required=True,
                        help="JSON list of per-period allocations.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing fixture without prompting.")
    args = parser.parse_args()

    exports_dir = Path(args.exports)
    if not exports_dir.exists():
        sys.exit(f"[!] Exports folder not found: {exports_dir}")

    # Validate actions JSON
    try:
        actions = json.loads(args.actions)
    except json.JSONDecodeError as e:
        sys.exit(f"[!] --actions is not valid JSON: {e}")

    dest_dir = GT_DIR / args.config / args.method
    if dest_dir.exists() and not args.overwrite:
        ans = input(f"  Fixture {args.config}/{args.method} already exists. "
                    f"Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy tables
    copied, skipped = [], []
    for table in HARNESS_TABLES:
        src = exports_dir / table
        if src.exists():
            shutil.copy2(src, dest_dir / table)
            copied.append(table)
        else:
            skipped.append(table)

    # Write actions.json
    actions_path = dest_dir / "actions.json"
    with open(actions_path, "w") as f:
        json.dump(actions, f, indent=4)

    print(f"\n  Fixture written to: {dest_dir}")
    print(f"  Tables copied  : {', '.join(copied)}")
    if skipped:
        print(f"  Tables skipped : {', '.join(skipped)}  (not present in exports)")
    print(f"  actions.json   : {actions}")
    print(f"\n  Run: pytest tests/regression/ -k \"{args.config}__{args.method}\" -v")


if __name__ == "__main__":
    main()