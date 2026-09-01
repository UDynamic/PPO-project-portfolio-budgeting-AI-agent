# src/db/export.py
#
# Export all tables from database.db to CSV files.
#
# Usage
# -----
#   python src/db/export.py                          # exports to src/db/exports/
#   python src/db/export.py --out path/to/folder     # custom output folder
#   python src/db/export.py --tables portfolios projects_status
#   python src/db/export.py --episode <episode_id>   # filter one episode
#   python src/db/export.py --format csv             # default
#   python src/db/export.py --format excel           # one .xlsx, one sheet per table
#   python src/db/export.py --format parquet         # one .parquet per table

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "database.db")

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export database.db tables.")
    p.add_argument(
        "--out", default=os.path.join(_DIR, "exports"),
        help="Output folder (created if missing). Default: src/db/exports/",
    )
    p.add_argument(
        "--tables", nargs="+", default=None,
        metavar="TABLE",
        help="Tables to export. Default: all tables.",
    )
    p.add_argument(
        "--episode", default=None,
        metavar="EPISODE_ID",
        help="Filter rows to a single episode_id where applicable.",
    )
    p.add_argument(
        "--format", choices=["csv", "excel", "parquet"],
        default="csv", dest="fmt",
        help="Output format. Default: csv.",
    )
    return p.parse_args()


def _load_table(conn: sqlite3.Connection,
                table: str,
                episode_id: str | None) -> "pd.DataFrame":
    import pandas as pd

    # Check if table has episode_id column
    cols = [
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    ]
    if episode_id and "episode_id" in cols:
        df = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE episode_id = ?",
            conn, params=(episode_id,),
        )
    else:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    return df


def export_csv(tables: dict[str, "pd.DataFrame"], out_dir: str) -> None:
    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name:<30} → {path}  ({len(df)} rows)")


def export_excel(tables: dict[str, "pd.DataFrame"], out_dir: str) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  [!] openpyxl required for Excel export: pip install openpyxl")
        sys.exit(1)

    path = os.path.join(out_dir, "database_export.xlsx")
    with __import__("pandas").ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            # Excel sheet names max 31 chars
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"  {name:<30} → sheet '{sheet}'  ({len(df)} rows)")
    print(f"\n  Saved: {path}")


def export_parquet(tables: dict[str, "pd.DataFrame"], out_dir: str) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("  [!] pyarrow required for Parquet export: pip install pyarrow")
        sys.exit(1)

    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"  {name:<30} → {path}  ({len(df)} rows)")


def main() -> None:
    args = _parse_args()

    if not os.path.exists(DB_PATH):
        print(f"  [!] Database not found: {DB_PATH}")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("  [!] pandas required: pip install pandas")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    tables_to_export = args.tables or ALL_TABLES

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify requested tables exist
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing = [t for t in tables_to_export if t not in existing]
    if missing:
        print(f"  [!] Tables not found in DB: {', '.join(missing)}")
        tables_to_export = [t for t in tables_to_export if t in existing]

    if not tables_to_export:
        print("  [!] No tables to export.")
        conn.close()
        sys.exit(1)

    print(f"\n  Exporting {len(tables_to_export)} table(s)")
    if args.episode:
        print(f"  Filter: episode_id = {args.episode}")
    print(f"  Format: {args.fmt}")
    print(f"  Output: {args.out}")
    print()

    data: dict[str, pd.DataFrame] = {}
    for table in tables_to_export:
        try:
            df = _load_table(conn, table, args.episode)
            data[table] = df
        except Exception as exc:
            print(f"  [!] Failed to load {table}: {exc}")

    conn.close()

    if args.fmt == "csv":
        export_csv(data, args.out)
    elif args.fmt == "excel":
        export_excel(data, args.out)
    elif args.fmt == "parquet":
        export_parquet(data, args.out)

    print("\n  Done.")


if __name__ == "__main__":
    main()