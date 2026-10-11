"""
src/tests/solvers/gap_plot.py
===========
Comparative cumulative-reward plot across configs and solvers.

Expected directory layout  (identical to dashboard.py)
-------------------------------------------------------
src/tests/solvers/                      ← this script lives here (next to dashboard.py)
src/tests/solvers/ground_truth/
    <config_name>/
        <solver>/
            portfolios.csv

Output
------
src/tests/solvers/ground_truth/gap_plot.png

Usage
-----
    python gap_plot.py
    python gap_plot.py --root path/to/ground_truth
    python gap_plot.py --out  path/to/output.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── colour palette (reuse dashboard palette) ──────────────────────────────────
C = dict(
    bg   = "#f8f9fb",
    grid = "#d8dce8",
)

SOLVER_COLOURS = [
    "#4361ee",   # indigo-blue
    "#06d6a0",   # emerald
    "#f77f00",   # amber
    "#ef233c",   # red
    "#8338ec",   # purple
    "#0096c7",   # cerulean
]

MARKERS = ["o", "s", "^", "D", "v", "P"]

DPI = 120


# ── data loading ──────────────────────────────────────────────────────────────

def _cum_reward(data_dir: Path) -> float | None:
    """
    Read portfolios.csv and return the terminal cumulative reward.

    Mirrors dashboard.py's _read_csv pattern — returns None when the file
    is missing or empty.  The terminal reward is the sum of per-period
    rewards (discount^t * net_cashflow), which equals the episode NPV.
    The last row where done==1 holds this as the running sum in the
    reward column only when cum_reward is stored; otherwise we sum
    the per-period reward column ourselves, matching how dashboard.py
    reads portfolios for the reward subplot.
    """
    path = data_dir / "portfolios.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    # Prefer cum_reward column if present (added after the schema fix)
    if "cum_reward" in df.columns and df["cum_reward"].notna().any():
        terminal = df.loc[df["cum_reward"].last_valid_index(), "cum_reward"]
        return float(terminal)

    # Fall back: sum per-period reward column (same quantity)
    if "reward" not in df.columns:
        return None
    return float(df["reward"].sum())


def _collect(ground_root: Path) -> pd.DataFrame:
    """
    Walk ground_truth/<config>/<solver>/portfolios.csv and collect
    (config, solver, cum_reward) rows.
    """
    records = []
    for config_dir in sorted(p for p in ground_root.iterdir() if p.is_dir()):
        config_name = config_dir.name
        for solver_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            solver_name = solver_dir.name
            if solver_name == "figures":          # skip output dirs
                continue
            val = _cum_reward(solver_dir)
            if val is None:
                print(f"  [SKIP] {config_name}/{solver_name} — portfolios.csv missing or empty")
                continue
            records.append({
                "config": config_name,
                "solver": solver_name,
                "cum_reward": val,
            })

    return pd.DataFrame(records)


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_gap(df: pd.DataFrame, out: Path) -> None:
    _CONFIG_ORDER = ["TC-10SF-R", "TC-1LMAIF-R", "TC-10MAIF-R", "TC-3XMAIF-R"]
    present = set(df["config"].unique())
    configs = [c for c in _CONFIG_ORDER if c in present] + \
              sorted(c for c in present if c not in _CONFIG_ORDER)
    solvers = sorted(df["solver"].unique())

    x_pos   = np.arange(len(configs))
    x_ticks = configs

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 2.2), 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.grid(True, linestyle=":", color=C["grid"], alpha=0.8, zorder=0)

    for s_idx, solver in enumerate(solvers):
        colour = SOLVER_COLOURS[s_idx % len(SOLVER_COLOURS)]
        marker = MARKERS[s_idx % len(MARKERS)]

        sub = df[df["solver"] == solver].set_index("config")
        y   = [sub.loc[c, "cum_reward"] if c in sub.index else np.nan
               for c in configs]

        ax.plot(
            x_pos, y,
            color=colour,
            marker=marker,
            markersize=7,
            linewidth=1.8,
            label=solver,
            zorder=3,
        )

        # annotate each point
        for xi, yi in zip(x_pos, y):
            if not np.isnan(yi):
                ax.annotate(
                    f"{yi:.2f}",
                    xy=(xi, yi),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=colour,
                )

    # zero reference
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks, fontsize=9, rotation=15, ha="right")
    ax.set_xlabel("Config", fontsize=10)
    ax.set_ylabel("Cumulative Reward  (episode NPV)", fontsize=10)
    ax.set_title("Solver Comparison — Cumulative Reward per Config",
                 fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(title="Solver", fontsize=8, title_fontsize=8,
              loc="best", framealpha=0.9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative reward comparison across configs and solvers."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Path to the ground_truth root directory. "
             "Defaults to <script_dir>/ground_truth.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. "
             "Defaults to <ground_truth>/gap_plot.png.",
    )
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    ground_root  = Path(args.root) if args.root else script_dir / "ground_truth"

    if not ground_root.exists():
        sys.exit(f"[ERROR] ground_truth directory not found: {ground_root}")

    out_path = Path(args.out) if args.out else ground_root / "gap_plot.png"

    print(f"  root : {ground_root}")
    df = _collect(ground_root)

    if df.empty:
        sys.exit("[ERROR] No data found — check that portfolios.csv files exist under ground_truth.")

    print(f"\n  configs : {sorted(df['config'].unique())}")
    print(f"  solvers : {sorted(df['solver'].unique())}")
    print()

    plot_gap(df, out_path)
    print("\n  Done.")


if __name__ == "__main__":
    main()