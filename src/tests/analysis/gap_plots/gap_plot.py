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

Output  (gap_plots/ beside ground_truth/)
------
gap_plot[_<theme>].png           — annotated  (original labels, may overlap)
gap_plot[_<theme>]_table.png     — table below plot instead of inline labels
gap_plot[_<theme>]_clean.png     — overlap-resolved labels

Themes: (none)=default  A=Graphite&Neon  B=Arctic Blueprint
        C=Ivory&Saffron  D=Amber&Slate  E=Print-Ready

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


# ── style themes ──────────────────────────────────────────────────────────────
#
#  Each theme mirrors the 3xdashboard theme set exactly:
#    A  Graphite & Neon   — dark charcoal, electric solver colours
#    B  Arctic Blueprint  — ice-white / deep navy, blueprint feel
#    C  Ivory & Saffron   — warm cream, editorial palette
#    D  Amber & Slate     — golden canvas, plum/teal/rust accents
#    E  Print-Ready       — white bg, black ink, hatch/dash differentiation
#
#  Keys per theme:
#    bg, panel_bg, grid, spine, tick_fg, label_fg, title_fg
#    zero_line       — colour of the y=0 reference line
#    solver_colours  — ordered list of line/marker colours (≥ 6)
#    annotation_alpha — opacity for point label text (0–1)

THEMES: dict[str, dict] = {

    # ── A · Graphite & Neon ───────────────────────────────────────────────────
    "A": dict(
        name             = "Graphite & Neon",
        bg               = "#1A1C22",
        panel_bg         = "#20232B",
        grid             = "#2E3240",
        spine            = "#3A3E52",
        tick_fg          = "#7A8099",
        label_fg         = "#9AA0B8",
        title_fg         = "#E8EAF0",
        legend_fc        = "#22252F",
        legend_ec        = "#3A3E52",
        legend_tc        = "#C8CCDE",
        zero_line        = "#4A5070",
        annotation_alpha = 1.0,
        solver_colours   = [
            "#00D4FF",   # neon cyan
            "#39FF6A",   # neon lime
            "#FF7B2C",   # neon orange
            "#FF4D6D",   # neon rose
            "#C77DFF",   # neon violet
            "#FFD700",   # gold
        ],
    ),

    # ── B · Arctic Blueprint ──────────────────────────────────────────────────
    "B": dict(
        name             = "Arctic Blueprint",
        bg               = "#F0F4FA",
        panel_bg         = "#E6EDF8",
        grid             = "#C8D4E8",
        spine            = "#96ADCC",
        tick_fg          = "#4A6080",
        label_fg         = "#3A5070",
        title_fg         = "#0D1F3C",
        legend_fc        = "#E6EDF8",
        legend_ec        = "#96ADCC",
        legend_tc        = "#0D1F3C",
        zero_line        = "#96ADCC",
        annotation_alpha = 0.9,
        solver_colours   = [
            "#1A3F6F",   # deep navy
            "#0E6655",   # deep teal
            "#7B3800",   # brown
            "#6C3483",   # violet
            "#1A7ABF",   # cerulean
            "#B7770D",   # ochre
        ],
    ),

    # ── C · Ivory & Saffron ───────────────────────────────────────────────────
    "C": dict(
        name             = "Ivory & Saffron",
        bg               = "#FAF7F0",
        panel_bg         = "#F4EFE4",
        grid             = "#E2D9C8",
        spine            = "#C8B99A",
        tick_fg          = "#7A6A52",
        label_fg         = "#5A4830",
        title_fg         = "#2C1A08",
        legend_fc        = "#F4EFE4",
        legend_ec        = "#C8B99A",
        legend_tc        = "#2C1A08",
        zero_line        = "#C8B99A",
        annotation_alpha = 0.9,
        solver_colours   = [
            "#7B4B2A",   # terracotta
            "#3B5E2B",   # forest green
            "#5C3A7E",   # plum
            "#C47C00",   # saffron
            "#1B5E6E",   # deep teal
            "#A83232",   # brick red
        ],
    ),

    # ── D · Amber & Slate ─────────────────────────────────────────────────────
    "D": dict(
        name             = "Amber & Slate",
        bg               = "#FDFAF3",
        panel_bg         = "#F6F1E4",
        grid             = "#E4DBC8",
        spine            = "#B8A880",
        tick_fg          = "#6A5C40",
        label_fg         = "#4E4228",
        title_fg         = "#2A2010",
        legend_fc        = "#F6F1E4",
        legend_ec        = "#B8A880",
        legend_tc        = "#2A2010",
        zero_line        = "#B8A880",
        annotation_alpha = 0.9,
        solver_colours   = [
            "#C98800",   # amber
            "#6B3FA0",   # plum
            "#1B6B5E",   # dark teal
            "#B84020",   # rust
            "#4A7C59",   # sage
            "#C05010",   # burnt orange
        ],
    ),

    # ── E · Print-Ready ───────────────────────────────────────────────────────
    "E": dict(
        name             = "Print-Ready",
        bg               = "#FFFFFF",
        panel_bg         = "#F2F2F2",
        grid             = "#C0C0C0",
        spine            = "#404040",
        tick_fg          = "#404040",
        label_fg         = "#202020",
        title_fg         = "#000000",
        legend_fc        = "#FFFFFF",
        legend_ec        = "#404040",
        legend_tc        = "#000000",
        zero_line        = "#808080",
        annotation_alpha = 1.0,
        solver_colours   = [
            "#1155AA",   # blue    — print-safe
            "#117744",   # green
            "#AA2211",   # red
            "#000000",   # black
            "#885500",   # brown
            "#554488",   # indigo
        ],
    ),
}


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


# ── plot helpers ──────────────────────────────────────────────────────────────

_CONFIG_ORDER = ["TC-10SF-R", "TC-1LMAIF-R", "TC-10MAIF-R", "TC-3XMAIF-R"]


def _resolve_theme(theme_key: str) -> dict:
    """Return a flat dict of all styling tokens for the given theme key."""
    if theme_key and theme_key in THEMES:
        th = THEMES[theme_key]
        return dict(
            bg             = th["bg"],
            grid_col       = th["grid"],
            spine_col      = th["spine"],
            tick_col       = th["tick_fg"],
            label_col      = th["label_fg"],
            title_col      = th["title_fg"],
            legend_fc      = th["legend_fc"],
            legend_ec      = th["legend_ec"],
            legend_tc      = th["legend_tc"],
            zero_col       = th["zero_line"],
            ann_alpha      = th["annotation_alpha"],
            solver_colours = th["solver_colours"],
            theme_label    = f"  ·  {th['name']}",
            line_styles    = (["-", "--", (0,(5,2)), (0,(2,2)), (0,(6,2,1,2)), (0,(1,1))]
                              if theme_key == "E" else ["-"] * 6),
        )
    return dict(
        bg             = C["bg"],
        grid_col       = C["grid"],
        spine_col      = "#aaaaaa",
        tick_col       = "#555555",
        label_col      = "#333333",
        title_col      = "#111111",
        legend_fc      = "#ffffff",
        legend_ec      = "#cccccc",
        legend_tc      = "#111111",
        zero_col       = "#aaaaaa",
        ann_alpha      = 0.9,
        solver_colours = SOLVER_COLOURS,
        theme_label    = "",
        line_styles    = ["-"] * 6,
    )


def _build_axes(df: pd.DataFrame, t: dict,
                fig_height: float = 5.0) -> tuple:
    """
    Create figure + axes, draw lines, return
    (fig, ax, configs, solvers, x_pos, series)
    where series = [(colour, x_positions, y_values), …].
    No labels are drawn here.
    """
    present = set(df["config"].unique())
    configs = [c for c in _CONFIG_ORDER if c in present] + \
              sorted(c for c in present if c not in _CONFIG_ORDER)
    solvers = sorted(df["solver"].unique())
    x_pos   = np.arange(len(configs))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 2.2), fig_height))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.grid(True, linestyle=":", color=t["grid_col"], alpha=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["spine_col"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["tick_col"], labelsize=8)

    series = []
    for s_idx, solver in enumerate(solvers):
        colour = t["solver_colours"][s_idx % len(t["solver_colours"])]
        marker = MARKERS[s_idx % len(MARKERS)]
        ls     = t["line_styles"][s_idx % len(t["line_styles"])]
        sub    = df[df["solver"] == solver].set_index("config")
        y      = [sub.loc[c, "cum_reward"] if c in sub.index else np.nan
                  for c in configs]
        ax.plot(x_pos, y, color=colour, marker=marker, markersize=7,
                linewidth=1.8, linestyle=ls, label=solver, zorder=3)
        series.append((solver, colour, x_pos.tolist(), y))

    ax.axhline(0, color=t["zero_col"], linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=9, rotation=15, ha="right",
                       color=t["tick_col"])
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Reward  (episode NPV)", fontsize=10,
                  color=t["label_col"])
    ax.tick_params(labelsize=8)
    leg = ax.legend(title="Solver", fontsize=8, title_fontsize=8,
                    loc="best", framealpha=0.9,
                    facecolor=t["legend_fc"], edgecolor=t["legend_ec"])
    leg.get_title().set_color(t["legend_tc"])
    for txt in leg.get_texts():
        txt.set_color(t["label_col"])

    return fig, ax, configs, solvers, x_pos, series


def _set_title(ax, base: str, variant: str, t: dict) -> None:
    suffix = f"  [{variant}]" if variant else ""
    ax.set_title(base + suffix,
                 fontsize=11, fontweight="bold", color=t["title_col"])


def _save(fig, out: Path, bg: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"  saved → {out}")


# ── variant 1: annotated (original — labels may overlap) ─────────────────────

def plot_annotated(df: pd.DataFrame, out: Path, theme_key: str = "") -> None:
    t = _resolve_theme(theme_key)
    fig, ax, configs, solvers, x_pos, series = _build_axes(df, t)
    _set_title(ax, f"Solver Comparison — Cumulative Reward per Config{t['theme_label']}",
               "annotated", t)

    for _solver, colour, xs, ys in series:
        for xi, yi in zip(xs, ys):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=colour, alpha=t["ann_alpha"])
    _save(fig, out, t["bg"])


# ── variant 2: table — no inline labels, clean data table below ───────────────

def plot_table(df: pd.DataFrame, out: Path, theme_key: str = "") -> None:
    t = _resolve_theme(theme_key)
    present = set(df["config"].unique())
    configs = [c for c in _CONFIG_ORDER if c in present] + \
              sorted(c for c in present if c not in _CONFIG_ORDER)
    solvers = sorted(df["solver"].unique())

    # ── figure layout via GridSpec ────────────────────────────────────────────
    # Row heights: plot panel (fixed 5 in) + explicit gap + table panel.
    # The gap row is empty whitespace — keeps x-axis labels and table header
    # from colliding regardless of font size or tick rotation.
    ROW_H     = 0.32          # inches per data row in the table
    HDR_H     = 0.38          # inches for the table header row
    GAP_H     = 0.70          # inches of breathing room between plot and table
    PLOT_H    = 5.0
    TABLE_H   = HDR_H + len(solvers) * ROW_H
    FIG_H     = PLOT_H + GAP_H + TABLE_H

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(max(8, len(configs) * 2.2), FIG_H))
    fig.patch.set_facecolor(t["bg"])

    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[PLOT_H, GAP_H, TABLE_H],
        hspace=0,           # we control spacing via height_ratios, not hspace
    )
    ax     = fig.add_subplot(gs[0])   # plot
    ax_gap = fig.add_subplot(gs[1])   # invisible spacer
    ax_tbl = fig.add_subplot(gs[2])   # table host

    # hide the gap and table axes frames entirely
    for a in (ax_gap, ax_tbl):
        a.set_visible(False)

    # ── plot panel ────────────────────────────────────────────────────────────
    ax.set_facecolor(t["bg"])
    ax.grid(True, linestyle=":", color=t["grid_col"], alpha=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["spine_col"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["tick_col"], labelsize=8)

    x_pos      = np.arange(len(configs))
    table_data = []

    for s_idx, solver in enumerate(solvers):
        colour = t["solver_colours"][s_idx % len(t["solver_colours"])]
        marker = MARKERS[s_idx % len(MARKERS)]
        ls     = t["line_styles"][s_idx % len(t["line_styles"])]
        sub    = df[df["solver"] == solver].set_index("config")
        y      = [sub.loc[c, "cum_reward"] if c in sub.index else np.nan
                  for c in configs]
        ax.plot(x_pos, y, color=colour, marker=marker, markersize=7,
                linewidth=1.8, linestyle=ls, label=solver, zorder=3)
        table_data.append([f"{v:.2f}" if not np.isnan(v) else "—" for v in y])

    ax.axhline(0, color=t["zero_col"], linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=9, rotation=15, ha="right",
                       color=t["tick_col"])
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Reward  (episode NPV)", fontsize=10,
                  color=t["label_col"])
    ax.tick_params(labelsize=8)
    leg = ax.legend(title="Solver", fontsize=8, title_fontsize=8,
                    loc="best", framealpha=0.9,
                    facecolor=t["legend_fc"], edgecolor=t["legend_ec"])
    leg.get_title().set_color(t["legend_tc"])
    for txt in leg.get_texts():
        txt.set_color(t["label_col"])
    _set_title(ax, f"Solver Comparison — Cumulative Reward per Config{t['theme_label']}",
               "table", t)

    # ── data table in its own axes ────────────────────────────────────────────
    bg_hex = t["bg"]
    cell_colours = [
        [bg_hex if ri % 2 == 0 else t["grid_col"] + "44"] * len(configs)
        for ri in range(len(solvers))
    ]

    # make ax_tbl visible just long enough to host the table, then rehide frame
    ax_tbl.set_visible(True)
    ax_tbl.set_facecolor(bg_hex)
    for spine in ax_tbl.spines.values():
        spine.set_visible(False)
    ax_tbl.set_xticks([])
    ax_tbl.set_yticks([])

    tbl = ax_tbl.table(
        cellText  = table_data,
        rowLabels = solvers,
        colLabels = [c.replace("TC-", "") for c in configs],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],        # fill the entire ax_tbl panel
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(t["spine_col"] + "55")
        cell.set_linewidth(0.4)
        if row == 0:                      # header
            cell.set_facecolor(t["grid_col"])
            cell.set_text_props(color=t["title_col"], fontweight="bold",
                                fontsize=7.5)
        elif col == -1:                   # row label
            sc = t["solver_colours"][(row - 1) % len(t["solver_colours"])]
            cell.set_facecolor(bg_hex)
            cell.set_text_props(color=sc, fontweight="bold", fontsize=7.5)
        else:
            cell.set_facecolor(cell_colours[row - 1][col])
            sc = t["solver_colours"][(row - 1) % len(t["solver_colours"])]
            cell.set_text_props(color=sc, fontsize=7.5)

    _save(fig, out, bg_hex)


# ── variant 3: clean — overlap-resolved labels ────────────────────────────────

def _resolve_label_positions(ax, fig, annotations: list) -> None:
    """
    Push overlapping annotation bounding boxes apart along the y axis.

    annotations: list of matplotlib Text objects (already added to ax).
    Strategy:
      1. Draw the canvas to get real pixel bboxes.
      2. Sort labels by their natural y position.
      3. Repeatedly scan for overlapping pairs and nudge the upper one up
         (and lower one down symmetrically) until no overlaps remain or
         max iterations hit.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    MAX_ITER = 60
    PAD_PT   = 1.5    # minimum gap in points between boxes

    def _bbox(ann):
        """Bounding box in display (pixel) coords."""
        return ann.get_window_extent(renderer=renderer)

    def _overlaps(b1, b2, pad):
        return not (b1.x1 + pad < b2.x0 or b2.x1 + pad < b1.x0 or
                    b1.y1 + pad < b2.y0 or b2.y1 + pad < b1.y0)

    # pixels per point
    dpi    = fig.dpi
    pt2px  = dpi / 72.0
    pad_px = PAD_PT * pt2px

    for _ in range(MAX_ITER):
        moved = False
        # sort by current y centre so we process bottom-up
        anns = sorted(annotations, key=lambda a: _bbox(a).y0)
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                b_i = _bbox(anns[i])
                b_j = _bbox(anns[j])
                if not _overlaps(b_i, b_j, pad_px):
                    continue
                # overlap: push them apart symmetrically
                overlap_y  = b_i.y1 + pad_px - b_j.y0
                shift_px   = overlap_y / 2.0
                # convert pixel shift to offset-points (they use offset points)
                shift_pt   = shift_px / pt2px

                ox_i, oy_i = anns[i].get_position()   # these are the offset coords
                ox_j, oy_j = anns[j].get_position()
                # oy is in offset-points relative to the anchor xy
                anns[i].set_position((ox_i, oy_i - shift_pt))
                anns[j].set_position((ox_j, oy_j + shift_pt))
                moved = True
        if not moved:
            break


def plot_clean(df: pd.DataFrame, out: Path, theme_key: str = "") -> None:
    t = _resolve_theme(theme_key)
    fig, ax, configs, solvers, x_pos, series = _build_axes(df, t)
    _set_title(ax, f"Solver Comparison — Cumulative Reward per Config{t['theme_label']}",
               "clean labels", t)

    # Place all annotations with a per-column stagger so same-x labels
    # start at slightly different offsets — reduces the nudge count needed.
    annotations = []
    n_solvers = len(series)
    for s_idx, (_solver, colour, xs, ys) in enumerate(series):
        # Initial vertical stagger: spread solvers evenly within ±6 pt
        stagger = (s_idx - (n_solvers - 1) / 2) * (12.0 / max(n_solvers, 1))
        base_offset = 9 + stagger
        for xi, yi in zip(xs, ys):
            if not np.isnan(yi):
                ann = ax.annotate(
                    f"{yi:.2f}", xy=(xi, yi),
                    xytext=(0, base_offset),
                    textcoords="offset points",
                    ha="center", fontsize=7,
                    color=colour, alpha=t["ann_alpha"],
                )
                annotations.append(ann)

    _resolve_label_positions(ax, fig, annotations)
    _save(fig, out, t["bg"])


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
        help="Output directory for gap plot PNGs. "
             "Defaults to <ground_truth>/../gap_plots/.",
    )
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    ground_root  = Path(args.root) if args.root else script_dir / "ground_truth"

    if not ground_root.exists():
        sys.exit(f"[ERROR] ground_truth directory not found: {ground_root}")

    # all outputs land in a gap_plots/ subdirectory beside ground_truth
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ground_root.parent / "gap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  root : {ground_root}")
    df = _collect(ground_root)

    if df.empty:
        sys.exit("[ERROR] No data found — check that portfolios.csv files exist under ground_truth.")

    print(f"\n  configs : {sorted(df['config'].unique())}")
    print(f"  solvers : {sorted(df['solver'].unique())}")
    print()

    for sk in ("", "A", "B", "C", "D", "E"):
        sfx  = f"_{sk}" if sk else ""
        stem = f"gap_plot{sfx}"
        plot_annotated(df, out_dir / f"{stem}.png",        theme_key=sk)
        plot_table(    df, out_dir / f"{stem}_table.png",  theme_key=sk)
        plot_clean(    df, out_dir / f"{stem}_clean.png",  theme_key=sk)

    print("\n  Done.")


if __name__ == "__main__":
    main()