"""
infograph.py
============
Renders a self-contained infographic for one episode's problem definition
and its solution from the standard CSV exports.

Usage
-----
    python infograph.py                        # uses CSVs next to this file
    python infograph.py --data path/to/csvs/  # explicit folder
    python infograph.py --out  my_plot.png

Layout (3 rows)
---------------
  Row 1 │ Config + Project Profile table  │  Milestone Profile table
  Row 2 │ Episode timeline & cash-flow diagram (full width)
  Row 3 │ Reward computation table         │  Timeline legend
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── palette ───────────────────────────────────────────────────────────────────
BG          = "#f8f9fb"
SURFACE     = "#f1efe8"
BORDER      = "#c8c6c0"
TEXT_PRI    = "#0b0b0b"
TEXT_SEC    = "#52514e"
TEXT_MUT    = "#898781"

C_PURPLE_BG = "#eeedfe";  C_PURPLE_BD = "#534ab7";  C_PURPLE_TX = "#26215c"
C_TEAL_BG   = "#e1f5ee";  C_TEAL_BD   = "#0f6e56";  C_TEAL_TX   = "#04342c"
C_BLUE_BG   = "#e6f1fb";  C_BLUE_BD   = "#185fa5";  C_BLUE_TX   = "#0c447c"

PAY_COLORS  = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]   # j=0,1,2,3
ALLOC_COLOR = "#8338ec"
PROJECT_COLORS = [C_PURPLE_BD, "#0f6e56", "#993c1d"]  # i=0,1,2

FONT = "DejaVu Sans"

# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}" if v != int(v) else str(int(v))
    return str(v)

def _table(ax, rows, col_widths, row_height=0.13, header=True,
           header_bg=SURFACE, cell_bg=BG, border=BORDER,
           header_tx=TEXT_PRI, cell_tx=TEXT_SEC):
    """Draw a simple table inside ax (axis units 0-1 x 0-1)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n_rows = len(rows)
    total_h = n_rows * row_height
    y0 = 1.0                                # top of table

    for r_idx, row in enumerate(rows):
        y_top = y0 - r_idx * row_height
        y_bot = y_top - row_height
        is_header = (r_idx == 0 and header)
        bg = header_bg if is_header else (cell_bg if r_idx % 2 == 0 else "#f5f3ec")
        tx = header_tx if is_header else cell_tx
        fw = "bold" if is_header else "normal"

        x = 0.0
        for c_idx, (cell, cw) in enumerate(zip(row, col_widths)):
            rect = mpatches.FancyBboxPatch(
                (x, y_bot), cw, row_height,
                boxstyle="square,pad=0", linewidth=0.5,
                edgecolor=border, facecolor=bg
            )
            ax.add_patch(rect)
            ax.text(x + cw / 2, (y_top + y_bot) / 2, str(cell),
                    ha="center", va="center", fontsize=7,
                    fontfamily=FONT, color=tx, fontweight=fw,
                    clip_on=True)
            x += cw

# ── main ──────────────────────────────────────────────────────────────────────
def build(data_dir: Path, out: Path):

    # ── load CSVs ─────────────────────────────────────────────────────────────
    pp  = pd.read_csv(data_dir / "projects_profile.csv")
    mp  = pd.read_csv(data_dir / "milestones_profile.csv")
    ms  = pd.read_csv(data_dir / "milestones_status.csv")
    ps  = pd.read_csv(data_dir / "projects_status.csv")
    pf  = pd.read_csv(data_dir / "portfolios.csv")

    episode_id = pp["episode_id"].iloc[0]
    config_id  = pp["config_id"].iloc[0]
    method     = ms["method"].iloc[0]
    n_proj     = len(pp)
    horizon    = int(pf["t_episode"].max())
    gamma      = 0.95   # read from env config if available; fallback
    budget     = pf["budget_available"].iloc[0]

    # efficiency per project: first active row per project
    eff_map = {}
    for i, grp in ps[ps["status"] == "active"].groupby("i"):
        eff_map[i] = grp.sort_values("t_episode")["efficiency"].tolist()

    # allocations per (i, t_episode)
    alloc_map = {}
    for _, row in ps.iterrows():
        alloc_map[(int(row["i"]), int(row["t_episode"]))] = float(row["allocation_action"])

    # certified payments per t_episode
    cert = defaultdict(list)  # t -> list of (i, j, net_payment)
    for _, row in ms.iterrows():
        cert[int(row["certified_t"])].append(
            (int(row["i"]), int(row["j"]), float(row["net_payment"]))
        )

    # portfolio reward timeline
    pf_sorted = pf.sort_values("t_episode").reset_index(drop=True)
    cum_reward = pf_sorted["reward"].cumsum().iloc[-1]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # GridSpec: row0=tables, row1=timeline, row2=reward+legend
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.6, 0.9],
        width_ratios=[1, 1],
        hspace=0.38, wspace=0.06,
        left=0.03, right=0.97, top=0.94, bottom=0.04
    )

    ax_prof  = fig.add_subplot(gs[0, 0])   # config + project profile
    ax_miles = fig.add_subplot(gs[0, 1])   # milestone profile
    ax_tl    = fig.add_subplot(gs[1, :])   # timeline (full width)
    ax_rew   = fig.add_subplot(gs[2, 0])   # reward table
    ax_leg   = fig.add_subplot(gs[2, 1])   # legend

    for ax in [ax_prof, ax_miles, ax_tl, ax_rew, ax_leg]:
        ax.set_facecolor(BG)

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, f"{episode_id}   ·   solver: {method}",
             ha="center", va="top", fontsize=13, fontweight="bold",
             fontfamily=FONT, color=TEXT_PRI)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0-LEFT  —  Config + Project Profile
    # ══════════════════════════════════════════════════════════════════════════
    ax_prof.set_title("Config & Project Profile", fontsize=9, color=TEXT_MUT,
                      fontfamily=FONT, loc="left", pad=4)

    header = ["", "config_id", "BAC", "price", "start", "finish", "dur",
              "adv%", "ret%", "profit%"]
    rows_p = [header]
    for _, r in pp.iterrows():
        rows_p.append([
            f"i={int(r['i'])}",
            config_id,
            _fmt(r["bac"]),
            _fmt(r["price"]),
            int(r["planned_start"]),
            int(r["planned_finish"]),
            int(r["planned_duration"]),
            f"{r['advance_percent']*100:.0f}%",
            f"{r['retention_rate']*100:.0f}%",
            f"{r['profit_percent']*100:.0f}%",
        ])

    cw = [0.08, 0.16, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08, 0.08, 0.12]
    _table(ax_prof, rows_p, cw, row_height=0.19,
           header_bg=C_PURPLE_BG, cell_bg=BG, border=C_PURPLE_BD,
           header_tx=C_PURPLE_TX, cell_tx=TEXT_SEC)

    # efficiency per project below
    eff_lines = []
    for i in range(n_proj):
        eff = eff_map.get(i, [])
        eff_str = ", ".join(f"{e:.2f}" for e in eff)
        eff_lines.append(f"  i={i}  η = [{eff_str}]")
    ax_prof.text(0.01, 1 - (len(rows_p)) * 0.19 - 0.05,
                 "Efficiency sequences:\n" + "\n".join(eff_lines),
                 va="top", ha="left", fontsize=7, fontfamily=FONT,
                 color=TEXT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0-RIGHT  —  Milestone Profile
    # ══════════════════════════════════════════════════════════════════════════
    ax_miles.set_title("Milestone Profile", fontsize=9, color=TEXT_MUT,
                       fontfamily=FONT, loc="left", pad=4)

    m_header = ["i", "j", "prog_thr", "ts_thr", "gross", "advance_rec",
                "retention", "net_pay"]
    rows_m = [m_header]
    for _, r in mp.iterrows():
        rows_m.append([
            int(r["i"]), int(r["j"]),
            _fmt(r["progress_threshold"]),
            int(r["timestep_threshold"]),
            _fmt(r["gross_payment"]),
            _fmt(r["advance_recovery"]),
            _fmt(r["retention_withheld"]),
            _fmt(r["net_payment"]),
        ])

    cw_m = [0.07, 0.07, 0.12, 0.10, 0.12, 0.16, 0.16, 0.20]
    _table(ax_miles, rows_m, cw_m, row_height=0.073,
           header_bg=C_TEAL_BG, cell_bg=BG, border=C_TEAL_BD,
           header_tx=C_TEAL_TX, cell_tx=TEXT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1  —  Episode Timeline
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_tl
    ax.set_title("Episode Cash-Flow Timeline", fontsize=9, color=TEXT_MUT,
                 fontfamily=FONT, loc="left", pad=4)
    ax.set_xlim(-0.5, horizon + 0.5)
    ax.set_ylim(-4.5, 4.0)
    ax.axis("off")

    T = list(range(horizon + 1))
    t_xs = {t: float(t) for t in T}

    # ── timeline axis ──
    ax.axhline(0, color=BORDER, lw=1.2, zorder=1)
    for t in T:
        ax.plot(t, 0, "|", color=BORDER, ms=6, mew=1.2, zorder=2)
        ax.text(t, -0.32, f"t={t}", ha="center", va="top",
                fontsize=7.5, fontfamily=FONT, color=TEXT_SEC)

    # ── efficiency labels below t labels ──
    for i in range(n_proj):
        proj_row = pp[pp["i"] == i].iloc[0]
        ps_t    = int(proj_row["planned_start"])
        pd_dur  = int(proj_row["planned_duration"])
        eff_seq = eff_map.get(i, [])
        for k, t in enumerate(range(ps_t, ps_t + pd_dur)):
            if k < len(eff_seq):
                offset = -0.62 - i * 0.22
                ax.text(t, offset, f"η={eff_seq[k]:.2f}",
                        ha="center", va="top", fontsize=6.5,
                        fontfamily=FONT,
                        color=PROJECT_COLORS[i % len(PROJECT_COLORS)])

    # ── project active bars ──
    bar_y_base = 0.18
    bar_h      = 0.32
    for i, row in pp.iterrows():
        ps_t = int(row["planned_start"])
        pf_t = int(row["planned_finish"])
        dur  = pf_t - ps_t
        col  = PROJECT_COLORS[i % len(PROJECT_COLORS)]
        rect = mpatches.FancyBboxPatch(
            (ps_t, bar_y_base + i * (bar_h + 0.08)),
            dur, bar_h,
            boxstyle="round,pad=0.04",
            linewidth=0.8, edgecolor=col,
            facecolor=col + "28",
            zorder=3
        )
        ax.add_patch(rect)
        ax.text(ps_t + dur / 2,
                bar_y_base + i * (bar_h + 0.08) + bar_h / 2,
                f"Project {i}  (BAC={int(row['bac'])}, price={int(row['price'])})",
                ha="center", va="center", fontsize=7.5,
                fontfamily=FONT, fontweight="bold", color=col, zorder=4)

    # ── inflow arrows + labels ──
    # group payments by certified_t so stacked ones don't overlap
    arrow_y_top = bar_y_base + n_proj * (bar_h + 0.08) + 0.15

    for t, pays in cert.items():
        # stack upward
        for stack_idx, (i, j, net_pay) in enumerate(pays):
            col   = PAY_COLORS[j % len(PAY_COLORS)]
            y_tip = arrow_y_top + stack_idx * 0.72
            y_base_arrow = arrow_y_top - 0.05

            ax.annotate("",
                xy=(t, y_tip), xytext=(t, y_base_arrow),
                arrowprops=dict(arrowstyle="->", color=col,
                                lw=1.4, mutation_scale=10),
                zorder=5)

            # milestone profile lookup for thresholds
            mp_row = mp[(mp["i"] == i) & (mp["j"] == j)]
            prog_thr = float(mp_row["progress_threshold"].iloc[0]) if not mp_row.empty else 0
            ts_thr   = int(mp_row["timestep_threshold"].iloc[0])   if not mp_row.empty else t

            label = (f"+{net_pay:.1f}\n"
                     f"i={i} j={j}\n"
                     f"prog≥{prog_thr:.0%}  ts≥{ts_thr}")
            ax.text(t, y_tip + 0.08, label,
                    ha="center", va="bottom", fontsize=6.2,
                    fontfamily=FONT, color=col, zorder=6,
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor=col + "18",
                              edgecolor=col + "60", lw=0.5))

    # ── allocation arrows (downward, red/purple) ──
    alloc_y_base = bar_y_base - 0.05
    seen_alloc_t = defaultdict(float)
    for (i, t), val in alloc_map.items():
        if val > 0:
            seen_alloc_t[t] += val

    for t, total_alloc in seen_alloc_t.items():
        y_tip = -1.5
        ax.annotate("",
            xy=(t, y_tip), xytext=(t, alloc_y_base),
            arrowprops=dict(arrowstyle="->", color=ALLOC_COLOR,
                            lw=1.4, mutation_scale=10,
                            linestyle="dashed"),
            zorder=5)
        ax.text(t, y_tip - 0.18,
                f"−{total_alloc:.1f}",
                ha="center", va="top", fontsize=7,
                fontfamily=FONT, color=ALLOC_COLOR, fontweight="bold", zorder=6)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2-LEFT  —  Reward Computation
    # ══════════════════════════════════════════════════════════════════════════
    ax_rew.set_title("Reward Computation  (γ = 0.95)", fontsize=9,
                     color=TEXT_MUT, fontfamily=FONT, loc="left", pad=4)

    rew_header = ["t", "net_cashflow", f"γᵗ", "period_reward", "cum_reward"]
    rew_rows   = [rew_header]
    running = 0.0
    for _, row in pf_sorted.iterrows():
        t    = int(row["t_episode"])
        ncf  = float(row["net_cashflow"])
        disc = gamma ** t
        pr   = float(row["reward"])
        running += pr
        rew_rows.append([
            t,
            f"{ncf:+.3f}",
            f"{disc:.4f}",
            f"{pr:+.4f}",
            f"{running:+.4f}",
        ])
    # summary row
    rew_rows.append(["", "", "", "cum_reward =", f"{running:+.4f}"])

    cw_r = [0.08, 0.22, 0.15, 0.28, 0.27]
    _table(ax_rew, rew_rows, cw_r, row_height=0.083,
           header_bg=C_BLUE_BG, cell_bg=BG, border=C_BLUE_BD,
           header_tx=C_BLUE_TX, cell_tx=TEXT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2-RIGHT  —  Legend
    # ══════════════════════════════════════════════════════════════════════════
    ax_leg.axis("off")
    ax_leg.set_title("Legend", fontsize=9, color=TEXT_MUT,
                     fontfamily=FONT, loc="left", pad=4)
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

    entries = []
    for j, col in enumerate(PAY_COLORS):
        entries.append((col, f"Inflow — milestone j={j}", "arrow_up"))
    entries.append((ALLOC_COLOR, "Outflow — allocation action", "arrow_down_dash"))
    for i, col in enumerate(PROJECT_COLORS[:n_proj]):
        entries.append((col, f"Project {i} active window", "bar"))

    step = 0.13
    y    = 0.93
    for col, label, style in entries:
        if style == "arrow_up":
            ax_leg.annotate("", xy=(0.06, y + 0.04), xytext=(0.06, y - 0.04),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.3,
                                            mutation_scale=9))
        elif style == "arrow_down_dash":
            ax_leg.annotate("", xy=(0.06, y - 0.04), xytext=(0.06, y + 0.04),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.3,
                                            linestyle="dashed", mutation_scale=9))
        elif style == "bar":
            rect = mpatches.FancyBboxPatch(
                (0.02, y - 0.035), 0.08, 0.07,
                boxstyle="round,pad=0.01",
                linewidth=0.8, edgecolor=col, facecolor=col + "28"
            )
            ax_leg.add_patch(rect)

        ax_leg.text(0.14, y, label, va="center", ha="left",
                    fontsize=8, fontfamily=FONT, color=TEXT_SEC)
        y -= step

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RL portfolio infographic")
    parser.add_argument("--data", default=None,
                        help="Folder containing the CSV exports.")
    parser.add_argument("--out",  default=None,
                        help="Output PNG path.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir   = Path(args.data) if args.data else script_dir
    out_path   = Path(args.out)  if args.out  else data_dir / "infograph.png"

    build(data_dir, out_path)
    print("Done.")


if __name__ == "__main__":
    main()