"""
infograph.py  —  Episode infographic output folder.

Outputs  (all written to  <out_dir>/infograph/ )
------------------------------------------------
  01_timeline.png              — cash-flow timeline (grid, arrows, efficiency, sidebar legend)
  01b_ncf_chart.png            — net cash-flow + budget curve (legend outside plot, right)
  02_portfolio_status.png      — period reward computation table (exact-fit framing)
  03_observations.png          — agent observation table + symbol legend (right column)

Usage
-----
  python infograph.py
  python infograph.py --data ground_truth/TC-3XMAIF-R/optimal/ --out ground_truth/
"""

from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
BG    = "#F8F7F3"
PANEL = "#EDEAE2"
RULE  = "#CCC8BE"
TH    = "#16151A"
TB    = "#363330"
TM    = "#78746A"

PC   = ["#3A5A82", "#2E7D52", "#944820"]   # project colours 0, 1, 2

SA = {                                      # (border-colour, header-bg)
    "tl":  ("#1A4878", "#E4EEF7"),
    "ps":  ("#7A3818", "#F4E8DF"),
    "obs": ("#1A6448", "#E3F1E9"),
    "leg": ("#3A3630", "#EEECEA"),
}

FONT = "DejaVu Sans"
DPI  = 300          # print-quality raster output

PROJ_ROWS = 3        # payments row / project box row / allocation row


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sf(v: float, d: int = 2) -> str:
    return "—" if v == 0.0 else f"{v:+.{d}f}"


def _hdr(ax, title: str, key: str, y: float, h: float = 0.30,
         fs: float = 8.0) -> float:
    """
    Draw a coloured header band in INCH-SCALE data coordinates where the
    y-axis is inverted (0 = top of figure, increasing downward).
    `y` is the top of the band; returns the bottom edge (new y).
    """
    brd, bg = SA[key]
    ax.add_patch(mpatches.Rectangle(
        (0, y), 1, h, lw=0, fc=bg, clip_on=False, zorder=8))
    ax.add_patch(mpatches.Rectangle(
        (0, y), 0.005, h, lw=0, fc=brd, clip_on=False, zorder=9))
    ax.text(0.012, y + h / 2, title,
            ha="left", va="center", fontsize=fs,
            fontfamily=FONT, fontweight="bold", color=brd,
            clip_on=False, zorder=10)
    return y + h


def _table(ax, rows, cw, rh: float, key: str,
           fs: float = 7.0, align=None, y0: float = 0.0,
           pad: float = 0.015, x0: float = 0.0, width: float = 1.0) -> float:
    """
    Render a table in the same inch-scale / inverted-y convention as `_hdr`.
    `y0` is the top edge; returns the bottom edge (new y).
    """
    brd, hbg = SA[key]
    y0 += pad
    if align is None:
        align = ["center"] * len(cw)
    for ri, row in enumerate(rows):
        yt = y0 + ri * rh
        yb = yt + rh
        is_hdr = ri == 0
        face = hbg if is_hdr else (BG if ri % 2 == 1 else PANEL)
        fc   = brd if is_hdr else TB
        fw   = "bold" if is_hdr else "normal"
        ax.add_patch(mpatches.Rectangle(
            (x0, yt), width, rh, lw=0, fc=face, clip_on=True, zorder=1))
        ax.plot([x0, x0 + width], [yb, yb], color=RULE, lw=0.3, zorder=2)
        x = x0
        for ci, (cell, w, ha) in enumerate(zip(row, cw, align)):
            colw = w * width
            if ci:
                ax.plot([x, x], [yt, yb], color=RULE, lw=0.2, zorder=2)
            p  = 0.006 * width
            tx = (x + p if ha == "left"
                  else x + colw - p if ha == "right"
                  else x + colw / 2)
            ax.text(tx, (yt + yb) / 2, str(cell),
                    ha=ha, va="center", fontsize=fs,
                    fontfamily=FONT, color=fc, fontweight=fw,
                    clip_on=True, zorder=3)
            x += colw
    nr = len(rows)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0 - pad), width, nr * rh + pad,
        boxstyle="square,pad=0", lw=0.7, ec=brd + "55", fc="none",
        clip_on=True, zorder=4))
    return y0 + nr * rh


def _fitted_axes(fw: float, fh: float):
    """
    Figure whose single axes fills the canvas exactly (no default mpl
    margins), using an INVERTED y-axis measured in inches from the top
    (y=0 top … y=fh bottom).  x stays in [0, 1] = fraction of width.
    This makes `fh` the true content height — no residual blank space.
    """
    fig = plt.figure(figsize=(fw, fh), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(fh, 0)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.06,
                facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  1 · TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
def _build_timeline(out_path: Path, data: dict) -> None:
    """
    Cash-flow timeline. Three-row grid band per project (earliest project on
    top): payments row / project box row / allocation row.

      • green arrows  — payments.    Base at the project box top edge,
        pointing UP.  When several payments land on the same project at the
        same timestep, a single shared arrow is drawn and the VALUES are
        stacked vertically above it (earliest payment on top), so nothing
        overlaps regardless of how many payments co-occur.
      • red arrows    — allocations. Base at the project box bottom edge,
        pointing DOWN.  Value printed under the arrow.
      • small circles sit exactly on the project box's bottom edge, one per
        active timestep, showing the efficiency η for that step.  Circle
        size is fixed in *physical inches* so the η value always stays
        inside the circle.

    All headers + the legend live in a single one-column card placed
    outside the grid, at the top-right of the figure.
    """
    cfg_id     = data["cfg_id"]
    ep_id      = data["ep_id"]
    method     = data["method"]
    n_proj     = data["n_proj"]
    horizon    = data["horizon"]
    gamma      = data["gamma"]
    bud0       = data["bud0"]
    proj_info  = data["proj_info"]
    proj_order = data["proj_order"]
    alloc_map  = data["alloc_map"]
    eff_map    = data["eff_map"]
    cert       = data["cert"]

    ARROW_GREEN = "#1A8C4E"
    ALLOC_RED   = "#C0392B"

    # ── layout constants (data-y units ≈ inches by construction) ───────────
    N_GRID    = n_proj * PROJ_ROWS
    ROW_H     = 0.82
    BOX_VPAD  = 0.16
    ARROW_LEN = 0.46
    LABEL_GAP = 0.10
    STACK_DY  = 0.23          # vertical spacing between stacked payment values

    FS_TIME   = 8.6
    FS_PLABEL = 7.8
    FS_BOX    = 9.8
    FS_ARROW  = 7.8
    FS_EFF    = 6.4

    GRID_TOP = N_GRID * ROW_H

    def proj_grid_top(rank: int) -> float:
        return (N_GRID - rank * PROJ_ROWS) * ROW_H

    # ── group payments by (project, certification-t) ───────────────────────
    pays_by_proj_t: dict[tuple, list] = defaultdict(list)
    for t_c, pays in cert.items():
        for pi, j, net in pays:
            pays_by_proj_t[(pi, t_c)].append((j, net))
    max_stack = max((len(v) for v in pays_by_proj_t.values()), default=1)

    # ── sidebar layout constants (drives both sizing + drawing) ────────────
    TOP_INSET      = 0.16
    BOTTOM_INSET   = 0.12
    EYEBROW_GAP    = 0.26
    TITLE_GAP      = 0.26
    SOLVER_GAP     = 0.24
    INFO_GAP       = 0.21
    N_INFO         = 4
    PRE_DIV_GAP    = 0.08
    POST_DIV_GAP   = 0.22
    LEGHDR_GAP     = 0.28
    ITEM_GAP       = 0.36
    N_LEGEND       = 4
    END_GAP        = 0.08

    SIDEBAR_H = (TOP_INSET + EYEBROW_GAP + TITLE_GAP + SOLVER_GAP
                 + N_INFO * INFO_GAP + PRE_DIV_GAP + POST_DIV_GAP
                 + LEGHDR_GAP + N_LEGEND * ITEM_GAP + END_GAP + BOTTOM_INSET)

    # ── vertical extent ──────────────────────────────────────────────────
    box_top_rank0 = GRID_TOP - ROW_H - BOX_VPAD
    payment_reach = ARROW_LEN + LABEL_GAP + (max_stack - 1) * STACK_DY + 0.20
    y_need_grid   = box_top_rank0 + payment_reach + 0.14

    TOP_PAD = 0.14
    Y_LO    = -0.36
    Y_HI    = TOP_PAD + max(y_need_grid, SIDEBAR_H)
    FH      = max(4.2, Y_HI - Y_LO)

    # ── horizontal extent (grid stretches with horizon) ────────────────────
    GRID_W = float(np.clip(0.80 * horizon + 4.2, 10.5, 18.0))
    X_LO   = -0.9
    X_HI   = horizon + 0.9
    din_x_grid = (X_HI - X_LO) / GRID_W

    SIDE_W_IN = 3.4
    GAP_IN    = 0.28
    sidebar_w = SIDE_W_IN * din_x_grid
    gap_w     = GAP_IN * din_x_grid
    X_SIDE_L  = X_HI + gap_w
    X_SIDE_R  = X_SIDE_L + sidebar_w

    FW = GRID_W + GAP_IN + SIDE_W_IN

    fig = plt.figure(figsize=(FW, FH), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(X_LO, X_SIDE_R)
    ax.set_ylim(Y_LO, Y_HI)

    din_x = (X_SIDE_R - X_LO) / FW
    din_y = (Y_HI - Y_LO) / FH

    T = list(range(horizon + 1))

    # ── vertical period guides — light dotted, classy ───────────────────────
    for t in T:
        ax.plot([t, t], [0, GRID_TOP], color=RULE, lw=0.5,
                 ls=(0, (1, 2.4)), zorder=0)

    # ── project-band separators (only at project boundaries) ───────────────
    for b in range(n_proj + 1):
        y = b * PROJ_ROWS * ROW_H
        ax.plot([X_LO + 0.05, X_HI - 0.05], [y, y],
                color="#B7B2A6", lw=0.9, zorder=1)

    # ── project label strip (left margin) ───────────────────────────────────
    for rank, pi in enumerate(proj_order):
        top_y = proj_grid_top(rank)
        mid_y = top_y - PROJ_ROWS / 2 * ROW_H
        ax.text(X_LO + 0.33, mid_y,
                f"P{pi}\nBAC {proj_info[pi]['bac']}\nprice {proj_info[pi]['price']}",
                ha="center", va="center", fontsize=FS_PLABEL,
                fontfamily=FONT, fontweight="bold", color=PC[pi],
                zorder=6, linespacing=1.35)

    # ── spine & time labels ──────────────────────────────────────────────
    ax.axhline(0, color="#888", lw=1.1, zorder=2)
    for t in T:
        ax.plot(t, 0, "|", color=TM, ms=5.5, mew=1.0, zorder=3)
        ax.text(t, -0.07, f"t{t}", ha="center", va="top",
                fontsize=FS_TIME, fontfamily=FONT, color=TM)

    # ── project boxes — narrow / short ──────────────────────────────────
    box_edges: dict[int, tuple[float, float]] = {}
    for rank, pi in enumerate(proj_order):
        pst, pft = proj_info[pi]["start"], proj_info[pi]["finish"]
        col   = PC[pi]
        top_y = proj_grid_top(rank)
        box_top = (top_y - ROW_H) - BOX_VPAD
        box_bot = (top_y - 2 * ROW_H) + BOX_VPAD
        box_edges[pi] = (box_top, box_bot)
        ax.add_patch(mpatches.FancyBboxPatch(
            (pst, box_bot), pft - pst, box_top - box_bot,
            boxstyle="round,pad=0.022",
            lw=1.5, ec=col, fc=col + "30", zorder=4))
        ax.text(pst + (pft - pst) / 2, (box_top + box_bot) / 2, f"P{pi}",
                ha="center", va="center", fontsize=FS_BOX,
                fontfamily=FONT, fontweight="bold", color=col, zorder=5)

    # ── payment arrows — green, UP, base on project-box TOP edge ───────────
    # Multiple payments at the same (project, t): ONE shared arrow, values
    # stacked vertically above it (earliest payment placed highest) so
    # nothing overlaps.
    for (pi, t_c), plist in pays_by_proj_t.items():
        box_top, _ = box_edges[pi]
        tip_y = box_top + ARROW_LEN
        ax.annotate("", xy=(t_c, tip_y), xytext=(t_c, box_top),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_GREEN,
                                    lw=1.6, mutation_scale=10,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ordered = sorted(plist, key=lambda jp: jp[0])     # earlier j first
        n = len(ordered)
        for idx, (j, net) in enumerate(ordered):
            y_lbl = tip_y + LABEL_GAP + (n - 1 - idx) * STACK_DY
            ax.text(t_c, y_lbl, f"{net:+.1f}",
                    ha="center", va="bottom", fontsize=FS_ARROW,
                    fontweight="bold", fontfamily=FONT,
                    color=ARROW_GREEN, zorder=7)

    # ── allocation arrows — red, DOWN, base on project-box BOTTOM edge ─────
    for pi in proj_order:
        _, box_bot = box_edges[pi]
        for t in T:
            alloc = alloc_map.get((pi, t), 0.0)
            if alloc <= 0:
                continue
            tip_y = box_bot - ARROW_LEN
            ax.annotate("", xy=(t, tip_y), xytext=(t, box_bot),
                        arrowprops=dict(arrowstyle="-|>", color=ALLOC_RED,
                                        lw=1.5, mutation_scale=9,
                                        shrinkA=0, shrinkB=0), zorder=6)
            ax.text(t, tip_y - LABEL_GAP, f"−{alloc:.0f}",
                    ha="center", va="top", fontsize=FS_ARROW,
                    fontweight="bold", fontfamily=FONT,
                    color=ALLOC_RED, zorder=7)

    # ── efficiency circles — fixed physical size, on box bottom edge ──────
    CIRC_DIAM_IN = 0.30
    circ_w = CIRC_DIAM_IN * din_x
    circ_h = CIRC_DIAM_IN * din_y
    for pi in proj_order:
        _, box_bot = box_edges[pi]
        pst, pft = proj_info[pi]["start"], proj_info[pi]["finish"]
        for t in range(pst, min(pft, horizon) + 1):
            key = (pi, t)
            if key not in eff_map:
                continue
            eta = eff_map[key]
            ax.add_patch(mpatches.Ellipse((t, box_bot), circ_w, circ_h,
                                           fc=BG, ec=ALLOC_RED, lw=1.2,
                                           zorder=8))
            ax.text(t, box_bot, f"{eta:.2f}", ha="center", va="center",
                    fontsize=FS_EFF, fontfamily=FONT, color=ALLOC_RED,
                    fontweight="bold", zorder=9)

    # ── sidebar card: headers + legend, one column, top-right ──────────────
    brd, side_bg = SA["tl"]
    pad_frac  = 0.085
    content_w = (X_SIDE_R - X_SIDE_L) * (1 - 2 * pad_frac)
    xc_l = X_SIDE_L + (X_SIDE_R - X_SIDE_L) * pad_frac
    xc_r = xc_l + content_w

    y0 = Y_HI - TOP_INSET

    def side_text(yy, s, fs, color, weight="normal"):
        ax.text(xc_l, yy, s, ha="left", va="top", fontsize=fs,
                fontweight=weight, fontfamily=FONT, color=color, zorder=5)

    y = y0
    side_text(y, "EPISODE CASH-FLOW TIMELINE", 8.4, brd, "bold")
    y -= EYEBROW_GAP
    side_text(y, f"{cfg_id}", 10.4, TH, "bold")
    y -= TITLE_GAP
    side_text(y, f"Solver: {method.upper()}", 8.2, TM)
    y -= SOLVER_GAP
    info_lines = [
        f"Episode {ep_id}",
        f"γ = {gamma}      T = {horizon}",
        f"{n_proj} projects",
        f"B₀ = {bud0:,.0f}",
    ]
    for line in info_lines:
        side_text(y, line, 7.9, TB)
        y -= INFO_GAP

    y -= PRE_DIV_GAP
    ax.plot([xc_l, xc_r], [y, y], color=brd + "60", lw=0.9, zorder=5)
    y -= POST_DIV_GAP

    side_text(y, "LEGEND", 8.2, brd, "bold")
    y -= LEGHDR_GAP

    icon_x = xc_l + 0.095 * content_w
    text_x = xc_l + 0.22 * content_w

    def icon_arrow(y_mid, up, col):
        d = 0.15
        if up:
            ax.annotate("", xy=(icon_x, y_mid + d), xytext=(icon_x, y_mid - d),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7,
                                        mutation_scale=10), zorder=6)
        else:
            ax.annotate("", xy=(icon_x, y_mid - d), xytext=(icon_x, y_mid + d),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7,
                                        mutation_scale=10), zorder=6)

    def icon_circle(y_mid, col):
        w, h = 0.28 * din_x, 0.28 * din_y
        ax.add_patch(mpatches.Ellipse((icon_x, y_mid), w, h,
                                       fc=BG, ec=col, lw=1.3, zorder=6))

    def icon_box(y_mid, col):
        w, h = 0.30 * din_x, 0.26 * din_y
        ax.add_patch(mpatches.FancyBboxPatch(
            (icon_x - w / 2, y_mid - h / 2), w, h,
            boxstyle="round,pad=0.01", lw=1.3, ec=col, fc=col + "30",
            zorder=6))

    legend_rows = [
        ("arrow_up",   ARROW_GREEN, "Payment\n(value on arrow)"),
        ("arrow_down", ALLOC_RED,   "Allocation\n(value under arrow)"),
        ("circle",     ALLOC_RED,   "η efficiency\nper timestep"),
        ("box",        PC[0],       "Project\nactive window"),
    ]
    for kind, col, label in legend_rows:
        y_mid = y - 0.10
        if kind == "arrow_up":
            icon_arrow(y_mid, True, col)
        elif kind == "arrow_down":
            icon_arrow(y_mid, False, col)
        elif kind == "circle":
            icon_circle(y_mid, col)
        elif kind == "box":
            icon_box(y_mid, col)
        ax.text(text_x, y_mid, label, ha="left", va="center",
                fontsize=7.6, fontfamily=FONT, color=TB, zorder=6,
                linespacing=1.25)
        y -= ITEM_GAP

    y -= END_GAP
    card_bottom = y

    ax.add_patch(mpatches.FancyBboxPatch(
        (X_SIDE_L, card_bottom), X_SIDE_R - X_SIDE_L, (y0 + TOP_INSET) - card_bottom,
        boxstyle="round,pad=0.018", lw=1.0, ec=brd + "80", fc=side_bg,
        zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (X_SIDE_L, card_bottom), 0.045 * din_x, (y0 + TOP_INSET) - card_bottom,
        lw=0, fc=brd, zorder=3))

    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  1b · NET CASH-FLOW CHART  (separate focused plot)
# ─────────────────────────────────────────────────────────────────────────────
def _build_ncf_chart(out_path: Path, data: dict) -> None:
    """
    Standalone net-cashflow + budget chart.  Legends are placed OUTSIDE the
    axes on the right (so they never overlap bars / value labels); the
    saved figure auto-expands to include them (bbox_inches='tight').
    """
    cfg_id  = data["cfg_id"]
    ep_id   = data["ep_id"]
    horizon = data["horizon"]
    pf_s    = data["pf_s"]

    T    = list(range(horizon + 1))
    ncfs = [float(pf_s[pf_s["t_episode"] == t]["net_cashflow"].iloc[0])
            if not pf_s[pf_s["t_episode"] == t].empty else 0.0
            for t in T]
    buds = [float(pf_s[pf_s["t_episode"] == t]["budget_available"].iloc[0])
            if not pf_s[pf_s["t_episode"] == t].empty else 0.0
            for t in T]

    FW = 8.6     # a touch wider to host the outside legend comfortably
    FH = max(4.2, min(6.0, 3.0 + horizon * 0.12))

    fig, (ax_ncf, ax_bud) = plt.subplots(
        2, 1, figsize=(FW, FH), facecolor=BG,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55})

    for ax in (ax_ncf, ax_bud):
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(RULE)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(colors=TM, labelsize=7.5)
        ax.set_xlim(-0.6, horizon + 0.6)
        ax.set_xticks(T)
        ax.xaxis.grid(True, color=RULE, lw=0.35, zorder=0)
        ax.set_axisbelow(True)

    # ── NCF bars ──────────────────────────────────────────────────────────────
    COL_POS = "#2060C0"
    COL_NEG = "#B03030"
    ax_ncf.axhline(0, color="#888", lw=0.8, zorder=2)
    ax_ncf.set_title("Net Cash-Flow per Period  (ΔBₜ)",
                     fontsize=8.5, fontfamily=FONT,
                     fontweight="bold", color=SA["ps"][0], pad=6)
    ax_ncf.set_ylabel("ΔBₜ", fontsize=7.5, fontfamily=FONT, color=TM)

    for t, ncf in zip(T, ncfs):
        if ncf == 0:
            continue
        col = COL_POS if ncf > 0 else COL_NEG
        ax_ncf.bar(t, ncf, color=col + "40", ec=col,
                   lw=0.8, width=0.55, zorder=3)
        va  = "bottom" if ncf > 0 else "top"
        off = ncf * 0.03
        ax_ncf.text(t, ncf + off, f"{ncf:+,.0f}",
                    ha="center", va=va, fontsize=6.5,
                    fontfamily=FONT, color=col, fontweight="bold", zorder=4)

    ax_ncf.set_xticklabels([f"t{t}" for t in T], fontsize=7.0, color=TM)
    ax_ncf.yaxis.set_tick_params(labelsize=7.0)

    ncf_abs_max = max(abs(v) for v in ncfs if v != 0) * 1.30 if any(ncfs) else 1
    ax_ncf.set_ylim(-ncf_abs_max, ncf_abs_max)

    pos_patch = mpatches.Patch(fc=COL_POS + "40", ec=COL_POS, lw=0.8,
                                label="Surplus  ΔBₜ > 0")
    neg_patch = mpatches.Patch(fc=COL_NEG + "40", ec=COL_NEG, lw=0.8,
                                label="Deficit  ΔBₜ < 0")
    ax_ncf.legend(handles=[pos_patch, neg_patch],
                  loc="upper left", bbox_to_anchor=(1.015, 1.02),
                  borderaxespad=0.0, fontsize=6.8, frameon=True,
                  facecolor=BG, edgecolor=RULE, framealpha=0.95,
                  handlelength=1.3, labelspacing=0.8)

    # ── Budget curve ──────────────────────────────────────────────────────────
    bmin = min(buds); bmax = max(buds)
    margin = max((bmax - bmin) * 0.18, 1)
    ax_bud.set_ylim(bmin - margin, bmax + margin)
    ax_bud.set_title("Portfolio Budget  (Bₜ)",
                     fontsize=8.5, fontfamily=FONT,
                     fontweight="bold", color=TM, pad=6)
    ax_bud.set_ylabel("Bₜ", fontsize=7.5, fontfamily=FONT, color=TM)

    ax_bud.plot(T, buds, color=TM, lw=1.4, ls="-", marker="o",
                ms=4.5, zorder=4, label="Bₜ  portfolio budget")
    ax_bud.fill_between(T, buds, bmin - margin,
                        color=TM + "18", zorder=1)

    for t, b in zip(T, buds):
        off = (bmax - bmin) * 0.035 + margin * 0.1
        ax_bud.text(t, b + off, f"{b:,.0f}",
                    ha="center", va="bottom", fontsize=6.0,
                    fontfamily=FONT, color=TM, zorder=5)

    ax_bud.set_xticks(T)
    ax_bud.set_xticklabels([f"t{t}" for t in T], fontsize=7.0, color=TM)
    ax_bud.yaxis.set_tick_params(labelsize=7.0)
    ax_bud.legend(loc="upper left", bbox_to_anchor=(1.015, 1.02),
                  borderaxespad=0.0, fontsize=6.8, frameon=True,
                  facecolor=BG, edgecolor=RULE, framealpha=0.95,
                  handlelength=1.6)

    fig.subplots_adjust(right=0.82)   # reserve room for outside legends

    fig.text(0.41, 0.01,
             f"{cfg_id}  ·  episode {ep_id}  ·  T={horizon}",
             ha="center", fontsize=7.0, fontfamily=FONT, color=TM)

    fig.patch.set_facecolor(BG)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  2 · PORTFOLIO STATUS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _build_portfolio_table(out_path: Path, data: dict) -> None:
    gamma    = data["gamma"]
    horizon  = data["horizon"]
    adv_t0   = data["adv_t0"]
    pf_s     = data["pf_s"]
    n_proj   = data["n_proj"]
    proj_ncf = data["proj_ncf"]

    ps_hdr = ["t", "γᵗ", "Bₜ", "ΔBₜ",
              *[f"P{i}·ΔB" for i in range(n_proj)],
              "r(t)=γᵗ·ΔBₜ", "ΣR"]
    pc_w   = round(0.200 / max(n_proj, 1), 3)
    ps_cw  = ([0.055, 0.075, 0.115, 0.095]
              + [pc_w] * n_proj
              + [0.150, 0.110])
    total  = sum(ps_cw)
    ps_cw  = [w / total for w in ps_cw]
    ps_al  = (["center", "center", "right", "right"]
              + ["right"] * n_proj
              + ["right", "right"])

    running = 0.0
    rows_ps = [ps_hdr]
    for _, row in pf_s.iterrows():
        t    = int(row["t_episode"])
        ncf  = float(row["net_cashflow"])
        bud  = float(row["budget_available"])
        disc = gamma ** t
        pr   = float(row["reward"])
        running += pr
        pcols = [_sf(proj_ncf.get((i, t), 0.0), 1) for i in range(n_proj)]
        rows_ps.append([
            f"t{t}", f"{disc:.3f}", f"{bud:,.0f}",
            _sf(ncf, 1) if ncf else "0",
            *pcols,
            _sf(pr, 2) if pr else "0",
            f"{running:+.2f}",
        ])
    rows_ps.append(["∑", "", "", ""] + [""] * n_proj
                   + [f"ΣR = {running:+.3f}", ""])

    # ── exact-fit layout (inches) — no residual blank space ───────────────
    FW     = 8.27
    MARGIN = 0.10
    HDR_H  = 0.28
    RH     = 0.195
    FOOT_H = 0.22
    N_ROWS = len(rows_ps)
    FH = MARGIN + HDR_H + N_ROWS * RH + FOOT_H + MARGIN

    fig, ax = _fitted_axes(FW, FH)
    y = MARGIN
    y = _hdr(ax, "PORTFOLIO STATUS  —  Period Reward Computation  (γ = 0.95)",
             "ps", y=y, h=HDR_H, fs=8.5)
    y = _table(ax, rows_ps, ps_cw, RH, "ps",
               fs=7.0, align=ps_al, y0=y, pad=0.015)
    ax.text(0.006, y + 0.09,
            f"* t=0 advance (+{adv_t0:.0f}) credited to Bₜ at reset — not a reward step.  "
            f"ΣR = Σ r(t), t ∈ [0, {horizon}]",
            ha="left", va="top", fontsize=6.3, fontfamily=FONT,
            color=TM, zorder=5)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  3 · OBSERVATION TABLE  (table + right-hand one-column legend)
# ─────────────────────────────────────────────────────────────────────────────
def _build_observation_table(out_path: Path, data: dict) -> None:
    horizon    = data["horizon"]
    obs        = data["obs"]
    proj_order = data["proj_order"]
    status_map = data["status_map"]

    OBS_FIELDS = [
        ("t",      "t_episode"),
        ("τᵢ",     "t_project"),
        ("ΔBᵢ",    "net_cashflow"),
        ("θᵣₑₘ",   "tolerance_remain"),
        ("xᶜᵘ",    "catchup_alloc_t"),
        ("xᶜᵘ⁺¹",  "catchup_alloc_next_t"),
        ("xʳᵖ",    "reach_plan_t"),
        ("Δψ*",    "target_progress_gap"),
        ("Δτ*",    "target_timestep_gap"),
        ("x*",     "target_required_alloc"),
        ("NPV*",   "target_npv"),
    ]
    INT_FIELDS = {"t_episode", "t_project", "tolerance_remain",
                  "target_timestep_gap"}

    ob_hdr = ["Proj", "t", "τᵢ", "ΔBᵢ", "θᵣₑₘ",
              "xᶜᵘ", "xᶜᵘ⁺¹", "xʳᵖ",
              "Δψ*", "Δτ*", "x*", "NPV*"]
    ob_cw  = [0.055, 0.052, 0.052, 0.085, 0.065,
              0.082, 0.082, 0.080,
              0.065, 0.055, 0.068, 0.059]
    total  = sum(ob_cw)
    ob_cw  = [w / total for w in ob_cw]
    ob_al  = ["center", "center", "center", "right", "center",
              "right", "right", "right",
              "right", "right", "right", "right"]

    active_obs = []
    for t in range(horizon + 1):
        for pi in proj_order:
            row = obs[(obs["i"] == pi) & (obs["t_episode"] == t)]
            if row.empty:
                continue
            r  = row.iloc[0]
            st = status_map.get((pi, t), "pending")
            if st != "active":
                continue
            has_data = (float(r.get("catchup_alloc_t", 0)) > 0
                        or float(r.get("target_required_alloc", 0)) > 0)
            if not has_data:
                continue
            entry = {"i": pi}
            for _, col_name in OBS_FIELDS:
                entry[col_name] = float(r[col_name]) if col_name in r.index else 0.0
            active_obs.append(entry)

    prev_i  = None
    rows_ob = [ob_hdr]
    for entry in active_obs:
        pi     = entry["i"]
        pi_lbl = f"P{pi}" if pi != prev_i else ""
        prev_i = pi
        vals   = []
        for _, code in OBS_FIELDS:
            v = entry[code]
            if code in INT_FIELDS:
                vals.append(str(int(v)))
            elif code == "target_progress_gap":
                vals.append(f"{v:.2f}")
            elif code == "target_npv":
                vals.append(f"{v:+.1f}")
            else:
                vals.append(f"{v:.1f}")
        rows_ob.append([pi_lbl] + vals)

    # ── layout: table (left) + one-column symbol legend (right) ──────────────
    TABLE_W = 0.72
    GAP     = 0.02
    LEG_X0  = TABLE_W + GAP
    LEG_W   = 1.0 - LEG_X0

    MARGIN = 0.10
    HDR_H  = 0.28
    RH     = 0.195
    N_ROWS = len(rows_ob)

    syms = [
        ("xᶜᵘ",   "Catch-up alloc (t)"),
        ("xᶜᵘ⁺¹", "Catch-up alloc (t+1)"),
        ("xʳᵖ",   "Reach-plan alloc"),
        ("Δψ*",   "Target progress gap"),
        ("Δτ*",   "Target timestep gap"),
        ("x*",    "Target required alloc"),
        ("NPV*",  "Target milestone NPV"),
        ("θᵣₑₘ",  "Tolerance remaining"),
    ]
    LEG_HDR_H  = 0.26
    LEG_ITEM_H = 0.33
    legend_content_h = LEG_HDR_H + len(syms) * LEG_ITEM_H + 0.06

    table_content_h = HDR_H + N_ROWS * RH + 0.02
    body_h = max(table_content_h, legend_content_h)
    FW = 8.27
    FH = MARGIN + body_h + MARGIN

    fig, ax = _fitted_axes(FW, FH)

    y0 = MARGIN
    y_after_hdr = _hdr(ax, "AGENT OBSERVATION  oₜ  —  Active Periods Only",
                        "obs", y=y0, h=HDR_H, fs=8.5)
    _table(ax, rows_ob, ob_cw, RH, "obs", fs=6.8, align=ob_al,
           y0=y_after_hdr, pad=0.012, x0=0.0, width=TABLE_W)

    # ── right-hand symbol legend card (sized to its own content) ──────────
    brd, hbg = SA["obs"]
    leg_top = y0
    leg_h   = legend_content_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (LEG_X0, leg_top), LEG_W, leg_h,
        boxstyle="round,pad=0.008", lw=0.8, ec=brd + "70", fc=hbg,
        clip_on=True, zorder=3))
    ax.text(LEG_X0 + LEG_W / 2, leg_top + LEG_HDR_H / 2, "SYMBOLS",
            ha="center", va="center", fontsize=7.2, fontweight="bold",
            fontfamily=FONT, color=brd, zorder=4)
    ax.plot([LEG_X0 + 0.012, LEG_X0 + LEG_W - 0.012],
            [leg_top + LEG_HDR_H, leg_top + LEG_HDR_H],
            color=brd + "50", lw=0.6, zorder=4)
    for i, (s, d) in enumerate(syms):
        ly = leg_top + LEG_HDR_H + 0.05 + i * LEG_ITEM_H
        ax.text(LEG_X0 + 0.02, ly, s, ha="left", va="top",
                fontsize=7.0, fontweight="bold", fontfamily=FONT,
                color=brd, zorder=4)
        ax.text(LEG_X0 + 0.02, ly + 0.145, d, ha="left", va="top",
                fontsize=6.0, fontfamily=FONT, color=TB, zorder=4)

    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def build(data_dir: Path, out_root: Path) -> None:
    """Load data, create infograph/ folder, write all output images."""

    # ── load ──────────────────────────────────────────────────────────────────
    pp  = pd.read_csv(data_dir / "projects_profile.csv")
    mp  = pd.read_csv(data_dir / "milestones_profile.csv")
    ms  = pd.read_csv(data_dir / "milestones_status.csv")
    ps  = pd.read_csv(data_dir / "projects_status.csv")
    pf  = pd.read_csv(data_dir / "portfolios.csv")
    obs = pd.read_csv(data_dir / "projects_observation.csv")

    # ── shared derived data ───────────────────────────────────────────────────
    cfg_id  = pp["config_id"].iloc[0]
    ep_id   = pp["episode_id"].iloc[0]
    method  = ms["method"].iloc[0]
    n_proj  = len(pp)
    pf_s    = pf.sort_values("t_episode").reset_index(drop=True)
    horizon = int(pf_s["t_episode"].max())
    gamma   = 0.95
    bud0    = pf_s["budget_available"].iloc[0]

    proj_info: dict[int, dict] = {}
    for _, r in pp.iterrows():
        i = int(r["i"])
        proj_info[i] = dict(
            bac    = int(r["bac"]),
            price  = int(r["price"]),
            start  = int(r["planned_start"]),
            finish = int(r["planned_finish"]),
        )

    alloc_map:  dict[tuple, float] = {}
    status_map: dict[tuple, str]   = {}
    eff_map:    dict[tuple, float] = {}
    for _, r in ps.iterrows():
        k = (int(r["i"]), int(r["t_episode"]))
        alloc_map[k]  = float(r["allocation_action"])
        status_map[k] = str(r["status"])
        eff_map[k]    = float(r["efficiency"])

    proj_ncf: dict[tuple, float] = {
        (int(r["i"]), int(r["t_episode"])): float(r["net_cashflow"])
        for _, r in ps.iterrows()
    }

    cert: dict[int, list] = defaultdict(list)
    for _, r in ms.sort_values(["certified_t", "j", "i"]).iterrows():
        cert[int(r["certified_t"])].append(
            (int(r["i"]), int(r["j"]), float(r["net_payment"])))

    adv_t0 = sum(
        float(mp[(mp["i"] == i) & (mp["j"] == 0)]["net_payment"].iloc[0])
        for i in range(n_proj)
        if proj_info[i]["start"] == 0
    )

    proj_order = sorted(range(n_proj), key=lambda i: proj_info[i]["start"])

    # ── create output folder ───────────────────────────────────────────────────
    out_dir = out_root / "infograph"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── shared data bundle ─────────────────────────────────────────────────────
    data = dict(
        cfg_id=cfg_id, ep_id=ep_id, method=method,
        n_proj=n_proj, horizon=horizon, gamma=gamma, bud0=bud0,
        proj_info=proj_info, proj_order=proj_order,
        alloc_map=alloc_map, status_map=status_map, eff_map=eff_map,
        proj_ncf=proj_ncf, cert=cert, adv_t0=adv_t0,
        pf_s=pf_s, obs=obs,
    )

    # ── emit each output ───────────────────────────────────────────────────────
    _build_timeline(         out_dir / "01_timeline.png",          data)
    _build_ncf_chart(        out_dir / "01b_ncf_chart.png",        data)
    _build_portfolio_table(  out_dir / "02_portfolio_status.png",  data)
    _build_observation_table(out_dir / "03_observations.png",      data)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate infograph/ folder for one episode.")
    p.add_argument("--data", default=None,
                   help="Path to the data directory (CSV files).")
    p.add_argument("--out", default=None,
                   help="Parent directory for the infograph/ folder.")
    args = p.parse_args()

    here     = Path(__file__).resolve().parent
    data_dir = Path(args.data) if args.data else (
        here / "ground_truth" / "TC-3XMAIF-R" / "optimal")
    out_root = Path(args.out) if args.out else data_dir.parent.parent

    print(f"data : {data_dir}")
    print(f"out  : {out_root / 'infograph'}/")
    build(data_dir, out_root)
    print("Done.")


if __name__ == "__main__":
    main()