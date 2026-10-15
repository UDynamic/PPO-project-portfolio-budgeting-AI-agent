"""
infograph.py  —  Episode infographic output folder.

Outputs  (all written to  <out_dir>/infograph/ )
------------------------------------------------
  01_timeline.png              — cash-flow timeline (arrows, bars, grid)
  01b_ncf_chart.png            — net cash-flow + budget curve (separate plot)
  02_portfolio_status.png      — period reward computation table
  03_observations.png          — agent observation table
  04_legend.png                — legend & notation card

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
PCL  = [c + "22" for c in PC]              # light fill
PCM  = [c + "55" for c in PC]              # medium fill

MC   = ["#1854A8", "#17966A", "#CC8A00", "#B52020"]   # milestone j=0..3

SA = {                                      # (border-colour, header-bg)
    "tl":  ("#1A4878", "#E4EEF7"),
    "ps":  ("#7A3818", "#F4E8DF"),
    "obs": ("#1A6448", "#E3F1E9"),
    "leg": ("#3A3630", "#EEECEA"),
}

FONT = "DejaVu Sans"
DPI  = 150

# Timeline grid constants
N_GRID    = 9
ROW_H     = 1.0
BAR_PAD   = 0.12
PROJ_ROWS = 3


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sf(v: float, d: int = 2) -> str:
    return "—" if v == 0.0 else f"{v:+.{d}f}"


def _hdr(ax, title: str, key: str, y: float = 1.0,
         h: float = 0.048, fs: float = 8.0) -> float:
    """Draw a coloured header band; return y of its bottom edge."""
    brd, bg = SA[key]
    ax.add_patch(mpatches.Rectangle(
        (0, y - h), 1, h, lw=0, fc=bg,
        transform=ax.transAxes, clip_on=False, zorder=8))
    ax.add_patch(mpatches.Rectangle(
        (0, y - h), 0.005, h, lw=0, fc=brd,
        transform=ax.transAxes, clip_on=False, zorder=9))
    ax.text(0.012, y - h / 2, title,
            ha="left", va="center", fontsize=fs,
            fontfamily=FONT, fontweight="bold", color=brd,
            transform=ax.transAxes, clip_on=False, zorder=10)
    return y - h


def _table(ax, rows, cw, rh: float, key: str,
           fs: float = 7.0, align=None, y0: float = 1.0,
           pad: float = 0.008) -> float:
    """Render a table; return y of bottom edge."""
    brd, hbg = SA[key]
    y0 -= pad
    if align is None:
        align = ["center"] * len(cw)
    for ri, row in enumerate(rows):
        yt = y0 - ri * rh
        yb = yt - rh
        is_hdr = ri == 0
        face = hbg if is_hdr else (BG if ri % 2 == 1 else PANEL)
        fc   = brd if is_hdr else TB
        fw   = "bold" if is_hdr else "normal"
        ax.add_patch(mpatches.Rectangle(
            (0, yb), 1, rh, lw=0, fc=face,
            transform=ax.transAxes, clip_on=True, zorder=1))
        ax.plot([0, 1], [yb, yb], color=RULE, lw=0.3, zorder=2,
                transform=ax.transAxes)
        x = 0.0
        for ci, (cell, w, ha) in enumerate(zip(row, cw, align)):
            if ci:
                ax.plot([x, x], [yb, yt], color=RULE, lw=0.2, zorder=2,
                        transform=ax.transAxes)
            p  = 0.006
            tx = (x + p if ha == "left"
                  else x + w - p if ha == "right"
                  else x + w / 2)
            ax.text(tx, (yt + yb) / 2, str(cell),
                    ha=ha, va="center", fontsize=fs,
                    fontfamily=FONT, color=fc, fontweight=fw,
                    transform=ax.transAxes, clip_on=True, zorder=3)
            x += w
    nr = len(rows)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, y0 - nr * rh), 1, nr * rh + pad,
        boxstyle="square,pad=0", lw=0.7, ec=brd + "55", fc="none",
        transform=ax.transAxes, clip_on=True, zorder=4))
    return y0 - nr * rh


def _bare_ax(fw: float, fh: float):
    """Return (fig, ax) with spines/ticks stripped."""
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  1 · TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
def _build_timeline(out_path: Path, data: dict) -> None:
    """
    Cash-flow timeline — no background row fills, vertical period grid,
    green upward payment arrows, red solid downward allocation lines,
    efficiency circles at arrow bases, project bars label-free.
    NCF chart removed — lives in 01b_ncf_chart.png.
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
    status_map = data["status_map"]
    eff_map    = data["eff_map"]
    cert       = data["cert"]

    ARROW_GREEN = "#1A8C4E"   # all payment arrows
    ALLOC_RED   = "#C0392B"   # all allocation lines

    def proj_grid_top(rank: int) -> float:
        return N_GRID - rank * PROJ_ROWS

    SPINE_Y = 0.0
    Y_LO    = -0.50           # just below spine — NCF removed from this plot
    Y_HI    = N_GRID + 0.85

    FW, FH = 8.27, 10.20
    fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
    ax.set_facecolor(BG)      # transparent grid — no row fills
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.75, horizon + 0.75)
    ax.set_ylim(Y_LO, Y_HI)

    # ── title band ────────────────────────────────────────────────────────────
    ttl_y = Y_HI - 0.05
    span  = horizon + 1.5
    ax.add_patch(mpatches.Rectangle(
        (-0.75, ttl_y - 0.58), span, 0.62,
        lw=0, fc="#182A48", zorder=0, clip_on=False))
    ax.text(horizon / 2, ttl_y - 0.19,
            f"Portfolio Budgeting  ·  {cfg_id}  ·  Solver: {method.upper()}",
            ha="center", va="center", fontsize=10.0, fontweight="bold",
            fontfamily=FONT, color="white")
    ax.text(horizon / 2, ttl_y - 0.46,
            f"episode {ep_id}   ·   γ={gamma}   ·   "
            f"{n_proj} projects   ·   T={horizon}   ·   B₀={bud0:,.0f}",
            ha="center", va="center", fontsize=7.0,
            fontfamily=FONT, color="#A4B8CC")

    # section label
    ax.text(-0.70, N_GRID + 0.56, "EPISODE CASH-FLOW TIMELINE",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            fontfamily=FONT, color=SA["tl"][0])

    T = list(range(horizon + 1))

    # ── vertical period grid (behind everything) ──────────────────────────────
    for t in T:
        ax.axvline(t, color=RULE, lw=0.35, zorder=0,
                   ymin=(Y_LO - Y_LO) / (Y_HI - Y_LO),
                   ymax=(N_GRID - Y_LO) / (Y_HI - Y_LO))

    # ── horizontal grid lines (no fills — BG shows through) ──────────────────
    for row_idx in range(N_GRID + 1):
        y   = row_idx * ROW_H
        lw  = 0.7 if row_idx % PROJ_ROWS == 0 else 0.25
        col = "#A8A29A" if row_idx % PROJ_ROWS == 0 else RULE
        ax.axhline(y, color=col, lw=lw, zorder=1, xmin=0.04, xmax=0.97)

    # ── project label strip (left margin) ────────────────────────────────────
    for rank, pi in enumerate(proj_order):
        top_y = proj_grid_top(rank)
        mid_y = top_y - PROJ_ROWS / 2
        ax.text(-0.55, mid_y,
                f"P{pi}\nBAC {proj_info[pi]['bac']}\nprice {proj_info[pi]['price']}",
                ha="center", va="center", fontsize=6.5,
                fontfamily=FONT, fontweight="bold", color=PC[pi],
                zorder=6, linespacing=1.35)

    # ── spine & time labels ───────────────────────────────────────────────────
    ax.axhline(SPINE_Y, color="#888", lw=1.0, zorder=2)
    for t in T:
        ax.plot(t, SPINE_Y, "|", color=TM, ms=5, mew=0.9, zorder=3)
        ax.text(t, SPINE_Y - 0.16, f"t{t}",
                ha="center", va="top", fontsize=7.2,
                fontfamily=FONT, color=TM)

    # ── project bars — label-free, just colour band ───────────────────────────
    for rank, pi in enumerate(proj_order):
        pst   = proj_info[pi]["start"]
        pft   = proj_info[pi]["finish"]
        col   = PC[pi]
        top_y = proj_grid_top(rank)
        bbot  = top_y - 2 * ROW_H + BAR_PAD
        btop  = top_y - 1 * ROW_H - BAR_PAD
        ax.add_patch(mpatches.FancyBboxPatch(
            (pst, bbot), pft - pst, btop - bbot,
            boxstyle="round,pad=0.04",
            lw=1.4, ec=col, fc=col + "30", zorder=4))
        # project name only — no BAC/price clutter
        ax.text(pst + (pft - pst) / 2, (bbot + btop) / 2,
                f"P{pi}",
                ha="center", va="center", fontsize=9.0,
                fontfamily=FONT, fontweight="bold", color=col, zorder=5)

    # ── efficiency circles at the base of allocation arrows ──────────────────
    # drawn before arrows so arrows sit on top
    ETA_R = 0.18   # circle radius in data units
    for rank, pi in enumerate(proj_order):
        top_y        = proj_grid_top(rank)
        alloc_row_bot = top_y - 3 * ROW_H
        for t in T:
            if alloc_map.get((pi, t), 0.0) <= 0:
                continue
            if status_map.get((pi, t)) != "active":
                continue
            eta = eff_map.get((pi, t), 1.0)
            cy  = alloc_row_bot + 0.22   # just above very bottom of alloc row
            # circle
            circ = plt.Circle((t, cy), ETA_R,
                               fc=BG, ec=ALLOC_RED, lw=1.1, zorder=7)
            ax.add_patch(circ)
            ax.text(t, cy, f"{eta:.2f}",
                    ha="center", va="center", fontsize=5.8,
                    fontfamily=FONT, color=ALLOC_RED,
                    fontweight="bold", zorder=8)

    # ── milestone payment arrows — all green, all upward ─────────────────────
    LABEL_H = 0.44
    for t_c, pays in sorted(cert.items()):
        pays_by_proj: dict[int, list] = defaultdict(list)
        for pi, j, net in pays:
            pays_by_proj[pi].append((j, net))
        for pi, proj_pays in pays_by_proj.items():
            rank    = proj_order.index(pi)
            top_y   = proj_grid_top(rank)
            bar_top = top_y - ROW_H - BAR_PAD
            for si, (j, net) in enumerate(proj_pays):
                label_base = top_y - 0.08 - si * LABEL_H
                # arrow tip lands just below bar top edge
                ax.annotate("",
                    xy=(t_c, bar_top),
                    xytext=(t_c, label_base),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=ARROW_GREEN,
                                    lw=1.5, mutation_scale=10,
                                    shrinkA=2, shrinkB=2), zorder=6)
                j_lbl = "Adv" if j == 0 else f"MS{j}"
                ax.text(t_c + 0.22, (label_base + bar_top) / 2,
                        f"+{net:.1f} {j_lbl}",
                        ha="left", va="center", fontsize=6.0,
                        fontfamily=FONT, color=ARROW_GREEN, zorder=7)

    # ── allocation lines — red, solid, from bar bottom down ──────────────────
    for rank, pi in enumerate(proj_order):
        top_y         = proj_grid_top(rank)
        alloc_row_top = top_y - 2 * ROW_H
        alloc_row_bot = top_y - 3 * ROW_H
        alloc_row_mid = (alloc_row_top + alloc_row_bot) / 2
        for t in T:
            alloc = alloc_map.get((pi, t), 0.0)
            if alloc <= 0:
                continue
            bar_bot  = top_y - 2 * ROW_H - BAR_PAD + 0.02
            arrow_tip = alloc_row_bot + ETA_R * 2 + 0.12  # stop above eta circle
            ax.annotate("",
                xy=(t, arrow_tip), xytext=(t, bar_bot),
                arrowprops=dict(arrowstyle="-|>",
                                color=ALLOC_RED,
                                lw=1.4, linestyle="solid",
                                mutation_scale=9,
                                shrinkA=2, shrinkB=2), zorder=6)
            ax.text(t + 0.22, alloc_row_mid + 0.10, f"−{alloc:.0f}",
                    ha="left", va="center", fontsize=6.0,
                    fontfamily=FONT, color=ALLOC_RED, zorder=7)

    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  1b · NET CASH-FLOW CHART  (separate focused plot)
# ─────────────────────────────────────────────────────────────────────────────
def _build_ncf_chart(out_path: Path, data: dict) -> None:
    """
    Standalone net-cashflow + budget chart.
    Two stacked subplots sharing the x-axis:
      top  — ΔBₜ bar chart with value labels
      bottom — Bₜ budget curve with value labels
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

    FW = 8.27
    # height proportional to horizon
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
        # vertical grid
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
        off = ncf * 0.03 if ncf > 0 else ncf * 0.03
        ax_ncf.text(t, ncf + off, f"{ncf:+,.0f}",
                    ha="center", va=va, fontsize=6.5,
                    fontfamily=FONT, color=col, fontweight="bold", zorder=4)

    ax_ncf.set_xticklabels([f"t{t}" for t in T], fontsize=7.0, color=TM)
    ax_ncf.yaxis.set_tick_params(labelsize=7.0)

    # y-axis ticks at nice round values
    ncf_abs_max = max(abs(v) for v in ncfs if v != 0) * 1.25 if any(ncfs) else 1
    ax_ncf.set_ylim(-ncf_abs_max, ncf_abs_max)

    # ── Budget curve ──────────────────────────────────────────────────────────
    bmin = min(buds); bmax = max(buds)
    margin = max((bmax - bmin) * 0.15, 1)
    ax_bud.set_ylim(bmin - margin, bmax + margin)
    ax_bud.set_title("Portfolio Budget  (Bₜ)",
                     fontsize=8.5, fontfamily=FONT,
                     fontweight="bold", color=TM, pad=6)
    ax_bud.set_ylabel("Bₜ", fontsize=7.5, fontfamily=FONT, color=TM)

    ax_bud.plot(T, buds, color=TM, lw=1.4, ls="-", marker="o",
                ms=4.5, zorder=4, label="Bₜ")
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

    # shared footer
    fig.text(0.5, 0.01,
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
    # column widths: distribute evenly for proj cols
    pc_w   = round(0.200 / max(n_proj, 1), 3)
    ps_cw  = ([0.055, 0.075, 0.115, 0.095]
              + [pc_w] * n_proj
              + [0.150, 0.110])
    # normalise to sum=1
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
            f"t{t}",
            f"{disc:.3f}",
            f"{bud:,.0f}",
            _sf(ncf, 1) if ncf else "0",
            *pcols,
            _sf(pr, 2) if pr else "0",
            f"{running:+.2f}",
        ])
    rows_ps.append(["∑", "", "", ""] + [""] * n_proj
                   + [f"ΣR = {running:+.3f}", ""])

    # auto-size height
    RH     = 0.052
    HDR_H  = 0.055
    N_ROWS = len(rows_ps)
    FH     = max((HDR_H + N_ROWS * RH + 0.12) * 9.0, 2.8)

    fig, ax = _bare_ax(8.27, FH)
    y0 = _hdr(ax, "PORTFOLIO STATUS  —  Period Reward Computation  (γ = 0.95)",
              "ps", y=0.97, h=HDR_H, fs=8.5)
    y_bot = _table(ax, rows_ps, ps_cw, RH, "ps",
                   fs=7.2, align=ps_al, y0=y0, pad=0.008)
    ax.text(0.005, y_bot - 0.016,
            f"* t=0 advance (+{adv_t0:.0f}) credited to Bₜ at reset — not a reward step.  "
            f"ΣR = Σ r(t), t ∈ [0, {horizon}]",
            ha="left", va="top", fontsize=6.5, fontfamily=FONT,
            color=TM, transform=ax.transAxes, zorder=3)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  3 · OBSERVATION TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _build_observation_table(out_path: Path, data: dict) -> None:
    horizon    = data["horizon"]
    obs        = data["obs"]
    proj_order = data["proj_order"]
    status_map = data["status_map"]
    n_proj     = data["n_proj"]

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

    codes = [c for _, c in OBS_FIELDS]

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

    RH     = 0.052
    HDR_H  = 0.055
    N_ROWS = len(rows_ob)
    FH     = max((HDR_H + N_ROWS * RH + 0.18) * 9.0, 2.8)

    fig, ax = _bare_ax(8.27, FH)
    y0 = _hdr(ax, "AGENT OBSERVATION  oₜ  —  Active Periods Only",
              "obs", y=0.97, h=HDR_H, fs=8.5)
    y_bot = _table(ax, rows_ob, ob_cw, RH, "obs",
                   fs=7.2, align=ob_al, y0=y0, pad=0.008)

    syms = [
        ("xᶜᵘ",   "catchup alloc (current t)"),
        ("xᶜᵘ⁺¹", "catchup alloc (next-t lookahead)"),
        ("xʳᵖ",   "reach-plan alloc"),
        ("Δψ*",   "target progress gap"),
        ("Δτ*",   "target timestep gap"),
        ("x*",    "target required alloc"),
        ("NPV*",  "target milestone NPV"),
        ("θᵣₑₘ",  "tolerance remaining"),
    ]
    sym_line = "     ".join(f"{s}: {d}" for s, d in syms)
    ax.text(0.005, y_bot - 0.018, sym_line,
            ha="left", va="top", fontsize=6.5, fontfamily=FONT,
            color=TM, transform=ax.transAxes, zorder=3)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  4 · LEGEND
# ─────────────────────────────────────────────────────────────────────────────
def _build_legend(out_path: Path, data: dict) -> None:
    n_proj  = data["n_proj"]
    horizon = data["horizon"]

    FW, FH = 8.27, 3.80
    HDR_H  = 0.062
    DY     = 0.096          # row pitch
    ICON_R = 0.036          # icon half-width (axes units)
    ICON_GAP = 0.026        # gap from icon right edge to text

    C1_IX = 0.060;  C1_TX = C1_IX + ICON_R + ICON_GAP
    C2_IX = 0.390;  C2_TX = C2_IX + ICON_R + ICON_GAP
    C3_IX = 0.715;  C3_TX = C3_IX + ICON_R + ICON_GAP

    Y0    = 1.0 - HDR_H - 0.030   # top of first icon row
    FS_LBL = 8.0

    fig, ax = _bare_ax(FW, FH)
    _hdr(ax, "LEGEND & NOTATION", "leg", y=1.0, h=HDR_H, fs=8.5)

    def _icon(kind: str, col: str, x: float, y: float) -> None:
        d = 0.028
        if kind == "arrow_up":
            ax.annotate("", xy=(x, y + d), xytext=(x, y - d),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.5, mutation_scale=11),
                        transform=ax.transAxes)
        elif kind == "arrow_dn_d":
            ax.annotate("", xy=(x, y - d), xytext=(x, y + d),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.5, linestyle="solid",
                                        mutation_scale=11),
                        transform=ax.transAxes)
        elif kind == "box":
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 0.034, y - 0.024), 0.068, 0.048,
                boxstyle="round,pad=0.004", lw=1.0,
                ec=col, fc=col + "28", transform=ax.transAxes))
        elif kind == "rect":
            ax.add_patch(mpatches.Rectangle(
                (x - 0.032, y - 0.022), 0.064, 0.044,
                lw=0, fc=col + "55", transform=ax.transAxes))
        elif kind == "dotline":
            ax.plot([x - 0.036, x + 0.036], [y, y],
                    color=col, lw=1.1, ls=":", marker="o", ms=4.0,
                    transform=ax.transAxes)
        elif kind == "shade":
            ax.add_patch(mpatches.Rectangle(
                (x - 0.032, y - 0.022), 0.064, 0.044,
                lw=0, fc=col + "22", transform=ax.transAxes))
        elif kind == "circle":
            ax.add_patch(mpatches.Circle(
                (x, y), 0.026,
                fc=BG, ec=col, lw=1.1,
                transform=ax.transAxes))
            ax.text(x, y, "η", ha="center", va="center",
                    fontsize=6.5, fontfamily=FONT, color=col,
                    transform=ax.transAxes)

    def _row(kind: str, col: str, ix: float, tx: float,
             y: float, label: str) -> None:
        _icon(kind, col, ix, y)
        ax.text(tx, y, label, ha="left", va="center", fontsize=FS_LBL,
                fontfamily=FONT, color=TB, transform=ax.transAxes)

    # ── column 1 — payment arrows (all green) ────────────────────────────
    ARROW_GREEN_LEG = "#1A8C4E"
    col1 = [
        ("arrow_up", ARROW_GREEN_LEG, "Advance payment  (j = 0)"),
        ("arrow_up", ARROW_GREEN_LEG, "Milestone payment  (j = 1)"),
        ("arrow_up", ARROW_GREEN_LEG, "Milestone payment  (j = 2)"),
        ("arrow_up", ARROW_GREEN_LEG, "Milestone payment  (j = 3)"),
    ]
    for k, (kind, col, lbl) in enumerate(col1):
        _row(kind, col, C1_IX, C1_TX, Y0 - k * DY, lbl)

    # ── column 2 — allocation arrows (solid red) + budget ────────────────
    ALLOC_RED_LEG = "#C0392B"
    col2 = [
        ("arrow_dn_d", ALLOC_RED_LEG, "Allocation outflow  (solid ↓, red)"),
        ("dotline",    TM,            "Bₜ  portfolio budget  (dotted)"),
    ]
    for k, (kind, col, lbl) in enumerate(col2):
        _row(kind, col, C2_IX, C2_TX, Y0 - k * DY, lbl)

    # ── column 3 — bars, cashflow rects, efficiency circle ───────────────
    col3 = [
        ("box",    PC[0],     "Project P0  active window"),
        ("box",    PC[1],     "Project P1  active window"),
        ("box",    PC[2],     "Project P2  active window"),
        ("rect",   "#2060C0", "ΔBₜ > 0  (period surplus)"),
        ("rect",   "#B03030", "ΔBₜ < 0  (period deficit)"),
        ("circle", "#C0392B", "η  efficiency  (circle at arrow base)"),
    ]
    for k, (kind, col, lbl) in enumerate(col3):
        _row(kind, col, C3_IX, C3_TX, Y0 - k * DY, lbl)

    # ── formula box ───────────────────────────────────────────────────────
    max_rows   = max(len(col1), len(col2), len(col3))
    y_last     = Y0 - (max_rows - 1) * DY
    y_box_top  = y_last - 2 * DY - 0.010   # 2-row gap
    BOX_H      = 0.210
    BOX_PAD    = 0.018

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.010, y_box_top - BOX_H), 0.980, BOX_H,
        boxstyle="round,pad=0.010", lw=0.8,
        ec=SA["ps"][0] + "80", fc=SA["ps"][1],
        transform=ax.transAxes))

    # 3 columns × 2 rows inside the box
    CX   = [0.025, 0.360, 0.695]
    FS_F = 7.6;  FS_O = 7.2
    y_r1 = y_box_top - BOX_PAD - 0.004
    y_r2 = y_r1 - 0.098

    row1 = [
        "r(t) = γᵗ × ΔBₜ",
        "ΔBₜ = Σ inflow(t) − Σ outflow(t)",
        f"ΣR = Σ r(t),   t ∈ [0, {horizon}]",
    ]
    row2 = [
        "outflow = allocation + interest(treasury_draw)",
        "o(t) per project: [θᵣₑₘ, xᶜᵘ, xᶜᵘ⁺¹, Δψ*, Δτ*, x*, NPV*]",
        "o(t) portfolio: [ΔBₜ/B₀, Bₜ/B₀]",
    ]
    for txt, cx in zip(row1, CX):
        ax.text(cx, y_r1, txt, ha="left", va="top",
                fontsize=FS_F, fontfamily=FONT,
                color=SA["ps"][0], fontweight="bold",
                transform=ax.transAxes)
    for txt, cx in zip(row2, CX):
        ax.text(cx, y_r2, txt, ha="left", va="top",
                fontsize=FS_O, fontfamily=FONT,
                color=SA["obs"][0],
                transform=ax.transAxes)

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
    _build_legend(           out_dir / "04_legend.png",            data)


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