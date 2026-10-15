"""
infograph.py
============
A4-portrait infographic for one episode's optimal portfolio-budgeting solution.

Usage
-----
    python infograph.py
    python infograph.py --data ground_truth/TC-3XMAIF-R/optimal/
    python infograph.py --data ... --out my_plot.png

Default output: ground_truth/infograph.png  (next to gap_plot.png)
Canvas: 210 × 297 mm @ 150 dpi
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

BG     = "#F9F8F5"
PANEL  = "#EFEDE5"
RULE   = "#D4D0C6"
TH     = "#17161A"   # text heavy
TB     = "#383530"   # text body
TM     = "#7A766C"   # text muted

PC   = ["#3A5A82", "#2E7D52", "#944820"]   # project colours
PCL  = [c + "24" for c in PC]             # translucent fills

MC   = ["#1E5FB5", "#17966A", "#CC8A00", "#B52020"]   # j=0,1,2,3

AC   = "#6B38A0"   # allocation / outflow

# Section accent tuples: (border, header-bg)
SA = {
    "pro": ("#453890", "#ECEAF8"),
    "mil": ("#1A6448", "#E4F2EB"),
    "tl":  ("#1A4878", "#E5EFF8"),
    "rew": ("#7A3818", "#F5EAE0"),
    "leg": ("#3A3630", "#EEECEA"),
}

FONT = "DejaVu Sans"
A4W, A4H, DPI = 8.27, 11.69, 150


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, dec=2):
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.{dec}f}"
    return str(v)


def _sf(v: float, dec=2) -> str:
    return "—" if v == 0.0 else f"{v:+.{dec}f}"


def _hdr_strip(ax, title, sa_key, y=1.0, h=0.038, fs=6.6):
    """Coloured section-header strip, axes-fraction coords."""
    brd, bg = SA[sa_key]
    ax.add_patch(mpatches.Rectangle(
        (0, y - h), 1, h, lw=0, fc=bg,
        transform=ax.transAxes, clip_on=False, zorder=8))
    ax.add_patch(mpatches.Rectangle(
        (0, y - h), 0.005, h, lw=0, fc=brd,
        transform=ax.transAxes, clip_on=False, zorder=9))
    ax.text(0.013, y - h / 2, title,
            ha="left", va="center", fontsize=fs,
            fontfamily=FONT, fontweight="bold", color=brd,
            transform=ax.transAxes, clip_on=False, zorder=10)
    return y - h   # content starts here


def _table(ax, rows, col_w, row_h, sa_key,
           fs=5.9, align=None, y0=1.0, pad_top=0.010):
    """
    Draw table from y0 downward.  Returns y at bottom of last row.
    rows[0] is the header.  col_w must sum ≤ 1.
    """
    brd, hbg = SA[sa_key]
    y0 -= pad_top
    if align is None:
        align = ["center"] * len(col_w)

    for ri, row in enumerate(rows):
        yt = y0 - ri * row_h
        yb = yt - row_h
        is_hdr = ri == 0
        face = hbg if is_hdr else (BG if ri % 2 == 1 else PANEL)
        fc   = brd if is_hdr else TB
        fw   = "bold" if is_hdr else "normal"

        ax.add_patch(mpatches.Rectangle(
            (0, yb), 1, row_h, lw=0, fc=face,
            transform=ax.transAxes, clip_on=True, zorder=1))
        ax.plot([0, 1], [yb, yb], color=RULE, lw=0.3, zorder=2,
                transform=ax.transAxes, clip_on=True)

        x = 0.0
        for ci, (cell, cw, ha) in enumerate(zip(row, col_w, align)):
            if ci:
                ax.plot([x, x], [yb, yt], color=RULE, lw=0.25, zorder=2,
                        transform=ax.transAxes, clip_on=True)
            pad = 0.006
            tx = (x + pad if ha == "left"
                  else x + cw - pad if ha == "right"
                  else x + cw / 2)
            ax.text(tx, (yt + yb) / 2, str(cell),
                    ha=ha, va="center", fontsize=fs,
                    fontfamily=FONT, color=fc, fontweight=fw,
                    transform=ax.transAxes, clip_on=True, zorder=3)
            x += cw

    nr = len(rows)
    h_tot = nr * row_h + pad_top
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, y0 - nr * row_h), 1, h_tot,
        boxstyle="square,pad=0", lw=0.7,
        ec=brd + "55", fc="none",
        transform=ax.transAxes, clip_on=True, zorder=4))
    return y0 - nr * row_h


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def build(data_dir: Path, out: Path) -> None:

    pp  = pd.read_csv(data_dir / "projects_profile.csv")
    mp  = pd.read_csv(data_dir / "milestones_profile.csv")
    ms  = pd.read_csv(data_dir / "milestones_status.csv")
    ps  = pd.read_csv(data_dir / "projects_status.csv")
    pf  = pd.read_csv(data_dir / "portfolios.csv")

    cfg_id  = pp["config_id"].iloc[0]
    ep_id   = pp["episode_id"].iloc[0]
    method  = ms["method"].iloc[0]
    n_proj  = len(pp)
    pf_s    = pf.sort_values("t_episode").reset_index(drop=True)
    horizon = int(pf_s["t_episode"].max())
    gamma   = 0.95
    bud0    = pf_s["budget_available"].iloc[0]

    eff_map: dict[int, list] = {}
    for iv, grp in ps[ps["status"] == "active"].groupby("i"):
        eff_map[int(iv)] = grp.sort_values("t_episode")["efficiency"].tolist()

    alloc_map: dict[tuple, float] = {
        (int(r["i"]), int(r["t_episode"])): float(r["allocation_action"])
        for _, r in ps.iterrows()
    }
    proj_ncf: dict[tuple, float] = {
        (int(r["i"]), int(r["t_episode"])): float(r["net_cashflow"])
        for _, r in ps.iterrows()
    }

    cert: dict[int, list] = defaultdict(list)
    for _, r in ms.sort_values(["certified_t", "j", "i"]).iterrows():
        cert[int(r["certified_t"])].append(
            (int(r["i"]), int(r["j"]), float(r["net_payment"])))

    # t=0 advance credited to budget at reset — does NOT appear as a reward row
    # in portfolios.csv (reward=0 at t=0).  ΣR is simply sum of reward column.
    adv_t0 = sum(
        float(mp[(mp["i"] == i) & (mp["j"] == 0)]["net_payment"].iloc[0])
        for i in range(n_proj)
        if int(pp[pp["i"] == i]["planned_start"].iloc[0]) == 0
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(A4W, A4H), facecolor=BG)

    # Row heights (proportions): title | timeline | reward+legend
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[0.036, 0.590, 0.374],
        width_ratios=[1, 1],
        hspace=0.0, wspace=0.04,
        left=0.032, right=0.968,
        top=0.978,  bottom=0.014,
    )

    ax_ttl = fig.add_subplot(gs[0, :])
    ax_tl  = fig.add_subplot(gs[1, :])
    ax_rew = fig.add_subplot(gs[2, 0])
    ax_leg = fig.add_subplot(gs[2, 1])

    for ax in (ax_ttl, ax_tl, ax_rew, ax_leg):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    # ══════════════════════════════════════════════════════════════════════════
    #  TITLE STRIP
    # ══════════════════════════════════════════════════════════════════════════
    ax_ttl.axis("off")
    ax_ttl.set_xlim(0, 1); ax_ttl.set_ylim(0, 1)
    ax_ttl.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, lw=0, fc="#182A48",
        transform=ax_ttl.transAxes))
    ax_ttl.text(0.5, 0.65,
                f"Portfolio Budgeting  ·  {cfg_id}  ·  Solver: {method.upper()}",
                ha="center", va="center", fontsize=9.0, fontweight="bold",
                fontfamily=FONT, color="white", transform=ax_ttl.transAxes)
    ax_ttl.text(0.5, 0.12,
                (f"episode  {ep_id}   ·   γ = {gamma}   ·   "
                 f"{n_proj} projects   ·   horizon T = {horizon}   ·   "
                 f"initial budget = {bud0:,.0f}"),
                ha="center", va="center", fontsize=5.9,
                fontfamily=FONT, color="#A4B8CC", transform=ax_ttl.transAxes)

    # ══════════════════════════════════════════════════════════════════════════
    #  ROW 2  —  TIMELINE  (data coords)
    # ══════════════════════════════════════════════════════════════════════════
    ax_tl.set_xlim(-0.8, horizon + 0.8)
    ax_tl.set_ylim(-6.2, 6.8)
    ax_tl.axis("off")

    _hdr_strip(ax_tl, "EPISODE CASH-FLOW TIMELINE",
               "tl", y=1.0, h=0.022, fs=7.0)

    T = list(range(horizon + 1))

    # ── spine ──
    ax_tl.axhline(0, color=RULE, lw=1.0, zorder=1)

    # ── t ticks ──
    for t in T:
        ax_tl.plot(t, 0, "|", color=TM, ms=6, mew=1.0, zorder=2)
        ax_tl.text(t, -0.22, f"t{t}",
                   ha="center", va="top", fontsize=7.5,
                   fontfamily=FONT, color=TM)

    # ── project bars ──
    BY0 = 0.18; BH = 0.50; BG_ = 0.12

    for _, row in pp.iterrows():
        i    = int(row["i"])
        ps_t = int(row["planned_start"])
        pf_t = int(row["planned_finish"])
        col  = PC[i % len(PC)]
        yb   = BY0 + i * (BH + BG_)
        ax_tl.add_patch(mpatches.FancyBboxPatch(
            (ps_t, yb), pf_t - ps_t, BH,
            boxstyle="round,pad=0.04",
            lw=1.0, ec=col, fc=PCL[i % len(PCL)], zorder=3))
        ax_tl.text(ps_t + (pf_t - ps_t) / 2, yb + BH / 2,
                   f"P{i}  BAC {int(row['bac'])}  price {int(row['price'])}",
                   ha="center", va="center", fontsize=7.5,
                   fontfamily=FONT, fontweight="bold", color=col, zorder=4)

    # ── efficiency labels ──
    for i in range(n_proj):
        row  = pp[pp["i"] == i].iloc[0]
        ps_t = int(row["planned_start"])
        eff  = eff_map.get(i, [])
        col  = PC[i % len(PC)]
        for k, et in enumerate(range(ps_t, ps_t + len(eff))):
            oy = -0.52 - i * 0.34
            ax_tl.text(et, oy, f"η{eff[k]:.2f}",
                       ha="center", va="top", fontsize=6.2,
                       fontfamily=FONT, color=col + "BB")

    # ── inflow arrows ──
    max_stack = max((len(v) for v in cert.values()), default=1)
    TOP_BASE  = BY0 + n_proj * (BH + BG_) + 0.14
    STACK_DY  = 1.10    # vertical spacing between stacked payment labels

    for t_c, pays in sorted(cert.items()):
        for si, (i, j, net) in enumerate(pays):
            col   = MC[j % len(MC)]
            y_anc = TOP_BASE + si * STACK_DY
            y_tip = y_anc + 0.58

            ax_tl.annotate(
                "", xy=(t_c, y_anc), xytext=(t_c, y_tip),
                arrowprops=dict(
                    arrowstyle="-|>", color=col,
                    lw=1.15, mutation_scale=8,
                    shrinkA=0, shrinkB=2),
                zorder=5)

            j_lbl = "Adv" if j == 0 else f"MS{j}"
            lbl   = f"+{net:.1f}  P{i}/{j_lbl}"
            ax_tl.text(t_c, y_tip + 0.08, lbl,
                       ha="center", va="bottom", fontsize=6.5,
                       fontfamily=FONT, color=col, zorder=6,
                       bbox=dict(boxstyle="round,pad=0.18",
                                 fc=col + "10", ec=col + "50", lw=0.5))

    # ── allocation arrows ──
    ATOP = BY0 - 0.12
    ABOT = -2.80

    per_t: dict[int, dict] = defaultdict(dict)
    for (i, t_ep), v in alloc_map.items():
        if v > 0:
            per_t[t_ep][i] = v

    for t_ep, pa in sorted(per_t.items()):
        total = sum(pa.values())
        ax_tl.annotate(
            "", xy=(t_ep, ABOT), xytext=(t_ep, ATOP),
            arrowprops=dict(
                arrowstyle="-|>", color=AC,
                lw=1.2, linestyle="dashed",
                mutation_scale=9, shrinkA=0, shrinkB=0),
            zorder=5)
        ax_tl.text(t_ep, ABOT - 0.16, f"−{total:.0f}",
                   ha="center", va="top", fontsize=7.2,
                   fontfamily=FONT, color=AC, fontweight="bold", zorder=6)
        bdn = "  ".join(f"P{i}:{v:.0f}" for i, v in sorted(pa.items()))
        ax_tl.text(t_ep, ABOT - 0.60, bdn,
                   ha="center", va="top", fontsize=5.8,
                   fontfamily=FONT, color=AC + "AA", zorder=6)

    # ── NCF bar chart ──
    NCF_Y0 = -4.00
    SCL    = 0.0046
    ax_tl.text(-0.75, NCF_Y0, "NCF",
               ha="left", va="center", fontsize=6.2,
               fontfamily=FONT, color=TM, fontstyle="italic")
    ax_tl.axhline(NCF_Y0, xmin=0.02, color=RULE, lw=0.5, ls="--", zorder=1)

    for _, row in pf_s.iterrows():
        t   = int(row["t_episode"])
        ncf = float(row["net_cashflow"])
        if ncf == 0:
            continue
        col = "#2060C0" if ncf > 0 else "#B03030"
        hb  = ncf * SCL
        ax_tl.bar(t, hb, bottom=NCF_Y0,
                  color=col + "40", ec=col, lw=0.7, width=0.46, zorder=3)
        vy = NCF_Y0 + hb + (0.09 if ncf > 0 else -0.09)
        ax_tl.text(t, vy, f"{ncf:+.1f}",
                   ha="center", va="bottom" if ncf > 0 else "top",
                   fontsize=6.0, fontfamily=FONT, color=col, zorder=6)

    # ── budget dotted curve ──
    bud_v = pf_s["budget_available"].tolist()
    bud_t = pf_s["t_episode"].tolist()
    bmin, bmax = min(bud_v), max(bud_v)
    brng = max(bmax - bmin, 1)
    BY0b = -5.20
    bscl = 0.72 / brng
    bys  = [BY0b + (b - bmin) * bscl for b in bud_v]
    ax_tl.plot(bud_t, bys, color=TM, lw=0.8, ls=":", marker="o", ms=3.0, zorder=3)
    ax_tl.text(-0.75, BY0b, "Bud",
               ha="left", va="center", fontsize=5.8,
               fontfamily=FONT, color=TM, fontstyle="italic")

    # ══════════════════════════════════════════════════════════════════════════
    #  ROW 3-LEFT  —  REWARD TABLE
    # ══════════════════════════════════════════════════════════════════════════
    ax_rew.axis("off"); ax_rew.set_xlim(0, 1); ax_rew.set_ylim(0, 1)

    y0r = _hdr_strip(ax_rew, f"REWARD COMPUTATION  (γ = {gamma})",
                     "rew", y=1.0, h=0.038)

    r_hdr = ["t", "γᵗ", "NCF", "P0", "P1", "P2", "r(t)=γᵗ·NCF", "ΣR"]
    r_cw  = [0.07, 0.10, 0.12, 0.10, 0.10, 0.10, 0.22, 0.19]
    r_al  = ["center","center","right","right","right","right","right","right"]

    running = 0.0
    rows_r  = [r_hdr]
    for _, row in pf_s.iterrows():
        t    = int(row["t_episode"])
        ncf  = float(row["net_cashflow"])
        disc = gamma ** t
        pr   = float(row["reward"])
        running += pr
        pcols = [_sf(proj_ncf.get((i, t), 0.0), 1) for i in range(n_proj)]
        rows_r.append([
            f"t{t}",
            f"{disc:.3f}",
            _sf(ncf, 2) if ncf else "0",
            *pcols,
            _sf(pr, 2) if pr else "0",
            f"{running:+.2f}",
        ])
    rows_r.append(["∑", "", "", "", "", "",
                   f"ΣR = {running:+.3f}", ""])

    y_bot_r = _table(ax_rew, rows_r, r_cw, 0.070, "rew",
                     fs=6.3, align=r_al, y0=y0r)

    ax_rew.text(0.01, y_bot_r - 0.018,
                f"*  t=0 advance (+{adv_t0:.0f}) credited to budget at reset — not a reward row.  "
                f"ΣR = Σ r(t),  t∈[0,{horizon}]",
                ha="left", va="top", fontsize=6.0, fontfamily=FONT,
                color=TM, transform=ax_rew.transAxes, zorder=3)

    # ══════════════════════════════════════════════════════════════════════════
    #  ROW 3-RIGHT  —  LEGEND + FORMULA
    # ══════════════════════════════════════════════════════════════════════════
    ax_leg.axis("off"); ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

    y0l = _hdr_strip(ax_leg, "LEGEND  &  NOTATION",
                     "leg", y=1.0, h=0.038)
    y0l -= 0.018

    IX = 0.06; TX = 0.15; FS = 6.8; DY = 0.082

    def _arw(y, col, up=True, dash=False):
        ls = "dashed" if dash else "solid"
        d  = 0.025
        kw = dict(arrowstyle="-|>", color=col, lw=1.15,
                  linestyle=ls, mutation_scale=8)
        if up:
            ax_leg.annotate("", xy=(IX, y + d), xytext=(IX, y - d),
                            arrowprops=kw, transform=ax_leg.transAxes)
        else:
            ax_leg.annotate("", xy=(IX, y - d), xytext=(IX, y + d),
                            arrowprops=kw, transform=ax_leg.transAxes)

    def _box(y, col):
        ax_leg.add_patch(mpatches.FancyBboxPatch(
            (IX - 0.042, y - 0.022), 0.084, 0.044,
            boxstyle="round,pad=0.004", lw=0.7,
            ec=col, fc=col + "28", transform=ax_leg.transAxes))

    def _rect(y, col):
        ax_leg.add_patch(mpatches.Rectangle(
            (IX - 0.036, y - 0.018), 0.072, 0.036,
            lw=0, fc=col + "50", transform=ax_leg.transAxes))

    def _dot_line(y):
        ax_leg.plot([IX - 0.04, IX + 0.04], [y, y],
                    color=TM, lw=0.7, ls=":", marker="o", ms=2.4,
                    transform=ax_leg.transAxes)

    def _lt(y, txt, col=None, fs=FS):
        ax_leg.text(TX, y, txt, ha="left", va="center",
                    fontsize=fs, fontfamily=FONT,
                    color=col or TB, transform=ax_leg.transAxes)

    y = y0l
    _arw(y, MC[0], up=True);           _lt(y, "Advance payment  (j=0, inflow)"); y -= DY
    _arw(y, MC[1], up=True);           _lt(y, "Milestone payment  (j=1, inflow)"); y -= DY
    _arw(y, MC[2], up=True);           _lt(y, "Milestone payment  (j=2, inflow)"); y -= DY
    _arw(y, MC[3], up=True);           _lt(y, "Milestone payment  (j=3, inflow)"); y -= DY
    _arw(y, AC, up=False, dash=True);  _lt(y, "Allocation outflow  (Σ per period)"); y -= DY
    for i in range(n_proj):
        _box(y, PC[i]);                _lt(y, f"Project P{i} active window"); y -= DY
    _rect(y, "#2060C0");               _lt(y, "Net cash-flow > 0  (period surplus)"); y -= DY
    _rect(y, "#B03030");               _lt(y, "Net cash-flow < 0  (period deficit)"); y -= DY
    _dot_line(y);                      _lt(y, "Portfolio budget  (dotted)"); y -= DY + 0.008

    # formula panel
    fp_h = 0.230
    y -= 0.004
    ax_leg.add_patch(mpatches.FancyBboxPatch(
        (0.01, y - fp_h), 0.98, fp_h,
        boxstyle="round,pad=0.008", lw=0.6,
        ec=SA["rew"][0] + "70", fc=SA["rew"][1],
        transform=ax_leg.transAxes))

    fy = y - 0.024
    def _fl(txt, dy, bold=False, col=None, fs=6.5):
        nonlocal fy
        ax_leg.text(0.05, fy, txt,
                    ha="left", va="top", fontsize=fs,
                    fontfamily=FONT, fontweight="bold" if bold else "normal",
                    color=col or SA["rew"][0],
                    transform=ax_leg.transAxes)
        fy -= dy

    _fl("Reward formula:", 0.038, bold=True, fs=6.8)
    _fl("r(t)  =  γᵗ  ×  NCF(t)", 0.042, col=SA["rew"][0], fs=7.2)
    _fl("NCF(t)  =  Σ inflow(t)  −  Σ outflow(t)", 0.036, col=TB, fs=6.4)
    _fl("outflow  =  allocation  +  interest(treasury_draw)", 0.036, col=TB, fs=6.4)
    _fl(f"ΣR  =  Σ r(t),   t ∈ [0, {horizon}]   =  +12.403", 0.036, col=TM, fs=6.4)

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="A4-portrait infographic for RL portfolio budgeting")
    p.add_argument("--data", default=None,
                   help="CSV folder (default: ground_truth/TC-3XMAIF-R/optimal/)")
    p.add_argument("--out",  default=None,
                   help="Output PNG (default: ground_truth/infograph.png)")
    args = p.parse_args()

    here     = Path(__file__).resolve().parent
    data_dir = Path(args.data) if args.data else (
        here / "ground_truth" / "TC-3XMAIF-R" / "optimal")
    out_path = Path(args.out)  if args.out  else (
        data_dir.parent.parent / "infograph.png")

    build(data_dir, out_path)
    print("Done.")


if __name__ == "__main__":
    main()