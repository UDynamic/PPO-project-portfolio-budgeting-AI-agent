"""
infograph.py  —  A4 portrait infographic for one optimal episode.

Outputs
-------
  <out_dir>/infograph.png   — main figure
  <out_dir>/legend.png      — horizontal legend card (separate)

Usage
-----
  python infograph.py
  python infograph.py --data ground_truth/TC-3XMAIF-R/optimal/ --out ground_truth/
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
from collections import defaultdict

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

PC   = ["#3A5A82", "#2E7D52", "#944820"]   # project 0,1,2
PCL  = [c + "22" for c in PC]
PCM  = [c + "55" for c in PC]              # medium fill

MC   = ["#1854A8", "#17966A", "#CC8A00", "#B52020"]   # j=0,1,2,3
AC   = "#6B38A0"                                        # allocation

SA = {                                      # (border, header-bg)
    "tl":  ("#1A4878", "#E4EEF7"),
    "ps":  ("#7A3818", "#F4E8DF"),
    "obs": ("#1A6448", "#E3F1E9"),
    "leg": ("#3A3630", "#EEECEA"),
}

FONT = "DejaVu Sans"
A4W, A4H, DPI = 8.27, 11.69, 150

# Grid row layout constants (9 data rows above spine, 1 below for alloc labels)
N_GRID    = 9          # rows above spine
ROW_H     = 1.0        # height of each grid row (data units)
BAR_PAD   = 0.12       # vertical pad inside each row for the project bar
PROJ_ROWS = 3          # grid rows per project


# ─────────────────────────────────────────────────────────────────────────────
#  TINY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _f(v, d=2):
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.{d}f}"
    return str(v)

def _sf(v: float, d=2):
    return "—" if v == 0.0 else f"{v:+.{d}f}"

def _hdr(ax, title, key, y=1.0, h=0.036, fs=6.6):
    brd, bg = SA[key]
    ax.add_patch(mpatches.Rectangle((0, y-h), 1, h, lw=0, fc=bg,
        transform=ax.transAxes, clip_on=False, zorder=8))
    ax.add_patch(mpatches.Rectangle((0, y-h), 0.005, h, lw=0, fc=brd,
        transform=ax.transAxes, clip_on=False, zorder=9))
    ax.text(0.012, y-h/2, title, ha="left", va="center", fontsize=fs,
            fontfamily=FONT, fontweight="bold", color=brd,
            transform=ax.transAxes, clip_on=False, zorder=10)
    return y - h

def _table(ax, rows, cw, rh, key, fs=5.8, align=None, y0=1.0, pad=0.008):
    brd, hbg = SA[key]
    y0 -= pad
    if align is None:
        align = ["center"]*len(cw)
    for ri, row in enumerate(rows):
        yt, yb = y0 - ri*rh, y0 - ri*rh - rh
        is_hdr = ri == 0
        face = hbg if is_hdr else (BG if ri%2==1 else PANEL)
        fc = brd if is_hdr else TB
        fw = "bold" if is_hdr else "normal"
        ax.add_patch(mpatches.Rectangle((0, yb), 1, rh, lw=0, fc=face,
            transform=ax.transAxes, clip_on=True, zorder=1))
        ax.plot([0,1],[yb,yb], color=RULE, lw=0.3, zorder=2,
                transform=ax.transAxes)
        x = 0.0
        for ci,(cell,w,ha) in enumerate(zip(row, cw, align)):
            if ci:
                ax.plot([x,x],[yb,yt], color=RULE, lw=0.2, zorder=2,
                        transform=ax.transAxes)
            p = 0.005
            tx = x+p if ha=="left" else x+w-p if ha=="right" else x+w/2
            ax.text(tx, (yt+yb)/2, str(cell), ha=ha, va="center",
                    fontsize=fs, fontfamily=FONT, color=fc, fontweight=fw,
                    transform=ax.transAxes, clip_on=True, zorder=3)
            x += w
    nr = len(rows)
    ax.add_patch(mpatches.FancyBboxPatch((0, y0-nr*rh), 1, nr*rh+pad,
        boxstyle="square,pad=0", lw=0.6, ec=brd+"55", fc="none",
        transform=ax.transAxes, clip_on=True, zorder=4))
    return y0 - nr*rh


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MAIN FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def build(data_dir: Path, out_dir: Path) -> None:

    # ── load ──────────────────────────────────────────────────────────────────
    pp  = pd.read_csv(data_dir / "projects_profile.csv")
    mp  = pd.read_csv(data_dir / "milestones_profile.csv")
    ms  = pd.read_csv(data_dir / "milestones_status.csv")
    ps  = pd.read_csv(data_dir / "projects_status.csv")
    pf  = pd.read_csv(data_dir / "portfolios.csv")
    obs = pd.read_csv(data_dir / "projects_observation.csv")

    cfg_id  = pp["config_id"].iloc[0]
    ep_id   = pp["episode_id"].iloc[0]
    method  = ms["method"].iloc[0]
    n_proj  = len(pp)
    pf_s    = pf.sort_values("t_episode").reset_index(drop=True)
    horizon = int(pf_s["t_episode"].max())
    gamma   = 0.95
    bud0    = pf_s["budget_available"].iloc[0]

    # per-project helpers
    proj_info = {}
    for _, r in pp.iterrows():
        i = int(r["i"])
        proj_info[i] = dict(
            bac   = int(r["bac"]),
            price = int(r["price"]),
            start = int(r["planned_start"]),
            finish= int(r["planned_finish"]),
        )

    alloc_map: dict[tuple,float] = {}
    status_map: dict[tuple,str]  = {}
    eff_map: dict[tuple,float]   = {}
    for _, r in ps.iterrows():
        k = (int(r["i"]), int(r["t_episode"]))
        alloc_map[k]  = float(r["allocation_action"])
        status_map[k] = str(r["status"])
        eff_map[k]    = float(r["efficiency"])

    proj_ncf: dict[tuple,float] = {
        (int(r["i"]), int(r["t_episode"])): float(r["net_cashflow"])
        for _, r in ps.iterrows()
    }

    cert: dict[int, list] = defaultdict(list)
    for _, r in ms.sort_values(["certified_t","j","i"]).iterrows():
        cert[int(r["certified_t"])].append(
            (int(r["i"]), int(r["j"]), float(r["net_payment"])))

    adv_t0 = sum(
        float(mp[(mp["i"]==i) & (mp["j"]==0)]["net_payment"].iloc[0])
        for i in range(n_proj)
        if proj_info[i]["start"] == 0
    )

    # grid row assignment: sort projects by planned_start ascending → row 0 = top
    proj_order = sorted(range(n_proj), key=lambda i: proj_info[i]["start"])

    # each project occupies 3 rows in the 9-row grid
    # row index (from top): proj_order[0] → rows 8,7,6 ; [1] → 5,4,3 ; [2] → 2,1,0
    def proj_grid_top(proj_rank: int) -> float:
        """Returns data-y of top of the top-row for this project (rank 0=first)."""
        return N_GRID - proj_rank * PROJ_ROWS   # e.g. 9, 6, 3

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(A4W, A4H), facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 2,
        height_ratios=[0.033, 0.462, 0.272, 0.233],
        width_ratios=[1, 1],
        hspace=0.0, wspace=0.04,
        left=0.030, right=0.970,
        top=0.978,  bottom=0.012,
    )
    ax_ttl = fig.add_subplot(gs[0, :])
    ax_tl  = fig.add_subplot(gs[1, :])
    ax_ps  = fig.add_subplot(gs[2, :])
    ax_obs = fig.add_subplot(gs[3, :])

    for ax in (ax_ttl, ax_tl, ax_ps, ax_obs):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    # ── TITLE ─────────────────────────────────────────────────────────────────
    ax_ttl.axis("off"); ax_ttl.set_xlim(0,1); ax_ttl.set_ylim(0,1)
    ax_ttl.add_patch(mpatches.Rectangle((0,0),1,1,lw=0,fc="#182A48",
        transform=ax_ttl.transAxes))
    ax_ttl.text(0.5, 0.66,
        f"Portfolio Budgeting  ·  {cfg_id}  ·  Solver: {method.upper()}",
        ha="center", va="center", fontsize=9.0, fontweight="bold",
        fontfamily=FONT, color="white", transform=ax_ttl.transAxes)
    ax_ttl.text(0.5, 0.13,
        f"episode  {ep_id}   ·   γ={gamma}   ·   {n_proj} projects"
        f"   ·   T={horizon}   ·   B₀={bud0:,.0f}",
        ha="center", va="center", fontsize=5.9,
        fontfamily=FONT, color="#A4B8CC", transform=ax_ttl.transAxes)

    # ══════════════════════════════════════════════════════════════════════════
    #  TIMELINE  (data coords: x=[−0.7 … horizon+0.7],  y=[−3.5 … 9.5])
    # ══════════════════════════════════════════════════════════════════════════
    SPINE_Y  = 0.0
    GRID_TOP = N_GRID        # 9.0
    ALLOC_Y  = -1.30         # centre of allocation arrow labels
    NCF_Y0   = -2.10         # NCF bar baseline
    BUD_Y0   = -3.10         # budget curve centre
    Y_LO     = -3.55
    Y_HI     = GRID_TOP + 0.8   # 9.8

    ax_tl.set_xlim(-0.75, horizon + 0.75)
    ax_tl.set_ylim(Y_LO, Y_HI)
    ax_tl.axis("off")
    _hdr(ax_tl, "EPISODE CASH-FLOW TIMELINE", "tl", y=1.0, h=0.022, fs=7.0)

    T = list(range(horizon + 1))

    # ── horizontal grid lines & row shading ──
    for row_idx in range(N_GRID + 1):
        y = row_idx * ROW_H
        lw = 0.6 if row_idx % PROJ_ROWS == 0 else 0.25
        col = RULE if row_idx % PROJ_ROWS != 0 else "#B0AA9E"
        ax_tl.axhline(y, color=col, lw=lw, zorder=1, xmin=0.03, xmax=0.97)

    # alternating row band shading (per project group)
    for rank, pi in enumerate(proj_order):
        top_y  = proj_grid_top(rank)
        bot_y  = top_y - PROJ_ROWS
        shade  = PCL[pi]
        ax_tl.add_patch(mpatches.Rectangle(
            (-0.75, bot_y), horizon+1.5, PROJ_ROWS,
            lw=0, fc=shade, zorder=0))

    # ── project label strip on left (row group labels) ──
    for rank, pi in enumerate(proj_order):
        top_y = proj_grid_top(rank)
        mid_y = top_y - PROJ_ROWS / 2
        col   = PC[pi]
        ax_tl.text(-0.55, mid_y,
                   f"P{pi}\nBAC {proj_info[pi]['bac']}\nprice {proj_info[pi]['price']}",
                   ha="center", va="center", fontsize=6.0,
                   fontfamily=FONT, fontweight="bold", color=col, zorder=6,
                   linespacing=1.35)

    # ── spine & t labels ──
    ax_tl.axhline(SPINE_Y, color="#888", lw=1.0, zorder=2)
    for t in T:
        ax_tl.plot(t, SPINE_Y, "|", color=TM, ms=5, mew=0.9, zorder=3)
        ax_tl.text(t, SPINE_Y - 0.14, f"t{t}",
                   ha="center", va="top", fontsize=6.8,
                   fontfamily=FONT, color=TM)

    # ── project bars (middle row of each 3-row group) ──
    for rank, pi in enumerate(proj_order):
        pst  = proj_info[pi]["start"]
        pft  = proj_info[pi]["finish"]
        col  = PC[pi]
        top_y = proj_grid_top(rank)
        # bar occupies the middle row of the 3-row group
        bar_row_bot = top_y - 2 * ROW_H + BAR_PAD
        bar_row_top = top_y - 1 * ROW_H - BAR_PAD
        bar_h = bar_row_top - bar_row_bot
        ax_tl.add_patch(mpatches.FancyBboxPatch(
            (pst, bar_row_bot), pft - pst, bar_h,
            boxstyle="round,pad=0.04",
            lw=1.1, ec=col, fc=PC[pi] + "40", zorder=4))
        ax_tl.text(pst + (pft-pst)/2, (bar_row_bot+bar_row_top)/2,
                   f"P{pi}  t=[{pst},{pft}]  BAC {proj_info[pi]['bac']}  P {proj_info[pi]['price']}",
                   ha="center", va="center", fontsize=7.2,
                   fontfamily=FONT, fontweight="bold", color=col, zorder=5)

    # ── efficiency labels (inside bar, small, so they don't clash with arrows) ──
    for rank, pi in enumerate(proj_order):
        top_y   = proj_grid_top(rank)
        bar_mid = top_y - 1.5 * ROW_H   # vertical centre of bar row
        col     = PC[pi]
        for t in T:
            if status_map.get((pi, t)) == "active":
                eta = eff_map.get((pi, t), 1.0)
                ax_tl.text(t, bar_mid + 0.28, f"η={eta:.2f}",
                           ha="center", va="center", fontsize=5.6,
                           fontfamily=FONT, color=col + "AA", zorder=5)

    # ── milestone payment arrows (upward, into top row of project group) ──
    # top row spans [top_y - ROW_H, top_y]; labels stack top-down from top_y
    LABEL_H  = 0.42   # vertical space per stacked label
    for t_c, pays in sorted(cert.items()):
        # group by project so each project's payments stack independently
        pays_by_proj: dict[int, list] = defaultdict(list)
        for pi, j, net in pays:
            pays_by_proj[pi].append((j, net))

        for pi, proj_pays in pays_by_proj.items():
            rank    = proj_order.index(pi)
            top_y   = proj_grid_top(rank)
            bar_top = top_y - ROW_H - BAR_PAD
            for si, (j, net) in enumerate(proj_pays):
                col       = MC[j % len(MC)]
                label_top = top_y - 0.06 - si * LABEL_H
                arrow_tip = label_top - 0.0
                ax_tl.annotate("",
                    xy=(t_c, bar_top), xytext=(t_c, arrow_tip),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.2, mutation_scale=8,
                                    shrinkA=2, shrinkB=2), zorder=6)
                j_lbl = "Adv" if j == 0 else f"MS{j}"
                # label placed to the right of the arrow tip, not on it
                ax_tl.text(t_c + 0.18, (label_top + bar_top) / 2,
                           f"+{net:.1f} {j_lbl}",
                           ha="left", va="center", fontsize=5.6,
                           fontfamily=FONT, color=col, zorder=7)

    # ── per-project allocation arrows (bottom row of each group) ──
    for rank, pi in enumerate(proj_order):
        top_y   = proj_grid_top(rank)
        alloc_row_top = top_y - 2*ROW_H
        alloc_row_bot = top_y - 3*ROW_H
        alloc_row_mid = (alloc_row_top + alloc_row_bot) / 2
        col = PC[pi]
        for t in T:
            alloc = alloc_map.get((pi, t), 0.0)
            if alloc <= 0:
                continue
            bar_bot = top_y - 2*ROW_H - BAR_PAD + 0.02
            ax_tl.annotate("",
                xy=(t, alloc_row_bot + 0.08), xytext=(t, bar_bot),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.0, linestyle="dashed",
                                mutation_scale=8,
                                shrinkA=2, shrinkB=2), zorder=6)
            # label placed to the right of the arrow, not on it
            ax_tl.text(t + 0.18, alloc_row_mid, f"−{alloc:.0f}",
                       ha="left", va="center", fontsize=5.8,
                       fontfamily=FONT, color=col, zorder=7)

    # ── NCF bar chart (below spine) ──
    NCF_SCL = 0.0036
    ax_tl.axhline(NCF_Y0, color=RULE, lw=0.4, ls="--", zorder=1,
                  xmin=0.04, xmax=0.96)
    ax_tl.text(-0.70, NCF_Y0, "ΔB",
               ha="left", va="center", fontsize=6.0, fontstyle="italic",
               fontfamily=FONT, color=TM)
    for _, row in pf_s.iterrows():
        t   = int(row["t_episode"])
        ncf = float(row["net_cashflow"])
        if ncf == 0: continue
        col = "#2060C0" if ncf > 0 else "#B03030"
        hb  = ncf * NCF_SCL
        ax_tl.bar(t, hb, bottom=NCF_Y0, color=col+"40", ec=col,
                  lw=0.65, width=0.44, zorder=3)
        # values live in the portfolio status table; omit from bars

    # ── budget curve ──
    bv = pf_s["budget_available"].tolist()
    bt = pf_s["t_episode"].tolist()
    bmin, bmax = min(bv), max(bv)
    brng = max(bmax-bmin, 1)
    bscl = 0.55/brng
    bys  = [BUD_Y0 + (b-bmin)*bscl for b in bv]
    ax_tl.plot(bt, bys, color=TM, lw=0.8, ls=":", marker="o",
               ms=2.8, zorder=4)
    ax_tl.text(-0.70, BUD_Y0, "Bₜ",
               ha="left", va="center", fontsize=6.0, fontstyle="italic",
               fontfamily=FONT, color=TM)

    # ══════════════════════════════════════════════════════════════════════════
    #  PORTFOLIO STATUS  (reward computation table — wide, full width)
    # ══════════════════════════════════════════════════════════════════════════
    ax_ps.axis("off"); ax_ps.set_xlim(0,1); ax_ps.set_ylim(0,1)
    y0ps = _hdr(ax_ps, "PORTFOLIO STATUS  —  Period Reward Computation  (γ = 0.95)",
                "ps", y=1.0, h=0.062, fs=7.0)

    # columns: t | γᵗ | Bₜ | ΔBₜ (NCF) | P0 ncf | P1 ncf | P2 ncf | r(t) | ΣR
    ps_hdr = ["t", "γᵗ", "Bₜ", "ΔBₜ", "P0·ΔB", "P1·ΔB", "P2·ΔB",
              "r(t)=γᵗ·ΔBₜ", "ΣR"]
    ps_cw  = [0.055, 0.080, 0.110, 0.095,
              0.100, 0.100, 0.100,
              0.155, 0.105]
    ps_al  = ["center","center","right","right",
              "right","right","right","right","right"]

    running = 0.0
    rows_ps = [ps_hdr]
    for _, row in pf_s.iterrows():
        t    = int(row["t_episode"])
        ncf  = float(row["net_cashflow"])
        bud  = float(row["budget_available"])
        disc = gamma**t
        pr   = float(row["reward"])
        running += pr
        pcols = [_sf(proj_ncf.get((i,t),0.0), 1) for i in range(n_proj)]
        rows_ps.append([
            f"t{t}",
            f"{disc:.3f}",
            f"{bud:,.0f}",
            _sf(ncf, 1) if ncf else "0",
            *pcols,
            _sf(pr, 2) if pr else "0",
            f"{running:+.2f}",
        ])
    rows_ps.append(["∑","","","","","","",
                    f"ΣR = {running:+.3f}",""])

    y_bot_ps = _table(ax_ps, rows_ps, ps_cw, 0.074, "ps",
                      fs=6.2, align=ps_al, y0=y0ps, pad=0.006)

    ax_ps.text(0.005, y_bot_ps - 0.018,
               f"*  t=0 advance (+{adv_t0:.0f}) credited to Bₜ at reset — not a reward step.  "
               f"ΣR = Σ r(t),  t∈[0,{horizon}]",
               ha="left", va="top", fontsize=5.4, fontfamily=FONT,
               color=TM, transform=ax_ps.transAxes, zorder=3)

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSERVATION BLOCK  (agent's input at each step, per-project)
    # ══════════════════════════════════════════════════════════════════════════
    ax_obs.axis("off"); ax_obs.set_xlim(0,1); ax_obs.set_ylim(0,1)
    y0ob = _hdr(ax_obs, "AGENT OBSERVATION  oₜ  —  Active Periods Only",
                "obs", y=1.0, h=0.060, fs=7.0)

    # Symbols from notation.yaml
    OBS_FIELDS = [
        ("t",               "t"),
        ("τᵢ",              "t_project"),
        ("ΔBᵢ",             "net_cashflow"),
        ("θᵣₑₘ",            "tolerance_remain"),
        ("xᶜᵘ",             "catchup_alloc_t"),
        ("xᶜᵘ⁺¹",           "catchup_alloc_next_t"),
        ("xʳᵖ",             "reach_plan_t"),
        ("Δψ*",             "target_progress_gap"),
        ("Δτ*",             "target_timestep_gap"),
        ("x*",              "target_required_alloc"),
        ("NPV*",            "target_npv"),
    ]
    sym_hdr = [s for s,_ in OBS_FIELDS]
    col_hdr = [f"P{i}" for i in range(n_proj)]

    # Build rows: one row per (project, t_episode) where project is active
    # Group by t_episode, within that per project in proj_order
    active_obs = []
    for t in range(horizon+1):
        for pi in proj_order:
            row = obs[(obs["i"]==pi) & (obs["t_episode"]==t)]
            if row.empty: continue
            r = row.iloc[0]
            st = status_map.get((pi, t), "pending")
            if st != "active": continue
            # skip rows where all observation signals are zero (completed project coasting)
            has_data = (
                float(r["catchup_alloc_t"]) > 0 or
                float(r["target_required_alloc"]) > 0
            )
            if not has_data: continue
            entry = {"i": pi, "t": t, "t_project": float(r["t_project"])}
            for sym, code in OBS_FIELDS:
                csv_code = code if code != "t" else "t_episode"
                entry[code] = float(r[csv_code]) if csv_code in r.index else 0.0
            active_obs.append(entry)

    # Layout: 3-column table (one col per project), rows are obs fields
    # Pivot: fields as rows, t-steps per project as sub-columns
    # Simpler: one wide table with rows = active periods, cols = symbol + P0 + P1 + P2
    # Each unique (i, t) is one row; we label project on first column

    # cw: id_col + 1 sym col + 11 value cols
    ob_hdr = ["Proj", "t", "t_proj", "DeltaB", "tol_rem",
               "x_cu", "x_cu+1", "x_rp",
               "d_psi*", "d_tau*", "x*", "NPV*"]
    ob_cw  = [0.055, 0.052, 0.052, 0.085,
               0.060,
               0.082, 0.082, 0.080,
               0.060, 0.055, 0.068, 0.069]
    ob_al  = ["center","center","center","right",
               "center",
               "right","right","right",
               "right","right","right","right"]

    ob_codes = ["t","t_project","net_cashflow","tolerance_remain",
                "catchup_alloc_t","catchup_alloc_next_t","reach_plan_t",
                "target_progress_gap","target_timestep_gap",
                "target_required_alloc","target_npv"]

    prev_i = None
    rows_ob = [ob_hdr]
    for entry in active_obs:
        pi = entry["i"]
        pi_lbl = f"P{pi}" if pi != prev_i else ""
        prev_i = pi
        vals = []
        for code in ob_codes:
            v = entry[code]
            if code in ("t","t_project","tolerance_remain","target_timestep_gap"):
                vals.append(str(int(v)))
            elif code == "target_progress_gap":
                vals.append(f"{v:.2f}")
            elif code == "target_npv":
                vals.append(f"{v:+.1f}")
            else:
                vals.append(f"{v:.1f}")
        rows_ob.append([pi_lbl] + vals)

    # add bottom note row
    _table(ax_obs, rows_ob, ob_cw, 0.097, "obs",
           fs=6.1, align=ob_al, y0=y0ob, pad=0.006)

    # symbol legend row below table
    syms = [
        ("x_cu",    "catchup alloc (current)"),
        ("x_cu+1",  "catchup alloc (next-t lookahead)"),
        ("x_rp",    "reach-plan alloc"),
        ("d_psi*",  "target progress gap"),
        ("d_tau*",  "target timestep gap"),
        ("x*",      "target required alloc"),
        ("NPV*",    "target milestone NPV"),
        ("tol_rem", "tolerance remaining"),
    ]
    n_bots = len(rows_ob)
    y_sym  = y0ob - n_bots * 0.097 - 0.006 - 0.026
    sym_parts = "   ".join(f"{s}: {d}" for s,d in syms)
    ax_obs.text(0.005, y_sym, sym_parts,
                ha="left", va="top", fontsize=5.3, fontfamily=FONT,
                color=TM, transform=ax_obs.transAxes, zorder=3)

    # ── save main figure ───────────────────────────────────────────────────────
    out_main = out_dir / "infograph.png"
    fig.savefig(out_main, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  main  → {out_main}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SEPARATE TABLE IMAGES
    # ══════════════════════════════════════════════════════════════════════════
    _build_portfolio_table(out_dir / "table_portfolio_status.png",
                           rows_ps, ps_cw, ps_al,
                           adv_t0, horizon, running, gamma)
    _build_observation_table(out_dir / "table_observations.png",
                             rows_ob, ob_cw, ob_al)

    # ══════════════════════════════════════════════════════════════════════════
    #  LEGEND  (separate horizontal image)
    # ══════════════════════════════════════════════════════════════════════════
    _build_legend(out_dir / "legend.png", n_proj, horizon)


def _build_portfolio_table(out_path: Path, rows_ps, ps_cw, ps_al,
                           adv_t0: float, horizon: int,
                           running: float, gamma: float) -> None:
    """Standalone Portfolio Status table image."""
    n_rows = len(rows_ps)
    RH = 0.048          # row height in axes-fraction
    HDR_H = 0.055
    NOTE_H = 0.06
    total_h = HDR_H + n_rows * RH + NOTE_H + 0.04

    FW = 8.27
    FH = max(total_h * 6.0, 2.4)   # scale axes-fraction → inches
    fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    y0 = _hdr(ax, "PORTFOLIO STATUS  —  Period Reward Computation  (γ = 0.95)",
              "ps", y=0.97, h=HDR_H, fs=8.0)
    y_bot = _table(ax, rows_ps, ps_cw, RH, "ps",
                   fs=7.2, align=ps_al, y0=y0, pad=0.008)
    ax.text(0.005, y_bot - 0.015,
            f"*  t=0 advance (+{adv_t0:.0f}) credited to Bₜ at reset — not a reward step.  "
            f"ΣR = Σ r(t),  t∈[0,{horizon}]",
            ha="left", va="top", fontsize=6.5, fontfamily=FONT,
            color=TM, transform=ax.transAxes, zorder=3)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  table_ps  → {out_path}")


def _build_observation_table(out_path: Path, rows_ob, ob_cw, ob_al) -> None:
    """Standalone Agent Observation table image."""
    n_rows = len(rows_ob)
    RH = 0.052
    HDR_H = 0.055
    NOTE_H = 0.08
    total_h = HDR_H + n_rows * RH + NOTE_H + 0.04

    FW = 8.27
    FH = max(total_h * 6.0, 2.4)
    fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    y0 = _hdr(ax, "AGENT OBSERVATION  oₜ  —  Active Periods Only",
              "obs", y=0.97, h=HDR_H, fs=8.0)
    y_bot = _table(ax, rows_ob, ob_cw, RH, "obs",
                   fs=7.0, align=ob_al, y0=y0, pad=0.008)

    syms = [
        ("x_cu",    "catchup alloc (current t)"),
        ("x_cu+1",  "catchup alloc (next t lookahead)"),
        ("x_rp",    "reach-plan alloc"),
        ("d_psi*",  "target progress gap"),
        ("d_tau*",  "target timestep gap"),
        ("x*",      "target required alloc"),
        ("NPV*",    "target milestone NPV"),
        ("tol_rem", "tolerance remaining"),
    ]
    sym_parts = "     ".join(f"{s}: {d}" for s, d in syms)
    ax.text(0.005, y_bot - 0.018, sym_parts,
            ha="left", va="top", fontsize=6.0, fontfamily=FONT,
            color=TM, transform=ax.transAxes, zorder=3)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  table_obs → {out_path}")


def _build_legend(out_path: Path, n_proj: int, horizon: int) -> None:
    """Separate legend card — wider spacing, no overlap, 3-col formula box."""

    # ── layout constants ──────────────────────────────────────────────────────
    FW, FH = 8.27, 3.60      # taller to give formula box room
    HDR_H  = 0.060
    DY     = 0.092            # row pitch for icon rows
    ICON_W = 0.040            # half-width of icon zone
    GAP_AFTER_ICON = 0.025    # gap between icon right edge and text

    # Column 1: x positions
    C1_IX  = 0.060            # icon centre x
    C1_TX  = C1_IX + ICON_W + GAP_AFTER_ICON   # text left edge

    # Column 2: starts at 0.36 — wide enough so col-1 text never reaches here
    C2_IX  = 0.395
    C2_TX  = C2_IX + ICON_W + GAP_AFTER_ICON

    # Column 3
    C3_IX  = 0.715
    C3_TX  = C3_IX + ICON_W + GAP_AFTER_ICON

    # icon rows start just below header
    Y0_ICONS = 1.0 - HDR_H - 0.025   # top of first icon row

    # two blank rows gap before formula box
    FORMULA_GAP = 2 * DY + 0.012

    FS_LBL = 7.8    # item label font size
    FS_HDR = 8.0    # header font size

    fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    _hdr(ax, "LEGEND & NOTATION", "leg", y=1.0, h=HDR_H, fs=FS_HDR)

    # ── icon helper ──────────────────────────────────────────────────────────
    def _icon(ax, kind, col, x, y):
        d = 0.028
        if kind == "arrow_up":
            ax.annotate("", xy=(x, y+d), xytext=(x, y-d),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.4, mutation_scale=10),
                        transform=ax.transAxes)
        elif kind == "arrow_dn_d":
            ax.annotate("", xy=(x, y-d), xytext=(x, y+d),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.4, linestyle="dashed",
                                        mutation_scale=10),
                        transform=ax.transAxes)
        elif kind == "box":
            ax.add_patch(mpatches.FancyBboxPatch(
                (x-0.032, y-0.022), 0.064, 0.044,
                boxstyle="round,pad=0.004", lw=0.9,
                ec=col, fc=col+"28", transform=ax.transAxes))
        elif kind == "rect":
            ax.add_patch(mpatches.Rectangle(
                (x-0.030, y-0.020), 0.060, 0.040,
                lw=0, fc=col+"55", transform=ax.transAxes))
        elif kind == "dotline":
            ax.plot([x-0.034, x+0.034], [y, y], color=col,
                    lw=1.0, ls=":", marker="o", ms=3.5,
                    transform=ax.transAxes)
        elif kind == "shade":
            ax.add_patch(mpatches.Rectangle(
                (x-0.030, y-0.020), 0.060, 0.040,
                lw=0, fc=col+"22", transform=ax.transAxes))

    def _row(ax, kind, col, ix, tx, y, label):
        _icon(ax, kind, col, ix, y)
        ax.text(tx, y, label, ha="left", va="center", fontsize=FS_LBL,
                fontfamily=FONT, color=TB, transform=ax.transAxes)

    # ── col 1 — payment arrows ─────────────────────────────────────────────
    col1_items = [
        ("arrow_up",   MC[0],  "Advance payment  (j = 0)"),
        ("arrow_up",   MC[1],  "Milestone payment  (j = 1)"),
        ("arrow_up",   MC[2],  "Milestone payment  (j = 2)"),
        ("arrow_up",   MC[3],  "Milestone payment  (j = 3)"),
    ]
    for k, (kind, col, lbl) in enumerate(col1_items):
        _row(ax, kind, col, C1_IX, C1_TX, Y0_ICONS - k*DY, lbl)

    # ── col 2 — allocation arrows + budget line ────────────────────────────
    col2_items = [
        ("arrow_dn_d", PC[0],     "P0 allocation outflow  (dashed ↓)"),
        ("arrow_dn_d", PC[1],     "P1 allocation outflow  (dashed ↓)"),
        ("arrow_dn_d", PC[2],     "P2 allocation outflow  (dashed ↓)"),
        ("dotline",    TM,        "Bₜ  portfolio budget  (dotted line)"),
    ]
    for k, (kind, col, lbl) in enumerate(col2_items):
        _row(ax, kind, col, C2_IX, C2_TX, Y0_ICONS - k*DY, lbl)

    # ── col 3 — project bars + cashflow rects ─────────────────────────────
    col3_items = [
        ("box",   PC[0],     "Project P0  active window"),
        ("box",   PC[1],     "Project P1  active window"),
        ("box",   PC[2],     "Project P2  active window"),
        ("rect",  "#2060C0", "ΔBₜ > 0  (period surplus)"),
        ("rect",  "#B03030", "ΔBₜ < 0  (period deficit)"),
        ("shade", PC[0],     "Row band shading per project group"),
    ]
    for k, (kind, col, lbl) in enumerate(col3_items):
        _row(ax, kind, col, C3_IX, C3_TX, Y0_ICONS - k*DY, lbl)

    # ── formula box (3 columns, 2 rows of gap above it) ───────────────────
    # bottom y of last icon row
    max_rows = max(len(col1_items), len(col2_items), len(col3_items))
    y_last_icon = Y0_ICONS - (max_rows - 1) * DY
    y_box_top   = y_last_icon - FORMULA_GAP

    # estimate box height: 2 text lines + padding
    BOX_H   = 0.195
    BOX_PAD = 0.018
    y_box_bot = y_box_top - BOX_H

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.010, y_box_bot), 0.980, BOX_H,
        boxstyle="round,pad=0.008", lw=0.7,
        ec=SA["ps"][0]+"80", fc=SA["ps"][1],
        transform=ax.transAxes))

    # 3-column formula layout
    COL_X = [0.025, 0.360, 0.690]
    FS_F  = 7.4
    FS_O  = 7.0

    # row 1
    row1 = [
        (SA["ps"][0], True,  "r(t) = γᵗ × ΔBₜ"),
        (SA["ps"][0], True,  "ΔBₜ = Σ inflow(t) − Σ outflow(t)"),
        (SA["ps"][0], True,  f"ΣR = Σ r(t),   t ∈ [0, {horizon}]"),
    ]
    # row 2
    row2 = [
        (SA["obs"][0], False, "outflow = allocation + interest(treasury_draw)"),
        (SA["obs"][0], False, "o(t) per project: [θᵣₑₘ, xᶜᵘ, xᶜᵘ⁺¹, Δψ*, Δτ*, x*, NPV*]"),
        (SA["obs"][0], False, "o(t) portfolio: [ΔBₜ/B₀, Bₜ/B₀]"),
    ]

    y_r1 = y_box_top - BOX_PAD - 0.005
    y_r2 = y_r1 - 0.092

    for (col, bold, txt), cx in zip(row1, COL_X):
        ax.text(cx, y_r1, txt, ha="left", va="top",
                fontsize=FS_F, fontfamily=FONT, color=col,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)

    for (col, bold, txt), cx in zip(row2, COL_X):
        ax.text(cx, y_r2, txt, ha="left", va="top",
                fontsize=FS_O, fontfamily=FONT, color=col,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  legend    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--out",  default=None)
    args = p.parse_args()

    here     = Path(__file__).resolve().parent
    data_dir = Path(args.data) if args.data else (
        here / "ground_truth" / "TC-3XMAIF-R" / "optimal")
    out_dir  = Path(args.out)  if args.out  else (
        data_dir.parent.parent)

    build(data_dir, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()