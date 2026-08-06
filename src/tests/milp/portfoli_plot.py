"""
portfolio_plot.py
-----------------
Reads cases.json from the same directory, produces one figure per case
and saves it to ./figures/<case_name>.png

Layout per case
---------------
  Row 1..N  : project panels (2 projects side-by-side per row)
               Each "project column" = top S-curve panel + bottom cash-flow panel
  Final row  : 1 × 3 portfolio summary (cash balance | period cash flows | cum NCF)

Usage
-----
  python portfolio_plot.py                    # reads cases.json in cwd
  python portfolio_plot.py path/to/cases.json # explicit path
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Okabe-Ito colorblind-safe palette ────────────────────────────────────────
C = {
    "BCWS":    "#0072B2",   # blue
    "BCWP":    "#CC79A7",   # pink-purple
    "ACWP":    "#D55E00",   # vermillion
    "PayExp":  "#E69F00",   # amber
    "PayRec":  "#009E73",   # green
    "Cap":     "#D55E00",   # vermillion (reuse for cap zones)
    "NCF":     "#0072B2",   # blue
    "Alloc":   "#D55E00",   # vermillion bars
    "Bt":      "#56B4E9",   # sky blue
}

ALPHA_CAP  = 0.12
ALPHA_BAR  = 0.75


# ═══════════════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════════════

def v(field):
    """Extract the scalar value from a {v, tol} dict or plain number."""
    if isinstance(field, dict):
        return field["v"]
    return field


def _step_xy(t_vals, y_vals, t_end=None):
    """
    Convert milestone step-chart points to (x, y) arrays that draw
    horizontal-first staircase lines.

      t_vals : list of times at which the level changes
      y_vals : list of cumulative levels *after* each step
      t_end  : extend the last level to this x (e.g. horizon H)
    """
    if not t_vals:
        return [], []
    xs, ys = [t_vals[0]], [y_vals[0]]
    for i in range(1, len(t_vals)):
        xs += [t_vals[i], t_vals[i]]
        ys += [y_vals[i - 1], y_vals[i]]
    if t_end is not None and t_end > t_vals[-1]:
        xs.append(t_end)
        ys.append(y_vals[-1])
    return xs, ys


# ═══════════════════════════════════════════════════════════════════════════════
#  per-project S-curve panel  (top)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_scurve(ax, proj, H):
    params  = proj["params"]
    tl      = proj["timeline"]
    BAC     = params["BAC"]
    fi      = params["fi"]
    mu      = params["mu"]       # cost overrun cap fraction
    Omega   = params["Omega"]    # schedule overrun cap (periods)
    ms_theta = params["ms_theta"]  # milestone progress fractions
    ms_e     = params["ms_e"]      # milestone expected times

    ts      = [v(e["t"])    for e in tl]
    bcws    = [v(e["BCWS"]) for e in tl]
    bcwp    = [v(e["BCWP"]) for e in tl]
    acwp    = [v(e["ACWP"]) for e in tl]
    eac     = [v(e["EAC_frac"]) for e in tl]

    t_max   = max(H, fi + Omega + 1)
    y_max   = max(1.0 + mu + 0.05, max(eac) + 0.05) if eac else 1.15

    # ── cap shading ───────────────────────────────────────────────────────────
    # schedule overrun zone
    ax.axvspan(fi, fi + Omega, ymin=0, ymax=1.0 / y_max,
               color=C["Cap"], alpha=ALPHA_CAP, zorder=0)
    # cost overrun zone  (horizontal band from 1.0 to 1+mu)
    ax.axhspan(1.0, 1.0 + mu, xmin=0, xmax=(fi + Omega) / t_max,
               color=C["Cap"], alpha=ALPHA_CAP, zorder=0)

    # ── BCWS ─────────────────────────────────────────────────────────────────
    ax.plot(ts, bcws, color=C["BCWS"], lw=1.8, label="BCWS")

    # ── BCWP ─────────────────────────────────────────────────────────────────
    ax.plot(ts, bcwp, color=C["BCWP"], lw=1.8, label="BCWP")

    # ── ACWP ─────────────────────────────────────────────────────────────────
    ax.plot(ts, acwp, color=C["ACWP"], lw=1.8, label="ACWP")

    # ── EAC forecast dashed from last point ──────────────────────────────────
    last_t    = ts[-1]
    last_acwp = acwp[-1]
    last_eac  = eac[-1]
    # forecast finish from SPI
    last_bcwp  = bcwp[-1]
    last_bcws  = bcws[-1]
    last_spi   = v(tl[-1]["SPI"]) if last_bcws > 0 else 1.0
    remaining  = (1.0 - last_bcwp) / max(last_spi, 0.01) if last_bcwp < 1.0 else 0
    f_hat      = last_t + remaining

    if last_bcwp < 1.0:
        ax.plot([last_t, f_hat], [last_bcwp, 1.0],
                color=C["BCWP"], lw=1.4, ls="--")
        ax.scatter([f_hat], [1.0], color=C["BCWP"], s=30, zorder=5)

    if last_acwp < last_eac:
        ax.plot([last_t, f_hat], [last_acwp, last_eac],
                color=C["ACWP"], lw=1.4, ls=":")
        ax.scatter([f_hat], [last_eac], color=C["ACWP"], s=30,
                   marker="o", zorder=5)

    # ── milestone payment steps (expected) ────────────────────────────────────
    # Build cumulative payment schedule from ms_theta & ms_e
    # ms_phi gives fractions but we show cumulative as fraction of BAC
    # Using ms_theta (progress at milestone) as proxy for expected payment level
    ms_phi = params.get("ms_phi", [1.0 / len(ms_theta)] * len(ms_theta))
    CP_frac = params.get("CP", BAC) / BAC  # contract price as fraction of BAC

    # cumulative expected payments: advance + each milestone net
    A_frac = params.get("A", 0) / BAC
    rho    = params.get("rho", 0)   # advance recovery rate

    cum_exp = [A_frac]
    t_exp   = [0]
    running = A_frac
    for k, (theta, phi, te) in enumerate(zip(ms_theta, ms_phi, ms_e)):
        gross  = CP_frac * phi
        recov  = rho * gross
        net_frac = max(gross - recov, 0)
        running += net_frac
        cum_exp.append(running)
        t_exp.append(te)

    xs_exp, ys_exp = _step_xy(t_exp, cum_exp, t_end=fi)
    ax.plot(xs_exp, ys_exp, color=C["PayExp"], lw=1.4, ls="-",
            label="Expected payments")

    # ── received payments from R_net in timeline ──────────────────────────────
    r_cum   = 0.0
    t_rec   = [0]
    cum_rec = [A_frac]   # advance assumed received at t=0
    for e in tl:
        rn = v(e["R_net"]) / BAC
        if rn > 1e-6:
            r_cum += rn
            t_rec.append(v(e["t"]))
            cum_rec.append(A_frac + r_cum)

    # extend to last t
    t_end_rec = max(ts[-1], fi)
    xs_rec, ys_rec = _step_xy(t_rec, cum_rec, t_end=t_end_rec)
    ax.plot(xs_rec, ys_rec, color=C["PayRec"], lw=1.6, ls="-",
            label="Received payments")

    # ── planned finish marker ─────────────────────────────────────────────────
    ax.axvline(fi, color=C["BCWS"], lw=0.8, ls="--", alpha=0.6)
    ax.text(fi, y_max * 0.97, f"$f_i={fi}$",
            color=C["BCWS"], fontsize=6, ha="center", va="top")

    # ── axis formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, y_max)
    ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.4)
    ax.set_ylabel("Fraction of BAC", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, ls="--", lw=0.3, alpha=0.4)
    ax.set_title(f"EVM  (BAC={BAC})", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════════════════
#  per-project cash-flow panel  (bottom)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_cashflow(ax, proj, H):
    params = proj["params"]
    tl     = proj["timeline"]
    BAC    = params["BAC"]

    ts         = [v(e["t"])          for e in tl]
    x_vals     = [v(e["x"]) / BAC   for e in tl]   # allocations (outflow)
    r_net_vals = [v(e["R_net"]) / BAC for e in tl]  # received (inflow)

    # expected periodic payments (from ms_phi / ms_e)
    ms_theta = params["ms_theta"]
    ms_phi   = params.get("ms_phi", [1.0 / len(ms_theta)] * len(ms_theta))
    ms_e     = params["ms_e"]
    CP_frac  = params.get("CP", BAC) / BAC
    rho      = params.get("rho", 0)
    A_frac   = params.get("A", 0) / BAC

    exp_by_t = {}
    if A_frac > 0:
        exp_by_t[0] = exp_by_t.get(0, 0) + A_frac
    for phi, te in zip(ms_phi, ms_e):
        gross   = CP_frac * phi
        recov   = rho * gross
        net_frac = max(gross - recov, 0)
        exp_by_t[te] = exp_by_t.get(te, 0) + net_frac

    t_max = max(H, max(ts) + 1)
    bar_w = max(0.25, t_max / 60)

    # ── allocation bars (below zero) ─────────────────────────────────────────
    ax.bar(ts, [-x for x in x_vals], width=bar_w,
           color=C["Alloc"], alpha=ALPHA_BAR, label="Allocation $x_{i,t}$",
           edgecolor=C["Alloc"], linewidth=0.4, zorder=3)

    # ── expected payment bars (above zero) ────────────────────────────────────
    t_exp_list = sorted(exp_by_t.keys())
    y_exp_list = [exp_by_t[t] for t in t_exp_list]
    ax.bar(t_exp_list, y_exp_list, width=bar_w,
           color=C["PayExp"], alpha=ALPHA_BAR, label="Expected payment",
           edgecolor=C["PayExp"], linewidth=0.4, zorder=3)

    # ── received payment bars (above zero, slightly offset) ───────────────────
    t_rec_nz = [ts[i] for i, r in enumerate(r_net_vals) if r > 1e-6]
    y_rec_nz = [r      for r in r_net_vals if r > 1e-6]
    if A_frac > 0:
        t_rec_nz = [0] + t_rec_nz
        y_rec_nz = [A_frac] + y_rec_nz
    ax.bar([t + bar_w * 0.5 for t in t_rec_nz], y_rec_nz, width=bar_w,
           color=C["PayRec"], alpha=ALPHA_BAR, label="Received payment",
           edgecolor=C["PayRec"], linewidth=0.4, zorder=4)

    # ── net cumulative cash-flow line ─────────────────────────────────────────
    ncf = 0.0
    ncf_t, ncf_y = [], []
    rec_by_t = {}
    if A_frac > 0:
        rec_by_t[0] = A_frac
    for i, e in enumerate(tl):
        rn = v(e["R_net"]) / BAC
        t  = v(e["t"])
        if rn > 1e-6:
            rec_by_t[t] = rec_by_t.get(t, 0) + rn

    for t in sorted(set([0] + ts)):
        ncf += rec_by_t.get(t, 0)
        ncf -= (x_vals[ts.index(t)] if t in ts else 0)
        ncf_t.append(t)
        ncf_y.append(ncf)

    ax.plot(ncf_t, ncf_y, color=C["NCF"], lw=1.4,
            marker="o", markersize=2.5, label="Net cum. cash flow", zorder=5)
    ax.axhline(0, color="black", lw=0.6)

    # ── axis formatting ───────────────────────────────────────────────────────
    all_y = [-x for x in x_vals] + y_exp_list + y_rec_nz + ncf_y
    y_abs = max(abs(y) for y in all_y) if all_y else 0.2
    y_lim = max(y_abs * 1.25, 0.05)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlim(0, t_max)
    ax.set_xlabel("Period $t$", fontsize=7)
    ax.set_ylabel("Cash flow / BAC", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, ls="--", lw=0.3, alpha=0.4, axis="y")
    ax.set_title("Cash flow profile", fontsize=7, pad=3)


# ═══════════════════════════════════════════════════════════════════════════════
#  portfolio summary panel  (1 row × 3 cols)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_portfolio_summary(axes, case):
    pf     = case["portfolio"]
    params = case["params"]
    meta   = case["meta"]
    H      = v(params.get("H", 10))
    B0     = v(params.get("B0", 300))
    Zstar  = meta.get("Zstar")

    ts       = [v(e["t"])          for e in pf]
    bt       = [v(e["B_t"])        for e in pf]
    inflow   = [v(e["sum_inflow"]) for e in pf]
    outflow  = [v(e["sum_outflow"])for e in pf]
    net      = [v(e["sum_net"])    for e in pf]
    disc_net = [v(e["disc_net"])   for e in pf]
    cum_z    = [v(e["cum_Z"])      for e in pf]

    ax1, ax2, ax3 = axes

    # ── panel 1: cash balance B_t ─────────────────────────────────────────────
    ax1.plot(ts, bt, color=C["Bt"], lw=1.8, marker="o", markersize=3)
    ax1.axhline(B0, color="gray", lw=0.7, ls="--", alpha=0.6,
                label=f"$B_0={B0}$")
    ax1.set_title("Cash balance $B_t$", fontsize=8)
    ax1.set_xlabel("Period $t$", fontsize=7)
    ax1.set_ylabel("Amount", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, ls="--", lw=0.3, alpha=0.4)
    ax1.legend(fontsize=6)

    # ── panel 2: period cash flows ────────────────────────────────────────────
    bar_w = max(0.2, H / 60)
    ax2.bar([t - bar_w * 0.5 for t in ts], outflow,
            width=bar_w, color=C["Alloc"], alpha=ALPHA_BAR,
            label="Outflow", edgecolor=C["Alloc"], lw=0.4)
    ax2.bar([t + bar_w * 0.5 for t in ts],
            [-o for o in outflow],
            width=bar_w, color=C["Alloc"], alpha=0.0)   # invisible — just for symmetry
    ax2.bar([t + bar_w * 0.5 for t in ts], inflow,
            width=bar_w, color=C["PayRec"], alpha=ALPHA_BAR,
            label="Inflow", edgecolor=C["PayRec"], lw=0.4)
    ax2.plot(ts, net, color=C["NCF"], lw=1.4,
             marker="o", markersize=2.5, label="Net")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_title("Period cash flows", fontsize=8)
    ax2.set_xlabel("Period $t$", fontsize=7)
    ax2.set_ylabel("Net amount", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, ls="--", lw=0.3, alpha=0.4, axis="y")
    ax2.legend(fontsize=6)

    # ── panel 3: cumulative discounted NCF ────────────────────────────────────
    ax3.plot(ts, cum_z, color=C["NCF"], lw=1.8, marker="o", markersize=3)
    if Zstar is not None:
        ax3.axhline(Zstar, color="gray", lw=0.7, ls="--", alpha=0.7,
                    label=f"$Z^*={Zstar}$")
        ax3.legend(fontsize=6)
    ax3.set_title("Cumul. discounted NCF", fontsize=8)
    ax3.set_xlabel("Period $t$", fontsize=7)
    ax3.set_ylabel("Amount", fontsize=7)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, ls="--", lw=0.3, alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
#  legend patches  (drawn once per figure)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_legend_handles():
    return [
        mlines.Line2D([], [], color=C["BCWS"],   lw=1.8, label="BCWS"),
        mlines.Line2D([], [], color=C["BCWP"],   lw=1.8, label="BCWP"),
        mlines.Line2D([], [], color=C["BCWP"],   lw=1.4, ls="--", label="BCWP forecast"),
        mlines.Line2D([], [], color=C["ACWP"],   lw=1.8, label="ACWP"),
        mlines.Line2D([], [], color=C["ACWP"],   lw=1.4, ls=":", label="ACWP forecast"),
        mlines.Line2D([], [], color=C["PayExp"], lw=1.4, label="Expected payments"),
        mlines.Line2D([], [], color=C["PayRec"], lw=1.6, label="Received payments"),
        mpatches.Patch(facecolor=C["Alloc"],  alpha=ALPHA_BAR, label="Allocation $x_{i,t}$"),
        mpatches.Patch(facecolor=C["PayExp"], alpha=ALPHA_BAR, label="Expected payment (bar)"),
        mpatches.Patch(facecolor=C["PayRec"], alpha=ALPHA_BAR, label="Received payment (bar)"),
        mlines.Line2D([], [], color=C["NCF"], lw=1.4,
                      marker="o", markersize=3, label="Net cum. cash flow"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  main figure builder for one case
# ═══════════════════════════════════════════════════════════════════════════════

def build_case_figure(case_name, case):
    projects = case["projects"]
    H        = case["params"]["H"]
    n_proj   = len(projects)

    # layout: 2 projects per row, each project = 2 sub-rows (scurve + cashflow)
    proj_cols = 2
    proj_rows = math.ceil(n_proj / proj_cols)   # rows of project pairs
    sub_rows  = 2                                # scurve + cashflow per project

    # total matplotlib rows = project rows * 2  +  1 portfolio row
    total_mpl_rows = proj_rows * sub_rows + sub_rows  # last sub_rows for portfolio

    # height ratios: scurve gets 2 units, cashflow gets 1 unit
    height_ratios = []
    for _ in range(proj_rows):
        height_ratios += [2, 1]    # scurve, cashflow
    height_ratios += [1.5, 1.5]   # portfolio panels (same height, 2 logical rows → 1 merged)

    fig = plt.figure(figsize=(14, 3.5 * proj_rows + 4))
    fig.suptitle(f"Case: {case_name}   (H={H})", fontsize=11, y=0.995, fontweight="bold")

    # build gridspec
    gs = fig.add_gridspec(
        nrows=total_mpl_rows,
        ncols=proj_cols,
        hspace=0.55,
        wspace=0.32,
        height_ratios=height_ratios,
    )

    # ── project panels ────────────────────────────────────────────────────────
    for idx, proj in enumerate(projects):
        col  = idx % proj_cols
        pair = idx // proj_cols          # which row-of-pairs
        mpl_row_sc = pair * sub_rows     # scurve row index
        mpl_row_cf = pair * sub_rows + 1 # cashflow row index

        ax_sc = fig.add_subplot(gs[mpl_row_sc, col])
        ax_cf = fig.add_subplot(gs[mpl_row_cf, col])

        ax_sc.set_title(f"Project {idx + 1}  —  EVM", fontsize=7, pad=2)
        _draw_scurve(ax_sc, proj, H)
        _draw_cashflow(ax_cf, proj, H)

        # share x-axis between scurve and cashflow
        ax_sc.sharex(ax_cf)
        plt.setp(ax_sc.get_xticklabels(), visible=False)
        ax_sc.set_xlabel("")

    # ── portfolio summary ─────────────────────────────────────────────────────
    pf_row_start = proj_rows * sub_rows
    # merge the two portfolio height rows into single axes
    ax_b  = fig.add_subplot(gs[pf_row_start:pf_row_start + 2, 0])
    ax_cf = fig.add_subplot(gs[pf_row_start:pf_row_start + 2, 1])
    ax_z  = fig.add_subplot(gs[pf_row_start:pf_row_start + 2, 2]
                             if proj_cols >= 3 else gs[pf_row_start:pf_row_start + 2, 1])

    # if only 2 columns, place portfolio across all cols differently
    # rebuild for portfolio row spanning correctly
    # remove the axes we just added and redo with colspan
    ax_b.remove()
    ax_cf.remove()
    try:
        ax_z.remove()
    except Exception:
        pass

    # portfolio as 3 equal columns spanning full width
    gs_pf = fig.add_gridspec(
        nrows=total_mpl_rows,
        ncols=3,
        hspace=0.55,
        wspace=0.38,
        height_ratios=height_ratios,
    )

    ax_pf1 = fig.add_subplot(gs_pf[pf_row_start:pf_row_start + 2, 0])
    ax_pf2 = fig.add_subplot(gs_pf[pf_row_start:pf_row_start + 2, 1])
    ax_pf3 = fig.add_subplot(gs_pf[pf_row_start:pf_row_start + 2, 2])

    _draw_portfolio_summary([ax_pf1, ax_pf2, ax_pf3], case)

    # separator line above portfolio
    line_y = 1.0 - (pf_row_start / total_mpl_rows) - 0.01
    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [line_y, line_y],
        transform=fig.transFigure,
        color="gray", lw=0.8, ls="--", alpha=0.5
    ))
    fig.text(0.5, line_y + 0.005, "Portfolio Summary",
             ha="center", va="bottom", fontsize=8,
             color="gray", style="italic",
             transform=fig.transFigure)

    # ── shared legend at bottom ───────────────────────────────────────────────
    handles = _make_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=6,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.patch.set_facecolor("white")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # resolve cases.json path
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = Path(__file__).parent / "cases.json"

    if not json_path.exists():
        print(f"ERROR: cannot find {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    # figures directory next to cases.json
    fig_dir = json_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    total = len(data)
    for i, (case_name, case) in enumerate(data.items(), 1):
        print(f"[{i}/{total}] {case_name} ...", end=" ", flush=True)
        try:
            fig = build_case_figure(case_name, case)
            out_path = fig_dir / f"{case_name}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            print(f"saved → {out_path.name}")
        except Exception as e:
            print(f"FAILED: {e}")
            plt.close("all")


if __name__ == "__main__":
    main()