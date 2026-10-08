"""
dashboard.py
============
Episode dashboard generator for portfolio solver ground-truth data.

Expected directory layout
-------------------------
src/tests/solvers/                      ← this script lives here
src/tests/solvers/ground_truth/
    <config_name>/
        <solver>/
            portfolios.csv
            projects_profile.csv
            projects_status.csv
            projects_observation.csv
            milestones_profile.csv
            milestones_status.csv
            portfolio_observation.csv
            environment_config.csv      (optional)
            training_log.csv            (optional)

Output
------
src/tests/solvers/ground_truth/<config_name>/<solver>/figures/
    portfolio_overview.png
    project_snapshot.png         ← notebook-style grid: one subplot per project
    project_<i>_evm.png          ← one per project
    project_<i>_cashflow.png     ← one per project
    portfolio_cashflow.png
    training_log.png             (only if non-empty)

Usage
-----
    # All configs & solvers (full run):
    python dashboard.py

    # Single config + solver (template / debug):
    python dashboard.py --config TC-3XMAIF-R --solver reach_t

    # Custom root:
    python dashboard.py --root path/to/ground_truth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (matches Appendix B LaTeX conventions)
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    bcws      = "#1a6bbf",   # blue   – planned value
    bcwp      = "#2da84a",   # green  – earned value
    acwp      = "#d4720a",   # orange – actual cost
    spi       = "#2da84a",   # green  – SPI line
    cpi       = "#d4720a",   # orange – CPI line
    ref_line  = "#e05050",   # red dashed – reference = 1
    threshold = "#2da84a",   # green dashed – milestone thresholds
    inflow    = "#2da84a",   # green bars  – inflows
    outflow   = "#e05050",   # red bars    – outflows
    net       = "#1a6bbf",   # blue line   – net cash flow
    balance   = "#0097a7",   # teal        – cash balance
    reward    = "#7b4fa6",   # purple      – cumul. discounted NCF / reward
    alloc     = "#e05050",   # red bars    – budget allocation spend
    bg        = "#f7f7f9",
    grid      = "#d0d0d8",
)

FIGSIZE_WIDE  = (16, 4.5)
FIGSIZE_PANEL = (16, 10)
DPI           = 120

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path, **kw) -> pd.DataFrame:
    """Read a CSV; return empty DataFrame on missing file."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def _savefig(fig: plt.Figure, out: Path, title: str | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out.relative_to(out.parents[4]) if out.parents[4].exists() else out}")


def _style_ax(ax: plt.Axes, xlabel="Period  t", ylabel="", title="") -> None:
    ax.set_facecolor(C["bg"])
    ax.grid(True, linestyle=":", color=C["grid"], alpha=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _s_curve(planned_start: int, planned_finish: int, bac: float,
             a: float = 1.5, b: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Beta-CDF S-curve (same parametrisation used by the environment).
    Returns (t_array, bcws_cumulative_fraction).
    """
    dur = planned_finish - planned_start
    if dur <= 0:
        return np.array([planned_start, planned_finish]), np.array([0.0, 1.0])
    t = np.linspace(0, 1, dur + 1)
    # Beta CDF approximation via regularised incomplete beta
    from scipy.special import betainc
    try:
        y = betainc(a, b, t)
    except Exception:
        y = t ** a / (t ** a + (1 - t) ** b)   # fallback
    y = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    ts = np.arange(planned_start, planned_finish + 1)
    return ts, y


# ─────────────────────────────────────────────────────────────────────────────
# Figure 0 – Project-level snapshot  (single figure, one row per project)
#
# Left y-axis  – normalised 0..1 (fraction of BAC / price):
#   • Yellow line + ○    → BCWS  cumulative planned progress  (progress_plan_t)
#   • White/grey line+○  → BCWP  cumulative earned value      (progress_actual)
#   • Teal line + △      → ACWP  cumulative actual cost / BAC
#   • Dashed green step  → Milestone profile: planned payment schedule
#                          (progress_threshold at timestep_threshold, /price)
#   • Orange line + ◇    → Net cash flow per period            (net_cashflow/BAC)
#
# Right y-axis – absolute currency units:
#   • Green bars         → Milestone payments received         (certified net_payment)
#   • Red bars           → Budget allocation spent             (allocation per period)
#   Both bar sets share each integer timestep; green is left-of-centre,
#   red is right-of-centre. Bar width auto-scales with x-span so they
#   never overlap and stay readable from a 2-tick to a 30-tick window.
#
# X-axis is clipped to the project's active window:
#   [min(planned_start, actual_start) − 1  …  max(planned_finish, actual_finish) + 1.5]
#   Integer ticks only; tick density adapts to span.
#
# Planned-finish dotted vertical line + fᵢ label near the top.
# Metrics text box (Status / CPI / SPI) pinned top-left.
# Legend outside the plot to the right.
# ─────────────────────────────────────────────────────────────────────────────

def _project_snapshot_ax(
    ax: plt.Axes,
    proj_i: int,
    ps_i: pd.DataFrame,
    pp_i: pd.Series,
    ms_status_i: pd.DataFrame,
    ms_profile_i: pd.DataFrame,
) -> None:
    bac            = float(pp_i["bac"])
    price          = float(pp_i.get("price", bac))
    planned_start  = int(pp_i["planned_start"])
    planned_finish = int(pp_i["planned_finish"])

    ps_sorted = ps_i.sort_values("t_episode").copy()

    # ── dynamic x-window ─────────────────────────────────────────────────
    non_pending = ps_sorted[
        ps_sorted["status"].isin(["active", "completed", "terminated"])
    ]
    if not non_pending.empty:
        actual_start  = int(non_pending["t_episode"].min())
        actual_finish = int(non_pending["t_episode"].max())
    else:
        actual_start  = planned_start
        actual_finish = planned_finish

    # x_lo: one tick before the project starts (but never < 0)
    # x_hi: the LAST timestep that actually has data — no padding beyond that
    x_lo   = max(0, min(planned_start, actual_start) - 1)
    x_hi   = max(planned_finish, actual_finish)          # ends at last data point
    x_span = max(x_hi - x_lo, 1)

    # Small visual padding on right so the last marker isn't clipped by the frame
    x_lim_r = x_hi + max(0.55, x_span * 0.05)

    # Slice to the window (only timesteps with data)
    win   = ps_sorted[
        (ps_sorted["t_episode"] >= x_lo) &
        (ps_sorted["t_episode"] <= x_hi)
    ].copy()
    t_win = win["t_episode"].values

    # ── responsive sizing ─────────────────────────────────────────────────
    # bar_w: two side-by-side bars fit in 80 % of one unit; each bar = 40 %
    bar_w = float(np.clip(0.38 / max(x_span, 1), 0.05, 0.36))
    msize = float(np.clip(5.5 - x_span * 0.12, 2.5, 5.5))
    lw    = float(np.clip(2.2 - x_span * 0.04, 1.2, 2.2))
    fs_tick = max(6, min(8, 8 - x_span // 10))

    # ── left-axis data (normalised, fraction of BAC / price) ─────────────
    bcws_cum = win["progress_plan_t"].values   # cumulative planned progress 0..1
    bcwp_cum = win["progress_actual"].values   # cumulative earned value    0..1
    acwp_cum = win["allocation"].cumsum().values / (bac + 1e-9)
    ncf_norm = win["net_cashflow"].values / (bac + 1e-9)   # can be negative

    # ── milestone profile: planned payment staircase (dashed green step) ──
    # Anchor at (x_lo, 0) then step up at each certified_t.
    # This gives the correct "flat then spike" shape even for single-period
    # projects where timestep_threshold == 0 but the payment arrives at t=1.
    ms_step_t: list[int]   = [x_lo]
    ms_step_y: list[float] = [0.0]
    if not ms_status_i.empty:
        ms_cert = ms_status_i.copy()
        if not ms_profile_i.empty:
            ms_cert = ms_status_i.merge(
                ms_profile_i[["j", "net_payment"]],
                on="j", how="left", suffixes=("", "_prof"),
            )
        pay_col  = "net_payment" if "net_payment" in ms_cert.columns else "net_payment_prof"
        ms_cert  = ms_cert.sort_values("certified_t")
        cum_pay  = 0.0
        for _, row in ms_cert.iterrows():
            ct       = int(row["certified_t"])
            cum_pay += float(row[pay_col])
            if x_lo <= ct <= x_hi:
                ms_step_t.append(ct)
                ms_step_y.append(cum_pay / (price + 1e-9))

    # ── right-axis data (absolute currency) ──────────────────────────────
    alloc_vals = win["allocation"].values   # one value per timestep in window

    # Actual milestone payment received at each certified_t (j > 0 only)
    ms_pay_t: list[int]   = []
    ms_pay_v: list[float] = []
    if not ms_status_i.empty:
        ms_merged = ms_status_i.copy()
        if not ms_profile_i.empty:
            ms_merged = ms_status_i.merge(
                ms_profile_i[["j", "net_payment"]],
                on="j", how="left", suffixes=("", "_prof"),
            )
        pay_col = "net_payment" if "net_payment" in ms_merged.columns else "net_payment_prof"
        for _, row in ms_merged[ms_merged["j"] > 0].iterrows():
            ct = int(row["certified_t"])
            if x_lo <= ct <= x_hi:
                ms_pay_t.append(ct)
                ms_pay_v.append(float(row[pay_col]))

    # ── right-axis scaling ────────────────────────────────────────────────
    # The right axis always starts at 0 and its scale is chosen so the
    # tallest bar reaches ~40 % of the panel height.  This keeps bars in the
    # lower portion and leaves the upper portion for normalised line charts.
    alloc_peak = float(alloc_vals.max()) if len(alloc_vals) else 0.0
    pay_peak   = max(ms_pay_v) if ms_pay_v else 0.0
    right_peak = max(alloc_peak, pay_peak, 1.0)
    right_max  = right_peak / 0.40   # bars occupy ≤ 40 % of panel height

    # ── left-axis y-limits (computed before plotting so we can align zeros) ─
    all_left_vals = np.concatenate([
        bcws_cum, bcwp_cum, acwp_cum, ncf_norm,
        np.array(ms_step_y),
    ])
    left_y_min = min(float(all_left_vals.min()), 0.0)
    left_y_max = max(float(all_left_vals.max()), 1.0)
    left_margin_top = (left_y_max - left_y_min) * 0.20
    left_margin_bot = (left_y_max - left_y_min) * 0.08
    left_bot = left_y_min - left_margin_bot
    left_top = left_y_max + left_margin_top

    # ── align right-axis zero with left-axis zero ─────────────────────────
    # The right axis must be set so that right_y = 0 maps to the same pixel
    # as left_y = 0.  Given the left axis spans [left_bot, left_top], the
    # fraction of the panel that corresponds to y=0 is:
    #   zero_frac = (0 - left_bot) / (left_top - left_bot)
    # The right axis needs:
    #   right_bot = -zero_frac * right_max / (1 - zero_frac)  [negative baseline]
    #   right_top = right_max
    left_span  = left_top - left_bot
    zero_frac  = (0.0 - left_bot) / left_span          # 0..1 position of zero
    # Extend right axis downward so its zero aligns; bars are still drawn from 0
    right_bot  = -zero_frac * right_max / max(1.0 - zero_frac, 1e-6)

    # ══════════════════════ twin right axis ══════════════════════════════
    ax_r = ax.twinx()
    ax_r.set_ylim(right_bot, right_max)
    ax_r.set_xlim(x_lo - 0.6, x_lim_r)

    # Milestone payment bars — green, left of centre at certified_t
    if ms_pay_t:
        ax_r.bar(
            np.array(ms_pay_t, dtype=float) - bar_w * 0.55,
            ms_pay_v,
            width=bar_w, color="#2ecc71", alpha=0.82,
            label="MS Payment (received)",
            zorder=3,
        )

    # Allocation bars — red, right of centre at each active timestep
    if len(alloc_vals):
        ax_r.bar(
            t_win + bar_w * 0.55,
            alloc_vals,
            width=bar_w, color="#e74c3c", alpha=0.75,
            label="Allocation (spent)",
            zorder=3,
        )

    ax_r.set_ylabel("Absolute amount", fontsize=7, color="#555555")
    ax_r.tick_params(axis="y", labelsize=fs_tick, labelcolor="#777777")
    ax_r.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=5, min_n_ticks=3))
    ax_r.grid(False)

    # ══════════════════════ primary left axis ════════════════════════════

    # Reference lines
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="-",  alpha=0.4, zorder=1)

    # BCWS — yellow step chart (pre: the plan is set at the START of each period)
    ax.step(t_win, bcws_cum,
            where="pre", linewidth=lw, color="#f1c40f",
            marker="o", markersize=msize,
            markeredgecolor="#b7950b", markeredgewidth=0.6,
            label="BCWS  (planned progress)", zorder=5)

    # BCWP — light grey step chart
    ax.step(t_win, bcwp_cum,
            where="post", linewidth=lw, color="#bdc3c7",
            marker="o", markersize=msize,
            markeredgecolor="#7f8c8d", markeredgewidth=0.6,
            label="BCWP  (earned value)", zorder=5)

    # ACWP — teal step chart + triangles
    ax.step(t_win, acwp_cum,
            where="post", linewidth=lw, color="#1abc9c",
            marker="^", markersize=msize,
            markeredgecolor="#148f77", markeredgewidth=0.6,
            label="ACWP  (actual cost / BAC)", zorder=5)

    # Milestone profile — dashed green step anchored from x_lo
    if len(ms_step_t) > 1:
        ax.step(ms_step_t, ms_step_y,
                where="post", linewidth=lw * 0.85, linestyle="--",
                color="#27ae60", alpha=0.88,
                label="MS Profile  (planned pmts / price)", zorder=4)

    # Net cash flow — orange step chart (can be negative)
    ax.step(t_win, ncf_norm,
            where="post", linewidth=max(lw * 0.8, 1.0),
            color="#e67e22", linestyle="-.",
            marker="D", markersize=max(msize * 0.75, 2.0),
            markeredgecolor="#ca6f1e", markeredgewidth=0.5,
            label="Net Cash Flow / BAC", zorder=5)

    # ── planned finish marker ─────────────────────────────────────────────
    ax.axvline(planned_finish, color="#5d6d7e", linewidth=1.0,
               linestyle=":", alpha=0.6, zorder=2)
    ax.text(planned_finish + 0.05, 0.98,
            f"fᵢ={planned_finish}",
            fontsize=max(6, min(8, 8 - x_span // 8)),
            color="#5d6d7e", va="top", ha="left",
            transform=ax.get_xaxis_transform(), zorder=6)

    # ── metrics box ───────────────────────────────────────────────────────
    last_row   = non_pending.sort_values("t_episode").iloc[-1] \
                 if not non_pending.empty else win.iloc[-1]
    cpi_val    = float(last_row["cpi"])
    spi_val    = float(last_row["spi"])
    status_val = str(last_row["status"]).upper()

    ax.text(0.01, 0.99,
            f"Status : {status_val}\nCPI    = {cpi_val:.3f}\nSPI    = {spi_val:.3f}",
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#aaaaaa", boxstyle="round,pad=0.3"),
            zorder=8)

    # ── axis limits & ticks ───────────────────────────────────────────────
    ax.set_xlim(x_lo - 0.6, x_lim_r)
    ax.set_ylim(left_bot, left_top)

    tick_step = max(1, x_span // 15)
    ax.set_xticks(range(x_lo, x_hi + 1, tick_step))
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=7)

    ax.set_xlabel("Episode Timestep", fontsize=8)
    ax.set_ylabel("Normalised (fraction of BAC / price)", fontsize=8)
    ax.set_title(
        f"Project {proj_i}  ·  BAC = {bac:.0f}  ·  "
        f"Plan [{planned_start} → {planned_finish}]  ·  {status_val}",
        fontsize=9, fontweight="bold", pad=6,
    )
    ax.set_facecolor(C["bg"])
    ax.grid(True, linestyle=":", color=C["grid"], alpha=0.6, zorder=0)

    # ── combined legend outside right ─────────────────────────────────────
    h_l, lb_l = ax.get_legend_handles_labels()
    h_r, lb_r = ax_r.get_legend_handles_labels()
    all_h  = [h for h, lb in zip(h_l + h_r, lb_l + lb_r) if not lb.startswith("_")]
    all_lb = [lb for lb in lb_l + lb_r if not lb.startswith("_")]
    ax.legend(all_h, all_lb,
              loc="upper left", bbox_to_anchor=(1.07, 1.0),
              fontsize=7.5, framealpha=0.92, edgecolor="#cccccc",
              title="Legend", title_fontsize=8, borderpad=0.7,
              labelspacing=0.5)
    if ax_r.get_legend():
        ax_r.get_legend().remove()


def plot_project_snapshot(
    ps: pd.DataFrame,
    pp: pd.DataFrame,
    ms_status: pd.DataFrame,
    ms_profile: pd.DataFrame,
    config_name: str,
    solver: str,
    out: Path,
) -> None:
    """
    Single figure, one row per project.
    Row height scales with x-span so short projects are not stretched
    and long projects are not cramped.
    The figure suptitle and per-subplot titles are given enough vertical
    space so they never overlap.
    """
    projects = sorted(ps["i"].unique())
    n_proj   = len(projects)

    # Per-project x-span → proportional row height
    row_heights = []
    for proj_i in projects:
        ps_i     = ps[ps["i"] == proj_i]
        pp_row   = pp[pp["i"] == proj_i]
        if pp_row.empty:
            row_heights.append(4.5)
            continue
        pp_i     = pp_row.iloc[0]
        pstart   = int(pp_i["planned_start"])
        pfinish  = int(pp_i["planned_finish"])
        non_pend = ps_i[ps_i["status"].isin(["active", "completed", "terminated"])]
        afinish  = int(non_pend["t_episode"].max()) if not non_pend.empty else pfinish
        x_span   = max(pfinish, afinish) - max(0, pstart - 1)
        # 4 in base + proportional growth, capped at 7
        row_heights.append(float(np.clip(4.0 + x_span * 0.06, 4.0, 7.0)))

    # Reserve fixed space at the top for the figure-level suptitle
    SUPTITLE_H = 0.55    # inches
    HSPACE_PER = 0.90    # inches of vertical gap between subplots
    fig_h = SUPTITLE_H + sum(row_heights) + HSPACE_PER * n_proj

    fig, axes = plt.subplots(
        n_proj, 1,
        figsize=(18, fig_h),
        gridspec_kw={"height_ratios": row_heights},
        squeeze=False,
    )
    fig.patch.set_facecolor(C["bg"])

    for row_idx, proj_i in enumerate(projects):
        ax        = axes[row_idx, 0]
        ps_i      = ps[ps["i"] == proj_i].copy()
        pp_rows   = pp[pp["i"] == proj_i]
        ms_stat_i = ms_status[ms_status["i"] == proj_i]   if not ms_status.empty   else pd.DataFrame()
        ms_prof_i = ms_profile[ms_profile["i"] == proj_i] if not ms_profile.empty  else pd.DataFrame()

        if pp_rows.empty:
            ax.set_visible(False)
            continue

        _project_snapshot_ax(ax, proj_i, ps_i, pp_rows.iloc[0],
                              ms_stat_i, ms_prof_i)

    # top fraction: leave SUPTITLE_H inches above the top subplot
    top_frac = 1.0 - SUPTITLE_H / fig_h
    fig.subplots_adjust(
        hspace   = HSPACE_PER * n_proj / sum(row_heights),
        right    = 0.78,    # 22 % margin for outside legend
        left     = 0.06,
        top      = top_frac,
        bottom   = 0.04,
    )

    # suptitle sits in the reserved space above top_frac (y=1 in figure coords)
    suptitle_y = top_frac + (SUPTITLE_H / fig_h) * 0.55
    fig.suptitle(
        f"Project-Level Snapshot  |  config: {config_name}  |  solver: {solver}",
        fontsize=11, fontweight="bold", y=suptitle_y,
    )
    # Use savefig without a second suptitle call
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out.relative_to(out.parents[4]) if out.parents[4].exists() else out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Portfolio overview  (cash balance · period flows · cumul. reward)
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_overview(
    portfolios: pd.DataFrame,
    config_name: str,
    solver: str,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(C["bg"])

    t = portfolios["t_episode"].values
    bal = portfolios["budget_available"].values
    ncf = portfolios["net_cashflow"].values
    rew = portfolios["reward"].values
    cum_rew = np.cumsum(rew)

    # ── Panel A: cash balance ─────────────────────────────────────────────
    ax = axes[0]
    ax.step(t, bal, where="post", color=C["balance"], linewidth=2)
    ax.axhline(bal[0], linestyle="--", color="grey", linewidth=1, label=f"B₀ = {bal[0]:.0f}")
    _style_ax(ax, ylabel="Budget Available", title="Cash Balance  Bₜ")
    ax.legend(fontsize=7)

    # ── Panel B: period cash flows (inflow / outflow bars + net line) ─────
    ax = axes[1]
    bar_w = 0.4
    pos_ncf = np.clip(ncf, 0, None)
    neg_ncf = np.clip(ncf, None, 0)
    ax.bar(t, pos_ncf, width=bar_w, color=C["inflow"],  alpha=0.85, label="Net inflow",  zorder=2)
    ax.bar(t, neg_ncf, width=bar_w, color=C["outflow"], alpha=0.85, label="Net outflow", zorder=2)
    ax.plot(t, ncf, marker="o", markersize=3, color=C["net"],
            linewidth=1.2, label="Net CF", zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    _style_ax(ax, ylabel="Amount", title="Period Net Cash Flow")
    ax.legend(fontsize=7)

    # ── Panel C: cumulative discounted reward ─────────────────────────────
    ax = axes[2]
    ax.plot(t, cum_rew, marker="o", markersize=3, color=C["reward"],
            linewidth=2, label="Cumul. reward")
    ax.axhline(0, linestyle="--", color=C["ref_line"], linewidth=0.8, alpha=0.6)
    _style_ax(ax, ylabel="Amount", title="Cumulative Reward  Σrₜ")
    ax.legend(fontsize=7)

    title = f"Portfolio Overview  |  config: {config_name}  |  solver: {solver}"
    _savefig(fig, out, title)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Per-project EVM panel  (BCWS · BCWP · ACWP · SPI · CPI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_project_evm(
    proj_i: int,
    ps_i: pd.DataFrame,          # projects_status filtered to project i
    pp_i: pd.Series,             # projects_profile row for project i
    ms_i: pd.DataFrame,          # milestones_profile for project i
    config_name: str,
    solver: str,
    out: Path,
) -> None:
    bac            = pp_i["bac"]
    planned_start  = int(pp_i["planned_start"])
    planned_finish = int(pp_i["planned_finish"])
    scurve_a       = pp_i.get("scurve_a", 1.5)
    scurve_b       = pp_i.get("scurve_b", 3.0)

    # active rows only (status != pending before project start)
    active = ps_i[ps_i["status"].isin(["active", "completed", "terminated"])].copy()
    t_ep   = ps_i["t_episode"].values           # all episode timesteps

    # Earned value (BCWP) and actual cost (ACWP) – normalised to BAC fraction
    t_act  = active["t_episode"].values
    bcwp   = active["progress_actual"].values            # already a fraction
    acwp   = active["allocation"].cumsum().values / (bac + 1e-9)   # cumulative spend fraction

    # BCWS from s-curve
    t_sc, y_sc = _s_curve(planned_start, planned_finish, bac, a=scurve_a, b=scurve_b)

    # SPI & CPI
    spi = active["spi"].values
    cpi = active["cpi"].values

    # milestone thresholds
    thresholds = ms_i[ms_i["j"] > 0]["progress_threshold"].values  # skip advance (j=0)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(C["bg"])

    # ── Panel A: EVM curves ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_sc, y_sc, color=C["bcws"], linewidth=2, label="BCWS (planned)")
    if len(t_act):
        ax.step(t_act, bcwp, where="post", color=C["bcwp"], linewidth=2,
                marker="o", markersize=3, label="BCWP (earned)")
        ax.plot(t_act, acwp, color=C["acwp"], linewidth=2,
                marker="^", markersize=3, linestyle="--", label="ACWP (actual)")

    # milestone threshold lines
    for θ in thresholds:
        ax.axhline(θ, color=C["threshold"], linestyle=":", linewidth=0.9, alpha=0.8)
        ax.text(planned_finish + 0.1, θ, f"θ={θ:.2f}", fontsize=6,
                color=C["threshold"], va="center")

    # planned finish marker
    ax.axvline(planned_finish, color=C["bcws"], linestyle="--",
               linewidth=0.9, alpha=0.7)
    ax.text(planned_finish + 0.1, 1.05, f"fᵢ={planned_finish}", fontsize=6,
            color=C["bcws"])

    ax.set_ylim(0, 1.35)
    _style_ax(ax, ylabel="Fraction of BAC", title=f"Project {proj_i} – EVM Signals")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel B: allocation (outflow) bars + milestone inflow bars ────────
    ax = axes[1]
    alloc = active["allocation"].values

    ax.bar(t_act, -alloc / (bac + 1e-9), width=0.4,
           color=C["alloc"], alpha=0.85, label="Allocation xᵢₜ")
    # milestone inflows at certification times
    for _, ms_row in ms_i.iterrows():
        # find certified_t from milestones_status if available
        net_pay = ms_row["net_payment"]
        j       = int(ms_row["j"])
        t_thresh = int(ms_row["timestep_threshold"])
        ax.bar(t_thresh, net_pay / (bac + 1e-9), width=0.4,
               color=C["inflow"], alpha=0.85)
        ax.text(t_thresh, net_pay / (bac + 1e-9) + 0.03,
                f"MS{j}", fontsize=6, ha="center", color=C["inflow"])

    ax.axhline(0, color="black", linewidth=0.5)
    ymax = max(abs(ax.get_ylim()[0]), ax.get_ylim()[1])
    ax.set_ylim(-ymax * 1.15, ymax * 1.15)
    _style_ax(ax, ylabel="Fraction of BAC  (outflow↓ / inflow↑)",
              title=f"Project {proj_i} – Cashflow Profile")
    ax.legend(fontsize=7)

    # ── Panel C: SPI / CPI ────────────────────────────────────────────────
    ax = axes[2]
    if len(t_act):
        ax.plot(t_act, spi, color=C["spi"], linewidth=1.8,
                marker="o", markersize=3, label="SPI")
        ax.plot(t_act, cpi, color=C["cpi"], linewidth=1.8,
                marker="s", markersize=3, linestyle="--", label="CPI")
    ax.axhline(1.0, color=C["ref_line"], linestyle="--",
               linewidth=1, alpha=0.7, label="ref = 1")
    ax.set_ylim(0, max(2.2, (spi.max() if len(spi) else 2.2) * 1.1))
    _style_ax(ax, ylabel="Index", title=f"Project {proj_i} – SPI / CPI")
    ax.legend(fontsize=7)

    title = (f"Project {proj_i} EVM  |  BAC={bac:.0f}  |  "
             f"plan [{planned_start}–{planned_finish}]  |  "
             f"config: {config_name}  |  solver: {solver}")
    _savefig(fig, out, title)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Per-project cashflow detail
# ─────────────────────────────────────────────────────────────────────────────

def plot_project_cashflow(
    proj_i: int,
    ps_i: pd.DataFrame,
    pp_i: pd.Series,
    ms_status_i: pd.DataFrame,
    ms_profile_i: pd.DataFrame,
    config_name: str,
    solver: str,
    out: Path,
) -> None:
    bac = pp_i["bac"]
    planned_finish = int(pp_i["planned_finish"])

    active = ps_i[ps_i["status"].isin(["active", "completed", "terminated"])].copy()
    t_act  = active["t_episode"].values

    inflow  = active["inflow"].values
    outflow = active["outflow"].values
    net     = active["net_cashflow"].values
    cum_net = np.cumsum(net)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(C["bg"])

    # ── Panel A: inflow / outflow bars ───────────────────────────────────
    ax = axes[0]
    bar_w = 0.35
    ax.bar(t_act - bar_w / 2, inflow,   width=bar_w, color=C["inflow"],  alpha=0.85, label="Inflow")
    ax.bar(t_act + bar_w / 2, -outflow, width=bar_w, color=C["outflow"], alpha=0.85, label="Outflow")
    ax.axhline(0, color="black", linewidth=0.5)
    _style_ax(ax, ylabel="Amount", title=f"Project {proj_i} – Gross Flows")
    ax.legend(fontsize=7)

    # ── Panel B: milestone payment breakdown ──────────────────────────────
    ax = axes[1]
    if not ms_status_i.empty and not ms_profile_i.empty:
        ms = ms_status_i.merge(
            ms_profile_i[["j", "gross_payment", "advance_recovery",
                          "retention_withheld", "retention_released", "net_payment"]],
            on="j", how="left",
        )
        t_cert = ms["certified_t"].values
        net_p  = ms["net_payment_x"].values if "net_payment_x" in ms.columns else ms["net_payment"].values
        gross  = ms.get("gross_payment", pd.Series(net_p)).values
        rec    = ms.get("advance_recovery", pd.Series(np.zeros_like(net_p))).values
        ret_w  = ms.get("retention_withheld", pd.Series(np.zeros_like(net_p))).values

        width = 0.3
        ax.bar(t_cert, gross, width=width, color=C["inflow"], alpha=0.8,  label="Gross")
        ax.bar(t_cert, -rec,  width=width, color="#ffaa00",   alpha=0.75, label="Adv. recovery",
               bottom=-gross, align="center")
        ax.bar(t_cert, -ret_w, width=width, color="#aa44cc",  alpha=0.7,  label="Retention",
               bottom=-gross + rec, align="center")
        ax.bar(t_cert, net_p, width=width * 0.5, color="white",
               edgecolor=C["inflow"], linewidth=1, label="Net pay.", zorder=3)
        ax.axhline(0, color="black", linewidth=0.5)
        for j_row, ms_row in enumerate(ms.itertuples()):
            ax.text(ms_row.certified_t, float(gross[j_row]) + 1,
                    f"MS{ms_row.j}", fontsize=6, ha="center", color=C["inflow"])
    _style_ax(ax, ylabel="Amount", title=f"Project {proj_i} – Milestone Payments")
    ax.legend(fontsize=6, loc="upper left")

    # ── Panel C: cumulative net cash flow ─────────────────────────────────
    ax = axes[2]
    ax.plot(t_act, cum_net, marker="o", markersize=3,
            color=C["reward"], linewidth=2, label="Cumul. NCF")
    ax.axhline(0, color=C["ref_line"], linestyle="--", linewidth=0.8, alpha=0.6)
    _style_ax(ax, ylabel="Amount", title=f"Project {proj_i} – Cumulative Net CF")
    ax.legend(fontsize=7)

    title = (f"Project {proj_i} Cashflow Detail  |  BAC={bac:.0f}  |  "
             f"planned finish={planned_finish}  |  "
             f"config: {config_name}  |  solver: {solver}")
    _savefig(fig, out, title)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Portfolio-level cashflow dashboard (3-row grid)
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_cashflow(
    portfolios: pd.DataFrame,
    ps: pd.DataFrame,
    pp: pd.DataFrame,
    config_name: str,
    solver: str,
    out: Path,
) -> None:
    n_proj = ps["i"].nunique()
    fig = plt.figure(figsize=(16, 4 + 3.5 * n_proj), facecolor=C["bg"])
    gs  = gridspec.GridSpec(n_proj + 1, 3, figure=fig,
                            hspace=0.55, wspace=0.35)

    t_port = portfolios["t_episode"].values
    bal    = portfolios["budget_available"].values
    ncf    = portfolios["net_cashflow"].values
    rew    = portfolios["reward"].values
    cum_rew = np.cumsum(rew)

    # ── Row 0: portfolio-level ────────────────────────────────────────────
    ax_bal = fig.add_subplot(gs[0, 0])
    ax_bal.step(t_port, bal, where="post", color=C["balance"], linewidth=2)
    ax_bal.axhline(bal[0], linestyle="--", color="grey", linewidth=0.9,
                   label=f"B₀={bal[0]:.0f}")
    _style_ax(ax_bal, ylabel="Budget Available", title="Portfolio – Cash Balance Bₜ")
    ax_bal.legend(fontsize=7)

    ax_ncf = fig.add_subplot(gs[0, 1])
    ax_ncf.bar(t_port, np.clip(ncf, 0, None), width=0.4,
               color=C["inflow"],  alpha=0.85, label="Inflow")
    ax_ncf.bar(t_port, np.clip(ncf, None, 0), width=0.4,
               color=C["outflow"], alpha=0.85, label="Outflow")
    ax_ncf.plot(t_port, ncf, marker="o", markersize=3,
                color=C["net"], linewidth=1.2, label="Net")
    ax_ncf.axhline(0, color="black", linewidth=0.4)
    _style_ax(ax_ncf, ylabel="Amount", title="Portfolio – Period Net CF")
    ax_ncf.legend(fontsize=7)

    ax_rew = fig.add_subplot(gs[0, 2])
    ax_rew.plot(t_port, cum_rew, marker="o", markersize=3,
                color=C["reward"], linewidth=2)
    ax_rew.axhline(0, linestyle="--", color=C["ref_line"], linewidth=0.8, alpha=0.6)
    _style_ax(ax_rew, ylabel="Cumul. reward", title="Portfolio – Cumulative Reward")

    # ── Rows 1..n: per-project allocation + progress ──────────────────────
    for row_idx, proj_i in enumerate(sorted(ps["i"].unique())):
        ps_i = ps[ps["i"] == proj_i].copy()
        pp_i = pp[pp["i"] == proj_i].iloc[0] if not pp[pp["i"] == proj_i].empty else None
        if pp_i is None:
            continue

        active  = ps_i[ps_i["status"].isin(["active", "completed", "terminated"])]
        t_act   = active["t_episode"].values
        alloc   = active["allocation"].values
        prog    = active["progress_actual"].values
        plan    = active["progress_plan_t"].values
        spi     = active["spi"].values
        cpi     = active["cpi"].values
        bac     = pp_i["bac"]

        # allocation bars
        ax_a = fig.add_subplot(gs[row_idx + 1, 0])
        ax_a.bar(t_act, alloc, width=0.5, color=C["outflow"], alpha=0.85)
        _style_ax(ax_a, ylabel="Allocation xᵢₜ",
                  title=f"Project {proj_i} – Budget Allocation  (BAC={bac:.0f})")

        # progress
        ax_p = fig.add_subplot(gs[row_idx + 1, 1])
        bpf_start = int(pp_i["planned_start"])
        bpf_end   = int(pp_i["planned_finish"])
        t_sc, y_sc = _s_curve(bpf_start, bpf_end, bac,
                              a=pp_i.get("scurve_a", 1.5),
                              b=pp_i.get("scurve_b", 3.0))
        ax_p.plot(t_sc, y_sc, color=C["bcws"], linewidth=1.5, label="BCWS")
        if len(t_act):
            ax_p.step(t_act, prog, where="post", color=C["bcwp"],
                      linewidth=1.5, marker="o", markersize=2, label="Progress")
        ax_p.set_ylim(0, 1.2)
        _style_ax(ax_p, ylabel="Progress", title=f"Project {proj_i} – Progress vs. Plan")
        ax_p.legend(fontsize=7)

        # SPI / CPI
        ax_s = fig.add_subplot(gs[row_idx + 1, 2])
        if len(t_act):
            ax_s.plot(t_act, spi, color=C["spi"], linewidth=1.5,
                      marker="o", markersize=2, label="SPI")
            ax_s.plot(t_act, cpi, color=C["cpi"], linewidth=1.5,
                      marker="s", markersize=2, linestyle="--", label="CPI")
        ax_s.axhline(1.0, color=C["ref_line"], linestyle="--",
                     linewidth=0.8, alpha=0.7)
        ax_s.set_ylim(0, max(2.2,
            (spi.max() if len(spi) else 2.2) * 1.1,
            (cpi.max() if len(cpi) else 2.2) * 1.1,
        ))
        _style_ax(ax_s, ylabel="Index", title=f"Project {proj_i} – SPI / CPI")
        ax_s.legend(fontsize=7)

    title = (f"Portfolio + Per-Project Summary  |  "
             f"config: {config_name}  |  solver: {solver}")
    _savefig(fig, out, title)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Training log  (only if rows present)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_log(log: pd.DataFrame, config_name: str, solver: str,
                      out: Path) -> None:
    if log.empty:
        return

    metrics = [c for c in ["reward_mean", "policy_loss", "value_loss",
                            "entropy", "kl_divergence", "clip_fraction"]
               if c in log.columns and log[c].notna().any()]
    if not metrics:
        return

    cols = 3
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    fig.patch.set_facecolor(C["bg"])
    axes = np.array(axes).flatten()
    x = log.get("timestep_global", log.get("update", log.index))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(x, log[metric], color=C["reward"], linewidth=1.2)
        if metric == "reward_mean" and "reward_std" in log.columns:
            ax.fill_between(x,
                            log["reward_mean"] - log["reward_std"],
                            log["reward_mean"] + log["reward_std"],
                            color=C["reward"], alpha=0.2)
        _style_ax(ax, xlabel="Timestep", title=metric.replace("_", " ").title())

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    title = f"Training Log  |  config: {config_name}  |  solver: {solver}"
    _savefig(fig, out, title)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner – single (config, solver) pair
# ─────────────────────────────────────────────────────────────────────────────

def run_one(data_dir: Path, config_name: str, solver: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  config : {config_name}")
    print(f"  solver : {solver}")
    print(f"  data   : {data_dir}")

    # ── load CSVs ──────────────────────────────────────────────────────────
    portfolios   = _read_csv(data_dir / "portfolios.csv")
    ps           = _read_csv(data_dir / "projects_status.csv")
    pp           = _read_csv(data_dir / "projects_profile.csv")
    ms_profile   = _read_csv(data_dir / "milestones_profile.csv")
    ms_status    = _read_csv(data_dir / "milestones_status.csv")
    training_log = _read_csv(data_dir / "training_log.csv")

    if portfolios.empty or ps.empty:
        print("  [SKIP] – portfolios.csv or projects_status.csv missing / empty")
        return

    figures_dir = data_dir / "figures"

    # ── Wipe existing figures so stale PNGs from a previous run don't linger
    if figures_dir.exists():
        for old_png in figures_dir.glob("*.png"):
            old_png.unlink()
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 0: notebook-style project snapshot (single figure) ───────────
    plot_project_snapshot(
        ps, pp, ms_status, ms_profile,
        config_name, solver,
        out=figures_dir / "project_snapshot.png",
    )

    # ── Fig 1: portfolio overview ──────────────────────────────────────────
    plot_portfolio_overview(
        portfolios, config_name, solver,
        out=figures_dir / "portfolio_overview.png",
    )

    # ── Figs 2 & 3: per-project ───────────────────────────────────────────
    for proj_i in sorted(ps["i"].unique()):
        ps_i        = ps[ps["i"] == proj_i].copy()
        pp_rows     = pp[pp["i"] == proj_i]
        ms_prof_i   = ms_profile[ms_profile["i"] == proj_i] if not ms_profile.empty else pd.DataFrame()
        ms_stat_i   = ms_status[ms_status["i"] == proj_i]   if not ms_status.empty  else pd.DataFrame()

        if pp_rows.empty:
            continue
        pp_i = pp_rows.iloc[0]

        plot_project_evm(
            proj_i, ps_i, pp_i, ms_prof_i,
            config_name, solver,
            out=figures_dir / f"project_{proj_i}_evm.png",
        )
        plot_project_cashflow(
            proj_i, ps_i, pp_i,
            ms_stat_i, ms_prof_i,
            config_name, solver,
            out=figures_dir / f"project_{proj_i}_cashflow.png",
        )

    # ── Fig 4: portfolio cashflow (multi-row) ─────────────────────────────
    plot_portfolio_cashflow(
        portfolios, ps, pp,
        config_name, solver,
        out=figures_dir / "portfolio_cashflow.png",
    )

    # ── Fig 5: training log ───────────────────────────────────────────────
    plot_training_log(
        training_log, config_name, solver,
        out=figures_dir / "training_log.png",
    )

    print(f"  done  → {figures_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point – iterate over all configs / solvers
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate episode dashboards from ground-truth CSV data."
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Path to the ground_truth root directory. "
            "Defaults to <script_dir>/ground_truth."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Run only this config name (default: all).",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Run only this solver name (default: all).",
    )
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    ground_root = Path(args.root) if args.root else script_dir / "ground_truth"

    if not ground_root.exists():
        sys.exit(f"[ERROR] ground_truth directory not found: {ground_root}")

    # Discover config dirs
    config_dirs = (
        [ground_root / args.config]
        if args.config
        else sorted(p for p in ground_root.iterdir() if p.is_dir())
    )

    processed = 0
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name

        # Discover solver dirs
        solver_dirs = (
            [config_dir / args.solver]
            if args.solver
            else sorted(p for p in config_dir.iterdir() if p.is_dir())
        )

        for solver_dir in solver_dirs:
            if not solver_dir.is_dir():
                continue
            solver_name = solver_dir.name
            # Skip the "figures" output folder
            if solver_name == "figures":
                continue
            run_one(solver_dir, config_name, solver_name)
            processed += 1

    print(f"\n{'═'*60}")
    print(f"  Processed {processed} (config, solver) pair(s).")
    print(f"  All figures written under .../figures/")


if __name__ == "__main__":
    main()