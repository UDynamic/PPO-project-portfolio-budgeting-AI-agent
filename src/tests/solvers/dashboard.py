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
# Colour palette  – refined, perceptually distinct, print-friendly
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    bcws      = "#4361ee",   # indigo-blue  – planned value / BCWS
    bcwp      = "#06d6a0",   # emerald      – earned value  / BCWP
    acwp      = "#f77f00",   # amber        – actual cost   / ACWP
    spi       = "#06d6a0",   # emerald      – SPI line
    cpi       = "#f77f00",   # amber        – CPI line
    ref_line  = "#ef233c",   # red          – reference = 1
    threshold = "#06d6a0",   # emerald      – milestone thresholds
    inflow    = "#06d6a0",   # emerald bars – inflows
    outflow   = "#ef233c",   # red bars     – outflows
    net       = "#4361ee",   # indigo       – net cash flow
    balance   = "#0096c7",   # cerulean     – cash balance
    reward    = "#7b2d8b",   # violet       – cumul. discounted NCF / reward
    alloc     = "#ef233c",   # red bars     – budget allocation spend
    ms_step   = "#8338ec",   # purple       – milestone profile staircase
    bg        = "#f8f9fb",   # near-white panel background
    grid      = "#d8dce8",   # subtle blue-grey grid
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
#   • Dashed violet step → MS Plan: full planned payment schedule
#                          (cumulative net_payment / price, at timestep_threshold)
#                          — always drawn in full, regardless of outcome.
#   • Solid  violet step → MS Certified: cumulative payments that were
#                          *actually* certified (at certified_t). Drawn on
#                          top of the dashed plan line, so wherever a
#                          milestone has been certified the solid line
#                          visually "covers" the dashed one. The line simply
#                          stops at the last certification — everything
#                          beyond that (e.g. after a project is terminated)
#                          is only visible as the uncovered dashed plan,
#                          flagging payments that were scheduled but never
#                          received.
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

    # Single-action project: planned_duration == 0 or 1 step with start==0
    is_single_action = (planned_finish - planned_start) <= 1 and planned_start == 0

    ps_sorted = ps_i.sort_values("t_episode").copy()

    # ── dynamic x-window ─────────────────────────────────────────────────
    non_pending = ps_sorted[
        ps_sorted["status"].isin(["active", "completed", "terminated"])
    ]
    if not non_pending.empty:
        actual_start = int(non_pending["t_episode"].min())

        # The environment keeps emitting "completed" rows for every subsequent
        # episode timestep after a project finishes, inflating actual_finish far
        # beyond the true end.  We want the chart to stop at the timestep the
        # project actually finished, not at the last episode timestep.
        #
        # Strategy: find the last *active* row; the project finished one step
        # later (the first "completed/terminated" row after the active period).
        active_rows = ps_sorted[ps_sorted["status"] == "active"]
        if not active_rows.empty:
            last_active_t = int(active_rows["t_episode"].max())
            first_done = ps_sorted[
                ps_sorted["status"].isin(["completed", "terminated"]) &
                (ps_sorted["t_episode"] > last_active_t)
            ]
            actual_finish = int(first_done["t_episode"].min()) \
                            if not first_done.empty else last_active_t + 1
        else:
            # Never went active — terminated before start, or instant completion
            first_done = ps_sorted[ps_sorted["status"].isin(["completed", "terminated"])]
            actual_finish = int(first_done["t_episode"].min()) \
                            if not first_done.empty else planned_finish
    else:
        actual_start  = planned_start
        actual_finish = planned_finish

    x_lo   = max(0, min(planned_start, actual_start) - 1)
    x_hi   = max(planned_finish, actual_finish)
    x_span = max(x_hi - x_lo, 1)

    # Small visual padding on right so markers aren't clipped
    x_lim_r = x_hi + max(0.65, x_span * 0.06)

    # Slice to the window
    win   = ps_sorted[
        (ps_sorted["t_episode"] >= x_lo) &
        (ps_sorted["t_episode"] <= x_hi)
    ].copy()
    t_win = win["t_episode"].values

    # ── responsive sizing ─────────────────────────────────────────────────
    msize   = float(np.clip(5.5 - x_span * 0.12, 2.5, 5.5))
    lw      = float(np.clip(2.2 - x_span * 0.04, 1.2, 2.2))
    fs_tick = max(6, min(8, 8 - x_span // 10))

    # ── EVM series (normalised 0..1) ──────────────────────────────────────
    bcwp_cum = win["progress_actual"].values
    acwp_cum = win["allocation"].cumsum().values / (bac + 1e-9)

    # BCWS: for a single-action project the plan is immediately 100 % at t=0,
    # so progress_plan_t jumps from 0→1 in one step and shows an ugly
    # vertical line at t=0.  Instead we draw a flat horizontal at y=1.0
    # from x_lo to planned_finish (the project was always expected to finish
    # in one action, so the "plan" is 100 % from the start).
    bcws_t: np.ndarray
    bcws_y: np.ndarray
    if is_single_action:
        bcws_t = np.array([x_lo, planned_finish], dtype=float)
        bcws_y = np.array([1.0, 1.0])
    else:
        bcws_t = t_win.astype(float)
        bcws_y = win["progress_plan_t"].values

    # ── milestone payment schedule: PLANNED (dashed) vs CERTIFIED (solid) ──
    #
    # PLANNED  → built from milestones_profile.csv using the milestone's
    #            planned trigger timestep (timestep_threshold). This is the
    #            *complete* schedule and is always drawn in full as a dashed
    #            step, independent of what actually happened.
    #
    # CERTIFIED → built from milestones_status.csv using the timestep the
    #            milestone was *actually* certified (certified_t). Only
    #            milestones that really got certified contribute. Drawn as a
    #            SOLID step, on top of the dashed plan line (same colour).
    #            Wherever a payment has been certified, the solid segment
    #            visually "covers" the dashed one up to that spike; the solid
    #            line then simply stops — any further (uncertified) vertical
    #            jumps remain visible only as dashed, flagging money that was
    #            scheduled but never received (e.g. after project
    #            termination). For fully completed projects every milestone
    #            is eventually certified, so the solid line tracks the dashed
    #            one the whole way and the plan appears fully "filled".
    plan_step_t: list[float] = [x_lo]
    plan_step_y: list[float] = [0.0]
    if not ms_profile_i.empty:
        time_col = next(
            (c for c in ("timestep_threshold", "planned_t", "threshold_t", "t_threshold")
             if c in ms_profile_i.columns),
            None,
        )
        if time_col is not None and "net_payment" in ms_profile_i.columns:
            ms_plan  = ms_profile_i.sort_values(time_col)
            cum_pay  = 0.0
            for _, row in ms_plan.iterrows():
                pt       = row[time_col]
                if pd.isna(pt):
                    continue
                pt       = float(pt)
                cum_pay += float(row["net_payment"])
                if x_lo <= pt <= x_hi:
                    plan_step_t.append(pt)
                    plan_step_y.append(cum_pay / (price + 1e-9))

    cert_step_t: list[float] = [x_lo]
    cert_step_y: list[float] = [0.0]
    if not ms_status_i.empty:
        ms_cert = ms_status_i.copy()
        if "certified" in ms_cert.columns:
            ms_cert = ms_cert[ms_cert["certified"].astype(bool)]
        if "certified_t" in ms_cert.columns:
            ms_cert = ms_cert[ms_cert["certified_t"].notna()]
            ms_cert = ms_cert[ms_cert["certified_t"] >= 0]
        if not ms_profile_i.empty and "net_payment" in ms_profile_i.columns:
            ms_cert = ms_cert.merge(
                ms_profile_i[["j", "net_payment"]],
                on="j", how="left", suffixes=("", "_prof"),
            )
        pay_col = "net_payment" if "net_payment" in ms_cert.columns else "net_payment_prof"
        if "certified_t" in ms_cert.columns and pay_col in ms_cert.columns:
            ms_cert  = ms_cert.sort_values("certified_t")
            cum_pay  = 0.0
            for _, row in ms_cert.iterrows():
                ct = row["certified_t"]
                if pd.isna(ct):
                    continue
                ct       = int(ct)
                cum_pay += float(row[pay_col])
                if x_lo <= ct <= x_hi:
                    cert_step_t.append(ct)
                    cert_step_y.append(cum_pay / (price + 1e-9))

    # ── y-axis limits ─────────────────────────────────────────────────────
    all_vals = np.concatenate([
        bcwp_cum, acwp_cum, bcws_y,
        np.array(plan_step_y), np.array(cert_step_y),
    ])
    y_min = min(float(all_vals.min()), 0.0)
    y_max = max(float(all_vals.max()), 1.0)
    margin_top = (y_max - y_min) * 0.22
    margin_bot = (y_max - y_min) * 0.10
    ax.set_ylim(y_min - margin_bot, y_max + margin_top)

    # ── derive final status metrics (used in both plotting and title) ────
    last_row   = non_pending.sort_values("t_episode").iloc[-1] \
                 if not non_pending.empty else win.iloc[-1]
    cpi_val    = float(last_row["cpi"])
    spi_val    = float(last_row["spi"])
    status_val = str(last_row["status"]).upper()
    status_color = {"COMPLETED": "#039972", "TERMINATED": "#ef233c"}.get(status_val, "#4a5568")

    # ══════════════════════ plotting ══════════════════════════════════════

    # Subtle reference lines
    ax.axhline(1.0, color="#c8ccd6", linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)
    ax.axhline(0.0, color="#b0b4be", linewidth=0.6, linestyle="-",  alpha=0.5, zorder=1)

    # BCWS – indigo line (horizontal for single-action, step otherwise)
    if is_single_action:
        ax.plot(bcws_t, bcws_y,
                linewidth=lw, color=C["bcws"],
                linestyle="-", alpha=0.9,
                label="BCWS  (planned progress)", zorder=5)
        # Mark both endpoints
        ax.scatter(bcws_t, bcws_y,
                   s=msize ** 2, color=C["bcws"],
                   edgecolors="#2340b8", linewidths=0.7,
                   zorder=6)
    else:
        ax.step(bcws_t, bcws_y,
                where="pre", linewidth=lw, color=C["bcws"],
                marker="o", markersize=msize,
                markeredgecolor="#2340b8", markeredgewidth=0.7,
                label="BCWS  (planned progress)", zorder=5)

    # BCWP – emerald step chart
    ax.step(t_win, bcwp_cum,
            where="post", linewidth=lw, color=C["bcwp"],
            marker="o", markersize=msize,
            markeredgecolor="#039972", markeredgewidth=0.7,
            label="BCWP  (earned value)", zorder=5)

    # ACWP – amber step chart + triangles
    ax.step(t_win, acwp_cum,
            where="post", linewidth=lw, color=C["acwp"],
            marker="^", markersize=msize,
            markeredgecolor="#c06200", markeredgewidth=0.7,
            label="ACWP  (actual cost / BAC)", zorder=5)

    # MS Plan – violet DASHED step: full planned payment schedule, always
    # drawn in full so any un-certified remainder stays visible.
    if len(plan_step_t) > 1:
        ax.step(plan_step_t, plan_step_y,
                where="post", linewidth=lw * 0.85, linestyle="--",
                color=C["ms_step"], alpha=0.65,
                label="MS Plan  (scheduled pmts / price)", zorder=4)

    # MS Certified – violet SOLID step: actually-received payments, drawn on
    # top of the dashed plan line. It "covers" the dashed line wherever a
    # milestone has been certified, and simply stops where certification
    # ends (e.g. at a termination timestep), exposing the dashed remainder.
    if len(cert_step_t) > 1:
        ax.step(cert_step_t, cert_step_y,
                where="post", linewidth=lw * 1.05, linestyle="-",
                color=C["ms_step"], alpha=0.95,
                label="MS Certified  (received pmts / price)", zorder=4.5)

    # ── planned-finish marker ─────────────────────────────────────────────
    ax.axvline(planned_finish, color="#8892a4", linewidth=1.0,
               linestyle=":", alpha=0.7, zorder=2)
    ax.text(planned_finish + 0.07, 0.97,
            f"fᵢ = {planned_finish}",
            fontsize=max(6, min(8, 8 - x_span // 8)),
            color="#6b758a", va="top", ha="left",
            transform=ax.get_xaxis_transform(), zorder=6)

    # ── termination marker ────────────────────────────────────────────────
    if status_val == "TERMINATED":
        ax.axvline(actual_finish, color=C["ref_line"], linewidth=1.4,
                   linestyle="--", alpha=0.9, zorder=7)
        ax.text(actual_finish + 0.07, 0.80,
                f"terminated\nt = {actual_finish}",
                fontsize=max(6, min(7, 7 - x_span // 12)),
                color=C["ref_line"], va="top", ha="left",
                transform=ax.get_xaxis_transform(), zorder=8)

    # ── metrics box ───────────────────────────────────────────────────────
    ax.text(0.012, 0.985,
            f"Status : {status_val}\nCPI    = {cpi_val:.3f}\nSPI    = {spi_val:.3f}",
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", fontfamily="monospace",
            color=status_color,
            bbox=dict(facecolor="white", alpha=0.88,
                      edgecolor="#d0d4de", boxstyle="round,pad=0.35"),
            zorder=8)

    # ── axis styling ──────────────────────────────────────────────────────
    ax.set_xlim(x_lo - 0.6, x_lim_r)

    tick_step = max(1, x_span // 15)
    ax.set_xticks(range(x_lo, x_hi + 1, tick_step))
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    ax.set_xlabel("Episode Timestep", fontsize=8, color="#4a5568")
    ax.set_ylabel("Fraction of BAC / price", fontsize=8, color="#4a5568")
    ax.set_title(
        f"Project {proj_i}  ·  BAC = {bac:.0f}  ·  "
        f"Plan [{planned_start} → {planned_finish}]  ·  {status_val}",
        fontsize=9, fontweight="bold", pad=7, color="#1a202c",
    )
    ax.set_facecolor(C["bg"])
    ax.grid(True, linestyle=":", color=C["grid"], alpha=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8ccd6")

    # ── legend outside right ──────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="upper left", bbox_to_anchor=(1.03, 1.0),
              fontsize=7.5, framealpha=0.95, edgecolor="#d0d4de",
              title="Legend", title_fontsize=8, borderpad=0.8,
              labelspacing=0.55, handlelength=1.8)


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

    HSPACE_PER = 0.90    # inches of vertical gap between subplots
    # Add a generous top pad (in inches) that sits between the suptitle and
    # the first subplot title.  This becomes the sole source of that spacing —
    # we no longer rely on bbox_inches arithmetic.
    TOP_PAD_IN = 0.70
    fig_h = TOP_PAD_IN + sum(row_heights) + HSPACE_PER * n_proj

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

    # top_frac: the fraction of the figure height that subplots occupy.
    # Subplots start just below the TOP_PAD_IN band and fill the rest.
    top_frac = 1.0 - TOP_PAD_IN / fig_h

    fig.subplots_adjust(
        hspace = HSPACE_PER * n_proj / sum(row_heights),
        right  = 0.80,
        left   = 0.06,
        top    = top_frac,
        bottom = 0.04,
    )

    # Place suptitle in the centre of the top pad band.
    # top_frac is the bottom of the pad; 1.0 is the top of the figure.
    # Midpoint in figure coords = top_frac + TOP_PAD_IN / fig_h * 0.5
    suptitle_y = top_frac + (TOP_PAD_IN / fig_h) * 0.50
    fig.suptitle(
        f"Project-Level Snapshot  |  config: {config_name}  |  solver: {solver}",
        fontsize=11, fontweight="bold", y=suptitle_y,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" would re-crop and break the spacing we just set,
    # so we save without it and let the pre-allocated margins do their job.
    fig.savefig(out, dpi=DPI, facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out.relative_to(out.parents[4]) if out.parents[4].exists() else out}")



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

    # ── Panel B: milestone payment breakdown ───────────────────────────────
    #
    # "Apartment levels" design:
    #   • Gross payment  → solid, faint, FULL-width background bar. This is
    #                      the total envelope / context: the building itself.
    #   • Net payment    → NARROW white box, green outline   (ground floor)
    #   • Adv. recovery  → NARROW white box, yellow outline  (next floor up)
    #   • Retention      → NARROW white box, purple outline  (top floor)
    #
    # The three narrow boxes are stacked bottom-to-top (Net → Recovery →
    # Retention) at the *same* narrow width, directly on top of one another
    # like floors of a building. Because their heights sum exactly to the
    # Gross amount, the stack visually proves
    #     Gross = Net + Advance-recovery + Retention
    # and the thickness of each floor shows its share of the total.
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

        width        = 0.3
        narrow_width = width * 0.5   # shared width for Net / Recovery / Retention

        col_net = C["inflow"]   # green  – net payment (ground floor)
        col_rec = "#f4c430"     # yellow – advance recovery (middle floor)
        col_ret = "#aa44cc"     # purple – retention withheld (top floor)

        # Gross – solid, faint, full-width backdrop (the "building")
        ax.bar(t_cert, gross, width=width, color=C["inflow"], alpha=0.80,
               label="Gross", zorder=1)

        # Stacked floor levels — Net (bottom) → Recovery → Retention (top)
        bottom_net = np.zeros_like(net_p, dtype=float)
        bottom_rec = net_p
        bottom_ret = net_p + rec

        ax.bar(t_cert, net_p, width=narrow_width, bottom=bottom_net,
               facecolor="white", edgecolor=col_net, linewidth=1.6,
               label="Net pay.", zorder=3)
        ax.bar(t_cert, rec, width=narrow_width, bottom=bottom_rec,
               facecolor="white", edgecolor=col_rec, linewidth=1.6,
               label="Adv. recovery", zorder=3)
        ax.bar(t_cert, ret_w, width=narrow_width, bottom=bottom_ret,
               facecolor="white", edgecolor=col_ret, linewidth=1.6,
               label="Retention", zorder=3)

        ax.axhline(0, color="black", linewidth=0.5)
        for j_row, ms_row in enumerate(ms.itertuples()):
            ax.text(ms_row.certified_t, float(gross[j_row]) + 1,
                    f"MS{ms_row.j}", fontsize=6, ha="center", color=C["inflow"])
    _style_ax(ax, ylabel="Amount",
              title=f"Project {proj_i} – Milestone Payments  "
                    f"(Gross = Net + Recovery + Retention)")
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

    # ── Figs 2: per-project cashflow ─────────────────────────────────────
    for proj_i in sorted(ps["i"].unique()):
        ps_i        = ps[ps["i"] == proj_i].copy()
        pp_rows     = pp[pp["i"] == proj_i]
        ms_prof_i   = ms_profile[ms_profile["i"] == proj_i] if not ms_profile.empty else pd.DataFrame()
        ms_stat_i   = ms_status[ms_status["i"] == proj_i]   if not ms_status.empty  else pd.DataFrame()

        if pp_rows.empty:
            continue
        pp_i = pp_rows.iloc[0]

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