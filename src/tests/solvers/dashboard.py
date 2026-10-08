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