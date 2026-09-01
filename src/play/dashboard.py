# src/env/render.py
#
# Minimal terminal observation display for PortfolioBudgetingEnv.
#
# Responsibility: show the agent/player exactly what the observation
# vector contains at each timestep, plus enough context to make an
# informed allocation decision.
#
# Deliberately excluded (handled by dashboard.py):
#   - timeseries panels
#   - cashflow breakdown tables
#   - milestone payment profile detail
#   - EVM history charts
#
# Two public functions:
#   reset_history(episode_id, n_projects)  → called from env.reset()
#   render(...)                            → called from env.render() and play.py

from __future__ import annotations

from typing import Optional

W = 90  # terminal width

# ── history store (minimal — only what render needs) ─────────────────────────

_episode_id: str = ""
_n_projects: int = 0


def reset_history(episode_id: str, n_projects: int) -> None:
    """Called once from env.reset(). Stores episode context for render."""
    global _episode_id, _n_projects
    _episode_id = episode_id
    _n_projects = n_projects


# ── format helpers ────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = W) -> str:
    return char * width


def _header(title: str, width: int = W) -> str:
    pad   = max(0, width - len(title) - 4)
    left  = pad // 2
    right = pad - left
    return f"{'═' * left}  {title}  {'═' * right}"


def _bar(value: float, width: int = 30, char: str = "█", empty: str = "░") -> str:
    filled = max(0, min(width, round(value * width)))
    return char * filled + empty * (width - filled)


def _status_tag(status: str) -> str:
    tags = {
        "active":     "[ACTIVE]    ",
        "pending":    "[PENDING]   ",
        "completed":  "[COMPLETED] ",
        "terminated": "[TERMINATED]",
    }
    return tags.get(status, f"[{status.upper():<10}]")


# ── portfolio header ──────────────────────────────────────────────────────────

def _print_portfolio_header(t: int,
                             budget: float,
                             initial_budget: float,
                             horizon: int,
                             cum_reward: float) -> None:
    budget_norm = budget / initial_budget if initial_budget > 0 else 1.0
    budget_bar  = _bar(min(budget_norm, 1.0))

    print()
    print(_sep("═"))
    print(
        f"  PERIOD {t:>3} / {horizon}"
        f"   Budget: {budget:>12,.2f}"
        f"  ({budget_norm:>6.1%} of initial)"
        f"   Cumulative reward: {cum_reward:+.4f}"
    )
    print(f"  Budget  [{budget_bar}]")
    print(_sep("═"))


# ── per-project observation ───────────────────────────────────────────────────

def _print_project_obs(proj: dict,
                        ps: dict,
                        milestones: list[dict],
                        milestone_state: list[dict],
                        budget: float) -> None:
    i      = proj["i"]
    status = ps.get("status", "pending")

    print()
    print(
        f"  Project {i}  {_status_tag(status)}"
        f"  BAC: {proj['bac']:,.0f}"
        f"  Price: {proj['price']:,.0f}"
        f"  t_proj: {ps.get('t_project', 0):>3}"
        f"  [{proj['planned_start']} → {proj['planned_finish']}]"
    )
    print(_sep("─"))

    # Progress bars
    prog  = ps.get("progress_actual",  0.0)
    plan  = ps.get("progress_plan_t",  0.0)
    delay = ps.get("progress_delay_t", 0.0)

    print(f"  Actual   [{_bar(prog)}]  {prog:>6.2%}")
    print(f"  Plan     [{_bar(plan)}]  {plan:>6.2%}   delay: {delay:>+.4f}")

    # EVM signals
    spi  = ps.get("spi",  1.0)
    cpi  = ps.get("cpi",  1.0)
    tcpi = ps.get("tcpi", 1.0)
    eac  = ps.get("eac",  proj["bac"])
    print(
        f"  SPI(t): {spi:>6.3f}"
        f"   CPI: {cpi:>6.3f}"
        f"   TCPI: {tcpi:>6.3f}"
        f"   EAC: {eac:>12,.2f}"
        f"   EAC/BAC: {ps.get('projected_cost_overrun', 1.0):.3f}x"
    )

    # Schedule projection
    pf_delay = ps.get("projected_finish_delay", 0.0)
    pf       = ps.get("projected_finish", float(proj["planned_finish"]))
    print(
        f"  Projected finish: {pf:.1f}"
        f"  (delay: {pf_delay:+.1f})"
        f"   Finish delay cap: {proj['finish_delay_cap']}"
    )

    # Tolerance
    tol      = ps.get("tolerance_remain", proj["termination_tolerance"])
    tol_max  = proj["termination_tolerance"]
    tol_norm = tol / tol_max if tol_max > 0 else 1.0
    low_flag = "  ⚠ LOW" if tol <= 2 else ""
    print(
        f"  Tolerance: {tol}/{tol_max}"
        f"  [{_bar(tol_norm, width=20)}]{low_flag}"
    )

    # Breach flags (only non-zero)
    flags = ps.get("breach_flags", {})
    active_flags = [
        k.replace("over_", "").replace("_", "-").upper()
        for k, v in flags.items()
        if v and k != "over_any"
    ]
    if active_flags:
        print(f"  ⚠  BREACHES: {', '.join(active_flags)}")

    # Next uncertified milestone
    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] == 0:
            continue
        if ms_state["certified"]:
            continue
        gap_prog = ms["progress_threshold"] - prog
        print(
            f"  Next milestone j={ms['j']}"
            f"  threshold: {ms['progress_threshold']:.0%}"
            f"  gap: {gap_prog:+.4f}"
            f"  net pay: {ms['net_payment']:,.2f}"
            f"  earliest t: {ms['timestep_threshold']}"
        )
        break

    # Cashflow summary (inflow / outflow)
    print(
        f"  Cum inflow: {ps.get('inflow', 0.0):>12,.2f}"
        f"   Cum outflow: {ps.get('outflow', 0.0):>12,.2f}"
    )


# ── public render entry point ─────────────────────────────────────────────────

def render(t: int,
           budget: float,
           horizon: int,
           initial_budget: float,
           projects: list[dict],
           proj_state: list[dict],
           milestones: list[list[dict]],
           milestone_state: list[list[dict]],
           episode_id: str = "",
           cum_reward: float = 0.0,
           **_kwargs) -> None:
    """
    Print the minimal observation display for one timestep.

    Parameters accepted but intentionally ignored (**_kwargs):
        proj_cf, period_inflow, period_outflow, reward
    These are shown in the dashboard, not the terminal.
    """
    _print_portfolio_header(t, budget, initial_budget, horizon, cum_reward)

    for proj, ps, ms_list, ms_state_list in zip(
        projects, proj_state, milestones, milestone_state
    ):
        _print_project_obs(proj, ps, ms_list, ms_state_list, budget)

    print()