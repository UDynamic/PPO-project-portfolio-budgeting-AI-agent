# src/env/render.py
#
# Compact two-column terminal observation display for PortfolioBudgetingEnv.
#
# Layout
# ------
# Portfolio header  : period / reward  |  net_cashflow / budget
# Per-project block : identity, catchup, target — all in two columns
# Thin ─ separator between logical groups within a project block
# Thick ═ separator between projects and at top/bottom
#
# Public API
# ----------
# reset_history(episode_id, n_projects)   called from env.reset()
# render(...)                             called from env.render() / play.py

from __future__ import annotations

# ── layout constants ──────────────────────────────────────────────────────────

W   = 80   # total line width
LW  = 17   # label column width
VW  = 12   # value column width
GAP =  3   # gap between the two columns


# ── format helpers ────────────────────────────────────────────────────────────

def _sep(char: str = "═") -> str:
    return char * W


def _row(l_label: str, l_value: str,
         r_label: str = "", r_value: str = "") -> str:
    """
    One two-column row.

    Left  : l_label padded to LW, l_value right-aligned to VW
    Right : r_label padded to LW, r_value right-aligned to VW
    Columns separated by GAP spaces.
    If right side is empty the line ends after the left value.
    """
    left = f"  {l_label:<{LW}} : {l_value:>{VW}}"
    if r_label or r_value:
        right = f"{' ' * GAP}{r_label:<{LW}} : {r_value:>{VW}}"
        return left + right
    return left


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def _status_tag(status: str) -> str:
    tags = {
        "active":     "ACTIVE",
        "pending":    "PENDING",
        "completed":  "COMPLETED",
        "terminated": "TERMINATED",
    }
    return tags.get((status or "").lower(), (status or "UNKNOWN").upper())


# ── episode context ───────────────────────────────────────────────────────────

_episode_id: str = ""
_n_projects: int = 0


def reset_history(episode_id: str, n_projects: int) -> None:
    global _episode_id, _n_projects
    _episode_id = episode_id
    _n_projects = n_projects


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
           net_cashflow: float = 0.0,
           cum_reward: float = 0.0,
           **_kwargs) -> None:

    # ── portfolio header ──────────────────────────────────────────────────────
    print()
    print(_sep())
    print(_row(
        f"PERIOD {t} / {horizon}", "",
        "net_cashflow",      _fmt(net_cashflow),
    ))
    print(_row(
        "reward",            f"{cum_reward:+.4f}",
        "budget_available",  _fmt(budget),
    ))
    print(_sep())

    # ── per-project blocks ────────────────────────────────────────────────────
    for proj, ps, ms_list, ms_state_list in zip(
        projects, proj_state, milestones, milestone_state
    ):
        i      = proj["i"]
        status = _status_tag(ps.get("status", "pending"))

        tol     = ps.get("tolerance_remain", proj["termination_tolerance"])
        tol_max = proj["termination_tolerance"]
        inflow  = ps.get("inflow",  0.0)
        outflow = ps.get("outflow", 0.0)

        # ── project heading ───────────────────────────────────────────────
        print(f"  [{status}]  Project {i}")

        # ── identity ──────────────────────────────────────────────────────
        print(_row(
            "BAC",   _fmt(proj["bac"]),
            "Price", _fmt(proj["price"]),
        ))
        print(_row(
            "t_proj",            str(ps.get("t_project", 0)),
            "tolerance_remain",  f"{tol}/{tol_max}",
        ))
        print(_row(
            "inflow",  _fmt(inflow),
            "outflow", _fmt(outflow),
        ))

        # ── catchup ───────────────────────────────────────────────────────
        print(_sep("─"))
        print(_row(
            "catchup_t",      _fmt(ps.get("catchup_alloc_t",      0.0)),
            "catchup_next_t", _fmt(ps.get("catchup_alloc_next_t", 0.0)),
        ))
        print(_row(
            "reach_plan_t",      _fmt(ps.get("reach_plan_t",      0.0)),
            "reach_plan_next_t", _fmt(ps.get("reach_plan_next_t", 0.0)),
        ))

        # ── target milestone ──────────────────────────────────────────────
        print(_sep("─"))
        t_j         = ps.get("target_milestone_j")
        t_net_pay   = ps.get("target_net_payment",    0.0)
        t_gap_prog  = ps.get("target_progress_gap",   0.0)
        t_gap_time  = ps.get("target_timestep_gap",   0)
        t_req_alloc = ps.get("target_required_alloc", 0.0)
        t_pay_rate  = ps.get("target_payment_rate",   0.0)

        ms_label = f"j={t_j}" if t_j is not None else "none"

        print(_row(
            "target",        ms_label,
            "net_payment",   _fmt(t_net_pay),
        ))
        print(_row(
            "progress_gap",  _fmt(t_gap_prog, 4),
            "timestep_gap",  str(t_gap_time),
        ))
        t_npv = ps.get("target_npv", 0.0)

        print(_row(
            "required_alloc", _fmt(t_req_alloc),
            "target_npv",     _fmt(t_npv, 4),
        ))
        print(_sep())