# src/env/render.py
#
# Terminal observation display for PortfolioBudgetingEnv.
#
# Shows exactly the observation vector defined in:
#   projects_observation  (per-project features)
#   portfolio_observation (portfolio-level features)
#
# Public API:
#   reset_history(episode_id, n_projects)   called from env.reset()
#   render(...)                             called from env.render() / play.py

from __future__ import annotations

W = 92  # terminal width

# ── episode context ───────────────────────────────────────────────────────────

_episode_id: str = ""
_n_projects: int = 0


def reset_history(episode_id: str, n_projects: int) -> None:
    global _episode_id, _n_projects
    _episode_id = episode_id
    _n_projects = n_projects


# ── format helpers ────────────────────────────────────────────────────────────

def _sep(char: str = "═", width: int = W) -> str:
    return char * width


def _status_tag(status: str) -> str:
    tags = {
        "active":     "ACTIVE",
        "pending":    "PENDING",
        "completed":  "COMPLETED",
        "terminated": "TERMINATED",
    }
    return tags.get((status or "").lower(), (status or "UNKNOWN").upper())


# ── render ────────────────────────────────────────────────────────────────────

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
    print(
        f"  PERIOD : {t} / {horizon}"
        f"   ---   net_cashflow : {net_cashflow:.2f}"
        f"   ---   budget_available : {budget:.2f}"
        f"   ---   Cumulative reward: {cum_reward:+.4f}"
    )
    print(_sep())

    # ── per-project blocks ────────────────────────────────────────────────────
    for proj, ps, ms_list, ms_state_list in zip(
        projects, proj_state, milestones, milestone_state
    ):
        i      = proj["i"]
        status = ps.get("status", "pending")

        tol     = ps.get("tolerance_remain", proj["termination_tolerance"])
        tol_max = proj["termination_tolerance"]

        proj_cf = ps.get("_last_cf_net", 0.0)   # per-project net; 0 if not set

        # Line 1 — status + identity
        print(
            f"  [{_status_tag(status)}] Project {i}"
        )

        # Line 2 — BAC / Price / t_proj / tolerance / net_cashflow
        print(
            f"  BAC: {proj['bac']:,.0f}"
            f"  ---  Price: {proj['price']:,.0f}"
            f"  ---  t_proj: {ps.get('t_project', 0)}"
            f"  ---  tolerance_remain : {tol}/{tol_max}"
            f"  ---  net_cashflow : {ps.get('inflow', 0.0) - ps.get('outflow', 0.0):.2f}"
        )

        print()

        # Line 3 — catchup
        print(
            f"    catchup_alloc_t  : {ps.get('catchup_alloc_t', 0.0):.2f}"
            f"    ---    catchup_alloc_next_t : {ps.get('catchup_alloc_next_t', 0.0):.2f}"
        )

        print()

        # Lines 4-7 — target milestone block
        t_j         = ps.get("target_milestone_j")
        t_net_pay   = ps.get("target_net_payment",    0.0)
        t_gap_prog  = ps.get("target_progress_gap",   0.0)
        t_gap_time  = ps.get("target_timestep_gap",   0)
        t_req_alloc = ps.get("target_required_alloc", 0.0)
        t_pay_rate  = ps.get("target_payment_rate",   0.0)

        ms_label = f"j={t_j}" if t_j is not None else "none"

        print(
            f"    target_milestone"
            f"              {ms_label}"
            f"           ----  net_payment: {t_net_pay:>12,.2f}"
        )
        print(
            f"    target_progress_gap  : {t_gap_prog:.4f}"
            f"  ---  target_timestep_gap  : {t_gap_time:>3}"
        )
        print(
            f"    target_required_alloc : {t_req_alloc:.2f}"
        )
        print(
            f"    target_payment_rate   : {t_pay_rate:.4f}"
        )

        # Breach flags — only if any active
        flags = ps.get("breach_flags", {})
        active_flags = [
            k.replace("over_", "").replace("_", "-").upper()
            for k, v in flags.items()
            if v and k != "over_any"
        ]
        if active_flags:
            print(f"    ⚠  BREACHES: {', '.join(active_flags)}")

        print(_sep())