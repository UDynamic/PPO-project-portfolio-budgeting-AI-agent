# src/env/db_logger.py
#
# SQLite audit ledger — side effects only.
#
# Design rules:
#   - Every write is wrapped in try/except — a DB failure must never crash
#     the environment or affect step return values.
#   - conn.commit() is called every step unconditionally.
#   - This module owns all SQL. env.py calls the public functions only;
#     it never touches conn directly for writes.
#   - The connection is passed in on each call (stateless module).
#   - All column names and dict keys match schema.sql exactly.

from __future__ import annotations

import sqlite3


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE PROFILES  (written once at reset)
# ─────────────────────────────────────────────────────────────────────────────

def write_profiles(conn: sqlite3.Connection,
                   episode_id: str,
                   config_id: str,
                   projects: list[dict],
                   milestones_list: list[list[dict]]) -> None:
    """
    Insert projects_profile and milestones_profile rows for a new episode.
    Called once from env.reset() after sampling.
    All column names match schema exactly.
    """
    try:
        for proj in projects:
            conn.execute("""
                INSERT INTO projects_profile (
                    episode_id, config_id, i,
                    planned_start, planned_finish, planned_duration,
                    bac, profit_percent, price,
                    scurve_a, scurve_b,
                    advance_percent, advance_trigger, advance_recovery,
                    retention_rate,
                    progress_delay_cap, finish_delay_cap,
                    cost_overrun_cap, termination_tolerance
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode_id, config_id, proj["i"],
                proj["planned_start"], proj["planned_finish"],
                proj["planned_duration"],
                proj["bac"], proj["profit_percent"], proj["price"],
                proj["scurve_a"], proj["scurve_b"],
                proj["advance_percent"], proj["advance_trigger"],
                proj["advance_recovery"], proj["retention_rate"],
                proj["progress_delay_cap"], proj["finish_delay_cap"],
                proj["cost_overrun_cap"], proj["termination_tolerance"],
            ))

        for i, ms_list in enumerate(milestones_list):
            for ms in ms_list:
                conn.execute("""
                    INSERT INTO milestones_profile (
                        episode_id, i, j,
                        progress_threshold, timestep_threshold,
                        payment_weight,
                        gross_payment,
                        advance_recovery, advance_recovery_remain,
                        retention_withheld, retention_released,
                        net_payment
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    episode_id, i, ms["j"],
                    ms["progress_threshold"], ms["timestep_threshold"],
                    ms["payment_weight"],
                    ms["gross_payment"],
                    ms["advance_recovery"], ms["advance_recovery_remain"],
                    ms["retention_withheld"], ms["retention_released"],
                    ms["net_payment"],
                ))

        conn.commit()

    except Exception:
        pass  # never load-bearing


# ─────────────────────────────────────────────────────────────────────────────
# STEP WRITE  (called once per step from env.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_step(conn: sqlite3.Connection,
               episode_id: str,
               config_id: str,
               t: int,
               method: str,
               budget: float,
               net_cashflow: float,
               reward: float,
               done: bool,
               projects: list[dict],
               proj_state: list[dict],
               milestones_list: list[list[dict]],
               proj_cf: list[dict]) -> None:
    """
    Write all rows for one completed step:
        - one portfolios row
        - one projects_status row per project
        - one milestones_status row per milestone certified this step

    Called from env.step() inside a try/except — never load-bearing.
    """
    try:
        _write_portfolio_row(
            conn, episode_id, config_id, t, method,
            budget, net_cashflow, reward, done,
        )

        for proj, ps, milestones, cf in zip(
            projects, proj_state, milestones_list, proj_cf
        ):
            _write_project_row(
                conn, episode_id, t, method, proj, ps, cf,
            )
            _write_certified_milestones(
                conn, episode_id, t, method, proj["i"], milestones,
            )

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_portfolio_row(conn: sqlite3.Connection,
                         episode_id: str,
                         config_id: str,
                         t: int,
                         method: str,
                         budget: float,
                         net_cashflow: float,
                         reward: float,
                         done: bool) -> None:
    conn.execute("""
        INSERT INTO portfolios (
            episode_id, config_id, t_episode, method,
            budget_available, net_cashflow,
            reward, done
        ) VALUES (?,?,?,?,?,?,?,?)
    """, (
        episode_id, config_id, t, method,
        budget, net_cashflow,
        reward, int(done),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT STATUS ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_project_row(conn: sqlite3.Connection,
                       episode_id: str,
                       t: int,
                       method: str,
                       proj: dict,
                       ps: dict,
                       cf: dict) -> None:
    """
    Insert one projects_status row.
    Reads ps keys that match schema column names exactly.
    cf is the per-project cashflow dict from env.step proj_cf[i].
    """
    flags = ps.get("breach_flags", {})

    conn.execute("""
        INSERT INTO projects_status (
            episode_id, i, t_episode, t_project, method,

            status,

            inflow, outflow, termination_settlement, net_cashflow,

            allocation_action, deficit, interest_cost, allocation,

            efficiency,

            spi, cpi, eac,

            progress_actual,

            progress_plan_t, progress_delay_t,

            projected_cost_overrun, projected_finish, projected_finish_delay,

            abandoned, over_duration_window,
            over_progress_delay, over_finish_delay, over_cost_overrun,
            over_any,

            tolerance_remain
        ) VALUES (
            ?,?,?,?,?,
            ?,
            ?,?,?,?,
            ?,?,?,?,
            ?,
            ?,?,?,
            ?,
            ?,?,
            ?,?,?,
            ?,?,?,?,?,?,
            ?
        )
    """, (
        episode_id, proj["i"], t, ps.get("t_project"), method,

        ps.get("status"),

        ps.get("inflow", 0.0),
        ps.get("outflow", 0.0),
        ps.get("termination_settlement", 0.0),
        cf.get("milestone_net", 0.0)
            + cf.get("advance", 0.0)
            + cf.get("settlement", 0.0)
            - cf.get("allocation", 0.0)
            - cf.get("interest", 0.0),   # period net_cashflow

        cf.get("allocation", 0.0),                      # allocation_action
        cf.get("treasury_draw", 0.0),                   # deficit
        cf.get("interest", 0.0),                        # interest_cost
        cf.get("allocation", 0.0) + cf.get("interest", 0.0),  # allocation (total outflow)

        ps.get("efficiency", 1.0),

        ps.get("spi", 1.0),
        ps.get("cpi", 1.0),
        ps.get("eac", proj["bac"]),

        ps.get("progress_actual", 0.0),

        ps.get("progress_plan_t", 0.0),
        ps.get("progress_delay_t", 0.0),

        ps.get("projected_cost_overrun", 1.0),
        ps.get("projected_finish", float(proj["planned_finish"])),
        ps.get("projected_finish_delay", 0.0),

        int(flags.get("abandoned", False)),
        int(flags.get("over_duration_window", False)),
        int(flags.get("over_progress_delay", False)),
        int(flags.get("over_finish_delay", False)),
        int(flags.get("over_cost_overrun", False)),
        int(flags.get("over_any", False)),

        ps.get("tolerance_remain", proj["termination_tolerance"]),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE STATUS  (written on certification)
# ─────────────────────────────────────────────────────────────────────────────

def _write_certified_milestones(conn: sqlite3.Connection,
                                 episode_id: str,
                                 t: int,
                                 method: str,
                                 i: int,
                                 milestones: list[dict]) -> None:
    """
    Insert milestones_status rows for milestones certified this step.
    Only writes rows where certified_t == t (certified this period).
    """
    for ms in milestones:
        if not ms["certified"]:
            continue
        if ms["certified_t"] != t:
            continue   # certified in a prior step; already written

        certification_delay = (
            ms["certified_t"] - ms["timestep_threshold"]
            if ms["certified_t"] is not None else None
        )

        conn.execute("""
            INSERT OR REPLACE INTO milestones_status (
                episode_id, i, j, method,
                certified_t, certification_delay, net_payment
            ) VALUES (?,?,?,?,?,?,?)
        """, (
            episode_id, i, ms["j"], method,
            ms["certified_t"],
            certification_delay,
            ms["payment_released"],
        ))


# ─────────────────────────────────────────────────────────────────────────────
# COMMIT  (unconditional, every step)
# ─────────────────────────────────────────────────────────────────────────────

def commit(conn: sqlite3.Connection) -> None:
    """Commit the current transaction. Swallows exceptions — never load-bearing."""
    try:
        conn.commit()
    except Exception:
        pass