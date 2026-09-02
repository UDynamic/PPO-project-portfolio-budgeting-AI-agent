# src/env/db_logger.py
#
# SQLite audit ledger — side effects only.
#
# Design rules:
#   - Every write is wrapped in try/except
#   - conn.commit() called after every step
#   - This module owns all SQL
#   - All column names match schema.sql exactly
#   - milestone profile and runtime state passed and read separately

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
        pass


# ─────────────────────────────────────────────────────────────────────────────
# STEP WRITE
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
               milestone_state_list: list[list[dict]],
               proj_cf: list[dict]) -> None:
    """
    Write all rows for one completed step.
    Called AFTER the late phase completes so all post-allocation
    values (progress, outflow, evm, breach flags) are current.
    """
    try:
        _write_portfolio_row(
            conn, episode_id, config_id, t, method,
            budget, net_cashflow, reward, done,
        )
        _write_portfolio_observation(
            conn, episode_id, t, method,
            net_cashflow, budget,
        )

        for proj, ps, milestones, ms_state_list, cf in zip(
            projects, proj_state, milestones_list,
            milestone_state_list, proj_cf
        ):
            _write_project_row(
                conn, episode_id, t, method, proj, ps, cf,
            )
            _write_project_observation(
                conn, episode_id, t, method, proj, ps,
            )
            _write_certified_milestones(
                conn, episode_id, t, method, proj["i"],
                milestones, ms_state_list,
            )

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_portfolio_row(conn, episode_id, config_id, t, method,
                         budget, net_cashflow, reward, done):
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
# PORTFOLIO OBSERVATION ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_portfolio_observation(conn, episode_id, t, method,
                                  net_cashflow, budget_available):
    conn.execute("""
        INSERT INTO portfolio_observation (
            episode_id, t_episode, method,
            net_cashflow, budget_available
        ) VALUES (?,?,?,?,?)
    """, (
        episode_id, t, method,
        net_cashflow, budget_available,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT STATUS ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_project_row(conn, episode_id, t, method, proj, ps, cf):
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
            progress_space_t, min_prog_t,
            progress_needed_t, catchup_alloc_t, reach_plan_t,

            progress_plan_next_t, progress_delay_next_t,
            progress_space_next_t, min_prog_next_t,
            progress_needed_next_t, catchup_alloc_next_t, reach_plan_next_t,

            target_milestone_j,
            target_progress_gap, target_timestep_gap,
            target_net_payment, target_required_alloc,
            target_payment_rate, target_npv,

            projected_cost_overrun, projected_finish, projected_finish_delay,

            over_duration_window,
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
            ?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,
            ?,
            ?,?,?,?,?,?,
            ?,?,?,
            ?,?,?,?,?,
            ?
        )
    """, (
        episode_id, proj["i"], t, ps.get("t_project"), method,

        ps.get("status"),

        ps.get("inflow",  0.0),
        ps.get("outflow", 0.0),
        ps.get("termination_settlement", 0.0),
        # net_cashflow for this project this period
        cf.get("milestone_net", 0.0)
            + cf.get("advance",    0.0)
            + cf.get("settlement", 0.0)
            - cf.get("allocation", 0.0)
            - cf.get("interest",   0.0),

        # allocation_action is the raw amount allocated this period
        cf.get("allocation",    0.0),
        # deficit = treasury draw (outflow - inflow before this period)
        cf.get("treasury_draw", 0.0),
        cf.get("interest",      0.0),
        # total deducted from budget = allocation + interest
        cf.get("allocation",    0.0) + cf.get("interest", 0.0),

        ps.get("efficiency", 1.0),

        ps.get("spi",  1.0),
        ps.get("cpi",  1.0),
        ps.get("eac",  proj["bac"]),

        ps.get("progress_actual", 0.0),

        # current period catchup fields
        ps.get("progress_plan_t",    0.0),
        ps.get("progress_delay_t",   0.0),
        ps.get("progress_space_t",   0.0),
        ps.get("min_prog_t",         0.0),
        ps.get("progress_needed_t",  0.0),
        ps.get("catchup_alloc_t",    0.0),
        ps.get("reach_plan_t",       0.0),

        # next period catchup fields
        ps.get("progress_plan_next_t",    0.0),
        ps.get("progress_delay_next_t",   0.0),
        ps.get("progress_space_next_t",   0.0),
        ps.get("min_prog_next_t",         0.0),
        ps.get("progress_needed_next_t",  0.0),
        ps.get("catchup_alloc_next_t",    0.0),
        ps.get("reach_plan_next_t",       0.0),

        # target milestone
        ps.get("target_milestone_j"),
        ps.get("target_progress_gap",    0.0),
        ps.get("target_timestep_gap",    0),
        ps.get("target_net_payment",     0.0),
        ps.get("target_required_alloc",  0.0),
        ps.get("target_payment_rate",    0.0),
        ps.get("target_npv",             0.0),

        ps.get("projected_cost_overrun", 1.0),
        ps.get("projected_finish",       float(proj["planned_finish"])),
        ps.get("projected_finish_delay", 0.0),

        int(flags.get("over_duration_window", False)),
        int(flags.get("over_progress_delay",  False)),
        int(flags.get("over_finish_delay",    False)),
        int(flags.get("over_cost_overrun",    False)),
        int(flags.get("over_any",             False)),

        ps.get("tolerance_remain", proj["termination_tolerance"]),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT OBSERVATION ROW
# ─────────────────────────────────────────────────────────────────────────────

def _write_project_observation(conn, episode_id, t, method, proj, ps):
    """
    Write one projects_observation row per project per step.
    These are exactly the features that go into the obs vector.
    net_cashflow here is per-project: inflow - outflow (cumulative delta).
    """
    conn.execute("""
        INSERT INTO projects_observation (
            episode_id, i, t_episode, t_project, method,
            net_cashflow,
            tolerance_remain,
            catchup_alloc_t,
            catchup_alloc_next_t,
            reach_plan_t,
            reach_plan_next_t,
            target_progress_gap,
            target_timestep_gap,
            target_required_alloc,
            target_npv
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        episode_id, proj["i"], t, ps.get("t_project"), method,
        ps.get("inflow", 0.0) - ps.get("outflow", 0.0),
        ps.get("tolerance_remain",       proj["termination_tolerance"]),
        ps.get("catchup_alloc_t",        0.0),
        ps.get("catchup_alloc_next_t",   0.0),
        ps.get("reach_plan_t",           0.0),
        ps.get("reach_plan_next_t",      0.0),
        ps.get("target_progress_gap",    0.0),
        ps.get("target_timestep_gap",    0),
        ps.get("target_required_alloc",  0.0),
        ps.get("target_npv",             0.0),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE STATUS
# ─────────────────────────────────────────────────────────────────────────────

def _write_certified_milestones(conn, episode_id, t, method, i,
                                 milestones, milestone_state):
    for ms, ms_state in zip(milestones, milestone_state):
        if not ms_state["certified"]:
            continue
        if ms_state["certified_t"] != t:
            continue

        certification_delay = (
            ms_state["certified_t"] - ms["timestep_threshold"]
            if ms_state["certified_t"] is not None else None
        )

        conn.execute("""
            INSERT OR REPLACE INTO milestones_status (
                episode_id, i, j, method,
                certified_t, certification_delay, net_payment
            ) VALUES (?,?,?,?,?,?,?)
        """, (
            episode_id, i, ms["j"], method,
            ms_state["certified_t"],
            certification_delay,
            ms_state["payment_released"],
        ))


# ─────────────────────────────────────────────────────────────────────────────
# COMMIT
# ─────────────────────────────────────────────────────────────────────────────

def commit(conn: sqlite3.Connection) -> None:
    try:
        conn.commit()
    except Exception:
        pass