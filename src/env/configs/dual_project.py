# src/env/configs/dual_project.py
#
# Dual-project portfolio baseline configuration — CFG-DUAL-001.
#
# Two interfaces:
#
#   CONFIG  (dict)
#       Pass directly to PortfolioBudgetingEnv(config=CONFIG).
#       No database required.
#
#   seed(conn)
#       Insert the config into the environment_config table so that
#       db_logger can reference it by config_id.  Safe to call multiple
#       times — skips silently if the row already exists.
#
# Parameters
# ----------
# Portfolio  : 2 projects, initial budget = 400 (kappa = 400/230 ~= 1.74)
# BAC        : uniform(80, 150) per project
# Profit %   : uniform(12%, 20%) per project
# Timeline   : planned_start = 0 for both; planned_duration uniform(10, 15)
# S-curve    : alpha uniform(2, 5), beta uniform(2, 5)
# Advance    : uniform(8%, 12%) of price per project
# Milestones : uniform(3, 4) per project; earliest_t at on-plan dates
#              advance recovery uniform(15%, 25%) per milestone gross
#              retention uniform(3%, 7%), released at final milestone
# Efficiency : uniform(0.75, 1.25) per period
# Discount   : 0.97 per period
# Interest   : 12% per annum → 1% per period

from __future__ import annotations

import sqlite3
from datetime import datetime

CONFIG_ID = "CFG-DUAL-001"

CONFIG: dict = {
    "config_id":   CONFIG_ID,
    "config_name": "dual_project_baseline",

    # ── portfolio ──────────────────────────────────────────────
    "n_projects_dist":            "fixed",
    "n_projects_p1":              2,
    "n_projects_p2":              None,
    "n_projects_p3":              None,
    "n_projects_p4":              None,

    "budget_available_dist":      "fixed",
    "budget_available_p1":        400.0,
    "budget_available_p2":        None,
    "budget_available_p3":        None,
    "budget_available_p4":        None,

    "budget_tightness_dist":      "fixed",
    "budget_tightness_p1":        1.74,
    "budget_tightness_p2":        None,
    "budget_tightness_p3":        None,
    "budget_tightness_p4":        None,

    "discount_dist":              "fixed",
    "discount_p1":                0.97,
    "discount_p2":                None,
    "discount_p3":                None,
    "discount_p4":                None,

    # ── project ────────────────────────────────────────────────
    "bac_dist":                   "uniform",
    "bac_p1":                     80.0,
    "bac_p2":                     150.0,
    "bac_p3":                     None,
    "bac_p4":                     None,

    "profit_percent_dist":        "uniform",
    "profit_percent_p1":          0.12,
    "profit_percent_p2":          0.20,
    "profit_percent_p3":          None,
    "profit_percent_p4":          None,

    "planned_start_dist":         "fixed",
    "planned_start_p1":           0,
    "planned_start_p2":           None,
    "planned_start_p3":           None,
    "planned_start_p4":           None,

    "planned_duration_dist":      "uniform",
    "planned_duration_p1":        10,
    "planned_duration_p2":        15,
    "planned_duration_p3":        None,
    "planned_duration_p4":        None,

    "scurve_a_dist":              "uniform",
    "scurve_a_p1":                2.0,
    "scurve_a_p2":                5.0,
    "scurve_a_p3":                None,
    "scurve_a_p4":                None,

    "scurve_b_dist":              "uniform",
    "scurve_b_p1":                2.0,
    "scurve_b_p2":                5.0,
    "scurve_b_p3":                None,
    "scurve_b_p4":                None,

    "advance_percent_dist":       "uniform",
    "advance_percent_p1":         0.08,
    "advance_percent_p2":         0.12,
    "advance_percent_p3":         None,
    "advance_percent_p4":         None,

    # Stored for schema completeness; not yet wired into certification logic.
    "advance_trigger_dist":       "fixed",
    "advance_trigger_p1":         0.20,
    "advance_trigger_p2":         None,
    "advance_trigger_p3":         None,
    "advance_trigger_p4":         None,

    "advance_recovery_dist":      "uniform",
    "advance_recovery_p1":        0.15,
    "advance_recovery_p2":        0.25,
    "advance_recovery_p3":        None,
    "advance_recovery_p4":        None,

    "retention_rate_dist":        "uniform",
    "retention_rate_p1":          0.03,
    "retention_rate_p2":          0.07,
    "retention_rate_p3":          None,
    "retention_rate_p4":          None,

    "progress_delay_cap_dist":    "fixed",
    "progress_delay_cap_p1":      0.10,
    "progress_delay_cap_p2":      None,
    "progress_delay_cap_p3":      None,
    "progress_delay_cap_p4":      None,

    "finish_delay_cap_dist":      "fixed",
    "finish_delay_cap_p1":        3,
    "finish_delay_cap_p2":        None,
    "finish_delay_cap_p3":        None,
    "finish_delay_cap_p4":        None,

    "cost_overrun_cap_dist":      "fixed",
    "cost_overrun_cap_p1":        1.30,
    "cost_overrun_cap_p2":        None,
    "cost_overrun_cap_p3":        None,
    "cost_overrun_cap_p4":        None,

    "termination_tolerance_dist": "fixed",
    "termination_tolerance_p1":   2,
    "termination_tolerance_p2":   None,
    "termination_tolerance_p3":   None,
    "termination_tolerance_p4":   None,

    "efficiency_dist":            "uniform",
    "efficiency_p1":              0.75,
    "efficiency_p2":              1.25,
    "efficiency_p3":              None,
    "efficiency_p4":              None,

    # ── milestones ─────────────────────────────────────────────
    "n_milestones_dist":          "uniform",
    "n_milestones_p1":            3,
    "n_milestones_p2":            4,
    "n_milestones_p3":            None,
    "n_milestones_p4":            None,

    "progress_threshold_dist":    "fixed",
    "progress_threshold_p1":      0.25,
    "progress_threshold_p2":      None,
    "progress_threshold_p3":      None,
    "progress_threshold_p4":      None,

    "payment_weight_dist":        "fixed",
    "payment_weight_p1":          0.25,
    "payment_weight_p2":          None,
    "payment_weight_p3":          None,
    "payment_weight_p4":          None,

    # fraction = 1.0 → earliest_t aligns with on-plan completion date
    "timestep_threshold_dist":   "fixed",
    "timestep_threshold_p1":     1.0,
    "timestep_threshold_p2":     None,
    "timestep_threshold_p3":     None,
    "timestep_threshold_p4":     None,

    "annual_interest_rate":       0.12,
}


def seed(conn: sqlite3.Connection) -> str:
    """
    Insert CFG-DUAL-001 into environment_config.
    Idempotent — skips if the row already exists.
    Returns the config_id.
    """
    from datetime import datetime, timezone

    existing = conn.execute(
        "SELECT config_id FROM environment_config WHERE config_id = ?",
        (CONFIG_ID,),
    ).fetchone()

    if existing:
        return CONFIG_ID

    conn.execute("""
        INSERT INTO environment_config (
            config_id, config_name, created_at,

            n_projects_dist, n_projects_p1,
            n_projects_p2, n_projects_p3, n_projects_p4,

            budget_available_dist, budget_available_p1,
            budget_available_p2, budget_available_p3, budget_available_p4,

            budget_tightness_dist, budget_tightness_p1,
            budget_tightness_p2, budget_tightness_p3, budget_tightness_p4,

            discount_dist, discount_p1,
            discount_p2, discount_p3, discount_p4,

            bac_dist, bac_p1,
            bac_p2, bac_p3, bac_p4,

            profit_percent_dist, profit_percent_p1,
            profit_percent_p2, profit_percent_p3, profit_percent_p4,

            planned_start_dist, planned_start_p1,
            planned_start_p2, planned_start_p3, planned_start_p4,

            planned_duration_dist, planned_duration_p1,
            planned_duration_p2, planned_duration_p3, planned_duration_p4,

            scurve_a_dist, scurve_a_p1,
            scurve_a_p2, scurve_a_p3, scurve_a_p4,

            scurve_b_dist, scurve_b_p1,
            scurve_b_p2, scurve_b_p3, scurve_b_p4,

            advance_percent_dist, advance_percent_p1,
            advance_percent_p2, advance_percent_p3, advance_percent_p4,

            advance_trigger_dist, advance_trigger_p1,
            advance_trigger_p2, advance_trigger_p3, advance_trigger_p4,

            advance_recovery_dist, advance_recovery_p1,
            advance_recovery_p2, advance_recovery_p3, advance_recovery_p4,

            retention_rate_dist, retention_rate_p1,
            retention_rate_p2, retention_rate_p3, retention_rate_p4,

            progress_delay_cap_dist, progress_delay_cap_p1,
            progress_delay_cap_p2, progress_delay_cap_p3, progress_delay_cap_p4,

            finish_delay_cap_dist, finish_delay_cap_p1,
            finish_delay_cap_p2, finish_delay_cap_p3, finish_delay_cap_p4,

            cost_overrun_cap_dist, cost_overrun_cap_p1,
            cost_overrun_cap_p2, cost_overrun_cap_p3, cost_overrun_cap_p4,

            termination_tolerance_dist, termination_tolerance_p1,
            termination_tolerance_p2, termination_tolerance_p3, termination_tolerance_p4,

            efficiency_dist, efficiency_p1,
            efficiency_p2, efficiency_p3, efficiency_p4,

            n_milestones_dist, n_milestones_p1,
            n_milestones_p2, n_milestones_p3, n_milestones_p4,

            progress_threshold_dist, progress_threshold_p1,
            progress_threshold_p2, progress_threshold_p3, progress_threshold_p4,

            payment_weight_dist, payment_weight_p1,
            payment_weight_p2, payment_weight_p3, payment_weight_p4,

            timestep_threshold_dist, timestep_threshold_p1,
            timestep_threshold_p2, timestep_threshold_p3, timestep_threshold_p4,

            annual_interest_rate
        )
        VALUES (
            ?, ?, ?,
            ?, ?, ?, ?, ?,   -- n_projects
            ?, ?, ?, ?, ?,   -- budget_available
            ?, ?, ?, ?, ?,   -- budget_tightness
            ?, ?, ?, ?, ?,   -- discount
            ?, ?, ?, ?, ?,   -- bac
            ?, ?, ?, ?, ?,   -- profit_percent
            ?, ?, ?, ?, ?,   -- planned_start
            ?, ?, ?, ?, ?,   -- planned_duration
            ?, ?, ?, ?, ?,   -- scurve_a
            ?, ?, ?, ?, ?,   -- scurve_b
            ?, ?, ?, ?, ?,   -- advance_percent
            ?, ?, ?, ?, ?,   -- advance_trigger
            ?, ?, ?, ?, ?,   -- advance_recovery
            ?, ?, ?, ?, ?,   -- retention_rate
            ?, ?, ?, ?, ?,   -- progress_delay_cap
            ?, ?, ?, ?, ?,   -- finish_delay_cap
            ?, ?, ?, ?, ?,   -- cost_overrun_cap
            ?, ?, ?, ?, ?,   -- termination_tolerance
            ?, ?, ?, ?, ?,   -- efficiency
            ?, ?, ?, ?, ?,   -- n_milestones
            ?, ?, ?, ?, ?,   -- progress_threshold
            ?, ?, ?, ?, ?,   -- payment_weight
            ?, ?, ?, ?, ?,   -- timestep_threshold
            ?                -- annual_interest_rate
        )
    """, (
        CONFIG_ID, "dual_project_baseline", datetime.now(timezone.utc).isoformat(),

        "fixed",   2,     None,  None, None,   # n_projects
        "fixed",   400.0, None,  None, None,   # budget_available
        "fixed",   1.74,  None,  None, None,   # budget_tightness
        "fixed",   0.97,  None,  None, None,   # discount

        "uniform", 80.0,  150.0, None, None,   # bac
        "uniform", 0.12,  0.20,  None, None,   # profit_percent
        "fixed",   0,     None,  None, None,   # planned_start
        "uniform", 10,    15,    None, None,   # planned_duration

        "uniform", 2.0,   5.0,   None, None,   # scurve_a
        "uniform", 2.0,   5.0,   None, None,   # scurve_b
        "uniform", 0.08,  0.12,  None, None,   # advance_percent
        "fixed",   0.20,  None,  None, None,   # advance_trigger

        "uniform", 0.15,  0.25,  None, None,   # advance_recovery
        "uniform", 0.03,  0.07,  None, None,   # retention_rate

        "fixed",   0.10,  None,  None, None,   # progress_delay_cap
        "fixed",   3,     None,  None, None,   # finish_delay_cap
        "fixed",   1.30,  None,  None, None,   # cost_overrun_cap
        "fixed",   2,     None,  None, None,   # termination_tolerance

        "uniform", 0.75,  1.25,  None, None,   # efficiency

        "uniform", 3,     4,     None, None,   # n_milestones
        "fixed",   0.25,  None,  None, None,   # progress_threshold
        "fixed",   0.25,  None,  None, None,   # payment_weight
        "fixed",   1.0,   None,  None, None,   # timestep_threshold

        0.12,                                  # annual_interest_rate
    ))
    conn.commit()
    return CONFIG_ID


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "db"))
    from db_init import init_db  # noqa: E402
    conn = init_db()
    seed(conn)
    print(f"Seeded {CONFIG_ID}")
    conn.close()