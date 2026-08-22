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
# Project 0  : larger, longer, back-loaded, tighter margins (informal target)
# Project 1  : smaller, shorter, front-loaded, richer margins (informal target)
# BAC        : uniform(80, 150) per project — episode draws two varied projects
# Margin     : uniform(12%, 20%) per project
# Timeline   : start = 0 for both; duration uniform(10, 15)
# S-curve    : alpha uniform(2, 5), beta uniform(2, 5) — varied shapes per draw
# Advance    : uniform(8%, 12%) of price per project
# Milestones : uniform(3, 4) per project; earliest_t at on-plan dates
#              advance recovery uniform(15%, 25%) per milestone gross
#              retention uniform(3%, 7%), released at completion
# Efficiency : uniform(0.75, 1.25) per period — wider spread than single project
# Discount   : 0.97 per period
# Interest   : 12% per annum -> 1% per period
#
# NOTE ON SINGLE-CONFIG MULTI-PROJECT DESIGN
# -------------------------------------------
# environment_config stores one distribution per parameter, shared across all
# projects sampled in an episode. Two projects with a fixed structural
# contrast (BAC, duration, s-curve) would require a per-project config table —
# a planned extension. For this MVP baseline, distributions are widened to
# span the desired contrast, so each episode independently draws two varied
# projects. Project count is fixed at 2.

from __future__ import annotations

import sqlite3
from datetime import datetime

CONFIG_ID = "CFG-DUAL-001"

CONFIG: dict = {
    "config_id":   CONFIG_ID,
    "config_name": "dual_project_baseline",

    # ── portfolio ──────────────────────────────────────────────
    "n_projects_dist":        "fixed",
    "n_projects_p1":          2,
    "n_projects_p2":          None,
    "n_projects_p3":          None,
    "n_projects_p4":          None,

    "initial_budget_dist":    "fixed",
    "initial_budget_p1":      400.0,
    "initial_budget_p2":      None,
    "initial_budget_p3":      None,
    "initial_budget_p4":      None,

    "budget_tightness_dist":  "fixed",
    "budget_tightness_p1":    1.74,
    "budget_tightness_p2":    None,
    "budget_tightness_p3":    None,
    "budget_tightness_p4":    None,

    "discount_dist":          "fixed",
    "discount_p1":            0.97,
    "discount_p2":            None,
    "discount_p3":            None,
    "discount_p4":            None,

    # ── project ────────────────────────────────────────────────
    "budget_dist":            "uniform",
    "budget_p1":              80.0,
    "budget_p2":              150.0,
    "budget_p3":              None,
    "budget_p4":              None,

    "margin_dist":            "uniform",
    "margin_p1":              0.12,
    "margin_p2":              0.20,
    "margin_p3":              None,
    "margin_p4":              None,

    "start_dist":             "fixed",
    "start_p1":               0,
    "start_p2":               None,
    "start_p3":               None,
    "start_p4":               None,

    "duration_dist":          "uniform",
    "duration_p1":            10,
    "duration_p2":            15,
    "duration_p3":            None,
    "duration_p4":            None,

    "scurve_a_dist":          "uniform",
    "scurve_a_p1":            2.0,
    "scurve_a_p2":            5.0,
    "scurve_a_p3":            None,
    "scurve_a_p4":            None,

    "scurve_b_dist":          "uniform",
    "scurve_b_p1":            2.0,
    "scurve_b_p2":            5.0,
    "scurve_b_p3":            None,
    "scurve_b_p4":            None,

    "advance_percent_dist":   "uniform",
    "advance_percent_p1":     0.08,
    "advance_percent_p2":     0.12,
    "advance_percent_p3":     None,
    "advance_percent_p4":     None,

    # Stored for schema completeness; not yet wired into certification logic.
    "advance_trigger_dist":   "fixed",
    "advance_trigger_p1":     0.20,
    "advance_trigger_p2":     None,
    "advance_trigger_p3":     None,
    "advance_trigger_p4":     None,

    "advance_recovery_dist":  "uniform",
    "advance_recovery_p1":    0.15,
    "advance_recovery_p2":    0.25,
    "advance_recovery_p3":    None,
    "advance_recovery_p4":    None,

    "retention_rate_dist":    "uniform",
    "retention_rate_p1":      0.03,
    "retention_rate_p2":      0.07,
    "retention_rate_p3":      None,
    "retention_rate_p4":      None,

    "plan_deviation_threshold": 0.10,

    "schedule_cap_dist":      "fixed",
    "schedule_cap_p1":        3,
    "schedule_cap_p2":        None,
    "schedule_cap_p3":        None,
    "schedule_cap_p4":        None,

    "cost_cap_dist":          "fixed",
    "cost_cap_p1":            1.30,
    "cost_cap_p2":            None,
    "cost_cap_p3":            None,
    "cost_cap_p4":            None,

    "cure_length_dist":       "fixed",
    "cure_length_p1":         2,
    "cure_length_p2":         None,
    "cure_length_p3":         None,
    "cure_length_p4":         None,

    "efficiency_dist":        "uniform",
    "efficiency_p1":          0.75,
    "efficiency_p2":          1.25,
    "efficiency_p3":          None,
    "efficiency_p4":          None,

    # ── milestones ─────────────────────────────────────────────
    "n_milestones_dist":      "uniform",
    "n_milestones_p1":        3,
    "n_milestones_p2":        4,
    "n_milestones_p3":        None,
    "n_milestones_p4":        None,

    "threshold_dist":         "fixed",
    "threshold_p1":           0.25,
    "threshold_p2":           None,
    "threshold_p3":           None,
    "threshold_p4":           None,

    "payment_weight_dist":    "fixed",
    "payment_weight_p1":      0.25,
    "payment_weight_p2":      None,
    "payment_weight_p3":      None,
    "payment_weight_p4":      None,

    # fraction = 1.0 -> earliest_t aligns with on-plan completion date
    "earliest_t_fraction_dist": "fixed",
    "earliest_t_fraction_p1":   1.0,
    "earliest_t_fraction_p2":   None,
    "earliest_t_fraction_p3":   None,
    "earliest_t_fraction_p4":   None,

    "annual_interest_rate":   0.12,
}


def seed(conn: sqlite3.Connection) -> str:
    """
    Insert CFG-DUAL-001 into environment_config.
    Idempotent — skips if the row already exists.
    Returns the config_id.
    """
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

            initial_budget_dist, initial_budget_p1,
            initial_budget_p2, initial_budget_p3, initial_budget_p4,

            budget_tightness_dist, budget_tightness_p1,
            budget_tightness_p2, budget_tightness_p3, budget_tightness_p4,

            discount_dist, discount_p1,
            discount_p2, discount_p3, discount_p4,

            budget_dist, budget_p1,
            budget_p2, budget_p3, budget_p4,

            margin_dist, margin_p1,
            margin_p2, margin_p3, margin_p4,

            start_dist, start_p1,
            start_p2, start_p3, start_p4,

            duration_dist, duration_p1,
            duration_p2, duration_p3, duration_p4,

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

            plan_deviation_threshold,

            schedule_cap_dist, schedule_cap_p1,
            schedule_cap_p2, schedule_cap_p3, schedule_cap_p4,

            cost_cap_dist, cost_cap_p1,
            cost_cap_p2, cost_cap_p3, cost_cap_p4,

            cure_length_dist, cure_length_p1,
            cure_length_p2, cure_length_p3, cure_length_p4,

            efficiency_dist, efficiency_p1,
            efficiency_p2, efficiency_p3, efficiency_p4,

            n_milestones_dist, n_milestones_p1,
            n_milestones_p2, n_milestones_p3, n_milestones_p4,

            threshold_dist, threshold_p1,
            threshold_p2, threshold_p3, threshold_p4,

            payment_weight_dist, payment_weight_p1,
            payment_weight_p2, payment_weight_p3, payment_weight_p4,

            earliest_t_fraction_dist, earliest_t_fraction_p1,
            earliest_t_fraction_p2, earliest_t_fraction_p3, earliest_t_fraction_p4,

            annual_interest_rate
        )
        VALUES (
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?
        )
    """, (
        CONFIG_ID, "dual_project_baseline", datetime.utcnow().isoformat(),

        "fixed", 2,       None, None, None,
        "fixed", 400.0,   None, None, None,
        "fixed", 1.74,    None, None, None,
        "fixed", 0.97,    None, None, None,

        "uniform", 80.0,  150.0, None, None,
        "uniform", 0.12,  0.20,  None, None,
        "fixed", 0,       None,  None, None,
        "uniform", 10,    15,    None, None,

        "uniform", 2.0,   5.0,   None, None,
        "uniform", 2.0,   5.0,   None, None,
        "uniform", 0.08,  0.12,  None, None,
        "fixed", 0.20,    None,  None, None,

        "uniform", 0.15,  0.25,  None, None,
        "uniform", 0.03,  0.07,  None, None,

        0.10,   # plan_deviation_threshold

        "fixed", 3,       None, None, None,
        "fixed", 1.30,    None, None, None,
        "fixed", 2,       None, None, None,

        "uniform", 0.75,  1.25,  None, None,

        "uniform", 3,     4,    None, None,
        "fixed", 0.25,    None, None, None,
        "fixed", 0.25,    None, None, None,
        "fixed", 1.0,     None, None, None,

        0.12,   # annual_interest_rate
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