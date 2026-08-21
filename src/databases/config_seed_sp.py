# config_seed_sp.py

import sqlite3
from datetime import datetime
from db_init import init_db

DB_PATH = "database.db"


def seed_single_project(conn: sqlite3.Connection) -> str:
    config_id = "CFG-SINGLE-001"
    existing = conn.execute(
        "SELECT config_id FROM environment_config WHERE config_id = ?",
        (config_id,)
    ).fetchone()

    if existing:
        print(f"Config '{config_id}' already exists. Skipping.")
        return config_id

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

            'fixed', 1, NULL, NULL, NULL,

            'fixed', 200, NULL, NULL, NULL,

            'fixed', 1.0, NULL, NULL, NULL,

            'fixed', 0.97, NULL, NULL, NULL,

            'fixed', 100, NULL, NULL, NULL,

            'fixed', 0.15, NULL, NULL, NULL,

            'fixed', 0, NULL, NULL, NULL,

            'fixed', 12, NULL, NULL, NULL,

            'fixed', 2.0, NULL, NULL, NULL,

            'fixed', 5.0, NULL, NULL, NULL,

            'fixed', 0.10, NULL, NULL, NULL,

            'fixed', 0.20, NULL, NULL, NULL,

            'fixed', 0.20, NULL, NULL, NULL,

            'fixed', 0.05, NULL, NULL, NULL,

            -- plan deviation threshold
            0.10,

            'fixed', 3, NULL, NULL, NULL,

            'fixed', 1.30, NULL, NULL, NULL,

            'fixed', 2, NULL, NULL, NULL,

            'uniform', 0.8, 1.2, NULL, NULL,

            'fixed', 4, NULL, NULL, NULL,

            'fixed', 0.25, NULL, NULL, NULL,

            'fixed', 0.25, NULL, NULL, NULL,

            'fixed', 1.0, NULL, NULL, NULL,

            -- annual interest rate: 12% per year → monthly_rate = 0.01
            0.12
        )
    """, (config_id, "single_project_baseline", datetime.utcnow().isoformat()))

    conn.commit()
    print(f"Config '{config_id}' inserted.")
    print()
    print("  Portfolio  : 1 project")
    print("  Budget     : initial=200, project BAC=100, price=115 (margin 15%)")
    print("  Timeline   : start=0, duration=12, finish=12")
    print("  Advance    : 10% of price (=11.50) credited at t=0")
    print("  Milestones : 4 evenly spaced (25%/50%/75%/100%)")
    print("               weights 25% each; final payment at t=12")
    print("               earliest_t at 3, 6, 9, 12 (fraction=1.0 of duration)")
    print("               advance recovery 20% per milestone payment")
    print("               retention 5% held per milestone, released at completion")
    print("  Efficiency : uniform(0.8, 1.2) per period")
    print("  Discount   : 0.97 per period")
    return config_id


if __name__ == "__main__":
    conn = init_db()
    seed_single_project(conn)
    conn.close()