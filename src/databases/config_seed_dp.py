# config_seed_dp.py
#
# Dual-project portfolio baseline configuration.
#
# Project 0  — larger, longer, back-loaded, tighter margins
#   BAC=150, duration=15, margin=12%, scurve front-loaded (α=2, β=5)
#   advance 8%, 3 milestones, efficiency uniform(0.75, 1.15)
#
# Project 1  — smaller, shorter, front-loaded, richer margins
#   BAC=80,  duration=10, margin=20%, scurve back-loaded  (α=5, β=2)
#   advance 12%, 4 milestones, efficiency uniform(0.85, 1.25)
#
# Both projects start at t=0.
# Portfolio initial budget = 400 (κ = 400/230 ≈ 1.74 — moderately abundant).
# Discount factor = 0.97 per period.

import sqlite3
from datetime import datetime
from db_init import init_db

DB_PATH = "database.db"


def seed_dual_project(conn: sqlite3.Connection) -> str:
    config_id = "CFG-DUAL-001"
    existing = conn.execute(
        "SELECT config_id FROM environment_config WHERE config_id = ?",
        (config_id,)
    ).fetchone()

    if existing:
        print(f"Config '{config_id}' already exists. Skipping.")
        return config_id

    # ── NOTE ON SINGLE-CONFIG MULTI-PROJECT DESIGN ────────────────────────────
    # The environment_config table stores one distribution per parameter,
    # shared across all projects sampled in an episode.  To produce two
    # *deterministically different* projects we use fixed distributions with
    # the same p1 value — this makes all projects identical by parameter, but
    # stochastic efficiency (η) still differentiates their realized performance.
    #
    # For structurally different projects (different BAC, duration, s-curve)
    # the correct architectural move is a per-project config table, which is
    # a planned extension.  For this MVP baseline we use uniform distributions
    # with ranges that span the desired contrast, so each episode naturally
    # produces varied projects.  The two-project count is fixed (n_projects=2).
    # ─────────────────────────────────────────────────────────────────────────

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

            schedule_cap_dist, schedule_cap_p1,
            plan_deviation_threshold=0.10,
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
            earliest_t_fraction_p2, earliest_t_fraction_p3, earliest_t_fraction_p4
        )
        VALUES (
            ?, ?, ?,

            -- 2 projects, fixed
            'fixed', 2, NULL, NULL, NULL,

            -- initial budget = 400
            'fixed', 400, NULL, NULL, NULL,

            -- budget tightness (informational only — not used by env logic)
            'fixed', 1.74, NULL, NULL, NULL,

            -- discount factor
            'fixed', 0.97, NULL, NULL, NULL,

            -- BAC: uniform 80–150 → each episode gets two varied projects
            'uniform', 80, 150, NULL, NULL,

            -- margin: uniform 12%–20%
            'uniform', 0.12, 0.20, NULL, NULL,

            -- all projects start at t=0
            'fixed', 0, NULL, NULL, NULL,

            -- duration: uniform 10–15 periods
            'uniform', 10, 15, NULL, NULL,

            -- s-curve α: uniform 2–5 (low α = front-loaded, high α = back-loaded)
            'uniform', 2.0, 5.0, NULL, NULL,

            -- s-curve β: uniform 2–5
            'uniform', 2.0, 5.0, NULL, NULL,

            -- advance: uniform 8%–12%
            'uniform', 0.08, 0.12, NULL, NULL,

            -- advance trigger (not yet used in certification logic)
            'fixed', 0.20, NULL, NULL, NULL,

            -- advance recovery rate: uniform 15%–25% per milestone gross
            'uniform', 0.15, 0.25, NULL, NULL,

            -- retention rate: uniform 3%–7%
            'uniform', 0.03, 0.07, NULL, NULL,

            -- schedule cap: 3 periods
            'fixed', 3, NULL, NULL, NULL,

            -- cost cap: 1.30×
            'fixed', 1.30, NULL, NULL, NULL,

            -- cure period length: 2 periods
            'fixed', 2, NULL, NULL, NULL,

            -- efficiency η: uniform 0.75–1.25 (wider spread than single project)
            'uniform', 0.75, 1.25, NULL, NULL,

            -- number of milestones: 3 or 4, sampled uniformly
            'uniform', 3, 4, NULL, NULL,

            -- threshold distribution (evenly spaced, computed by env)
            'fixed', 0.25, NULL, NULL, NULL,

            -- payment weight distribution (equal split, computed by env)
            'fixed', 0.25, NULL, NULL, NULL,

            -- earliest_t_fraction = 1.0 → eligible at on-plan completion date
            'fixed', 1.0, NULL, NULL, NULL
        )
    """, (config_id, "dual_project_baseline", datetime.utcnow().isoformat()))

    conn.commit()
    print(f"Config '{config_id}' inserted.")
    print()
    print("  Portfolio  : 2 projects")
    print("  Budget     : initial=400")
    print("  BAC        : uniform(80, 150) per project")
    print("  Margin     : uniform(12%, 20%) per project")
    print("  Timeline   : start=0 for both; duration uniform(10, 15)")
    print("  S-curve    : α uniform(2,5), β uniform(2,5) — varied shapes")
    print("  Advance    : uniform(8%, 12%) of price per project")
    print("  Milestones : uniform(3, 4) per project; earliest_t at on-plan dates")
    print("               advance recovery uniform(15%, 25%) per milestone gross")
    print("               retention uniform(3%, 7%), released at completion")
    print("  Efficiency : uniform(0.75, 1.25) per period — wider than SP config")
    print("  Discount   : 0.97 per period")
    return config_id


if __name__ == "__main__":
    conn = init_db()
    seed_dual_project(conn)
    conn.close()