# src/env/db_logger.py
#
# SQLite audit ledger — side effect only.
#
# Design rules (from spec):
#   - Every write is wrapped in try/except — a DB failure must never crash
#     the environment or affect the step return values.
#   - conn.commit() is called every step unconditionally (Bug #2 fix).
#   - This module owns all SQL.  env.py calls the public functions;
#     it never touches self.conn directly for writes.
#   - The connection is passed in on each call (stateless module).

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
    Insert project and milestone profile rows for a new episode.
    Called once from env.reset() after sampling.
    """
    try:
        for proj in projects:
            conn.execute("""
                INSERT INTO projects_profile
                    (episode_id, config_id, i, budget, price, margin,
                     start, finish, duration, scurve_a, scurve_b,
                     advance_percent, advance_trigger, advance_recovery,
                     retention_rate, schedule_cap, plan_deviation_threshold,
                     cost_cap, cure_length)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode_id, config_id, proj["i"],
                proj["budget"], proj["price"], proj["margin"],
                proj["start"], proj["finish"], proj["duration"],
                proj["scurve_a"], proj["scurve_b"],
                proj["advance_percent"], proj["advance_trigger"],
                proj["advance_recovery"], proj["retention_rate"],
                proj["schedule_cap"], proj["plan_deviation_threshold"],
                proj["cost_cap"], proj["cure_length"],
            ))

        for i, ms_list in enumerate(milestones_list):
            for ms in ms_list:
                conn.execute("""
                    INSERT INTO milestones_profile
                        (episode_id, i, j, threshold, earliest_t, payment_weight)
                    VALUES (?,?,?,?,?,?)
                """, (
                    episode_id, i, ms["j"],
                    ms["threshold"], ms["earliest_t"], ms["payment_weight"],
                ))

        conn.commit()

    except Exception:
        pass  # never load-bearing


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ROW  (written every step)
# ─────────────────────────────────────────────────────────────────────────────

def write_portfolio_row(conn: sqlite3.Connection,
                        episode_id: str,
                        config_id: str,
                        t: int,
                        method: str,
                        budget: float,
                        inflow: float,
                        outflow: float,
                        reward: float,
                        done: bool) -> None:
    try:
        conn.execute("""
            INSERT INTO portfolios
                (episode_id, config_id, t_episode, method,
                 budget, inflow, outflow, reward, done)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            episode_id, config_id, t, method,
            budget, inflow, outflow, reward, int(done),
        ))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT STATUS ROW  (written every step, per project)
# ─────────────────────────────────────────────────────────────────────────────

def write_project_row(conn: sqlite3.Connection,
                      episode_id: str,
                      config_id: str,
                      t: int,
                      method: str,
                      i: int,
                      proj: dict,
                      ps: dict,
                      allocation: float,
                      efficiency,           # float | None
                      advance_amount: float,
                      payment_net,          # float | None
                      retention_release,    # float | None
                      settlement,           # float | None
                      interest_cost: float = 0.0,
                      treasury_draw: float = 0.0) -> None:
    """
    Insert one row into projects_status.

    Bug #1 fix: callers that hit the early-continue path (t < proj["start"]
    or already terminal) must pass interest_cost=0.0 and treasury_draw=0.0
    explicitly — the defaults guarantee no NameError even if the caller
    omits them.
    """
    try:
        conn.execute("""
            INSERT INTO projects_status
                (episode_id, i, t_episode, t_project, method, status,
                 allocation, efficiency,
                 progress, progress_plan, progress_increment,
                 spi, cpi, tcpi, eac,
                 schedule_slip, plan_deviation, cost_overrun, forecast_finish,
                 cure_remaining,
                 advance_amount, payment_net, retention_release, settlement,
                 interest_cost, treasury_draw)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            episode_id, i, t, ps.get("t_project"), method, ps.get("status"),
            allocation, efficiency,
            ps.get("progress", 0.0), ps.get("progress_plan", 0.0),
            ps.get("progress_increment", 0.0),
            ps.get("spi", 1.0), ps.get("cpi", 1.0),
            ps.get("tcpi", 1.0), ps.get("eac"),
            ps.get("schedule_slip", 0.0), ps.get("plan_deviation", 0.0),
            ps.get("cost_overrun", 0.0), ps.get("forecast_finish"),
            ps.get("cure_remaining"),
            advance_amount, payment_net, retention_release, settlement,
            interest_cost, treasury_draw,
        ))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE STATUS  (written on certification)
# ─────────────────────────────────────────────────────────────────────────────

def write_milestone_status(conn: sqlite3.Connection,
                           episode_id: str,
                           method: str,
                           i: int,
                           j: int,
                           ms: dict) -> None:
    try:
        conn.execute("""
            INSERT OR REPLACE INTO milestones_status
                (episode_id, i, j, method,
                 certified, certified_t, payment_released)
            VALUES (?,?,?,?,?,?,?)
        """, (
            episode_id, i, j, method,
            int(ms["certified"]), ms["certified_t"], ms["payment_released"],
        ))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# COMMIT  (unconditional, every step — Bug #2 fix)
# ─────────────────────────────────────────────────────────────────────────────

def commit(conn: sqlite3.Connection) -> None:
    """Commit the current transaction.  Swallows exceptions — never load-bearing."""
    try:
        conn.commit()
    except Exception:
        pass