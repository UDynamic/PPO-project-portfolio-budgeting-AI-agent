# src/env/env.py
#
# PortfolioBudgetingEnv — Gymnasium-compatible environment.
#
# Bugs fixed (all six from spec):
#
#   #1  Early-continue crash: _write_project_row called with undefined
#       interest_cost / treasury_draw when t < proj["start"].
#       Fix: db_logger.write_project_row defaults to 0.0, 0.0.
#
#   #2  DB commit only on done.
#       Fix: db_logger.commit() called unconditionally at end of every step.
#
#   #3  Double termination: cure and deadline both fire same period.
#       Fix: deadline branch is elif, not if.
#
#   #4  Retention release same period as completion.
#       Fix: completion sets _release_retention_next_period = True;
#            payments.release_retention() fires in Phase 1c next period.
#
#   #5  Interest base included current allocation.
#       Fix: treasury_draw computed from ps["cumulative_cost"] BEFORE
#            alloc is added (Phase 2 precedes Phase 3).
#
#   #6  advance_trigger sampled but never used.
#       Fix: sampled and stored in project profile for schema completeness;
#            explicitly documented as not-yet-wired (planned extension).
#            No silent omission, no silent use.

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import db_logger
import evm as evm_mod
import payments as pay
import render as rnd
from sampler import (
    sample, sample_int,
    sample_project, sample_milestones, sample_efficiency,
    planned_progress,
)


class PortfolioBudgetingEnv(gym.Env):
    """
    Portfolio budget allocation under cash-flow uncertainty.

    Observation space : Box(shape=(8n + 1,), dtype=float32)
        Per project (8 signals, in order):
            progress, progress_plan, plan_deviation,
            spi, cpi, eac/BAC, schedule_slip, cure_remaining
        Portfolio (1):
            budget / initial_budget

    Action space : Box(low=0, high=1, shape=(n,), dtype=float32)
        Allocation fractions.  Scaled to actual budget inside env.
        Clipped and rescaled if sum > 1.

    Reward : discount^t × (total_inflow_t − total_outflow_t)

    Termination  : all projects completed or terminated
    Truncation   : t >= horizon
    """

    metadata = {"render_modes": ["ansi"]}

    # ── INIT ─────────────────────────────────────────────────

    def __init__(self, config: dict, render_mode: Optional[str] = None,
                 conn=None, method: str = "rl"):
        """
        Parameters
        ----------
        config      : plain dict with all environment_config fields
                      (loaded once by the caller; never re-read from DB)
        render_mode : "ansi" or None
        conn        : optional sqlite3.Connection for audit logging
                      (None → logging silently skipped)
        method      : label written to DB rows ("rl", "manual", "milp", …)
        """
        super().__init__()

        self.config      = config
        self.render_mode = render_mode
        self.conn        = conn
        self.method      = method

        # Derive n from config so spaces can be declared before reset()
        n = max(1, round(config.get("n_projects_p1", 1)))
        self._n_projects_hint = n
        self._declare_spaces(n)

        # Episode state — populated by reset()
        self.episode_id    : str         = ""
        self.t             : int         = 0
        self.budget        : float       = 0.0
        self.initial_budget: float       = 0.0
        self.discount      : float       = 1.0
        self.horizon       : int         = 0
        self.projects      : list[dict]  = []
        self.milestones    : list[list]  = []
        self.proj_state    : list[dict]  = []

    def _declare_spaces(self, n: int) -> None:
        """Set observation_space and action_space for n projects."""
        obs_dim = 8 * n + 1
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape= (obs_dim,),
            dtype= np.float32,
        )
        self.action_space = spaces.Box(
            low  = 0.0,
            high = 1.0,
            shape= (n,),
            dtype= np.float32,
        )

    # ── RESET ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)   # sets self.np_random

        cfg = self.config
        self.episode_id = str(uuid.uuid4())
        self.t          = 0

        self.discount = max(0.0, min(1.0, sample(
            self.np_random,
            cfg["discount_dist"], cfg["discount_p1"],
            cfg["discount_p2"], cfg["discount_p3"], cfg["discount_p4"],
        )))

        n_projects = sample_int(
            self.np_random,
            cfg["n_projects_dist"], cfg["n_projects_p1"],
            cfg["n_projects_p2"], cfg["n_projects_p3"], cfg["n_projects_p4"],
        )

        self.initial_budget = max(0.0, sample(
            self.np_random,
            cfg["initial_budget_dist"], cfg["initial_budget_p1"],
            cfg["initial_budget_p2"], cfg["initial_budget_p3"], cfg["initial_budget_p4"],
        ))
        self.budget = self.initial_budget

        # Re-declare spaces if n changed (e.g. between episodes with uniform n)
        if n_projects != self._n_projects_hint:
            self._declare_spaces(n_projects)
            self._n_projects_hint = n_projects

        self.projects   = []
        self.milestones = []

        for i in range(n_projects):
            proj = sample_project(self.np_random, cfg, i)
            self.projects.append(proj)
            self.milestones.append(sample_milestones(self.np_random, cfg, proj))

        self.horizon = max(
            p["finish"] + p["schedule_cap"] for p in self.projects
        )

        self.proj_state = [
            self._init_proj_state(p) for p in self.projects
        ]

        # t=0 advance: projects that start at period 0 receive their advance
        # before the first step.  This is credited directly to budget and
        # tracked in advance_received; it is NOT a step cash-flow event.
        for ps, proj in zip(self.proj_state, self.projects):
            if proj["start"] == 0:
                ps["status"]    = "active"
                ps["t_project"] = 1
                ps["progress_plan"] = planned_progress(
                    1, proj["duration"], proj["scurve_a"], proj["scurve_b"]
                )
                advance              = pay.compute_advance(proj)
                self.budget         += advance
                ps["advance_received"] = advance

        # DB profiles
        if self.conn is not None:
            db_logger.write_profiles(
                self.conn, self.episode_id, self.config.get("config_id", ""),
                self.projects, self.milestones,
            )

        # Render history
        rnd.reset_history(self.episode_id, len(self.projects))

        return self._get_obs(), self._get_info()

    # ── STEP ─────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one period.  Fixed phase order — do not reorder.

        Returns (obs, reward, terminated, truncated, info)
        """
        n = len(self.projects)

        # ── scale action fractions to budget ─────────────────
        raw = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)

        # Only active projects consume budget
        active_mask = np.array([
            ps["status"] == "active" for ps in self.proj_state
        ], dtype=bool)

        active_sum = float(raw[active_mask].sum())
        if active_sum > 1.0 + 1e-9:
            raw = raw / active_sum          # rescale fractions to sum <= 1
        allocations = raw * self.budget     # fractions → amounts

        # Zero out non-active projects
        for i in range(n):
            if not active_mask[i]:
                allocations[i] = 0.0

        discount_factor = self.discount ** self.t
        total_inflow    = 0.0
        total_outflow   = 0.0

        # Per-project cashflow accumulator (used by info dict and render)
        proj_cashflow: list[dict] = [
            {
                "i":                 proj["i"],
                "allocation":        0.0,
                "advance":           0.0,
                "milestone_gross":   0.0,
                "milestone_net":     0.0,
                "retention_release": 0.0,
                "settlement":        0.0,
                "interest_cost":     0.0,
                "treasury_draw":     0.0,
            }
            for proj in self.projects
        ]

        eta_by_proj: list[float | None] = [None] * n

        for i, (proj, ps) in enumerate(zip(self.projects, self.proj_state)):

            alloc = float(allocations[i])

            # ── not yet started ───────────────────────────────
            if self.t < proj["start"]:
                ps["status"] = None
                # Bug #1 fix: interest_cost=0.0, treasury_draw=0.0 explicit
                if self.conn is not None:
                    db_logger.write_project_row(
                        self.conn, self.episode_id,
                        self.config.get("config_id", ""), self.t,
                        self.method, i, proj, ps,
                        0.0, None, 0.0, None, None, None,
                        interest_cost=0.0, treasury_draw=0.0,
                    )
                continue

            # ── already completed or terminated ───────────────
            if ps["status"] in ("completed", "terminated"):
                if self.conn is not None:
                    db_logger.write_project_row(
                        self.conn, self.episode_id,
                        self.config.get("config_id", ""), self.t,
                        self.method, i, proj, ps,
                        0.0, None, 0.0, None, None, None,
                        interest_cost=0.0, treasury_draw=0.0,
                    )
                continue

            # ═════════════════════════════════════════════════
            # PHASE 1 — INFLOWS
            # ═════════════════════════════════════════════════

            advance_amount    = 0.0
            payment_net       = 0.0
            retention_release = 0.0

            # 1a. Advance on project start (start > 0 only; start==0 handled
            #     at reset time)
            if self.t == proj["start"] and proj["start"] > 0:
                ps["status"]           = "active"
                ps["t_project"]        = 1
                advance_amount         = pay.compute_advance(proj)
                ps["advance_received"] = advance_amount
                total_inflow          += advance_amount
                proj_cashflow[i]["advance"] = advance_amount

            # 1b. Milestone certification
            ms_net, ms_gross = pay.process_milestones(
                self.t, proj, ps, self.milestones[i]
            )
            payment_net  = ms_net
            total_inflow += ms_net
            proj_cashflow[i]["milestone_net"]   = ms_net
            proj_cashflow[i]["milestone_gross"] = ms_gross

            # Write milestone status rows for newly certified milestones
            if self.conn is not None:
                for j, ms in enumerate(self.milestones[i]):
                    if ms["certified"] and ms["certified_t"] == self.t:
                        db_logger.write_milestone_status(
                            self.conn, self.episode_id,
                            self.method, i, j, ms,
                        )

            # 1c. Retention release — fires if project completed last period
            #     Bug #4 fix: flag set in Phase 5; release fires here next period
            if ps.get("_release_retention_next_period", False):
                ret = pay.release_retention(ps)
                if ret > 0.0:
                    retention_release                  = ret
                    total_inflow                      += ret
                    proj_cashflow[i]["retention_release"] = ret

            # ═════════════════════════════════════════════════
            # PHASE 2 — INTEREST
            # Bug #5 fix: treasury_draw uses cumulative_cost BEFORE alloc
            # ═════════════════════════════════════════════════

            monthly_rate       = self.config["annual_interest_rate"] / 12.0
            cumulative_inflows = ps["advance_received"] + ps["milestone_inflows"]
            treasury_draw      = max(0.0, ps["cumulative_cost"] - cumulative_inflows)
            interest_cost      = monthly_rate * treasury_draw

            ps["treasury_draw"] = treasury_draw
            proj_cashflow[i]["treasury_draw"] = treasury_draw
            proj_cashflow[i]["interest_cost"] = interest_cost
            total_outflow += interest_cost

            # ═════════════════════════════════════════════════
            # PHASE 3 — ALLOCATION & PROGRESS
            # ═════════════════════════════════════════════════

            ps["t_project"] = self.t - proj["start"] + 1

            eta = sample_efficiency(self.np_random, self.config)
            eta_by_proj[i] = eta

            increment       = (alloc / proj["budget"]) * eta if proj["budget"] > 0 else 0.0
            ps["progress"]  = min(1.0, ps["progress"] + increment)
            ps["progress_increment"] = increment

            ps["cumulative_cost"] += alloc
            ps["acwp"]            += alloc + interest_cost   # ACWP includes financing

            total_outflow                   += alloc
            proj_cashflow[i]["allocation"]   = alloc

            # ═════════════════════════════════════════════════
            # PHASE 4 — EVM UPDATE
            # ═════════════════════════════════════════════════

            ps["progress_plan"] = planned_progress(
                ps["t_project"], proj["duration"],
                proj["scurve_a"], proj["scurve_b"],
            )
            evm_mod.update_evm(ps, proj)   # mutates ps in-place

            # ═════════════════════════════════════════════════
            # PHASE 5 — TERMINATION CHECK
            # ═════════════════════════════════════════════════

            settlement = None

            # Completion: progress reached 1.0 this period
            if ps["progress"] >= 1.0 and ps["status"] == "active":
                ps["status"] = "completed"
                # Bug #4 fix: retention releases NEXT period
                ps["_release_retention_next_period"] = True

            # Breach evaluation (active projects only)
            if ps["status"] == "active":
                idle_breach      = alloc < 1e-9
                deviation_breach = ps["plan_deviation"] > self.config["plan_deviation_threshold"]
                schedule_breach  = ps["schedule_slip"]  > proj["schedule_cap"]
                cost_breach      = ps["eac"]             > proj["cost_cap"] * proj["budget"]
                deadline_breach  = self.t >= proj["finish"] + proj["schedule_cap"]

                any_breach = (
                    idle_breach
                    or deviation_breach
                    or (schedule_breach and cost_breach)
                    or deadline_breach
                )

                if any_breach:
                    ps["cure_remaining"] -= 1
                else:
                    ps["cure_remaining"] = proj["cure_length"]

                # Bug #3 fix: elif — cure exhaustion and deadline are mutually
                # exclusive branches; both cannot fire in the same period.
                if ps["cure_remaining"] <= 0:
                    ps["status"] = "terminated"
                    settlement   = pay.termination_settlement(proj, ps)
                    total_inflow += settlement
                    proj_cashflow[i]["settlement"] = settlement
                elif deadline_breach:
                    ps["status"] = "terminated"
                    settlement   = pay.termination_settlement(proj, ps)
                    total_inflow += settlement
                    proj_cashflow[i]["settlement"] = settlement

            # ═════════════════════════════════════════════════
            # PHASE 6 — RECORD
            # ═════════════════════════════════════════════════

            if self.conn is not None:
                db_logger.write_project_row(
                    self.conn, self.episode_id,
                    self.config.get("config_id", ""), self.t,
                    self.method, i, proj, ps,
                    alloc, eta,
                    advance_amount,
                    payment_net if payment_net else None,
                    retention_release if retention_release else None,
                    settlement,
                    interest_cost=interest_cost,
                    treasury_draw=treasury_draw,
                )

        # ── portfolio close ───────────────────────────────────

        self.budget  = self.budget - total_outflow + total_inflow
        net_cashflow = total_inflow - total_outflow
        reward       = float(discount_factor * net_cashflow)

        all_terminal = all(
            ps["status"] in ("completed", "terminated")
            for ps in self.proj_state
        )
        budget_exhausted = self.budget <= 0.0 and any(
            ps["status"] == "active" for ps in self.proj_state
        )

        terminated = all_terminal or budget_exhausted
        truncated  = self.t >= self.horizon

        if self.conn is not None:
            db_logger.write_portfolio_row(
                self.conn, self.episode_id,
                self.config.get("config_id", ""), self.t,
                self.method, self.budget,
                total_inflow, total_outflow, reward,
                terminated or truncated,
            )
            # Bug #2 fix: commit every step unconditionally
            db_logger.commit(self.conn)

        self.t += 1

        # Pre-compute next-period planned progress for active projects
        # (used by render and _get_obs at the start of next step)
        for ps, proj in zip(self.proj_state, self.projects):
            if ps["status"] == "active":
                next_tp = self.t - proj["start"] + 1
                if next_tp <= proj["duration"]:
                    ps["progress_plan"] = planned_progress(
                        next_tp, proj["duration"],
                        proj["scurve_a"], proj["scurve_b"],
                    )

        info = {
            "cashflow":          proj_cashflow,
            "portfolio_inflow":  total_inflow,
            "portfolio_outflow": total_outflow,
            "_reward":           reward,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── GYMNASIUM API ─────────────────────────────────────────

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None

        cf_by_proj = {i: {} for i in range(len(self.projects))}

        rnd.print_portfolio_summary(
            t            = self.t,
            budget       = self.budget,
            horizon      = self.horizon,
            total_reward = 0.0,   # cumulative reward tracked externally
        )
        rnd.render_all_projects(
            episode_id       = self.episode_id,
            projects         = self.projects,
            proj_states      = self.proj_state,
            milestones_list  = self.milestones,
            cfg              = self.config,
            cashflow_by_proj = cf_by_proj,
            t                = self.t,
        )
        return None   # ANSI printed to stdout; nothing to return

    def close(self) -> None:
        pass

    # ── OBSERVATION & INFO ────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat float32 observation vector.

        State space: 8n + 1
        Per project (8): progress, progress_plan, plan_deviation,
                         spi, cpi, eac/BAC, schedule_slip, cure_remaining
        Portfolio  (1):  budget / initial_budget
        """
        obs: list[float] = []
        for proj, ps in zip(self.projects, self.proj_state):
            bac     = proj["budget"]
            eac_bac = ps["eac"] / bac if bac > 0 else 1.0
            cure_norm = float(ps["cure_remaining"]) / max(1, proj["cure_length"])
            obs.extend([
                ps["progress"],
                ps["progress_plan"],
                ps["plan_deviation"],
                ps["spi"],
                ps["cpi"],
                eac_bac,
                ps["schedule_slip"],
                cure_norm,
            ])
        budget_norm = self.budget / self.initial_budget if self.initial_budget > 0 else 1.0
        obs.append(budget_norm)
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        """
        Return a rich state dict compatible with the old play.py _get_state()
        format, plus the gym-standard info keys.
        """
        projects_info = []
        for i, (proj, ps) in enumerate(zip(self.projects, self.proj_state)):
            ms_obs     = self._next_milestone_obs(i)
            ms_history = []
            for ms in self.milestones[i]:
                gross     = ms["payment_weight"] * proj["price"]
                recovery  = gross * proj["advance_recovery"]
                retention = gross * proj["retention_rate"]
                net       = gross - recovery - retention
                ms_history.append({
                    "j":           ms["j"],
                    "threshold":   ms["threshold"],
                    "earliest_t":  ms["earliest_t"],
                    "certified":   ms["certified"],
                    "certified_t": ms["certified_t"],
                    "gross":       gross,
                    "net":         net,
                    "is_final":    ms["threshold"] == 1.0,
                })
            projects_info.append({
                "i":                     proj["i"],
                "status":                ps["status"],
                "ms_history":            ms_history,
                "budget":                proj["budget"],
                "start":                 proj["start"],
                "finish":                proj["finish"],
                "progress":              ps["progress"],
                "progress_plan":         ps["progress_plan"],
                "plan_deviation":        ps["plan_deviation"],
                "spi":                   ps["spi"],
                "cpi":                   ps["cpi"],
                "tcpi":                  ps["tcpi"],
                "eac":                   ps["eac"],
                "forecast_finish":       ps["forecast_finish"],
                "schedule_slip":         ps["schedule_slip"],
                "cure_remaining":        ps["cure_remaining"],
                "t_project":             ps["t_project"],
                **ms_obs,
            })
        return {
            "t_episode":  self.t,
            "budget":     self.budget,
            "horizon":    self.horizon,
            "projects":   projects_info,
        }

    def _next_milestone_obs(self, i: int) -> dict:
        proj           = self.projects[i]
        ps             = self.proj_state[i]
        retention_held = ps["retention_held"]

        next_ms = next(
            (ms for ms in self.milestones[i] if not ms["certified"]),
            None,
        )

        if next_ms is None:
            return {
                "next_ms_threshold_gap": 0.0,
                "next_ms_net_payment":   0.0,
                "next_ms_earliest_t":    None,
                "next_ms_is_final":      False,
                "retention_held":        retention_held,
            }

        gross         = next_ms["payment_weight"] * proj["price"]
        recovery      = gross * proj["advance_recovery"]
        retention     = gross * proj["retention_rate"]
        net           = gross - recovery - retention
        threshold_gap = max(0.0, next_ms["threshold"] - ps["progress"])
        is_final      = next_ms["threshold"] == 1.0

        return {
            "next_ms_threshold_gap": threshold_gap,
            "next_ms_net_payment":   net,
            "next_ms_earliest_t":    next_ms["earliest_t"],
            "next_ms_is_final":      is_final,
            "retention_held":        retention_held,
        }

    # ── PROJECT INITIALISATION ────────────────────────────────

    @staticmethod
    def _init_proj_state(proj: dict) -> dict:
        return {
            "status":                        None,
            "t_project":                     None,
            "progress":                      0.0,
            "progress_plan":                 0.0,
            "plan_deviation":                0.0,
            "progress_increment":            0.0,
            "acwp":                          0.0,
            "spi":                           1.0,
            "cpi":                           1.0,
            "tcpi":                          1.0,
            "eac":                           proj["budget"],
            "schedule_slip":                 0.0,
            "cost_overrun":                  0.0,
            "forecast_finish":               float(proj["finish"]),
            "cure_remaining":                proj["cure_length"],
            "advance_received":              0.0,
            "advance_recovered":             0.0,
            "milestone_inflows":             0.0,
            "retention_held":                0.0,
            "retention_released":            False,
            "_release_retention_next_period":False,
            "cumulative_cost":               0.0,
            "treasury_draw":                 0.0,
        }