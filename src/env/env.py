# src/env/env.py
#
# PortfolioBudgetingEnv — Gymnasium-compatible environment.
#
# Phase order per timestep (per active project):
#
#   ── EARLY PHASE (pre-allocation) ──────────────────────────────────────────
#   1. Advance delivery     if t == planned_start (j=0 milestone)
#   2. EVM update           based on state carried from previous period
#   3. Certification check  flag milestones that qualify (no payment yet)
#   4. Breach evaluation    pre-allocation flags (abandoned always False)
#   5. Tolerance update     decrement or reset
#   6. Termination check    fires on deadline or tolerance exhaustion
#   7. Completion check     progress_actual >= 1.0
#
#   GET OBSERVATION  ← snapshot here; agent acts on pre-allocation state
#
#   ── LATE PHASE (post-allocation) ──────────────────────────────────────────
#   8. Scale & validate action
#   9. Interest             treasury_draw computed BEFORE outflow update
#  10. Allocation & progress cumulative_cost and progress_actual updated
#  11. EVM update           recomputed on post-allocation state
#  12. Payment delivery     certified milestones paid; inflow updated
#  13. Breach evaluation    post-allocation (abandoned checked here)
#  14. Tolerance update
#  15. Termination check    may fire on under-allocation breach
#  16. Completion check     progress_actual >= 1.0 after allocation
#
#   RECORD
#  17. net_cashflow, budget, reward
#  18. DB write (try/except — never load-bearing)

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import db_logger
import evm as evm_mod
import payments as pay
import breaches as br
import render as rnd
from sampler import (
    sample, sample_int,
    sample_project, sample_milestones, sample_efficiency,
    planned_progress,
)


class PortfolioBudgetingEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    # ── INIT ──────────────────────────────────────────────────────────────────

    def __init__(self, config: dict,
                 render_mode: Optional[str] = None,
                 conn=None,
                 method: str = "rl"):
        super().__init__()

        self.config      = config
        self.render_mode = render_mode
        self.conn        = conn
        self.method      = method

        n = max(1, round(config.get("n_projects_p1", 1)))
        self._n_projects_hint = n
        self._declare_spaces(n)

        self.episode_id      : str             = ""
        self.t               : int             = 0
        self.budget          : float           = 0.0
        self.initial_budget  : float           = 0.0
        self.discount        : float           = 1.0
        self.horizon         : int             = 0
        self.projects        : list[dict]      = []
        self.milestones      : list[list]      = []   # profile (static)
        self.milestone_state : list[list]      = []   # runtime (per episode)
        self.proj_state      : list[dict]      = []

    def _declare_spaces(self, n: int) -> None:
        # Per-project feature bounds (8 features each)
        proj_low = np.tile([
            0.0,   # progress_actual
            0.0,   # progress_plan_t
        -1.0,   # progress_delay_t
            0.0,   # spi
            0.0,   # cpi
            0.0,   # projected_cost_overrun (EAC/BAC)
            0.0,   # catchup_norm
            0.0,   # tol_norm
        ], n)

        proj_high = np.tile([
            1.0,   # progress_actual
            1.0,   # progress_plan_t
            1.0,   # progress_delay_t
            3.0,   # spi  — clamp in _build_obs if needed
            3.0,   # cpi
            3.0,   # projected_cost_overrun
            1.0,   # catchup_norm
            1.0,   # tol_norm
        ], n)

        # Portfolio-level feature (1 feature)
        port_low  = np.array([0.0])   # budget_norm
        port_high = np.array([2.0])   # budget can grow beyond initial

        self.observation_space = spaces.Box(
            low  = np.concatenate([proj_low,  port_low]).astype(np.float32),
            high = np.concatenate([proj_high, port_high]).astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n,),
            dtype=np.float32,
        )

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        cfg             = self.config
        self.episode_id = str(uuid.uuid4())
        self.t          = 0

        self.discount = max(0.0, min(1.0, sample(
            self.np_random,
            cfg["discount_dist"], cfg["discount_p1"],
            cfg["discount_p2"],   cfg["discount_p3"], cfg["discount_p4"],
        )))

        n_projects = sample_int(
            self.np_random,
            cfg["n_projects_dist"], cfg["n_projects_p1"],
            cfg["n_projects_p2"],   cfg["n_projects_p3"], cfg["n_projects_p4"],
        )

        self.initial_budget = max(0.0, sample(
            self.np_random,
            cfg["budget_available_dist"], cfg["budget_available_p1"],
            cfg["budget_available_p2"],   cfg["budget_available_p3"],
            cfg["budget_available_p4"],
        ))
        self.budget = self.initial_budget

        if n_projects != self._n_projects_hint:
            self._declare_spaces(n_projects)
            self._n_projects_hint = n_projects

        # Sample projects, milestone profiles, and milestone runtime state
        self.projects        = []
        self.milestones      = []
        self.milestone_state = []
        for i in range(n_projects):
            proj = sample_project(self.np_random, cfg, i)
            ms_list = sample_milestones(self.np_random, cfg, proj)
            ms_state_list = [
                {"certified": False, "certified_t": None, "payment_released": 0.0}
                for _ in ms_list
            ]
            self.projects.append(proj)
            self.milestones.append(ms_list)
            self.milestone_state.append(ms_state_list)

        self.horizon = max(
            p["planned_finish"] + p["finish_delay_cap"]
            for p in self.projects
        )

        self.proj_state = [
            self._init_proj_state(proj) for proj in self.projects
        ]

        # Deliver advance payments for projects starting at t=0
        for ps, proj, milestones, ms_state_list in zip(
            self.proj_state, self.projects,
            self.milestones, self.milestone_state
        ):
            if proj["planned_start"] == 0:
                self._deliver_advance(ps, proj, milestones, ms_state_list, t=0)

        if self.conn is not None:
            try:
                db_logger.write_profiles(
                    self.conn, self.episode_id,
                    self.config.get("config_id", ""),
                    self.projects, self.milestones,
                )
            except Exception:
                pass

        rnd.reset_history(self.episode_id, len(self.projects))

        return self._build_obs(), self._build_info()

    # ── STEP ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, bool, dict]:

        n   = len(self.projects)
        cfg = self.config

        raw = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)

        active_mask = np.array(
            [ps["status"] == "active" for ps in self.proj_state], dtype=bool
        )
        active_sum = float(raw[active_mask].sum())
        if active_sum > 1.0 + 1e-9:
            raw = raw / active_sum

        allocations = raw * self.budget
        for i in range(n):
            if not active_mask[i]:
                allocations[i] = 0.0

        discount_factor = self.discount ** self.t
        period_inflow   = 0.0
        period_outflow  = 0.0

        proj_cf: list[dict] = [
            {
                "advance":       0.0,
                "milestone_net": 0.0,
                "settlement":    0.0,
                "allocation":    0.0,
                "interest":      0.0,
                "treasury_draw": 0.0,
            }
            for _ in self.projects
        ]

        # ══════════════════════════════════════════════════════════════════════
        # EARLY PHASE
        # ══════════════════════════════════════════════════════════════════════

        certified_js_by_proj: list[list[int]] = [[] for _ in self.projects]

        for i, (proj, ps, milestones, ms_state_list) in enumerate(zip(
            self.projects, self.proj_state,
            self.milestones, self.milestone_state
        )):
            if self.t < proj["planned_start"]:
                ps["status"] = "pending"
                continue

            if ps["status"] in ("completed", "terminated"):
                continue

            # 1. Advance delivery for projects starting this period (start > 0)
            if self.t == proj["planned_start"] and proj["planned_start"] > 0:
                advance = self._deliver_advance(
                    ps, proj, milestones, ms_state_list, t=self.t
                )
                period_inflow        += advance
                proj_cf[i]["advance"] = advance

            # 2. EVM update
            ps["progress_plan_t"] = planned_progress(
                ps["t_project"],
                proj["planned_duration"],
                proj["scurve_a"],
                proj["scurve_b"],
            )
            evm_mod.update_evm(ps, proj)

            # 3. Certification check
            certified_js_by_proj[i] = pay.check_certifications(
                self.t, ps, milestones, ms_state_list
            )

            # 4. Breach evaluation
            flags = br.evaluate_breaches(ps, proj, alloc=None)

            # 5. Tolerance update
            br.update_tolerance(ps, proj, flags)

            # 6. Termination check
            terminated, over_deadline = br.check_termination(
                self.t, ps, proj, flags
            )
            ps["breach_flags"] = flags

            if terminated:
                ps["status"] = "terminated"
                settlement   = pay.compute_termination_settlement(proj, ps)
                ps["termination_settlement"] = settlement
                ps["inflow"]               += max(0.0, settlement)
                period_inflow              += max(0.0, settlement)
                proj_cf[i]["settlement"]    = settlement
                continue

            # 7. Completion check
            if ps["progress_actual"] >= 1.0 and ps["status"] == "active":
                ps["status"] = "completed"
                continue

        # ── GET OBSERVATION ────────────────────────────────────────────────
        obs = self._build_obs()

        # ══════════════════════════════════════════════════════════════════════
        # LATE PHASE
        # ══════════════════════════════════════════════════════════════════════

        for i, (proj, ps, milestones, ms_state_list) in enumerate(zip(
            self.projects, self.proj_state,
            self.milestones, self.milestone_state
        )):
            if ps["status"] != "active":
                continue

            alloc = float(allocations[i])

            # 9. Interest
            monthly_rate  = cfg["annual_interest_rate"] / 12.0
            treasury_draw = max(0.0, ps["outflow"] - ps["inflow"])
            interest      = monthly_rate * treasury_draw

            proj_cf[i]["treasury_draw"] = treasury_draw
            proj_cf[i]["interest"]      = interest
            period_outflow             += interest

            # 10. Allocation & progress
            eta       = sample_efficiency(self.np_random, cfg)
            increment = (alloc / proj["bac"]) * eta if proj["bac"] > 0 else 0.0

            ps["progress_actual"]   = min(1.0, ps["progress_actual"] + increment)
            ps["outflow"]          += alloc + interest
            ps["efficiency"]        = eta
            ps["allocation_action"] = alloc

            period_outflow          += alloc
            proj_cf[i]["allocation"] = alloc

            ps["t_project"] += 1

            # 11. EVM update
            ps["progress_plan_t"] = planned_progress(
                ps["t_project"],
                proj["planned_duration"],
                proj["scurve_a"],
                proj["scurve_b"],
            )
            evm_mod.update_evm(ps, proj)

            # 12. Payment delivery
            ms_net = pay.deliver_payments(
                certified_js_by_proj[i], self.t,
                ps, milestones, ms_state_list
            )
            period_inflow               += ms_net
            proj_cf[i]["milestone_net"]  = ms_net

            # 13. Breach evaluation
            flags = br.evaluate_breaches(ps, proj, alloc=alloc)

            # 14. Tolerance update
            br.update_tolerance(ps, proj, flags)

            # 15. Termination check
            terminated, over_deadline = br.check_termination(
                self.t, ps, proj, flags
            )
            ps["breach_flags"] = flags

            if terminated:
                ps["status"] = "terminated"
                settlement   = pay.compute_termination_settlement(proj, ps)
                ps["termination_settlement"] = settlement
                ps["inflow"]               += max(0.0, settlement)
                period_inflow              += max(0.0, settlement)
                proj_cf[i]["settlement"]    = settlement
                continue

            # 16. Completion check
            if ps["progress_actual"] >= 1.0:
                ps["status"] = "completed"

        # ══════════════════════════════════════════════════════════════════════
        # RECORD
        # ══════════════════════════════════════════════════════════════════════

        net_cashflow   = period_inflow - period_outflow
        self.budget   += net_cashflow
        reward         = float(discount_factor * net_cashflow)

        all_terminal = all(
            ps["status"] in ("completed", "terminated")
            for ps in self.proj_state
        )
        terminated_ep = all_terminal
        truncated_ep  = self.t >= self.horizon

        if self.conn is not None:
            try:
                db_logger.write_step(
                    self.conn, self.episode_id,
                    self.config.get("config_id", ""),
                    self.t, self.method,
                    self.budget, net_cashflow, reward,
                    terminated_ep or truncated_ep,
                    self.projects, self.proj_state,
                    self.milestones, self.milestone_state,
                    proj_cf,
                )
                db_logger.commit(self.conn)
            except Exception:
                pass

        self.t += 1

        info = self._build_info()
        info["cashflow"]      = proj_cf
        info["period_inflow"] = period_inflow
        info["period_outflow"]= period_outflow

        return obs, reward, terminated_ep, truncated_ep, info

    # ── GYMNASIUM API ──────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        rnd.render(
            t              = self.t,
            budget         = self.budget,
            horizon        = self.horizon,
            initial_budget = self.initial_budget,
            projects       = self.projects,
            proj_state     = self.proj_state,
            milestones     = self.milestones,
        )
        return None

    def close(self) -> None:
        pass

    # ── OBSERVATION ────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        obs: list[float] = []

        for proj, ps in zip(self.projects, self.proj_state):
            bac          = proj["bac"]
            tol_norm     = (
                ps["tolerance_remain"] / proj["termination_tolerance"]
                if proj["termination_tolerance"] > 0 else 1.0
            )
            catchup_norm = (
                ps.get("catchup_alloc_t", 0.0) / bac
                if bac > 0 else 0.0
            )
            obs.extend([
                np.clip(ps["progress_actual"],          0.0, 1.0),
                np.clip(ps["progress_plan_t"],          0.0, 1.0),
                np.clip(ps["progress_delay_t"],        -1.0, 1.0),
                np.clip(ps["spi"],                      0.0, 3.0),
                np.clip(ps["cpi"],                      0.0, 3.0),
                np.clip(ps["projected_cost_overrun"],   0.0, 3.0),
                np.clip(catchup_norm,                   0.0, 1.0),
                np.clip(tol_norm,                       0.0, 1.0),
            ])

        budget_norm = (
            self.budget / self.initial_budget
            if self.initial_budget > 0 else 1.0
        )
        obs.append(np.clip(budget_norm, 0.0, 2.0))

        return np.array(obs, dtype=np.float32)

    # ── INFO ───────────────────────────────────────────────────────────────────

    def _build_info(self) -> dict:
        return {
            "t_episode": self.t,
            "budget":    self.budget,
            "horizon":   self.horizon,
            "projects": [
                {
                    "i":                      proj["i"],
                    "status":                 ps["status"],
                    "bac":                    proj["bac"],
                    "planned_start":          proj["planned_start"],
                    "planned_finish":         proj["planned_finish"],
                    "progress_actual":        ps["progress_actual"],
                    "progress_plan_t":        ps["progress_plan_t"],
                    "progress_delay_t":       ps["progress_delay_t"],
                    "spi":                    ps["spi"],
                    "cpi":                    ps["cpi"],
                    "tcpi":                   ps["tcpi"],
                    "eac":                    ps["eac"],
                    "projected_finish":       ps["projected_finish"],
                    "projected_finish_delay": ps["projected_finish_delay"],
                    "projected_cost_overrun": ps["projected_cost_overrun"],
                    "tolerance_remain":       ps["tolerance_remain"],
                    "t_project":              ps["t_project"],
                    "inflow":                 ps["inflow"],
                    "outflow":                ps["outflow"],
                }
                for proj, ps in zip(self.projects, self.proj_state)
            ],
        }

    # ── ADVANCE DELIVERY HELPER ────────────────────────────────────────────────

    def _deliver_advance(self, ps: dict, proj: dict,
                         milestones: list[dict],
                         milestone_state: list[dict],
                         t: int) -> float:
        """
        Deliver the advance payment (j=0) for a project starting at *t*.
        Marks the advance milestone runtime state as certified.
        Updates ps["inflow"] and ps["status"].
        Returns the advance amount delivered.
        """
        ms_state = milestone_state[0]   # j=0 is always first

        if ms_state["certified"]:
            return 0.0                  # already delivered (safety guard)

        amount = milestones[0]["net_payment"]   # read from profile

        ms_state["certified"]        = True
        ms_state["certified_t"]      = t
        ms_state["payment_released"] = amount

        ps["status"]  = "active"
        ps["inflow"] += amount

        return amount

    # ── PROJECT STATE INITIALISATION ───────────────────────────────────────────

    @staticmethod
    def _init_proj_state(proj: dict) -> dict:
        return {
            "status":                   "pending",
            "t_project":                0,
            "inflow":                   0.0,
            "outflow":                  0.0,
            "termination_settlement":   0.0,
            "allocation_action":        0.0,
            "efficiency":               1.0,
            "progress_actual":          0.0,
            "progress_plan_t":          0.0,
            "progress_delay_t":         0.0,
            "catchup_alloc_t":          0.0,
            "catchup_alloc_next_t":     0.0,
            "spi":                      1.0,
            "cpi":                      1.0,
            "tcpi":                     1.0,
            "eac":                      proj["bac"],
            "projected_finish":         float(proj["planned_finish"]),
            "projected_finish_delay":   0.0,
            "projected_cost_overrun":   1.0,
            "breach_flags": {
                "abandoned":            False,
                "over_progress_delay":  False,
                "over_finish_delay":    False,
                "over_cost_overrun":    False,
                "over_duration_window": False,
                "over_any":             False,
            },
            "tolerance_remain":         proj["termination_tolerance"],
        }