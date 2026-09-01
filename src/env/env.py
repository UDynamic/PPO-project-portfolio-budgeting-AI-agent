# src/env/env.py
#
# PortfolioBudgetingEnv — Gymnasium-compatible environment.
#
# Observation vector (matches projects_observation + portfolio_observation):
#
#   Per project (7 features each):
#     tolerance_remain_norm    tolerance_remain / termination_tolerance
#     catchup_alloc_t          allocation needed this period to avoid breach
#     catchup_alloc_next_t     allocation needed next period to avoid breach
#     target_progress_gap      next_ms.progress_threshold - progress_actual
#                              (clamped to 0 when progress past threshold;
#                               milestone may still be pending on time gate)
#     target_timestep_gap      next_ms.timestep_threshold - t_episode
#     target_required_alloc    target_progress_gap * bac
#     target_payment_rate      next_ms.net_payment / target_required_alloc
#
#   Portfolio (2 features):
#     net_cashflow             period inflow - outflow across all projects
#     budget_available         current budget after net_cashflow applied
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
#   8. Obs fields update    computed on post-EVM pre-allocation state
#
#   GET OBSERVATION  ← snapshot here
#
#   ── LATE PHASE (post-allocation) ──────────────────────────────────────────
#   9.  Scale & validate action
#   10. Interest
#   11. Allocation & progress
#   12. EVM update (post-allocation)
#   13. Payment delivery
#   14. Breach evaluation (post-allocation)
#   15. Tolerance update
#   16. Termination check
#   17. Completion check
#   18. Obs fields update   always runs — even on termination/completion
#
#   RECORD
#   19. net_cashflow, budget, reward
#   20. DB write

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

        self.episode_id         : str        = ""
        self.t                  : int        = 0
        self.budget             : float      = 0.0
        self.initial_budget     : float      = 0.0
        self.discount           : float      = 1.0
        self.horizon            : int        = 0
        self.projects           : list[dict] = []
        self.milestones         : list[list] = []
        self.milestone_state    : list[list] = []
        self.proj_state         : list[dict] = []
        self._last_net_cashflow : float      = 0.0

    def _declare_spaces(self, n: int) -> None:
        proj_low = np.tile([
            0.0,    # tolerance_norm
            0.0,    # catchup_alloc_t_norm
            0.0,    # catchup_alloc_next_t_norm
            0.0,    # target_progress_gap        (always >= 0)
           -1.0,    # target_timestep_gap_norm   (negative = overdue)
            0.0,    # target_required_alloc_norm
            0.0,    # target_payment_rate
        ], n)

        proj_high = np.tile([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            5.0,    # payment rate clipped at 5
        ], n)

        port_low  = np.array([-2.0, 0.0])
        port_high = np.array([ 2.0, 2.0])

        self.observation_space = spaces.Box(
            low  = np.concatenate([proj_low,  port_low]).astype(np.float32),
            high = np.concatenate([proj_high, port_high]).astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32,
        )

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        cfg             = self.config
        self.episode_id = str(uuid.uuid4())
        self.t          = 0
        self._last_net_cashflow = 0.0

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

        self.projects        = []
        self.milestones      = []
        self.milestone_state = []
        for i in range(n_projects):
            proj = sample_project(self.np_random, cfg, i)
            ms_list = sample_milestones(self.np_random, cfg, proj)
            ms_state_list = [
                {"certified": False,
                 "certified_t": None,
                 "payment_released": 0.0}
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

        # Initial EVM + obs fields
        for ps, proj, milestones, ms_state_list in zip(
            self.proj_state, self.projects,
            self.milestones, self.milestone_state
        ):
            ps["progress_plan_t"] = planned_progress(
                ps["t_project"],
                proj["planned_duration"],
                proj["scurve_a"],
                proj["scurve_b"],
            )
            evm_mod.update_evm(ps, proj)
            self._update_obs_fields(ps, proj, milestones, ms_state_list)

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
        # Termination fires here ONLY for over_duration_window (hard deadline).
        # All other breach conditions are evaluated and tolerance is updated
        # so the agent sees the degraded state in the observation, but
        # termination for those conditions is deferred to the late phase
        # after the agent has had a chance to allocate.
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

            # 1. Advance delivery for projects starting this period
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

            # 4. Breach evaluation (pre-allocation; abandoned always False)
            flags = br.evaluate_breaches(ps, proj, alloc=None)

            # 5. Hard deadline check — only over_duration_window terminates
            #    here. Tolerance is NOT updated in early phase — the agent
            #    must be given the chance to allocate before tolerance is
            #    decremented and termination can fire on exhaustion.

            #    here. All other breach conditions wait for late phase.
            over_duration_window = (
                self.t >= proj["planned_finish"] + proj["finish_delay_cap"]
            )
            flags["over_duration_window"] = over_duration_window
            ps["breach_flags"] = flags

            if over_duration_window:
                ps["status"] = "terminated"
                settlement   = pay.compute_termination_settlement(proj, ps)
                ps["termination_settlement"] = settlement
                ps["inflow"]               += max(0.0, settlement)
                period_inflow              += max(0.0, settlement)
                proj_cf[i]["settlement"]    = settlement
                self._update_obs_fields(ps, proj, milestones, ms_state_list)
                continue

            # 7. Completion check
            if ps["progress_actual"] >= 1.0:
                ps["status"] = "completed"
                self._update_obs_fields(ps, proj, milestones, ms_state_list)
                continue

            # 8. Obs fields — computed after EVM update, before allocation
            self._update_obs_fields(ps, proj, milestones, ms_state_list)

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

            # 10. Interest
            monthly_rate  = cfg["annual_interest_rate"] / 12.0
            treasury_draw = max(0.0, ps["outflow"] - ps["inflow"])
            interest      = monthly_rate * treasury_draw

            proj_cf[i]["treasury_draw"] = treasury_draw
            proj_cf[i]["interest"]      = interest
            period_outflow             += interest

            # 11. Allocation & progress
            eta       = sample_efficiency(self.np_random, cfg)
            increment = (alloc / proj["bac"]) * eta if proj["bac"] > 0 else 0.0

            ps["progress_actual"]   = min(1.0, ps["progress_actual"] + increment)
            ps["outflow"]          += alloc + interest
            ps["efficiency"]        = eta
            ps["allocation_action"] = alloc

            period_outflow          += alloc
            proj_cf[i]["allocation"] = alloc

            ps["t_project"] += 1

            # 12. EVM update (post-allocation)
            ps["progress_plan_t"] = planned_progress(
                ps["t_project"],
                proj["planned_duration"],
                proj["scurve_a"],
                proj["scurve_b"],
            )
            evm_mod.update_evm(ps, proj)

            # 13. Payment delivery
            ms_net = pay.deliver_payments(
                certified_js_by_proj[i], self.t,
                ps, milestones, ms_state_list
            )
            period_inflow               += ms_net
            proj_cf[i]["milestone_net"]  = ms_net

            # 14. Breach evaluation (post-allocation)
            flags = br.evaluate_breaches(ps, proj, alloc=alloc)

            # 15. Tolerance update — only here, after allocation is applied
            br.update_tolerance(ps, proj, flags)

            # 16. Termination check
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

            # 17. Completion check (only if not already terminated)
            elif ps["progress_actual"] >= 1.0:
                ps["status"] = "completed"

            # 18. Obs fields refresh — ALWAYS runs regardless of status
            #     so render shows post-step state and next early phase
            #     starts with current values
            self._update_obs_fields(ps, proj, milestones, ms_state_list)

    # ══════════════════════════════════════════════════════════════════════
    # RECORD  — all late-phase mutations are complete at this point
    # ══════════════════════════════════════════════════════════════════════

        net_cashflow            = period_inflow - period_outflow
        self._last_net_cashflow = net_cashflow
        self.budget            += net_cashflow
        reward                  = float(discount_factor * net_cashflow)

        all_terminal = all(
            ps["status"] in ("completed", "terminated")
            for ps in self.proj_state
        )
        terminated_ep = all_terminal
        truncated_ep  = self.t >= self.horizon

        # DB write — proj_state is fully post-allocation here
        if self.conn is not None:
            try:
                db_logger.write_step(
                    self.conn,
                    self.episode_id,
                    self.config.get("config_id", ""),
                    self.t,                  # current period (pre-increment)
                    self.method,
                    self.budget,             # post net_cashflow
                    net_cashflow,
                    reward,
                    terminated_ep or truncated_ep,
                    self.projects,
                    self.proj_state,         # fully post-allocation
                    self.milestones,
                    self.milestone_state,
                    proj_cf,
                )
                db_logger.commit(self.conn)
            except Exception:
                pass

        self.t += 1                          # increment AFTER DB write

        info = self._build_info()
        info["cashflow"]       = proj_cf
        info["period_inflow"]  = period_inflow
        info["period_outflow"] = period_outflow

        return obs, reward, terminated_ep, truncated_ep, info

    # ── GYMNASIUM API ──────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        rnd.render(
            t               = self.t,
            budget          = self.budget,
            horizon         = self.horizon,
            initial_budget  = self.initial_budget,
            projects        = self.projects,
            proj_state      = self.proj_state,
            milestones      = self.milestones,
            milestone_state = self.milestone_state,
            episode_id      = self.episode_id,
            net_cashflow    = self._last_net_cashflow,
            cum_reward      = 0.0,
        )
        return None

    def close(self) -> None:
        pass

    # ── OBSERVATION FIELD COMPUTATION ─────────────────────────────────────────

    def _update_obs_fields(self, ps: dict, proj: dict,
                       milestones: list[dict],
                       milestone_state: list[dict]) -> None:
        """
        Compute and store all catchup and target observation fields into ps.

        Current-period catchup (assuming efficiency = 1):
            progress_plan_t       planned_progress(t_project, duration, a, b)
                                NOTE: already set by EVM update; read from ps
            progress_delay_t      progress_plan_t - progress_actual
                                NOTE: already set by EVM update; read from ps
            progress_space_t      progress_delay_cap - progress_delay_t
            min_prog_t            max(0, progress_plan_t - progress_delay_cap)
            progress_needed_t     max(0, min_prog_t - progress_actual)
            catchup_alloc_t       progress_needed_t * bac

        Next-period catchup (assuming efficiency = 1):
            progress_plan_next_t  planned_progress(t_project + 1, duration, a, b)
            progress_delay_next_t progress_plan_next_t - progress_actual
            progress_space_next_t progress_delay_cap - progress_delay_next_t
            min_prog_next_t       max(0, progress_plan_next_t - progress_delay_cap)
            progress_needed_next_t max(0, min_prog_next_t - progress_actual)
            catchup_alloc_next_t  progress_needed_next_t * bac

        Target milestone:
            first uncertified milestone (j > 0) by milestone_state
        """
        bac       = proj["bac"]
        prog      = ps.get("progress_actual", 0.0)
        t_proj    = ps.get("t_project", 0)
        duration  = proj["planned_duration"]
        a         = proj["scurve_a"]
        b         = proj["scurve_b"]
        delay_cap = proj["progress_delay_cap"]

        # ── current period ────────────────────────────────────────────────────
        # progress_plan_t and progress_delay_t already set by evm.update_evm
        plan_t   = ps.get("progress_plan_t", 0.0)
        delay_t  = plan_t - prog                        # progress_delay_t

        space_t  = delay_cap - delay_t                  # progress_space_t
        min_t    = max(0.0, plan_t - delay_cap)         # min_prog_t
        needed_t = max(0.0, min_t - prog)               # progress_needed_t
        catch_t  = needed_t * bac                       # catchup_alloc_t

        reach_t   = max(0.0, plan_t - prog) * bac      # reach_plan_t

        ps["progress_delay_t"]    = delay_t             # keep in sync
        ps["progress_space_t"]    = space_t
        ps["min_prog_t"]          = min_t
        ps["progress_needed_t"]   = needed_t
        ps["catchup_alloc_t"]     = catch_t
        ps["reach_plan_t"]        = reach_t

        # ── next period ───────────────────────────────────────────────────────
        plan_nt   = planned_progress(t_proj + 1, duration, a, b)
        delay_nt  = plan_nt - prog                      # progress_delay_next_t
        space_nt  = delay_cap - delay_nt                # progress_space_next_t
        min_nt    = max(0.0, plan_nt - delay_cap)       # min_prog_next_t
        needed_nt = max(0.0, min_nt - prog)             # progress_needed_next_t
        catch_nt  = needed_nt * bac                     # catchup_alloc_next_t
        reach_nt  = max(0.0, plan_nt - prog) * bac     # reach_plan_next_t

        ps["progress_plan_next_t"]    = plan_nt
        ps["progress_delay_next_t"]   = delay_nt
        ps["progress_space_next_t"]   = space_nt
        ps["min_prog_next_t"]         = min_nt
        ps["progress_needed_next_t"]  = needed_nt
        ps["catchup_alloc_next_t"]    = catch_nt
        ps["reach_plan_next_t"]       = reach_nt

        # ── target milestone ──────────────────────────────────────────────────
        target_j              = None
        target_progress_gap   = 0.0
        target_timestep_gap   = 0
        target_net_payment    = 0.0
        target_required_alloc = 0.0
        target_payment_rate   = 0.0

        for ms, ms_state in zip(milestones, milestone_state):
            if ms["j"] == 0:
                continue
            if ms_state["certified"]:
                continue
            target_j              = ms["j"]
            target_progress_gap   = max(0.0, ms["progress_threshold"] - prog)
            target_timestep_gap   = ms["timestep_threshold"] - self.t
            target_net_payment    = ms["net_payment"]
            target_required_alloc = target_progress_gap * bac
            target_payment_rate   = (
                target_net_payment / target_required_alloc
                if target_required_alloc > 1e-9 else 0.0
            )
            break

        ps["target_milestone_j"]    = target_j
        ps["target_progress_gap"]   = target_progress_gap
        ps["target_timestep_gap"]   = target_timestep_gap
        ps["target_net_payment"]    = target_net_payment
        ps["target_required_alloc"] = target_required_alloc
        ps["target_payment_rate"]   = target_payment_rate
    # ── OBSERVATION BUILDER ────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Build the flat observation vector from pre-computed ps fields.

        Layout (7 × n_projects + 2 portfolio):
          Per project:
            [0] tolerance_norm              = tolerance_remain / tol_max
            [1] catchup_alloc_t_norm        = catchup_alloc_t / initial_budget
            [2] catchup_alloc_next_t_norm   = catchup_alloc_next_t / initial_budget
            [3] target_progress_gap         raw [0, 1]
            [4] target_timestep_gap_norm    = target_timestep_gap / horizon
            [5] target_required_alloc_norm  = target_required_alloc / initial_budget
            [6] target_payment_rate         clipped [0, 5]
          Portfolio:
            [7n+0] net_cashflow_norm        = last_net_cashflow / initial_budget
            [7n+1] budget_available_norm    = budget / initial_budget
        """
        obs: list[float] = []
        ib = self.initial_budget if self.initial_budget > 0 else 1.0

        for proj, ps in zip(self.projects, self.proj_state):
            tol_max  = proj["termination_tolerance"]
            tol_norm = ps["tolerance_remain"] / tol_max if tol_max > 0 else 1.0

            obs.extend([
                np.clip(tol_norm,                                       0.0, 1.0),
                np.clip(ps["catchup_alloc_t"]      / ib,               0.0, 1.0),
                np.clip(ps["catchup_alloc_next_t"] / ib,               0.0, 1.0),
                np.clip(ps["target_progress_gap"],                      0.0, 1.0),
                np.clip(ps["target_timestep_gap"]  / self.horizon,     -1.0, 1.0),
                np.clip(ps["target_required_alloc"]/ ib,               0.0, 1.0),
                np.clip(ps["target_payment_rate"],                      0.0, 5.0),
            ])

        obs.extend([
            np.clip(self._last_net_cashflow / ib, -2.0,  2.0),
            np.clip(self.budget             / ib,  0.0,  2.0),
        ])

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
                    "catchup_alloc_t":        ps["catchup_alloc_t"],
                    "catchup_alloc_next_t":   ps["catchup_alloc_next_t"],
                    "reach_plan_t":           ps["reach_plan_t"],
                    "reach_plan_next_t":      ps["reach_plan_next_t"],
                    "target_progress_gap":    ps["target_progress_gap"],
                    "target_timestep_gap":    ps["target_timestep_gap"],
                    "target_required_alloc":  ps["target_required_alloc"],
                    "target_payment_rate":    ps["target_payment_rate"],
                }
                for proj, ps in zip(self.projects, self.proj_state)
            ],
        }

    # ── ADVANCE DELIVERY HELPER ────────────────────────────────────────────────

    def _deliver_advance(self, ps: dict, proj: dict,
                         milestones: list[dict],
                         milestone_state: list[dict],
                         t: int) -> float:
        ms_state = milestone_state[0]

        if ms_state["certified"]:
            return 0.0

        amount = milestones[0]["net_payment"]

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
            # current period
            "progress_plan_t":          0.0,
            "progress_delay_t":         0.0,
            "progress_space_t":         0.0,
            "min_prog_t":               0.0,
            "progress_needed_t":        0.0,
            "catchup_alloc_t":          0.0,
            "reach_plan_t":             0.0,
            # next period
            "progress_plan_next_t":     0.0,
            "progress_delay_next_t":    0.0,
            "progress_space_next_t":    0.0,
            "min_prog_next_t":          0.0,
            "progress_needed_next_t":   0.0,
            "catchup_alloc_next_t":     0.0,
            "reach_plan_next_t":        0.0,
            # target milestone
            "target_milestone_j":       None,
            "target_progress_gap":      0.0,
            "target_timestep_gap":      0,
            "target_net_payment":       0.0,
            "target_required_alloc":    0.0,
            "target_payment_rate":      0.0,
            # evm
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