# src/env/env.py
#
# PortfolioBudgetingEnv — Gymnasium environment for project portfolio budgeting.
# DB logging (SQLite) is inlined below the environment class.
#
# External imports: numpy, gymnasium, sqlite3, uuid
# Internal imports: helper  (all pure logic — sampling, EVM, breaches, payments, render)

from __future__ import annotations

import sqlite3
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import helper as h


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioBudgetingEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict,
                 render_mode: Optional[str] = None,
                 conn=None,
                 method: str = "rl",
                 run_id: str = ""):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode
        self.conn        = conn
        self.method      = method
        self.run_id      = run_id

        n = max(1, round(config.get("n_projects_p1", 1)))
        self._n_projects_hint = n
        self._declare_spaces(n)

        self.episode_id             : str        = ""
        self.t                      : int        = 0
        self.budget                 : float      = 0.0
        self.initial_budget         : float      = 0.0
        self.discount               : float      = 1.0
        self.horizon                : int        = 0
        self.projects               : list[dict] = []
        self.milestones             : list[list] = []
        self.milestone_state        : list[list] = []
        self.proj_state             : list[dict] = []
        self._last_net_cashflow     : float      = 0.0
        self._pending_advance_inflow: float      = 0.0
        self._cum_reward            : float      = 0.0

    # ── spaces ────────────────────────────────────────────────────────────────

    def _declare_spaces(self, n: int) -> None:
        proj_low  = np.tile([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0], n)
        proj_high = np.tile([1.0, 1.0, 1.0, 1.0,  1.0, 1.0,  1.0], n)
        self.observation_space = spaces.Box(
            low  = np.concatenate([proj_low,  [-2.0, 0.0]]).astype(np.float32),
            high = np.concatenate([proj_high, [ 2.0, 2.0]]).astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        cfg             = self.config
        config_id = self.config.get("config_id", "unknown")
        self.episode_id = f"{config_id}__{self.method}__{self.run_id}"
        
        self.t          = 0
        self._last_net_cashflow      = 0.0
        self._pending_advance_inflow = 0.0
        self._cum_reward             = 0.0

        self.discount = max(0.0, min(1.0, h.sample(
            self.np_random, cfg["discount_dist"],
            cfg["discount_p1"], cfg["discount_p2"],
            cfg["discount_p3"], cfg["discount_p4"],
        )))

        n_projects = h.sample_int(
            self.np_random, cfg["n_projects_dist"],
            cfg["n_projects_p1"], cfg["n_projects_p2"],
            cfg["n_projects_p3"], cfg["n_projects_p4"],
        )

        if n_projects != self._n_projects_hint:
            self._declare_spaces(n_projects)
            self._n_projects_hint = n_projects

        self.projects, self.milestones, self.milestone_state = [], [], []
        for i in range(n_projects):
            proj     = h.sample_project(self.np_random, cfg, i)
            ms_list  = h.sample_milestones(self.np_random, cfg, proj)
            ms_state = [{"certified": False, "certified_t": None, "payment_released": 0.0}
                        for _ in ms_list]
            self.projects.append(proj)
            self.milestones.append(ms_list)
            self.milestone_state.append(ms_state)

        # budget_available = sum(BAC) * budget_tightness
        # budget_tightness < 1 → constrained (less budget than total BAC)
        budget_tightness = max(0.0, h.sample(
            self.np_random, cfg["budget_tightness_dist"],
            cfg["budget_tightness_p1"], cfg["budget_tightness_p2"],
            cfg["budget_tightness_p3"], cfg["budget_tightness_p4"],
        ))
        sum_bac = sum(p["bac"] for p in self.projects)
        self.initial_budget = sum_bac * budget_tightness
        self.budget         = self.initial_budget

        self.horizon = max(p["planned_finish"] + p["finish_delay_cap"] for p in self.projects)
        self.proj_state = [self._init_proj_state(proj) for proj in self.projects]

        # Advance payments for t=0 projects — credited to budget immediately.
        # _pending_advance_inflow carries the amount so step() can write the
        # correct DB row without double-counting it in period_inflow.
        for ps, proj, ms_list, ms_sl in zip(
                self.proj_state, self.projects, self.milestones, self.milestone_state):
            if proj["planned_start"] == 0:
                adv = self._deliver_advance(ps, proj, ms_list, ms_sl, t=0)
                self._pending_advance_inflow += adv
                self.budget                  += adv

        self._last_net_cashflow = self._pending_advance_inflow

        # Prime EVM + obs fields
        for ps, proj, ms_list, ms_sl in zip(
                self.proj_state, self.projects, self.milestones, self.milestone_state):
            ps["progress_plan_t"] = h.planned_progress(
                ps["t_project"], proj["planned_duration"],
                proj["scurve_a"], proj["scurve_b"])
            h.update_evm(ps, proj)
            self._update_obs_fields(ps, proj, ms_list, ms_sl)

        if self.conn is not None:
            try:
                _db_write_profiles(self.conn, self.episode_id,
                                   self.config.get("config_id", ""),
                                   self.projects, self.milestones)
            except Exception:
                pass

        self._cum_reward = self._pending_advance_inflow   # advance already in budget

        h.reset_history(self.episode_id, len(self.projects))
        return self._build_obs(), self._build_info()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.config
        n   = len(self.projects)

        raw         = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
        active_mask = np.array([ps["status"] == "active" for ps in self.proj_state], dtype=bool)

        # ── RULE 1: zero out non-active projects before anything else ─────────
        # Applies to both RL and manual play; completed/terminated slots get nothing.
        raw[~active_mask] = 0.0

        # ── RULE 2: per-project allocation cap ───────────────────────────────
        # Cap = bac / eta_min so that a full single-step allocation always
        # guarantees project completion regardless of the efficiency draw.
        # For fixed/uniform/custom_ordinal this is exact; for other distributions
        # eta_min falls back to 0.01 (the global floor), making the cap large
        # and letting Rule 3 (budget cap) dominate instead.
        budget_ref = max(self.budget, 1e-9)
        raw_abs    = raw * budget_ref
        for i, proj in enumerate(self.projects):
            if active_mask[i]:
                eta_min   = h.efficiency_lower_bound(cfg, proj_idx=i)
                alloc_cap = proj["bac"] / eta_min
                raw_abs[i] = min(raw_abs[i], alloc_cap)
        raw = raw_abs / budget_ref

        # ── RULE 3: total allocation cannot exceed available budget ───────────
        # Normalise so the sum of fractions ≤ 1.0, then scale to budget.
        active_sum = float(raw[active_mask].sum())
        if active_sum > 1.0 + 1e-9:
            raw = raw / active_sum   # rescale proportionally, preserving relative weights

        allocations = raw * self.budget
        # Hard floor: if budget is non-positive no spending is possible at all.
        if self.budget <= 0.0:
            allocations[:] = 0.0

        # ── RULE 4: bankrupt episode termination ──────────────────────────────
        # If the budget is exhausted AND there are no pending inflows this period
        # (advances for projects starting now), the portfolio cannot progress.
        # Truncate immediately so the agent doesn't spin on a dead state.
        # We check this BEFORE spending so a zero-budget period that happens to
        # receive an advance inflow is still allowed to continue.
        pending_advances = sum(
            milestones[0]["net_payment"]
            for proj, ps, milestones, ms_sl in zip(
                self.projects, self.proj_state, self.milestones, self.milestone_state)
            if (proj["planned_start"] == self.t
                and ps["status"] != "completed"
                and not ms_sl[0]["certified"])
        )
        if self.budget <= 0.0 and pending_advances <= 0.0:
            obs  = self._build_obs()
            info = self._build_info()
            info["truncation_reason"] = "budget_exhausted"
            return obs, 0.0, False, True, info

        discount_factor = self.discount ** self.t
        period_inflow   = 0.0
        period_outflow  = 0.0
        self._pending_advance_inflow = 0.0

        proj_cf: list[dict] = [
            {"advance": 0.0, "milestone_net": 0.0, "settlement": 0.0,
             "allocation": 0.0, "interest": 0.0, "treasury_draw": 0.0}
            for _ in self.projects
        ]

        # Restore advance already credited at reset into proj_cf for t=0 DB row.
        # Also accumulate into _t0_display_inflow so the post-action render shows
        # the correct net_cashflow without touching budget or reward (already done at reset).
        _t0_display_inflow = 0.0
        for i, (proj, ps, ms_list, ms_sl) in enumerate(zip(
                self.projects, self.proj_state, self.milestones, self.milestone_state)):
            if proj["planned_start"] == 0 and self.t == 0 and ms_sl[0]["certified_t"] == 0:
                adv = ms_sl[0]["payment_released"]
                proj_cf[i]["advance"]  = adv
                _t0_display_inflow    += adv

        # ── EARLY PHASE ──────────────────────────────────────────────────────
        # Hard deadline fires here. Other breaches update obs only (no tolerance,
        # no non-deadline termination) — the agent gets one period to allocate first.

        certified_js_by_proj: list[list[int]] = [[] for _ in self.projects]

        for i, (proj, ps, ms_list, ms_sl) in enumerate(zip(
                self.projects, self.proj_state, self.milestones, self.milestone_state)):

            if self.t < proj["planned_start"]:
                ps["status"] = "pending"
                continue
            if ps["status"] in ("completed", "terminated"):
                continue

            # Advance for projects starting this period (t > 0)
            if self.t == proj["planned_start"] and proj["planned_start"] > 0:
                adv = self._deliver_advance(ps, proj, ms_list, ms_sl, t=self.t)
                period_inflow        += adv
                proj_cf[i]["advance"] = adv

            # EVM update
            ps["progress_plan_t"] = h.planned_progress(
                ps["t_project"], proj["planned_duration"],
                proj["scurve_a"], proj["scurve_b"])
            h.update_evm(ps, proj)

            # Certification check
            certified_js_by_proj[i] = h.check_certifications(self.t, ps, ms_list, ms_sl,
                                                              planned_start=proj["planned_start"])

            # Breach flags (pre-allocation, obs only)
            flags = h.evaluate_breaches(ps, proj)

            # Hard deadline — only over_duration_window terminates in early phase
            odw = self.t >= proj["planned_finish"] + proj["finish_delay_cap"]
            flags["over_duration_window"] = odw
            ps["breach_flags"] = flags

            if odw:
                ps["status"] = "terminated"
                s = h.compute_termination_settlement(proj, ps)
                ps["termination_settlement"] = s
                ps["inflow"]  += max(0.0, s)
                period_inflow += max(0.0, s)
                proj_cf[i]["settlement"] = s
                self._update_obs_fields(ps, proj, ms_list, ms_sl)
                continue

            if ps["progress_actual"] >= 1.0 and _all_certified(ms_sl):
                ps["status"] = "completed"
                self._update_obs_fields(ps, proj, ms_list, ms_sl)
                continue

            self._update_obs_fields(ps, proj, ms_list, ms_sl)

        obs = self._build_obs()

        # ── LATE PHASE ───────────────────────────────────────────────────────
        # Tolerance update + non-deadline termination after allocation is applied.

        for i, (proj, ps, ms_list, ms_sl) in enumerate(zip(
                self.projects, self.proj_state, self.milestones, self.milestone_state)):

            if ps["status"] != "active":
                continue

            alloc = float(allocations[i])

            # Interest on treasury draw: computed on the deficit AFTER adding this
            # period's allocation, because the allocation itself is drawn from treasury
            # when no inflow surplus covers it — it must bear interest this period.
            monthly_rate  = cfg["annual_interest_rate"] / 12.0
            treasury_draw = max(0.0, ps["outflow"] + alloc - ps["inflow"])
            interest      = monthly_rate * treasury_draw
            proj_cf[i]["treasury_draw"] = treasury_draw
            proj_cf[i]["interest"]      = interest
            period_outflow += interest

            # Allocation & progress
            eta       = h.sample_efficiency(self.np_random, cfg, proj_idx=i, t_project=ps["t_project"])
            increment = (alloc / proj["bac"]) * eta if proj["bac"] > 0 else 0.0
            ps["progress_actual"]   = min(1.0, ps["progress_actual"] + increment)
            ps["outflow"]          += alloc + interest
            ps["efficiency"]        = eta
            ps["allocation_action"] = alloc
            period_outflow         += alloc
            proj_cf[i]["allocation"] = alloc

            # EVM update (post-allocation; t_project not yet incremented)
            ps["progress_plan_t"] = h.planned_progress(
                ps["t_project"], proj["planned_duration"],
                proj["scurve_a"], proj["scurve_b"])
            h.update_evm(ps, proj)

            # Payment delivery — re-check after allocation in case progress just crossed
            # a threshold that was not met in the pre-allocation early-phase check.
            late_certified = h.check_certifications(self.t, ps, ms_list, ms_sl,
                                                     planned_start=proj["planned_start"])
            all_certified_js = list(set(certified_js_by_proj[i]) | set(late_certified))
            ms_net = h.deliver_payments(all_certified_js, self.t, ps, ms_list, ms_sl)
            period_inflow              += ms_net
            proj_cf[i]["milestone_net"] = ms_net

            # Breach evaluation (post-allocation — authoritative)
            flags = h.evaluate_breaches(ps, proj)
            h.update_tolerance(ps, proj, flags)
            terminated, _ = h.check_termination(self.t, ps, proj, flags)
            ps["breach_flags"] = flags

            if terminated:
                ps["status"] = "terminated"
                s = h.compute_termination_settlement(proj, ps)
                ps["termination_settlement"] = s
                ps["inflow"]  += max(0.0, s)
                period_inflow += max(0.0, s)
                proj_cf[i]["settlement"] = s
            elif ps["progress_actual"] >= 1.0 and _all_certified(ms_sl):
                ps["status"] = "completed"

        # ── RECORD ───────────────────────────────────────────────────────────

        net_cashflow            = period_inflow - period_outflow
        self._last_net_cashflow = net_cashflow
        self.budget            += net_cashflow
        reward                  = float(discount_factor * net_cashflow)
        self._cum_reward       += reward

        all_terminal  = all(ps["status"] in ("completed", "terminated") for ps in self.proj_state)
        terminated_ep = all_terminal
        truncated_ep  = self.t >= self.horizon

        if self.conn is not None:
            try:
                _db_write_step(
                    self.conn, self.episode_id, self.config.get("config_id", ""),
                    self.t, self.method, self.budget, net_cashflow, reward,
                    terminated_ep or truncated_ep,
                    self.projects, self.proj_state,
                    self.milestones, self.milestone_state, proj_cf,
                )
                _db_commit(self.conn)
            except Exception:
                pass

        # Snapshot BEFORE incrementing t_project so render_post shows the
        # period-T view: progress, plan fields, and breach state as they stood
        # when the allocation was applied — not the T+1 lookahead.
        # _recorded_ncf adds back the t=0 advance for display only — it was
        # already in budget and _cum_reward from reset, so it must not re-enter
        # period_inflow, budget, or reward.
        _recorded_t          = self.t
        _recorded_ncf        = net_cashflow + _t0_display_inflow
        _recorded_reward     = reward
        _recorded_proj_state = [dict(ps) for ps in self.proj_state]
        _recorded_budget     = self.budget

        # Advance t_project and refresh obs fields for T+1 pre-action view
        self.t += 1
        for ps, proj, ms_list, ms_sl in zip(
                self.proj_state, self.projects, self.milestones, self.milestone_state):
            if ps["status"] == "active":
                ps["t_project"] += 1
            if ps["status"] == "pending" and self.t == proj["planned_start"]:
                # Activate now so pre-action render at t=planned_start shows ACTIVE.
                # _deliver_advance is idempotent (returns 0 if already certified),
                # so the early phase in the next step() call is safe.
                adv = self._deliver_advance(ps, proj, ms_list, ms_sl, t=self.t)
                self.budget                  += adv
                self._pending_advance_inflow += adv
                self._update_obs_fields(ps, proj, ms_list, ms_sl)
            elif ps["status"] != "pending":
                ps["allocation_action"] = 0.0   # clear stale value before pre-action render
                self._update_obs_fields(ps, proj, ms_list, ms_sl)

        info = self._build_info()
        info["cashflow"]              = proj_cf
        info["period_inflow"]         = period_inflow
        info["period_outflow"]        = period_outflow
        info["recorded_t"]            = _recorded_t
        info["recorded_net_cashflow"] = _recorded_ncf
        info["recorded_reward"]       = _recorded_reward
        info["recorded_proj_state"]   = _recorded_proj_state
        info["recorded_budget"]       = _recorded_budget
        info["cum_reward"]            = self._cum_reward
        info["render_post"]           = self._build_render_state(
            t=_recorded_t,
            budget=_recorded_budget,
            proj_state=_recorded_proj_state,
            net_cashflow=_recorded_ncf,
            cum_reward=self._cum_reward,
        )
        if not (terminated_ep or truncated_ep):
            info["render_pre"] = self._build_render_state(
                t=self.t,
                budget=self.budget,
                proj_state=self.proj_state,
                net_cashflow=0.0,
                cum_reward=self._cum_reward,
            )
        return obs, reward, terminated_ep, truncated_ep, info

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        h.render(
            t=self.t, budget=self.budget, horizon=self.horizon,
            initial_budget=self.initial_budget,
            projects=self.projects, proj_state=self.proj_state,
            milestones=self.milestones, milestone_state=self.milestone_state,
            episode_id=self.episode_id,
            net_cashflow=self._last_net_cashflow, cum_reward=0.0,
        )
        return None

    def close(self) -> None:
        pass

    # ── play interface ────────────────────────────────────────────────────────

    @property
    def prompt_hint(self) -> dict:
        """State the prompt needs — no env internals leak into play.py."""
        n_active = sum(1 for ps in self.proj_state if ps["status"] == "active")
        return {"n_projects": len(self.projects), "n_active": n_active}

    def parse_input(self, raw: str) -> "np.ndarray | None":
        """
        Parse a raw input string into an action array.
        Empty string → equal split across active projects.
        Invalid input → None (caller should re-prompt).
        """
        n           = len(self.projects)
        active_mask = [ps["status"] == "active" for ps in self.proj_state]
        if raw == "":
            return h.default_action(n, active_mask)
        return h.parse_action(raw, n, self.budget)

    # ── observation field computation ─────────────────────────────────────────

    def _update_obs_fields(self, ps: dict, proj: dict,
                           milestones: list[dict], milestone_state: list[dict]) -> None:
        bac      = proj["bac"]
        prog     = ps.get("progress_actual", 0.0)
        t_proj   = ps.get("t_project", 0)
        dur      = proj["planned_duration"]
        a, b     = proj["scurve_a"], proj["scurve_b"]
        dcap     = proj["progress_delay_cap"]
        plan_t   = ps.get("progress_plan_t", 0.0)

        # ── PENDING: zero all catchup/reach fields; suppress target milestone ──
        if ps["status"] == "pending":
            ps["progress_delay_t"]        = 0.0
            ps["progress_space_t"]        = 0.0
            ps["min_prog_t"]              = 0.0
            ps["progress_needed_t"]       = 0.0
            ps["catchup_alloc_t"]         = 0.0
            ps["reach_plan_t"]            = 0.0
            ps["progress_plan_next_t"]    = 0.0
            ps["progress_delay_next_t"]   = 0.0
            ps["progress_space_next_t"]   = 0.0
            ps["min_prog_next_t"]         = 0.0
            ps["progress_needed_next_t"]  = 0.0
            ps["catchup_alloc_next_t"]    = 0.0
            ps["reach_plan_next_t"]       = 0.0
            ps["target_milestone_j"]      = None
            ps["target_progress_gap"]     = 0.0
            ps["target_timestep_gap"]     = 0
            ps["target_net_payment"]      = 0.0
            ps["target_required_alloc"]   = 0.0
            ps["target_payment_rate"]     = 0.0
            ps["target_npv"]              = 0.0
            return

        # Current-period catchup fields.
        # delay/space use plan_t (actual delay relative to the plan at this instant).
        # min/needed/catchup/reach use plan_nt: the allocation made THIS step must
        # satisfy the plan target at END of this step (= planned_progress(t_project+1)).
        plan_nt  = h.planned_progress(t_proj + 1, dur, a, b)
        delay_t  = plan_t - prog
        ps["progress_delay_t"]  = delay_t
        ps["progress_space_t"]  = dcap - delay_t
        ps["min_prog_t"]        = max(0.0, plan_nt - dcap)
        ps["progress_needed_t"] = max(0.0, ps["min_prog_t"] - prog)
        ps["catchup_alloc_t"]   = ps["progress_needed_t"] * bac
        ps["reach_plan_t"]      = max(0.0, plan_nt - prog) * bac

        # Next-period catchup fields — lookahead one further step (t_project+2).
        plan_nt2 = h.planned_progress(t_proj + 2, dur, a, b)
        delay_nt = plan_nt - prog
        min_nt   = max(0.0, plan_nt2 - dcap)
        ps["progress_plan_next_t"]    = plan_nt
        ps["progress_delay_next_t"]   = delay_nt
        ps["progress_space_next_t"]   = dcap - delay_nt
        ps["min_prog_next_t"]         = min_nt
        ps["progress_needed_next_t"]  = max(0.0, min_nt - prog)
        ps["catchup_alloc_next_t"]    = ps["progress_needed_next_t"] * bac
        ps["reach_plan_next_t"]       = max(0.0, plan_nt2 - prog) * bac

        # Target milestone
        # timestep_threshold is in project-time; convert to episode-time for the gap.
        target_j = target_pg = target_tg = target_np = target_ra = target_pr = target_npv = None
        for ms, ms_state in zip(milestones, milestone_state):
            if ms["j"] == 0 or ms_state["certified"]:
                continue
            target_j  = ms["j"]
            target_pg = max(0.0, ms["progress_threshold"] - prog)
            target_tg = (proj["planned_start"] + ms["timestep_threshold"]) - self.t
            target_np = ms["net_payment"]
            target_ra = target_pg * bac
            gap       = max(0, target_tg)
            disc_pay  = (target_np / (self.discount ** gap)
                         if self.discount > 1e-9 else target_np)
            target_npv = disc_pay - target_ra
            target_pr  = target_np / target_ra if target_ra > 1e-9 else 0.0
            break

        ps["target_milestone_j"]    = target_j
        ps["target_progress_gap"]   = target_pg   or 0.0
        ps["target_timestep_gap"]   = target_tg   or 0
        ps["target_net_payment"]    = target_np   or 0.0
        ps["target_required_alloc"] = target_ra   or 0.0
        ps["target_payment_rate"]   = target_pr   or 0.0
        ps["target_npv"]            = target_npv  or 0.0

    # ── observation builder ───────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        ib       = self.initial_budget if self.initial_budget > 0 else 1.0
        npv_vals = [ps["target_npv"] for ps in self.proj_state]
        npv_den  = max(abs(v) for v in npv_vals)
        npv_den  = npv_den if npv_den > 1e-9 else 1.0

        obs: list[float] = []
        for proj, ps in zip(self.projects, self.proj_state):
            tol_max = proj["termination_tolerance"]
            obs.extend([
                np.clip(ps["tolerance_remain"],                             0.0, float(tol_max)),
                np.clip(ps["catchup_alloc_t"]      / ib,                   0.0, 1.0),
                np.clip(ps["catchup_alloc_next_t"] / ib,                   0.0, 1.0),
                np.clip(ps["target_progress_gap"],                          0.0, 1.0),
                np.clip(ps["target_timestep_gap"]  / self.horizon,         -1.0, 1.0),
                np.clip(ps["target_required_alloc"]/ ib,                   0.0, 1.0),
                np.clip(ps["target_npv"]           / npv_den,              -1.0, 1.0),
            ])
        obs.extend([
            np.clip(self._last_net_cashflow / ib, -2.0, 2.0),
            np.clip(self.budget             / ib,  0.0, 2.0),
        ])
        return np.array(obs, dtype=np.float32)

    # ── info builder ──────────────────────────────────────────────────────────

    def _build_render_state(self, t: int, budget: float,
                            proj_state: list[dict],
                            net_cashflow: float,
                            cum_reward: float) -> dict:
        """Assemble the kwargs dict that h.render() expects."""
        return {
            "t":              t,
            "budget":         budget,
            "horizon":        self.horizon,
            "initial_budget": self.initial_budget,
            "projects":       self.projects,
            "proj_state":     proj_state,
            "milestones":     self.milestones,
            "milestone_state": self.milestone_state,
            "episode_id":     self.episode_id,
            "net_cashflow":   net_cashflow,
            "cum_reward":     cum_reward,
        }

    def _build_info(self) -> dict:
        n_active = sum(1 for ps in self.proj_state if ps["status"] == "active")
        return {
            "t_episode": self.t, "budget": self.budget, "horizon": self.horizon,
            "n_active":  n_active,
            "period_inflow":  self._pending_advance_inflow,
            "period_outflow": 0.0,
            "episode_header": {
                "episode_id": self.episode_id,
                "n_projects": len(self.projects),
                "horizon":    self.horizon,
                "budget":     self.budget,
                "run_dir":    None,   # play.py fills this in after env.reset()
            },
            "render_state": self._build_render_state(
                t=self.t,
                budget=self.budget,
                proj_state=self.proj_state,
                net_cashflow=self._last_net_cashflow,
                cum_reward=self._cum_reward,
            ),
            "projects": [{
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
                "target_npv":             ps["target_npv"],
            } for proj, ps in zip(self.projects, self.proj_state)],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _deliver_advance(self, ps: dict, proj: dict,
                         milestones: list[dict], milestone_state: list[dict],
                         t: int) -> float:
        ms_state = milestone_state[0]
        if ms_state["certified"]:
            return 0.0
        amount = milestones[0]["net_payment"]
        ms_state.update(certified=True, certified_t=t, payment_released=amount)
        ps["status"]  = "active"
        ps["inflow"] += amount
        return amount

    @staticmethod
    def _init_proj_state(proj: dict) -> dict:
        return {
            "status": "pending", "t_project": 0,
            "inflow": 0.0, "outflow": 0.0,
            "termination_settlement": 0.0, "allocation_action": 0.0,
            "efficiency": 1.0, "progress_actual": 0.0,
            "progress_plan_t": 0.0,   "progress_delay_t": 0.0,
            "progress_space_t": 0.0,  "min_prog_t": 0.0,
            "progress_needed_t": 0.0, "catchup_alloc_t": 0.0, "reach_plan_t": 0.0,
            "progress_plan_next_t": 0.0,   "progress_delay_next_t": 0.0,
            "progress_space_next_t": 0.0,  "min_prog_next_t": 0.0,
            "progress_needed_next_t": 0.0, "catchup_alloc_next_t": 0.0,
            "reach_plan_next_t": 0.0,
            "target_milestone_j": None, "target_progress_gap": 0.0,
            "target_timestep_gap": 0,   "target_net_payment": 0.0,
            "target_required_alloc": 0.0, "target_payment_rate": 0.0,
            "target_npv": 0.0,
            "spi": 1.0, "cpi": 1.0, "tcpi": 1.0,
            "eac": proj["bac"],
            "breach_flags": {
                "over_progress_delay": False, "over_duration_window": False,
                "over_any": False,
            },
            "tolerance_remain": proj["termination_tolerance"],
        }


# ── module-level helper ───────────────────────────────────────────────────────

def _all_certified(milestone_state: list[dict]) -> bool:
    return all(ms["certified"] for ms in milestone_state)


# ═══════════════════════════════════════════════════════════════════════════════
# DB LOGGER  (SQLite audit ledger — inlined from db_logger.py)
# All writes are fire-and-forget; callers wrap in try/except.
# Column names match schema.sql exactly.
# ═══════════════════════════════════════════════════════════════════════════════

def _db_write_profiles(conn: sqlite3.Connection, episode_id: str, config_id: str,
                       projects: list[dict], milestones_list: list[list[dict]]) -> None:
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
                    termination_tolerance
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode_id, config_id, proj["i"],
                proj["planned_start"], proj["planned_finish"], proj["planned_duration"],
                proj["bac"], proj["profit_percent"], proj["price"],
                proj["scurve_a"], proj["scurve_b"],
                proj["advance_percent"], proj["advance_trigger"], proj["advance_recovery"],
                proj["retention_rate"],
                proj["progress_delay_cap"], proj["finish_delay_cap"],
                proj["termination_tolerance"],
            ))
        for i, ms_list in enumerate(milestones_list):
            for ms in ms_list:
                conn.execute("""
                    INSERT INTO milestones_profile (
                        episode_id, i, j,
                        progress_threshold, timestep_threshold, payment_weight,
                        gross_payment, advance_recovery, advance_recovery_remain,
                        retention_withheld, retention_released, net_payment
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    episode_id, i, ms["j"],
                    ms["progress_threshold"], ms["timestep_threshold"],
                    ms["payment_weight"], ms["gross_payment"],
                    ms["advance_recovery"], ms["advance_recovery_remain"],
                    ms["retention_withheld"], ms["retention_released"], ms["net_payment"],
                ))
        conn.commit()
    except Exception:
        pass


def _db_write_step(conn: sqlite3.Connection, episode_id: str, config_id: str,
                   t: int, method: str, budget: float, net_cashflow: float,
                   reward: float, done: bool,
                   projects: list[dict], proj_state: list[dict],
                   milestones_list: list[list[dict]],
                   milestone_state_list: list[list[dict]],
                   proj_cf: list[dict]) -> None:
    try:
        conn.execute("""
            INSERT INTO portfolios (
                episode_id, config_id, t_episode, method,
                budget_available, net_cashflow, reward, done
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (episode_id, config_id, t, method, budget, net_cashflow, reward, int(done)))

        conn.execute("""
            INSERT INTO portfolio_observation (
                episode_id, t_episode, method, net_cashflow, budget_available
            ) VALUES (?,?,?,?,?)
        """, (episode_id, t, method, net_cashflow, budget))

        for proj, ps, milestones, ms_sl, cf in zip(
                projects, proj_state, milestones_list, milestone_state_list, proj_cf):
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
                    over_duration_window,
                    over_progress_delay,
                    over_any,
                    tolerance_remain
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                episode_id, proj["i"], t, ps.get("t_project"), method,
                ps.get("status"),
                ps.get("inflow", 0.0), ps.get("outflow", 0.0),
                ps.get("termination_settlement", 0.0),
                (cf.get("milestone_net", 0.0) + cf.get("advance", 0.0)
                 + cf.get("settlement", 0.0)
                 - cf.get("allocation", 0.0) - cf.get("interest", 0.0)),
                cf.get("allocation", 0.0), cf.get("treasury_draw", 0.0),
                cf.get("interest", 0.0),
                cf.get("allocation", 0.0) + cf.get("interest", 0.0),
                ps.get("efficiency", 1.0),
                ps.get("spi", 1.0), ps.get("cpi", 1.0), ps.get("eac", proj["bac"]),
                ps.get("progress_actual", 0.0),
                ps.get("progress_plan_t", 0.0),   ps.get("progress_delay_t", 0.0),
                ps.get("progress_space_t", 0.0),  ps.get("min_prog_t", 0.0),
                ps.get("progress_needed_t", 0.0), ps.get("catchup_alloc_t", 0.0),
                ps.get("reach_plan_t", 0.0),
                ps.get("progress_plan_next_t", 0.0),   ps.get("progress_delay_next_t", 0.0),
                ps.get("progress_space_next_t", 0.0),  ps.get("min_prog_next_t", 0.0),
                ps.get("progress_needed_next_t", 0.0), ps.get("catchup_alloc_next_t", 0.0),
                ps.get("reach_plan_next_t", 0.0),
                ps.get("target_milestone_j"),
                ps.get("target_progress_gap", 0.0),   ps.get("target_timestep_gap", 0),
                ps.get("target_net_payment", 0.0),    ps.get("target_required_alloc", 0.0),
                ps.get("target_payment_rate", 0.0),   ps.get("target_npv", 0.0),
                int(flags.get("over_duration_window", False)),
                int(flags.get("over_progress_delay",  False)),
                int(flags.get("over_any",             False)),
                ps.get("tolerance_remain", proj["termination_tolerance"]),
            ))

            conn.execute("""
                INSERT INTO projects_observation (
                    episode_id, i, t_episode, t_project, method,
                    net_cashflow, tolerance_remain,
                    catchup_alloc_t, catchup_alloc_next_t,
                    reach_plan_t, reach_plan_next_t,
                    target_progress_gap, target_timestep_gap,
                    target_required_alloc, target_npv
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode_id, proj["i"], t, ps.get("t_project"), method,
                ps.get("inflow", 0.0) - ps.get("outflow", 0.0),
                ps.get("tolerance_remain", proj["termination_tolerance"]),
                ps.get("catchup_alloc_t", 0.0),    ps.get("catchup_alloc_next_t", 0.0),
                ps.get("reach_plan_t", 0.0),        ps.get("reach_plan_next_t", 0.0),
                ps.get("target_progress_gap", 0.0), ps.get("target_timestep_gap", 0),
                ps.get("target_required_alloc", 0.0), ps.get("target_npv", 0.0),
            ))

            for ms, ms_state in zip(milestones, ms_sl):
                if not ms_state["certified"] or ms_state["certified_t"] != t:
                    continue
                episode_threshold = proj["planned_start"] + ms["timestep_threshold"]
                delay = (ms_state["certified_t"] - episode_threshold
                         if ms_state["certified_t"] is not None else None)
                conn.execute("""
                    INSERT OR REPLACE INTO milestones_status (
                        episode_id, i, j, method,
                        certified_t, certification_delay, net_payment
                    ) VALUES (?,?,?,?,?,?,?)
                """, (
                    episode_id, proj["i"], ms["j"], method,
                    ms_state["certified_t"], delay, ms_state["payment_released"],
                ))
    except Exception:
        pass


def _db_commit(conn: sqlite3.Connection) -> None:
    try:
        conn.commit()
    except Exception:
        pass