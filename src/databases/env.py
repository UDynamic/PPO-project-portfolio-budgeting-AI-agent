# env.py

import sqlite3
import random
import math
import uuid
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# SAMPLER
# ─────────────────────────────────────────────────────────────

def sample(dist: str, p1, p2=None, p3=None, p4=None) -> float:
    if dist == "fixed":
        return p1
    elif dist == "uniform":
        return random.uniform(p1, p2)
    elif dist == "normal":
        return random.gauss(p1, p2)
    elif dist == "lognormal":
        return random.lognormvariate(p1, p2)
    elif dist == "triangular":
        return random.triangular(p1, p3, p2)   # min, max, mode
    elif dist == "truncated_normal":
        while True:
            v = random.gauss(p1, p2)
            if p3 <= v <= p4:
                return v
    elif dist == "beta":
        return random.betavariate(p1, p2)
    elif dist == "categorical":
        r = random.random()
        if r < p1:
            return int(p3)
        elif r < p1 + p2:
            return int(p4)
        else:
            return int(p3)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def sample_int(dist: str, p1, p2=None, p3=None, p4=None) -> int:
    return max(1, round(sample(dist, p1, p2, p3, p4)))


# ─────────────────────────────────────────────────────────────
# S-CURVE PLANNED PROGRESS
# ─────────────────────────────────────────────────────────────

def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function via simple numerical integration."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    steps = 200
    dx = x / steps
    total = 0.0
    for k in range(steps):
        t = (k + 0.5) * dx
        total += (t ** (a - 1)) * ((1 - t) ** (b - 1)) * dx
    import math
    B = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return min(1.0, total / B)


def planned_progress(t_project: int, duration: int, a: float, b: float) -> float:
    """S-curve planned cumulative progress at end of project period t_project."""
    x = t_project / duration
    return beta_cdf(x, a, b)


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class PortfolioEnv:
    """
    Timestep convention
    ───────────────────
    t counts completed periods.  t=0 is "before any period has elapsed."

    reset() → returns the observation at t=0:
        - Portfolio generated, advance payments received for start=0 projects.
        - progress_plan set to planned_progress(t_project=1, ...) — what the
          plan says should be complete by the end of the coming period.
        - progress / acwp are 0.0 — no work has been done yet.
        - DB is NOT written yet. The agent/human observes this state and
          chooses allocations for period 0.

    step(alloc) → executes the current period t, then increments t:
        - Draws η, advances progress, runs EVM, checks milestones/completion.
        - Writes the t row to portfolios and projects_status.
        - Increments self.t.
        - Returns observation for period t+1 (next decision point).

    Reward (NPV)
    ────────────
    reward = discount_factor^t × (total_inflow − total_outflow)

    Both inflows (advance payments, milestone payments, retention releases)
    and outflows (budget allocations) are discounted at the same rate.
    This is the correct NPV formulation: spending early is penalised,
    earning early is rewarded.  The agent maximises the discounted sum of
    net cash flows over the episode.

    Advance payment at t=0
    ──────────────────────
    For projects with start=0 the advance is credited to self.budget during
    reset() (before any allocation).  To include it in the period-0 reward
    it is stored in self._t0_advances and added to total_inflow inside the
    first step() call so the NPV calculation picks it up correctly.

    So the sequence is:
        obs₀          = reset()          # plan shown, no work done, t=0
        obs₁, r₀, …  = step(alloc₀)     # period 0 executed, stored; t→1
        obs₂, r₁, …  = step(alloc₁)     # period 1 executed, stored; t→2
        …
    """

    def __init__(self, conn: sqlite3.Connection, config_id: str, method: str = "rl"):
        self.conn = conn
        self.config_id = config_id
        self.method = method

        self.episode_id = None
        self.cfg = None
        self.projects = []
        self.milestones = []

        self.t = 0
        self.budget = 0.0
        self.discount = 1.0
        self.proj_state = []

        # Advance payments credited in reset() for start=0 projects.
        # Consumed during the t=0 step so they appear in the reward.
        self._t0_advances: dict = {}   # {project_index: advance_amount}

    # ── RESET ────────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Generate the portfolio and return the t=0 observation.

        For projects starting at episode t=0:
          - status set to "active"
          - advance payment credited to self.budget AND stored in
            self._t0_advances so step() can include it in the period-0 reward
          - progress_plan set to planned_progress(t_project=1) so the agent
            can see the planned target for the coming period
          - progress / acwp remain 0.0 (no work executed yet)

        No DB rows are written here. The DB row for t=0 is written by the
        first step() call after the agent has chosen allocations.
        """
        self.episode_id = str(uuid.uuid4())
        self.t = 0
        self._t0_advances = {}

        # 1. Load config
        self.cfg = dict(self.conn.execute(
            "SELECT * FROM environment_config WHERE config_id = ?",
            (self.config_id,)
        ).fetchone())

        cfg = self.cfg

        # 2. Sample portfolio-level parameters
        self.discount = sample(
            cfg["discount_dist"], cfg["discount_p1"],
            cfg["discount_p2"], cfg["discount_p3"], cfg["discount_p4"]
        )
        self.discount = max(0.0, min(1.0, self.discount))

        n_projects = sample_int(
            cfg["n_projects_dist"], cfg["n_projects_p1"],
            cfg["n_projects_p2"], cfg["n_projects_p3"], cfg["n_projects_p4"]
        )

        initial_budget = sample(
            cfg["initial_budget_dist"], cfg["initial_budget_p1"],
            cfg["initial_budget_p2"], cfg["initial_budget_p3"], cfg["initial_budget_p4"]
        )
        self.budget = max(0.0, initial_budget)

        # 3. Generate projects
        self.projects = []
        self.milestones = []

        for i in range(n_projects):
            proj = self._sample_project(i)
            self.projects.append(proj)
            self.milestones.append(self._sample_milestones(i, proj))

        # 4. Derive horizon — latest possible finish across all projects.
        self.horizon = max(p["finish"] + p["schedule_cap"] for p in self.projects)

        # 5. Write static tables (profiles + milestones)
        self._write_profiles()

        # 6. Initialise mutable project states
        self.proj_state = [
            self._init_proj_state(p, ms)
            for p, ms in zip(self.projects, self.milestones)
        ]

        # 7. Pre-step initialisation for projects that start at t=0:
        #    - credit advance payment to budget AND store for step() reward
        #    - set status + t_project so the observation is meaningful
        #    - set progress_plan to the plan target for the coming period
        for ps, proj in zip(self.proj_state, self.projects):
            if proj["start"] == 0:
                ps["status"] = "active"
                ps["t_project"] = 1          # period 1 is the coming period
                ps["progress_plan"] = planned_progress(
                    1, proj["duration"], proj["scurve_a"], proj["scurve_b"]
                )
                advance = proj["advance_percent"] * proj["price"]
                self.budget += advance
                self._t0_advances[proj["i"]] = advance   # ← store for step()

        # 8. Return the pre-action observation. No DB writes yet.
        return self._get_state()

    # ── STEP ─────────────────────────────────────────────────

    def step(self, allocations: list) -> tuple:
        """
        Execute period self.t with the given allocations.

        Flow per call:
          1. Clip allocations to available budget.
          2. For each project active this period: draw η, advance progress,
             update EVM, check milestones, check completion/termination.
          3. Update self.budget.
          4. Write portfolios and projects_status rows for self.t.
          5. Increment self.t.
          6. Return next observation (pre-action state for period t+1).

        Reward
        ------
        reward = discount^t × (total_inflow − total_outflow)

        Allocations reduce the reward; payments increase it.  This is the
        correct discounted NPV signal: the agent is penalised for spending
        early and rewarded for collecting payments early.

        Returns
        -------
        (state, reward, done, info)

        info["cashflow"] is a list of per-project dicts:
            {
              "i":                int,
              "allocation":       float,   # outflow (cost spent)
              "advance":          float,   # advance payment received
              "milestone_gross":  float,   # sum of gross milestone payments certified this period
              "milestone_net":    float,   # sum of net milestone payments received
              "retention_release":float,   # retention released on completion
              "settlement":       float,   # termination penalty (negative)
            }
        info["portfolio_inflow"]:  float
        info["portfolio_outflow"]: float
        """
        # ── 1. Clip allocations ──────────────────────────────
        allocations = [max(0.0, a) for a in allocations]
        total_alloc = sum(
            allocations[i] for i in range(len(self.projects))
            if self.proj_state[i]["status"] == "active"
        )
        if total_alloc > self.budget:
            scale = self.budget / total_alloc if total_alloc > 0 else 0.0
            allocations = [a * scale for a in allocations]

        total_inflow = 0.0
        total_outflow = 0.0
        discount_factor = self.discount ** self.t

        # Per-project cash flow tracking for the info dict
        proj_cashflow = [
            {
                "i": proj["i"],
                "allocation": 0.0,
                "advance": 0.0,
                "milestone_gross": 0.0,
                "milestone_net": 0.0,
                "retention_release": 0.0,
                "settlement": 0.0,
            }
            for proj in self.projects
        ]

        # ── 2. Per-project update ────────────────────────────
        for i, (proj, ps) in enumerate(zip(self.projects, self.proj_state)):

            alloc = allocations[i] if i < len(allocations) else 0.0
            advance_amount = 0.0
            payment_net = None
            retention_release = None
            settlement = None
            eta = None

            # ── Not yet started ──────────────────────────────
            if self.t < proj["start"]:
                ps["status"] = None
                self._write_project_row(i, proj, ps, alloc, eta,
                                        advance_amount, payment_net,
                                        retention_release, settlement)
                continue

            # ── Project start: credit advance (start > 0 only) ──
            if self.t == proj["start"] and proj["start"] > 0:
                ps["status"] = "active"
                ps["t_project"] = 1
                advance_amount = proj["advance_percent"] * proj["price"]
                total_inflow += advance_amount
                proj_cashflow[i]["advance"] = advance_amount

            elif self.t == 0 and proj["start"] == 0 and i in self._t0_advances:
                # Include the t=0 advance in this period's inflow/reward
                advance_amount = self._t0_advances.pop(i)
                total_inflow += advance_amount
                proj_cashflow[i]["advance"] = advance_amount

            # ── Already done ─────────────────────────────────
            if ps["status"] in ("completed", "terminated"):
                self._write_project_row(i, proj, ps, 0.0, None,
                                        0.0, None, None, None)
                continue

            # ── Active: execute this period ──────────────────
            ps["t_project"] = self.t - proj["start"] + 1

            # Draw efficiency
            cfg = self.cfg
            eta = sample(
                cfg["efficiency_dist"], cfg["efficiency_p1"],
                cfg["efficiency_p2"], cfg["efficiency_p3"], cfg["efficiency_p4"]
            )
            eta = max(0.01, eta)

            # Progress increment
            increment = (alloc / proj["budget"]) * eta if proj["budget"] > 0 else 0.0
            ps["progress"] = min(1.0, ps["progress"] + increment)
            ps["progress_increment"] = increment

            # Planned progress at end of this period
            ps["progress_plan"] = planned_progress(
                ps["t_project"], proj["duration"],
                proj["scurve_a"], proj["scurve_b"]
            )

            # Accumulate actual cost
            ps["acwp"] += alloc
            total_outflow += alloc
            proj_cashflow[i]["allocation"] = alloc

            # EVM
            bcws = ps["progress_plan"] * proj["budget"]
            bcwp = ps["progress"] * proj["budget"]
            acwp = ps["acwp"]

            ps["spi"] = bcwp / bcws if bcws > 1e-9 else 1.0
            ps["cpi"] = bcwp / acwp if acwp > 1e-9 else 1.0
            ps["eac"] = (proj["budget"] / ps["cpi"]
                         if ps["cpi"] > 1e-9
                         else proj["budget"] * proj["cost_cap"])

            # Forecast finish
            ps["forecast_finish"] = (
                proj["start"] + proj["duration"] / ps["spi"]
                if ps["spi"] > 1e-9
                else proj["finish"] + proj["schedule_cap"] + 1
            )
            ps["schedule_slip"] = ps["forecast_finish"] - proj["finish"]
            ps["cost_overrun"] = ps["eac"] - proj["budget"]

            # ── Milestone check ──────────────────────────────
            for j, ms in enumerate(self.milestones[i]):
                if ms["certified"]:
                    continue
                if ps["progress"] >= ms["threshold"] and self.t >= ms["earliest_t"]:
                    ms["certified"] = True
                    ms["certified_t"] = self.t
                    gross = ms["payment_weight"] * proj["price"]
                    recovery = gross * proj["advance_recovery"]
                    retention_held = gross * proj["retention_rate"]
                    net = gross - recovery - retention_held
                    ms["payment_released"] = net
                    payment_net = (payment_net or 0.0) + net
                    ps["advance_recovered"] += recovery
                    ps["retention_held"] += retention_held
                    total_inflow += net
                    proj_cashflow[i]["milestone_gross"] += gross
                    proj_cashflow[i]["milestone_net"]   += net
                    self._write_milestone_status(i, j, ms)
                    self.conn.commit()

            # ── Completion ───────────────────────────────────
            if ps["progress"] >= 1.0 and ps["status"] == "active":
                ps["status"] = "completed"
                retention_release = ps["retention_held"]
                total_inflow += retention_release
                ps["retention_released"] = True
                proj_cashflow[i]["retention_release"] = retention_release

            # ── Breach and cure ──────────────────────────────
            schedule_breach = ps["schedule_slip"] > proj["schedule_cap"]
            cost_breach = ps["eac"] > proj["cost_cap"] * proj["budget"]

            if (schedule_breach or cost_breach) and ps["status"] == "active":
                ps["cure_remaining"] -= 1
            else:
                ps["cure_remaining"] = proj["cure_length"]

            if ps["cure_remaining"] <= 0 and ps["status"] == "active":
                ps["status"] = "terminated"
                settlement = -(ps["acwp"] * 0.05)
                total_inflow += settlement
                proj_cashflow[i]["settlement"] = settlement

            self._write_project_row(i, proj, ps, alloc, eta,
                                    advance_amount, payment_net,
                                    retention_release, settlement)

        # ── 3. Portfolio update ──────────────────────────────
        self.budget = self.budget - total_outflow + total_inflow

        # ── 4. Compute reward (NPV of net cash flow) ─────────
        # reward = δ^t × (inflow − outflow)
        # Allocations reduce the reward; payments increase it.
        # This is the correct NPV signal: the agent is penalised for early
        # spending and rewarded for early payment collection.
        net_cashflow = total_inflow - total_outflow
        reward = discount_factor * net_cashflow

        # ── 5. Write portfolio row ───────────────────────────
        all_terminal = all(
            ps["status"] in ("completed", "terminated")
            for ps in self.proj_state
        )
        budget_exhausted = self.budget <= 0 and any(
            ps["status"] == "active" for ps in self.proj_state
        )

        done = self.t >= self.horizon or all_terminal or budget_exhausted

        self._write_portfolio_row(
            inflow=total_inflow,
            outflow=total_outflow,
            reward=reward,
            done=int(done)
        )

        if done:
            self.conn.commit()

        # ── 6. Advance clock ─────────────────────────────────
        self.t += 1

        # ── 7. Pre-arm progress_plan for the next period ─────
        for ps, proj in zip(self.proj_state, self.projects):
            if ps["status"] == "active":
                next_t_project = self.t - proj["start"] + 1
                if next_t_project <= proj["duration"]:
                    ps["progress_plan"] = planned_progress(
                        next_t_project, proj["duration"],
                        proj["scurve_a"], proj["scurve_b"]
                    )

        info = {
            "cashflow": proj_cashflow,
            "portfolio_inflow":  total_inflow,
            "portfolio_outflow": total_outflow,
        }

        return self._get_state(), reward, done, info

    # ── STATE ────────────────────────────────────────────────

    def _next_milestone_obs(self, i: int) -> dict:
        """
        Return observable fields for the next uncertified milestone of project i.
        """
        proj = self.projects[i]
        ps   = self.proj_state[i]

        retention_held = ps["retention_held"]

        next_ms = None
        for ms in self.milestones[i]:
            if not ms["certified"]:
                next_ms = ms
                break

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

    def _get_state(self) -> dict:
        state = {
            "t_episode": self.t,
            "budget": self.budget,
            "horizon": self.horizon,
            "projects": []
        }
        for i, (proj, ps) in enumerate(zip(self.projects, self.proj_state)):
            ms_obs = self._next_milestone_obs(i)
            state["projects"].append({
                "i": proj["i"],
                "status": ps["status"],
                "budget": proj["budget"],
                "start": proj["start"],
                "finish": proj["finish"],
                "progress": ps["progress"],
                "progress_plan": ps["progress_plan"],
                "spi": ps["spi"],
                "cpi": ps["cpi"],
                "eac": ps["eac"],
                "forecast_finish": ps["forecast_finish"],
                "schedule_slip": ps["schedule_slip"],
                "cure_remaining": ps["cure_remaining"],
                "t_project": ps["t_project"],
                "next_ms_threshold_gap": ms_obs["next_ms_threshold_gap"],
                "next_ms_net_payment":   ms_obs["next_ms_net_payment"],
                "next_ms_earliest_t":    ms_obs["next_ms_earliest_t"],
                "next_ms_is_final":      ms_obs["next_ms_is_final"],
                "retention_held":        ms_obs["retention_held"],
            })
        return state

    # ── GENERATORS ───────────────────────────────────────────

    def _sample_project(self, i: int) -> dict:
        cfg = self.cfg

        budget = max(1.0, sample(cfg["budget_dist"], cfg["budget_p1"],
                                 cfg["budget_p2"], cfg["budget_p3"], cfg["budget_p4"]))
        margin = max(0.0, sample(cfg["margin_dist"], cfg["margin_p1"],
                                 cfg["margin_p2"], cfg["margin_p3"], cfg["margin_p4"]))
        price = budget * (1 + margin)

        start = max(0, int(round(sample(cfg["start_dist"], cfg["start_p1"],
                                       cfg["start_p2"], cfg["start_p3"], cfg["start_p4"]))))
        duration = max(1, sample_int(cfg["duration_dist"], cfg["duration_p1"],
                                     cfg["duration_p2"], cfg["duration_p3"], cfg["duration_p4"]))
        finish = start + duration

        scurve_a = max(0.5, sample(cfg["scurve_a_dist"], cfg["scurve_a_p1"],
                                   cfg["scurve_a_p2"], cfg["scurve_a_p3"], cfg["scurve_a_p4"]))
        scurve_b = max(0.5, sample(cfg["scurve_b_dist"], cfg["scurve_b_p1"],
                                   cfg["scurve_b_p2"], cfg["scurve_b_p3"], cfg["scurve_b_p4"]))

        advance_percent = sample(cfg["advance_percent_dist"], cfg["advance_percent_p1"],
                                 cfg["advance_percent_p2"], cfg["advance_percent_p3"],
                                 cfg["advance_percent_p4"])
        advance_trigger = sample(cfg["advance_trigger_dist"], cfg["advance_trigger_p1"],
                                 cfg["advance_trigger_p2"], cfg["advance_trigger_p3"],
                                 cfg["advance_trigger_p4"])
        advance_recovery = sample(cfg["advance_recovery_dist"], cfg["advance_recovery_p1"],
                                  cfg["advance_recovery_p2"], cfg["advance_recovery_p3"],
                                  cfg["advance_recovery_p4"])
        retention_rate = sample(cfg["retention_rate_dist"], cfg["retention_rate_p1"],
                                cfg["retention_rate_p2"], cfg["retention_rate_p3"],
                                cfg["retention_rate_p4"])
        schedule_cap = max(1, sample_int(cfg["schedule_cap_dist"], cfg["schedule_cap_p1"],
                                         cfg["schedule_cap_p2"], cfg["schedule_cap_p3"],
                                         cfg["schedule_cap_p4"]))
        cost_cap = max(1.0, sample(cfg["cost_cap_dist"], cfg["cost_cap_p1"],
                                   cfg["cost_cap_p2"], cfg["cost_cap_p3"], cfg["cost_cap_p4"]))
        cure_length = max(1, sample_int(cfg["cure_length_dist"], cfg["cure_length_p1"],
                                        cfg["cure_length_p2"], cfg["cure_length_p3"],
                                        cfg["cure_length_p4"]))

        return {
            "i": i,
            "budget": budget,
            "price": price,
            "margin": margin,
            "start": start,
            "finish": finish,
            "duration": duration,
            "scurve_a": scurve_a,
            "scurve_b": scurve_b,
            "advance_percent": max(0.0, min(0.3, advance_percent)),
            "advance_trigger": max(0.0, min(1.0, advance_trigger)),
            "advance_recovery": max(0.0, min(1.0, advance_recovery)),
            "retention_rate": max(0.0, min(0.1, retention_rate)),
            "schedule_cap": schedule_cap,
            "cost_cap": cost_cap,
            "cure_length": cure_length,
        }

    def _sample_milestones(self, i: int, proj: dict) -> list:
        cfg = self.cfg
        n_ms = max(1, sample_int(cfg["n_milestones_dist"], cfg["n_milestones_p1"],
                                 cfg["n_milestones_p2"], cfg["n_milestones_p3"],
                                 cfg["n_milestones_p4"]))

        thresholds = [round((j + 1) / n_ms, 4) for j in range(n_ms)]
        thresholds[-1] = 1.0

        weights = [round(1.0 / n_ms, 6)] * n_ms
        weights[-1] = round(1.0 - sum(weights[:-1]), 6)

        milestones = []
        for j in range(n_ms):
            is_final = (j == n_ms - 1)
            if is_final:
                earliest_t = proj["finish"]
            else:
                earliest_t = proj["start"] + max(0, round(thresholds[j] * proj["duration"] * 0.5))
            milestones.append({
                "j": j,
                "threshold": thresholds[j],
                "earliest_t": earliest_t,
                "payment_weight": weights[j],
                "certified": False,
                "certified_t": None,
                "payment_released": None,
            })
        return milestones

    def _init_proj_state(self, proj: dict, milestones: list) -> dict:
        return {
            "status": None,
            "t_project": None,
            "progress": 0.0,
            "progress_plan": 0.0,
            "progress_increment": 0.0,
            "acwp": 0.0,
            "spi": 1.0,
            "cpi": 1.0,
            "eac": proj["budget"],
            "schedule_slip": 0.0,
            "cost_overrun": 0.0,
            "forecast_finish": float(proj["finish"]),
            "cure_remaining": proj["cure_length"],
            "advance_recovered": 0.0,
            "retention_held": 0.0,
            "retention_released": False,
        }

    # ── DB WRITERS ───────────────────────────────────────────

    def _write_profiles(self):
        for proj in self.projects:
            self.conn.execute("""
                INSERT INTO projects_profile
                    (episode_id, config_id, i, budget, price, margin,
                     start, finish, duration, scurve_a, scurve_b,
                     advance_percent, advance_trigger, advance_recovery,
                     retention_rate, schedule_cap, cost_cap, cure_length)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                self.episode_id, self.config_id, proj["i"],
                proj["budget"], proj["price"], proj["margin"],
                proj["start"], proj["finish"], proj["duration"],
                proj["scurve_a"], proj["scurve_b"],
                proj["advance_percent"], proj["advance_trigger"],
                proj["advance_recovery"], proj["retention_rate"],
                proj["schedule_cap"], proj["cost_cap"], proj["cure_length"]
            ))

        for i, ms_list in enumerate(self.milestones):
            for ms in ms_list:
                self.conn.execute("""
                    INSERT INTO milestones_profile
                        (episode_id, i, j, threshold, earliest_t, payment_weight)
                    VALUES (?,?,?,?,?,?)
                """, (self.episode_id, i, ms["j"],
                      ms["threshold"], ms["earliest_t"], ms["payment_weight"]))
        self.conn.commit()

    def _write_portfolio_row(self, inflow, outflow, reward, done):
        self.conn.execute("""
            INSERT INTO portfolios
                (episode_id, config_id, t_episode, method,
                 budget, inflow, outflow, reward, done)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            self.episode_id, self.config_id, self.t, self.method,
            self.budget, inflow, outflow, reward, done
        ))

    def _write_project_row(self, i, proj, ps, allocation, efficiency,
                           advance_amount, payment_net,
                           retention_release, settlement):
        self.conn.execute("""
            INSERT INTO projects_status
                (episode_id, i, t_episode, t_project, method, status,
                 allocation, efficiency,
                 progress, progress_plan, progress_increment,
                 spi, cpi, eac,
                 schedule_slip, cost_overrun, forecast_finish,
                 cure_remaining,
                 advance_amount, payment_net, retention_release, settlement)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            self.episode_id, i, self.t, ps["t_project"], self.method, ps["status"],
            allocation, efficiency,
            ps["progress"], ps["progress_plan"], ps["progress_increment"],
            ps["spi"], ps["cpi"], ps["eac"],
            ps["schedule_slip"], ps["cost_overrun"], ps["forecast_finish"],
            ps["cure_remaining"],
            advance_amount, payment_net, retention_release, settlement
        ))

    def _write_milestone_status(self, i, j, ms):
        self.conn.execute("""
            INSERT OR REPLACE INTO milestones_status
                (episode_id, i, j, method, certified, certified_t, payment_released)
            VALUES (?,?,?,?,?,?,?)
        """, (
            self.episode_id, i, j, self.method,
            int(ms["certified"]), ms["certified_t"], ms["payment_released"]
        ))