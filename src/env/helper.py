# src/env/helper.py
#
# Pure, stateless helpers for PortfolioBudgetingEnv.
# No I/O, no global state, no randomness except through the rng argument.
#
# Sections
# --------
#   DISTRIBUTIONS   — sample / sample_int
#   S-CURVE         — beta_cdf / planned_progress / earned_schedule
#   SAMPLERS        — sample_project / sample_milestones / sample_efficiency
#   EVM             — compute_* / update_evm
#   BREACHES        — evaluate_breaches / update_tolerance / check_termination
#   PAYMENTS        — check_certifications / deliver_payments / compute_termination_settlement
#   RENDER          — reset_history / render

from __future__ import annotations
import math

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sample(rng, dist: str, p1, p2=None, p3=None, p4=None) -> float:
    """Draw one float from the named distribution using rng (numpy Generator).

    fixed            → p1
    uniform          → U(p1, p2)
    normal           → N(p1, p2)
    lognormal        → LogN(p1, p2)
    triangular       → Tri(p1=low, p2=high, p3=mode)
    truncated_normal → N(p1, p2) clipped to [p3, p4]
    beta             → Beta(p1, p2)
    categorical      → p1 chance of int(p3), p2 chance of int(p4), else int(p3)
    """
    if dist == "fixed":             return float(p1)
    if dist == "uniform":           return float(rng.uniform(p1, p2))
    if dist == "normal":            return float(rng.normal(p1, p2))
    if dist == "lognormal":         return float(rng.lognormal(p1, p2))
    if dist == "triangular":        return float(rng.triangular(p1, p3, p2))
    if dist == "beta":              return float(rng.beta(p1, p2))
    if dist == "categorical":
        r = float(rng.random())
        if r < p1:        return int(p3)
        if r < p1 + p2:   return int(p4)
        return int(p3)
    if dist == "truncated_normal":
        while True:
            v = float(rng.normal(p1, p2))
            if p3 <= v <= p4:
                return v
    raise ValueError(f"Unknown distribution: {dist!r}")


def sample_int(rng, dist: str, p1, p2=None, p3=None, p4=None) -> int:
    """Round the float draw to the nearest integer; always >= 1."""
    return max(1, round(sample(rng, dist, p1, p2, p3, p4)))


# ═══════════════════════════════════════════════════════════════════════════════
# S-CURVE  (Beta CDF — no scipy)
# ═══════════════════════════════════════════════════════════════════════════════

def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a,b) via 200-point midpoint quadrature.
    Error < 1e-4 for a,b ∈ [0.5, 10]."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    dx    = x / 200
    total = 0.0
    for k in range(200):
        t      = (k + 0.5) * dx
        total += (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0)) * dx
    B = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return min(1.0, total / B)


def planned_progress(t_project: int, duration: int, a: float, b: float) -> float:
    """Planned cumulative progress at project-period t_project: BetaCDF(t/duration, a, b)."""
    return beta_cdf(t_project / duration, a, b)


def earned_schedule(progress_actual: float, duration: int, a: float, b: float) -> float:
    """Earned Schedule via binary search over [0, duration] (50 iterations)."""
    if progress_actual <= 0.0: return 0.0
    if progress_actual >= 1.0: return float(duration)
    lo, hi = 0.0, float(duration)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid / duration, a, b) < progress_actual:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _s(rng, cfg, key):
    """Shorthand: sample one field using cfg[key_dist/p1..p4] pattern."""
    return sample(rng, cfg[f"{key}_dist"],
                  cfg[f"{key}_p1"], cfg[f"{key}_p2"],
                  cfg[f"{key}_p3"], cfg[f"{key}_p4"])

def _si(rng, cfg, key):
    return sample_int(rng, cfg[f"{key}_dist"],
                      cfg[f"{key}_p1"], cfg[f"{key}_p2"],
                      cfg[f"{key}_p3"], cfg[f"{key}_p4"])


def sample_project(rng, cfg: dict, i: int) -> dict:
    """Draw all project-level parameters from cfg distributions."""
    bac              = max(1.0, _s(rng, cfg, "bac"))
    profit_percent   = max(0.0, _s(rng, cfg, "profit_percent"))
    price            = bac * (1.0 + profit_percent)
    planned_start    = max(0, int(round(_s(rng, cfg, "planned_start"))))
    planned_duration = max(1, _si(rng, cfg, "planned_duration"))
    planned_finish   = planned_start + planned_duration
    scurve_a         = max(0.5, _s(rng, cfg, "scurve_a"))
    scurve_b         = max(0.5, _s(rng, cfg, "scurve_b"))
    advance_percent  = max(0.0, min(0.3, _s(rng, cfg, "advance_percent")))
    advance_trigger  = max(0.0, min(1.0, _s(rng, cfg, "advance_trigger")))
    advance_recovery = max(0.0, min(1.0, _s(rng, cfg, "advance_recovery")))
    retention_rate   = max(0.0, min(0.1, _s(rng, cfg, "retention_rate")))
    progress_delay_cap   = max(0.0, min(1.0, _s(rng, cfg, "progress_delay_cap")))
    finish_delay_cap     = max(1, _si(rng, cfg, "finish_delay_cap"))
    cost_overrun_cap     = max(1.0, _s(rng, cfg, "cost_overrun_cap"))
    termination_tolerance = max(1, _si(rng, cfg, "termination_tolerance"))
    return {
        "i": i,
        "planned_start": planned_start, "planned_finish": planned_finish,
        "planned_duration": planned_duration,
        "bac": bac, "profit_percent": profit_percent, "price": price,
        "scurve_a": scurve_a, "scurve_b": scurve_b,
        "advance_percent": advance_percent, "advance_trigger": advance_trigger,
        "advance_recovery": advance_recovery, "retention_rate": retention_rate,
        "progress_delay_cap": progress_delay_cap,
        "finish_delay_cap": finish_delay_cap,
        "cost_overrun_cap": cost_overrun_cap,
        "termination_tolerance": termination_tolerance,
    }


def sample_milestones(rng, cfg: dict, proj: dict) -> list[dict]:
    """Build the full milestone profile (j=0 advance + j=1..n milestones)."""
    n_ms = max(1, _si(rng, cfg, "n_milestones"))

    thresholds = [round((j + 1) / n_ms, 4) for j in range(n_ms)]
    thresholds[-1] = 1.0
    weights = [round(1.0 / n_ms, 6)] * n_ms
    weights[-1] = round(1.0 - sum(weights[:-1]), 6)

    ts_frac = max(0.0, min(1.0, _s(rng, cfg, "timestep_threshold")))

    price          = proj["price"]
    advance_amount = proj["advance_percent"] * price
    adv_rec_rate   = proj["advance_recovery"]
    ret_rate       = proj["retention_rate"]

    milestones = [{
        "j": 0,
        "progress_threshold": 0.0,
        "timestep_threshold": proj["planned_start"],
        "payment_weight": 0.0,
        "gross_payment": advance_amount,
        "advance_recovery": 0.0,
        "advance_recovery_remain": advance_amount,
        "retention_withheld": 0.0,
        "retention_released": 0.0,
        "net_payment": advance_amount,
    }]

    remaining_advance    = advance_amount
    cumulative_retention = 0.0

    for idx in range(n_ms):
        j        = idx + 1
        is_final = (idx == n_ms - 1)

        ts = (proj["planned_finish"] if is_final else
              proj["planned_start"] + max(1, round(
                  thresholds[idx] * proj["planned_duration"] * ts_frac)))

        gross    = weights[idx] * price
        recovery = remaining_advance if is_final else min(gross * adv_rec_rate, remaining_advance)
        remaining_advance -= recovery

        ret_held = gross * ret_rate
        cumulative_retention += ret_held
        ret_released = cumulative_retention if is_final else 0.0

        milestones.append({
            "j": j,
            "progress_threshold":      thresholds[idx],
            "timestep_threshold":      ts,
            "payment_weight":          weights[idx],
            "gross_payment":           gross,
            "advance_recovery":        recovery,
            "advance_recovery_remain": remaining_advance,
            "retention_withheld":      ret_held,
            "retention_released":      ret_released,
            "net_payment":             gross - recovery - ret_held + ret_released,
        })

    return milestones


def sample_efficiency(rng, cfg: dict) -> float:
    """Draw period efficiency η; clamped to min 0.01."""
    return max(0.01, float(_s(rng, cfg, "efficiency")))


# ═══════════════════════════════════════════════════════════════════════════════
# EVM
# ═══════════════════════════════════════════════════════════════════════════════

def _spi(progress: float, t_project: int, duration: int, a: float, b: float) -> float:
    es = earned_schedule(progress, duration, a, b)
    at = float(t_project)
    return es / at if at > 1e-9 else 1.0

def _cpi(bcwp: float, acwp: float) -> float:
    return bcwp / acwp if acwp > 1e-9 else 1.0

def _eac(acwp: float, bac: float, bcwp: float, cpi: float, spi: float,
         cost_overrun_cap: float) -> float:
    composite = cpi * spi
    return (acwp + (bac - bcwp) / composite) if composite > 1e-9 else bac * cost_overrun_cap

def _tcpi(bac: float, bcwp: float, acwp: float) -> float:
    work_rem   = bac - bcwp
    budget_rem = bac - acwp
    if budget_rem > 1e-9: return work_rem / budget_rem
    return 0.0 if work_rem <= 0.0 else float("inf")

def _forecast_finish(planned_start: int, planned_duration: int, spi: float,
                     planned_finish: int, finish_delay_cap: int) -> float:
    if spi > 1e-9:
        return planned_start + planned_duration / spi
    return float(planned_finish + finish_delay_cap + 1)


def update_evm(ps: dict, proj: dict) -> None:
    """Recompute all EVM signals in ps in-place."""
    bac  = proj["bac"]
    bcwp = ps["progress_actual"] * bac
    acwp = ps["outflow"]

    spi  = _spi(ps["progress_actual"], ps["t_project"],
                proj["planned_duration"], proj["scurve_a"], proj["scurve_b"])
    cpi  = _cpi(bcwp, acwp)
    eac  = _eac(acwp, bac, bcwp, cpi, spi, proj["cost_overrun_cap"])
    tcpi = _tcpi(bac, bcwp, acwp)

    proj_finish = _forecast_finish(proj["planned_start"], proj["planned_duration"],
                                   spi, proj["planned_finish"], proj["finish_delay_cap"])

    ps["spi"]                    = spi
    ps["cpi"]                    = cpi
    ps["eac"]                    = eac
    ps["tcpi"]                   = tcpi
    ps["projected_finish"]       = proj_finish
    ps["projected_finish_delay"] = proj_finish - float(proj["planned_finish"])
    ps["projected_cost_overrun"] = eac / bac
    ps["progress_delay_t"]       = ps["progress_plan_t"] - ps["progress_actual"]


# ═══════════════════════════════════════════════════════════════════════════════
# BREACHES
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_breaches(ps: dict, proj: dict) -> dict:
    """Evaluate breach flags (pre- or post-allocation). over_duration_window always False here."""
    op = ps["progress_delay_t"]       > proj["progress_delay_cap"]
    of = ps["projected_finish_delay"] > proj["finish_delay_cap"]
    oc = ps["projected_cost_overrun"] > proj["cost_overrun_cap"]
    return {
        "over_progress_delay":  op,
        "over_finish_delay":    of,
        "over_cost_overrun":    oc,
        "over_any":             op or of or oc,
        "over_duration_window": False,
    }


def update_tolerance(ps: dict, proj: dict, flags: dict) -> None:
    """Decrement tolerance on any breach; reset to max when fully healthy."""
    if flags["over_any"]:
        ps["tolerance_remain"] = max(0, ps["tolerance_remain"] - 1)
    else:
        ps["tolerance_remain"] = proj["termination_tolerance"]


def check_termination(t: int, ps: dict, proj: dict,
                      flags: dict) -> tuple[bool, bool]:
    """Return (terminated, over_duration_window). Mutates flags['over_duration_window']."""
    odw = t >= proj["planned_finish"] + proj["finish_delay_cap"]
    flags["over_duration_window"] = odw
    if odw:
        return True, True
    if ps["tolerance_remain"] <= 0:
        return True, False
    return False, False


# ═══════════════════════════════════════════════════════════════════════════════
# PAYMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def check_certifications(t: int, ps: dict,
                         milestones: list[dict],
                         milestone_state: list[dict]) -> list[int]:
    """Return j indices of milestones qualifying for certification this period."""
    result = []
    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] == 0 or ms_state["certified"]:
            continue
        if (ps["progress_actual"] >= ms["progress_threshold"]
                and t >= ms["timestep_threshold"]):
            result.append(ms["j"])
    return result


def deliver_payments(certified_js: list[int], t: int, ps: dict,
                     milestones: list[dict],
                     milestone_state: list[dict]) -> float:
    """Deliver net payments for certified milestones; mutate ms_state and ps['inflow']."""
    total = 0.0
    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] not in certified_js:
            continue
        net = ms["net_payment"]
        ms_state.update(certified=True, certified_t=t, payment_released=net)
        ps["inflow"] += net
        total        += net
    return total


def compute_termination_settlement(proj: dict, ps: dict) -> float:
    """settlement = progress_actual × price − cumulative_inflow."""
    return ps["progress_actual"] * proj["price"] - ps["inflow"]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

_W, _LW, _VW, _GAP = 80, 17, 12, 3

def _sep(c="═"):  return c * _W

def _row(ll, lv, rl="", rv=""):
    left = f"  {ll:<{_LW}} : {lv:>{_VW}}"
    return left + (f"{' '*_GAP}{rl:<{_LW}} : {rv:>{_VW}}" if rl or rv else "")

def _fmt(v, d=2): return f"{v:,.{d}f}"

def _tag(s):
    return {"active":"ACTIVE","pending":"PENDING",
            "completed":"COMPLETED","terminated":"TERMINATED"}.get(
            (s or "").lower(), (s or "UNKNOWN").upper())


_episode_id = ""
_n_projects = 0


def reset_history(episode_id: str, n_projects: int) -> None:
    global _episode_id, _n_projects
    _episode_id, _n_projects = episode_id, n_projects


def render(t: int, budget: float, horizon: int, initial_budget: float,
           projects: list[dict], proj_state: list[dict],
           milestones: list[list[dict]], milestone_state: list[list[dict]],
           episode_id: str = "", net_cashflow: float = 0.0,
           cum_reward: float = 0.0, **_) -> None:

    print()
    print(_sep())
    print(_row(f"PERIOD {t} / {horizon}", "", "net_cashflow", _fmt(net_cashflow)))
    print(_row("reward", f"{cum_reward:+.4f}", "budget_available", _fmt(budget)))
    print(_sep())

    for proj, ps, ms_list, ms_sl in zip(projects, proj_state, milestones, milestone_state):
        tol     = ps.get("tolerance_remain", proj["termination_tolerance"])
        tol_max = proj["termination_tolerance"]
        print(f"  [{_tag(ps.get('status','pending'))}]  Project {proj['i']}")
        print(_row("BAC",    _fmt(proj["bac"]),   "Price",             _fmt(proj["price"])))
        print(_row("t_proj", str(ps.get("t_project", 0)),
                   "tolerance_remain", f"{tol}/{tol_max}"))
        print(_row("inflow", _fmt(ps.get("inflow", 0.0)),
                   "outflow", _fmt(ps.get("outflow", 0.0))))
        print(_sep("─"))
        print(_row("catchup_t",      _fmt(ps.get("catchup_alloc_t",      0.0)),
                   "catchup_next_t", _fmt(ps.get("catchup_alloc_next_t", 0.0))))
        print(_row("reach_plan_t",      _fmt(ps.get("reach_plan_t",      0.0)),
                   "reach_plan_next_t", _fmt(ps.get("reach_plan_next_t", 0.0))))
        print(_sep("─"))
        j       = ps.get("target_milestone_j")
        print(_row("target",         f"j={j}" if j is not None else "none",
                   "net_payment",    _fmt(ps.get("target_net_payment",    0.0))))
        print(_row("progress_gap",   _fmt(ps.get("target_progress_gap",   0.0), 4),
                   "timestep_gap",   str(ps.get("target_timestep_gap",   0))))
        print(_row("required_alloc", _fmt(ps.get("target_required_alloc", 0.0)),
                   "target_npv",     _fmt(ps.get("target_npv",            0.0), 4)))
        print(_sep())