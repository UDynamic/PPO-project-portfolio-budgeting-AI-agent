# src/env/evm.py
#
# Earned Value Management signal computation.
#
# All functions are pure (no state, no randomness, no I/O).
# They operate on scalar floats and return scalars or a dict.
#
# Imported by env.py only.
# beta_cdf / earned_schedule are re-exported from sampler to avoid
# duplicating the numerical code; this module does not redefine them.
#
# Formulas match the domain model in the spec exactly:
#
#   SPI(t)  = ES / AT                  (Earned Schedule method)
#   CPI     = BCWP / ACWP
#   TCPI    = (BAC − BCWP) / (BAC − ACWP)
#   EAC     = ACWP + (BAC − BCWP) / (CPI × SPI(t))
#   forecast_finish = planned_start + planned_duration / SPI(t)
#   schedule_slip   = forecast_finish − planned_finish
#
# Key name conventions (match schema exactly):
#   proj["bac"]               Budget at Completion
#   proj["planned_start"]     planned start period
#   proj["planned_finish"]    planned finish period
#   proj["planned_duration"]  planned duration in periods
#   proj["scurve_a/b"]        Beta shape parameters
#   proj["cost_overrun_cap"]  EAC/BAC fallback ceiling
#   proj["finish_delay_cap"]  schedule slip fallback ceiling
#   ps["outflow"]             cumulative cost (ACWP) — schema: outflow
#   ps["progress_actual"]     cumulative progress (BCWP = progress × BAC)

from __future__ import annotations

from sampler import earned_schedule  # re-use; no duplication


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_spi(progress: float, t_project: int,
                duration: int, scurve_a: float, scurve_b: float) -> float:
    """
    SPI(t) via the Earned Schedule method.

    ES = the planned time at which the s-curve equals current actual progress.
    AT = t_project (actual elapsed periods on the project clock).

    SPI(t) < 1  → behind schedule.
    SPI(t) > 1  → ahead of schedule.
    SPI(t) continues to degrade past the planned finish, correcting the
    classical SPI freeze pathology.
    """
    es = earned_schedule(progress, duration, scurve_a, scurve_b)
    at = float(t_project)
    return es / at if at > 1e-9 else 1.0


def compute_cpi(bcwp: float, acwp: float) -> float:
    """
    Cost Performance Index.

    CPI = BCWP / ACWP.
    Returns 1.0 when ACWP is effectively zero (no spend yet).
    """
    return bcwp / acwp if acwp > 1e-9 else 1.0


def compute_eac(acwp: float, bac: float, bcwp: float,
                cpi: float, spi: float, cost_overrun_cap: float) -> float:
    """
    Estimate at Completion using the composite CPI × SPI denominator.

    EAC = ACWP + (BAC − BCWP) / (CPI × SPI(t))

    Falls back to bac × cost_overrun_cap when the composite is effectively
    zero (degenerate case: no spend and/or no schedule performance).
    """
    composite = cpi * spi
    if composite > 1e-9:
        return acwp + (bac - bcwp) / composite
    return bac * cost_overrun_cap


def compute_tcpi(bac: float, bcwp: float, acwp: float) -> float:
    """
    To-Complete Performance Index.

    TCPI = (BAC − BCWP) / (BAC − ACWP)

    Returns 0.0 when work remaining is zero (project effectively complete).
    Returns inf when budget is exhausted but work remains.
    """
    work_remaining   = bac - bcwp
    budget_remaining = bac - acwp

    if budget_remaining > 1e-9:
        return work_remaining / budget_remaining
    return 0.0 if work_remaining <= 0.0 else float("inf")


def compute_forecast_finish(planned_start: int, planned_duration: int,
                             spi: float,
                             planned_finish: int,
                             finish_delay_cap: int) -> float:
    """
    Forecast completion date.

    forecast_finish = planned_start + planned_duration / SPI(t)

    Falls back to planned_finish + finish_delay_cap + 1 when SPI is
    effectively zero (signals terminal delay — further past the deadline
    than the contract allows).
    """
    if spi > 1e-9:
        return planned_start + planned_duration / spi
    return float(planned_finish + finish_delay_cap + 1)


def compute_schedule_slip(forecast_finish: float, planned_finish: int) -> float:
    """
    schedule_slip = forecast_finish − planned_finish.

    Positive → forecast is later than planned (delay).
    Negative → forecast is earlier (ahead of plan).
    """
    return forecast_finish - float(planned_finish)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE UPDATE  (called once per active project per step)
# ─────────────────────────────────────────────────────────────────────────────

def update_evm(ps: dict, proj: dict) -> None:
    """
    Recompute all EVM signals in *ps* in-place.

    Parameters
    ----------
    ps   : project state dict (mutated in-place)
    proj : project parameter dict (read-only; keys match projects_profile)

    Reads from ps
    -------------
    ps["progress_actual"]   cumulative progress fraction [0, 1]
    ps["outflow"]           cumulative cost spent (ACWP)
    ps["t_project"]         elapsed periods on the project clock
    ps["progress_plan_t"]   planned progress at t (set by caller before this)

    Writes to ps
    ------------
    ps["spi"], ps["cpi"], ps["tcpi"], ps["eac"],
    ps["projected_finish"], ps["projected_finish_delay"],
    ps["projected_cost_overrun"], ps["progress_delay_t"]
    """
    bac  = proj["bac"]
    bcwp = ps["progress_actual"] * bac
    acwp = ps["outflow"]

    spi = compute_spi(
        ps["progress_actual"],
        ps["t_project"],
        proj["planned_duration"],
        proj["scurve_a"],
        proj["scurve_b"],
    )
    cpi  = compute_cpi(bcwp, acwp)
    eac  = compute_eac(acwp, bac, bcwp, cpi, spi, proj["cost_overrun_cap"])
    tcpi = compute_tcpi(bac, bcwp, acwp)

    projected_finish = compute_forecast_finish(
        proj["planned_start"],
        proj["planned_duration"],
        spi,
        proj["planned_finish"],
        proj["finish_delay_cap"],
    )
    projected_finish_delay = compute_schedule_slip(
        projected_finish, proj["planned_finish"]
    )

    ps["spi"]                   = spi
    ps["cpi"]                   = cpi
    ps["eac"]                   = eac
    ps["tcpi"]                  = tcpi
    ps["projected_finish"]      = projected_finish
    ps["projected_finish_delay"]= projected_finish_delay
    ps["projected_cost_overrun"]= eac / bac          # ratio; compared against cost_overrun_cap
    ps["progress_delay_t"]      = ps["progress_plan_t"] - ps["progress_actual"]