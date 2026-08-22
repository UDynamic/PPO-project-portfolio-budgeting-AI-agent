# src/env/evm.py
#
# Earned Value Management signal computation.
#
# All functions are pure (no state, no randomness, no I/O).
# They operate on scalar floats and return scalars or a dict.
#
# Imported by env.py (Phase 4) only.
# beta_cdf / earned_schedule are re-exported from sampler to avoid
# duplicating the numerical code; this module does not redefine them.
#
# Formulas match the domain model in the spec exactly:
#
#   SPI(t)  = ES / AT                  (Earned Schedule method)
#   CPI     = BCWP / ACWP
#   TCPI    = (BAC − BCWP) / (BAC − ACWP)
#   EAC     = ACWP + (BAC − BCWP) / (CPI × SPI(t))
#   forecast_finish = start + duration / SPI(t)
#   schedule_slip   = forecast_finish − planned_finish

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
                cpi: float, spi: float, cost_cap: float) -> float:
    """
    Estimate at Completion using the composite CPI × SPI denominator.

    EAC = ACWP + (BAC − BCWP) / (CPI × SPI(t))

    Falls back to bac × cost_cap when the composite is effectively zero
    (degenerate case: no spend and/or no schedule performance).
    """
    composite = cpi * spi
    if composite > 1e-9:
        return acwp + (bac - bcwp) / composite
    return bac * cost_cap


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


def compute_forecast_finish(start: int, duration: int,
                             spi: float,
                             finish: int, schedule_cap: int) -> float:
    """
    Forecast completion date.

    forecast_finish = start + duration / SPI(t)

    Falls back to finish + schedule_cap + 1 when SPI is effectively zero
    (signals terminal delay — further past the deadline than allowed).
    """
    if spi > 1e-9:
        return start + duration / spi
    return float(finish + schedule_cap + 1)


def compute_schedule_slip(forecast_finish: float, planned_finish: int) -> float:
    """
    schedule_slip = forecast_finish − planned_finish.

    Positive → forecast is later than planned (delay).
    Negative → forecast is earlier (ahead of plan).
    """
    return forecast_finish - float(planned_finish)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE UPDATE  (called once per active project per step, Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def update_evm(ps: dict, proj: dict) -> None:
    """
    Recompute all EVM signals in *ps* in-place after Phase 3 has updated
    progress and cumulative cost.

    Parameters
    ----------
    ps   : project state dict (mutated in-place)
    proj : project parameter dict (read-only)

    Updates
    -------
    ps["spi"], ps["cpi"], ps["tcpi"], ps["eac"],
    ps["forecast_finish"], ps["schedule_slip"], ps["cost_overrun"]
    ps["plan_deviation"]  (progress_plan already set by Phase 4 caller)
    """
    bac  = proj["budget"]
    bcwp = ps["progress"] * bac
    acwp = ps["acwp"]

    spi = compute_spi(
        ps["progress"], ps["t_project"],
        proj["duration"], proj["scurve_a"], proj["scurve_b"]
    )
    cpi = compute_cpi(bcwp, acwp)
    eac = compute_eac(acwp, bac, bcwp, cpi, spi, proj["cost_cap"])
    tcpi = compute_tcpi(bac, bcwp, acwp)

    forecast_finish = compute_forecast_finish(
        proj["start"], proj["duration"], spi,
        proj["finish"], proj["schedule_cap"]
    )
    schedule_slip = compute_schedule_slip(forecast_finish, proj["finish"])

    ps["spi"]            = spi
    ps["cpi"]            = cpi
    ps["eac"]            = eac
    ps["tcpi"]           = tcpi
    ps["forecast_finish"]= forecast_finish
    ps["schedule_slip"]  = schedule_slip
    ps["cost_overrun"]   = eac - bac
    ps["plan_deviation"] = ps["progress_plan"] - ps["progress"]