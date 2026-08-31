# src/env/sampler.py
#
# All stochastic sampling for the environment.
#
# Design rules:
#   - Every random draw goes through np_random (a numpy.random.Generator
#     passed in from env.reset via super().reset(seed=seed)).
#   - No import of `random` anywhere in this file.
#   - beta_cdf and earned_schedule are deterministic; they live here because
#     sampler.py is the only module that needs them besides evm.py, which
#     imports them directly.  No circular imports.
#   - advance_trigger is accepted as a parameter (it is stored in the project
#     profile for schema completeness) but is NOT sampled in sample_project;
#     the caller passes the raw cfg value through.  Wiring it into
#     certification logic is a planned extension.
#
# Milestone profile structure (j index):
#   j = 0        → advance payment (t=planned_start, no threshold, no weight)
#   j = 1..n-1   → interim milestones (threshold, weight, retention withheld)
#   j = n        → final milestone (threshold=1.0, retention withheld + retention released)
#
# Payment breakdown per milestone:
#   advance:
#     gross = advance_percent × price
#     advance_recovery = 0, retention_withheld = 0, retention_released = 0
#     net = gross
#
#   interim (j=1..n-1):
#     gross              = payment_weight × price
#     advance_recovery   = min(gross × advance_recovery_rate, remaining_advance)
#     retention_withheld = gross × retention_rate
#     retention_released = 0
#     net                = gross - advance_recovery - retention_withheld
#
#   final (j=n):
#     gross              = payment_weight × price
#     advance_recovery   = min(gross × advance_recovery_rate, remaining_advance)
#     retention_withheld = gross × retention_rate   ← same rule as interim
#     retention_released = sum of ALL retention_withheld (j=1..n, including final)
#     net                = gross - advance_recovery - retention_withheld + retention_released

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CORE DISTRIBUTION SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample(rng, dist: str, p1, p2=None, p3=None, p4=None) -> float:
    """
    Draw one float from the named distribution using *rng* (numpy Generator).

    Supported distributions
    -----------------------
    fixed           → p1  (deterministic)
    uniform         → U(p1, p2)
    normal          → N(p1, p2)          p1=mean, p2=std
    lognormal       → LogN(p1, p2)       p1=mu, p2=sigma (log-space)
    triangular      → Tri(p1, p3, p2)    p1=low, p2=high, p3=mode
    truncated_normal→ N(p1,p2) clipped to [p3, p4]
    beta            → Beta(p1, p2)       result in [0, 1]
    categorical     → p1 chance of int(p3), p2 chance of int(p4),
                      else int(p3)
    """
    if dist == "fixed":
        return float(p1)

    elif dist == "uniform":
        return float(rng.uniform(p1, p2))

    elif dist == "normal":
        return float(rng.normal(p1, p2))

    elif dist == "lognormal":
        return float(rng.lognormal(p1, p2))

    elif dist == "triangular":
        # numpy triangular: left, mode, right
        return float(rng.triangular(p1, p3, p2))

    elif dist == "truncated_normal":
        # Rejection sampling — acceptable because [p3,p4] is expected to
        # cover several sigma and the loop terminates quickly in practice.
        while True:
            v = float(rng.normal(p1, p2))
            if p3 <= v <= p4:
                return v

    elif dist == "beta":
        return float(rng.beta(p1, p2))

    elif dist == "categorical":
        r = float(rng.random())
        if r < p1:
            return int(p3)
        elif r < p1 + p2:
            return int(p4)
        else:
            return int(p3)

    else:
        raise ValueError(f"Unknown distribution: {dist!r}")


def sample_int(rng, dist: str, p1, p2=None, p3=None, p4=None) -> int:
    """Round the float draw to the nearest integer; always >= 1."""
    return max(1, round(sample(rng, dist, p1, p2, p3, p4)))


# ─────────────────────────────────────────────────────────────────────────────
# S-CURVE  (Beta CDF — no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def beta_cdf(x: float, a: float, b: float) -> float:
    """
    Regularised incomplete beta function I_x(a, b) via midpoint quadrature.

    200 sub-intervals give < 1e-4 absolute error for the (a, b) ranges used
    by the s-curve (a, b ∈ [0.5, 10]).  No external dependencies.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    steps = 200
    dx    = x / steps
    total = 0.0
    for k in range(steps):
        t      = (k + 0.5) * dx
        total += (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0)) * dx

    B = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return min(1.0, total / B)


def planned_progress(t_project: int, duration: int,
                     a: float, b: float) -> float:
    """
    Planned cumulative progress at the end of project-period *t_project*.

    P̄_i(t) = BetaCDF(t_project / duration, a, b)
    """
    return beta_cdf(t_project / duration, a, b)


def earned_schedule(progress_actual: float, duration: int,
                    a: float, b: float) -> float:
    """
    Earned Schedule (ES) — the planned time at which the s-curve would
    have reached *progress_actual*.

    Inverts the Beta CDF by binary search over [0, duration].
    50 iterations → precision < duration / 2^50 ≈ negligible.

    ES / AT gives SPI(t), which continues to degrade after the planned
    finish date even when progress is frozen — correcting the classical
    SPI freeze pathology.
    """
    if progress_actual <= 0.0:
        return 0.0
    if progress_actual >= 1.0:
        return float(duration)

    lo, hi = 0.0, float(duration)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid / duration, a, b) < progress_actual:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_project(rng, cfg: dict, i: int) -> dict:
    """
    Draw all project-level parameters from *cfg* distributions.

    Keys match projects_profile schema columns exactly.
    DB-only columns (episode_id, config_id) are not included here —
    they are added by db_logger at write time.

    advance_trigger is sampled and stored (schema requires it) but is NOT
    currently used in certification logic. Wiring it into certification
    is a planned extension.
    """
    bac = max(1.0, sample(rng, cfg["bac_dist"],
                          cfg["bac_p1"], cfg["bac_p2"],
                          cfg["bac_p3"], cfg["bac_p4"]))

    profit_percent = max(0.0, sample(rng, cfg["profit_percent_dist"],
                                     cfg["profit_percent_p1"], cfg["profit_percent_p2"],
                                     cfg["profit_percent_p3"], cfg["profit_percent_p4"]))
    price = bac * (1.0 + profit_percent)

    planned_start    = max(0, int(round(sample(
        rng, cfg["planned_start_dist"],
        cfg["planned_start_p1"], cfg["planned_start_p2"],
        cfg["planned_start_p3"], cfg["planned_start_p4"]
    ))))
    planned_duration = max(1, sample_int(
        rng, cfg["planned_duration_dist"],
        cfg["planned_duration_p1"], cfg["planned_duration_p2"],
        cfg["planned_duration_p3"], cfg["planned_duration_p4"]
    ))
    planned_finish = planned_start + planned_duration

    scurve_a = max(0.5, sample(rng, cfg["scurve_a_dist"],
                               cfg["scurve_a_p1"], cfg["scurve_a_p2"],
                               cfg["scurve_a_p3"], cfg["scurve_a_p4"]))
    scurve_b = max(0.5, sample(rng, cfg["scurve_b_dist"],
                               cfg["scurve_b_p1"], cfg["scurve_b_p2"],
                               cfg["scurve_b_p3"], cfg["scurve_b_p4"]))

    advance_percent = max(0.0, min(0.3, sample(
        rng, cfg["advance_percent_dist"],
        cfg["advance_percent_p1"], cfg["advance_percent_p2"],
        cfg["advance_percent_p3"], cfg["advance_percent_p4"]
    )))

    # Sampled for profile storage; not yet wired into certification.
    advance_trigger = max(0.0, min(1.0, sample(
        rng, cfg["advance_trigger_dist"],
        cfg["advance_trigger_p1"], cfg["advance_trigger_p2"],
        cfg["advance_trigger_p3"], cfg["advance_trigger_p4"]
    )))

    advance_recovery = max(0.0, min(1.0, sample(
        rng, cfg["advance_recovery_dist"],
        cfg["advance_recovery_p1"], cfg["advance_recovery_p2"],
        cfg["advance_recovery_p3"], cfg["advance_recovery_p4"]
    )))

    retention_rate = max(0.0, min(0.1, sample(
        rng, cfg["retention_rate_dist"],
        cfg["retention_rate_p1"], cfg["retention_rate_p2"],
        cfg["retention_rate_p3"], cfg["retention_rate_p4"]
    )))

    progress_delay_cap = max(0.0, min(1.0, sample(
        rng, cfg["progress_delay_cap_dist"],
        cfg["progress_delay_cap_p1"], cfg["progress_delay_cap_p2"],
        cfg["progress_delay_cap_p3"], cfg["progress_delay_cap_p4"]
    )))

    finish_delay_cap = max(1, sample_int(
        rng, cfg["finish_delay_cap_dist"],
        cfg["finish_delay_cap_p1"], cfg["finish_delay_cap_p2"],
        cfg["finish_delay_cap_p3"], cfg["finish_delay_cap_p4"]
    ))

    cost_overrun_cap = max(1.0, sample(
        rng, cfg["cost_overrun_cap_dist"],
        cfg["cost_overrun_cap_p1"], cfg["cost_overrun_cap_p2"],
        cfg["cost_overrun_cap_p3"], cfg["cost_overrun_cap_p4"]
    ))

    termination_tolerance = max(1, sample_int(
        rng, cfg["termination_tolerance_dist"],
        cfg["termination_tolerance_p1"], cfg["termination_tolerance_p2"],
        cfg["termination_tolerance_p3"], cfg["termination_tolerance_p4"]
    ))

    return {
        "i":                      i,
        "planned_start":          planned_start,
        "planned_finish":         planned_finish,
        "planned_duration":       planned_duration,
        "bac":                    bac,
        "profit_percent":         profit_percent,
        "price":                  price,
        "scurve_a":               scurve_a,
        "scurve_b":               scurve_b,
        "advance_percent":        advance_percent,
        "advance_trigger":        advance_trigger,
        "advance_recovery":       advance_recovery,
        "retention_rate":         retention_rate,
        "progress_delay_cap":     progress_delay_cap,
        "finish_delay_cap":       finish_delay_cap,
        "cost_overrun_cap":       cost_overrun_cap,
        "termination_tolerance":  termination_tolerance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_milestones(rng, cfg: dict, proj: dict) -> list[dict]:
    """
    Build the full milestone profile for *proj*, including advance (j=0)
    and the final milestone with retention release (j=n).

    j index structure
    -----------------
    j = 0        advance payment at planned_start
    j = 1..n-1   interim milestones
    j = n        final milestone (progress_threshold=1.0)

    Payment breakdown
    -----------------
    All payment values are precomputed here so db_logger can write
    milestones_profile without any financial logic.

    advance (j=0):
      gross_payment           = advance_percent × price
      advance_recovery        = 0
      advance_recovery_remain = advance_amount  (full amount outstanding)
      retention_withheld      = 0
      retention_released      = 0
      net_payment             = gross_payment

    interim (j=1..n-1):
      gross_payment           = payment_weight × price
      advance_recovery        = min(gross × advance_recovery_rate, remaining_advance)
      advance_recovery_remain = remaining_advance after this milestone
      retention_withheld      = gross × retention_rate
      retention_released      = 0
      net_payment             = gross - advance_recovery - retention_withheld

    final (j=n):
      gross_payment           = payment_weight × price
      advance_recovery        = min(gross × advance_recovery_rate, remaining_advance)
      advance_recovery_remain = 0  (any residual is forgiven at completion)
      retention_withheld      = gross × retention_rate  (same rule as interim)
      retention_released      = sum of ALL retention_withheld (j=1..n, including final)
      net_payment             = gross - advance_recovery - retention_withheld + retention_released

    Keys match milestones_profile schema columns exactly.
    Runtime-only fields (certified, certified_t, payment_released) are
    NOT included — they belong to episode state, not the profile.
    """
    n_ms = max(1, sample_int(rng, cfg["n_milestones_dist"],
                             cfg["n_milestones_p1"], cfg["n_milestones_p2"],
                             cfg["n_milestones_p3"], cfg["n_milestones_p4"]))

    # Evenly spaced thresholds; final pinned to 1.0
    thresholds     = [round((j + 1) / n_ms, 4) for j in range(n_ms)]
    thresholds[-1] = 1.0

    # Equal weights; final absorbs rounding residual
    weights        = [round(1.0 / n_ms, 6)] * n_ms
    weights[-1]    = round(1.0 - sum(weights[:-1]), 6)

    # Sample earliest_t fraction once per project
    earliest_t_fraction = max(0.0, min(1.0, sample(
        rng, cfg["earliest_t_fraction_dist"],
        cfg["earliest_t_fraction_p1"], cfg["earliest_t_fraction_p2"],
        cfg["earliest_t_fraction_p3"], cfg["earliest_t_fraction_p4"]
    )))

    price            = proj["price"]
    advance_amount   = proj["advance_percent"] * price
    advance_rec_rate = proj["advance_recovery"]
    retention_rate   = proj["retention_rate"]

    milestones = []

    # ── j = 0 : advance ──────────────────────────────────────────────────────
    milestones.append({
        "j":                      0,
        "progress_threshold":     0.0,        # advance has no progress gate
        "timestep_threshold":     proj["planned_start"],
        "payment_weight":         0.0,        # advance is not a weight-based payment
        "gross_payment":          advance_amount,
        "advance_recovery":       0.0,
        "advance_recovery_remain": advance_amount,
        "retention_withheld":     0.0,
        "retention_released":     0.0,
        "net_payment":            advance_amount,
    })

    # ── j = 1..n : milestones ────────────────────────────────────────────────
    remaining_advance   = advance_amount   # tracks how much is still unrecovered
    cumulative_retention = 0.0            # accumulates across all milestones

    for idx in range(n_ms):
        j        = idx + 1               # j=1 is first interim, j=n_ms is final
        is_final = (idx == n_ms - 1)

        # timestep_threshold
        if is_final:
            timestep_threshold = proj["planned_finish"]
        else:
            timestep_threshold = proj["planned_start"] + max(1, round(
                thresholds[idx] * proj["planned_duration"] * earliest_t_fraction
            ))

        gross = weights[idx] * price

        # advance recovery — residual forgiven at final milestone
        if is_final:
            recovery = remaining_advance   # clear whatever remains
        else:
            recovery = min(gross * advance_rec_rate, remaining_advance)
        remaining_advance -= recovery

        retention_withheld = gross * retention_rate
        cumulative_retention += retention_withheld

        # retention is released only at the final milestone
        if is_final:
            retention_released = cumulative_retention   # includes this milestone's withheld
        else:
            retention_released = 0.0

        net = gross - recovery - retention_withheld + retention_released

        milestones.append({
            "j":                      j,
            "progress_threshold":     thresholds[idx],
            "timestep_threshold":     timestep_threshold,
            "payment_weight":         weights[idx],
            "gross_payment":          gross,
            "advance_recovery":       recovery,
            "advance_recovery_remain": remaining_advance,
            "retention_withheld":     retention_withheld,
            "retention_released":     retention_released,
            "net_payment":            net,
        })

    return milestones


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def sample_efficiency(rng, cfg: dict) -> float:
    """
    Draw period efficiency η from config distribution.
    Called once per active project per period.
    Clamped to a minimum of 0.01 to prevent zero-progress deadlock.
    """
    eta = sample(rng, cfg["efficiency_dist"],
                 cfg["efficiency_p1"], cfg["efficiency_p2"],
                 cfg["efficiency_p3"], cfg["efficiency_p4"])
    return max(0.01, float(eta))