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
#     profile for schema completeness) but is NOT sampled in _sample_project;
#     the caller passes the raw cfg value through.  Wiring it into
#     certification logic is a planned extension — see Bug #6 fix notes.

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
# PROJECT & MILESTONE SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_project(rng, cfg: dict, i: int) -> dict:
    """
    Draw all project-level parameters from *cfg* distributions.

    advance_trigger is sampled and stored (schema requires it) but is NOT
    currently used in certification logic.  Bug #6 from the spec: the field
    is kept rather than removed so the DB profile remains consistent; wiring
    it into Phase 1b is a planned extension.
    """
    budget = max(1.0, sample(rng, cfg["budget_dist"],
                             cfg["budget_p1"], cfg["budget_p2"],
                             cfg["budget_p3"], cfg["budget_p4"]))

    margin = max(0.0, sample(rng, cfg["margin_dist"],
                             cfg["margin_p1"], cfg["margin_p2"],
                             cfg["margin_p3"], cfg["margin_p4"]))
    price  = budget * (1.0 + margin)

    start    = max(0, int(round(sample(rng, cfg["start_dist"],
                                       cfg["start_p1"], cfg["start_p2"],
                                       cfg["start_p3"], cfg["start_p4"]))))
    duration = max(1, sample_int(rng, cfg["duration_dist"],
                                 cfg["duration_p1"], cfg["duration_p2"],
                                 cfg["duration_p3"], cfg["duration_p4"]))
    finish   = start + duration

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

    # Sampled for profile storage; not yet wired into certification (Bug #6).
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

    schedule_cap = max(1, sample_int(rng, cfg["schedule_cap_dist"],
                                     cfg["schedule_cap_p1"], cfg["schedule_cap_p2"],
                                     cfg["schedule_cap_p3"], cfg["schedule_cap_p4"]))

    cost_cap = max(1.0, sample(rng, cfg["cost_cap_dist"],
                               cfg["cost_cap_p1"], cfg["cost_cap_p2"],
                               cfg["cost_cap_p3"], cfg["cost_cap_p4"]))

    cure_length = max(1, sample_int(rng, cfg["cure_length_dist"],
                                    cfg["cure_length_p1"], cfg["cure_length_p2"],
                                    cfg["cure_length_p3"], cfg["cure_length_p4"]))

    return {
        "i":                i,
        "budget":           budget,          # BAC
        "price":            price,           # BAC × (1 + margin)
        "margin":           margin,
        "start":            start,
        "finish":           finish,
        "duration":         duration,
        "scurve_a":         scurve_a,
        "scurve_b":         scurve_b,
        "advance_percent":  advance_percent,
        "advance_trigger":  advance_trigger,
        "advance_recovery": advance_recovery,
        "retention_rate":   retention_rate,
        "schedule_cap":     schedule_cap,
        "plan_deviation_threshold": cfg.get("plan_deviation_threshold", 0.10),
        "cost_cap":         cost_cap,
        "cure_length":      cure_length,
    }


def sample_milestones(rng, cfg: dict, proj: dict) -> list[dict]:
    """
    Build the milestone schedule for *proj*.

    Thresholds are evenly spaced; the final one is pinned to 1.0.
    Weights are equal; the final one absorbs rounding residual.

    earliest_t logic
    ----------------
    Intermediate milestones:
        earliest_t[j] = start + max(1, round(threshold[j] × duration × fraction))
    Final milestone:
        earliest_t = proj["finish"]  (contract invariant — never overridden)

    fraction = 1.0 → earliest_t aligns with the on-plan date for that threshold.
    fraction < 1.0 → allows early certification for high performers.
    """
    n_ms = max(1, sample_int(rng, cfg["n_milestones_dist"],
                             cfg["n_milestones_p1"], cfg["n_milestones_p2"],
                             cfg["n_milestones_p3"], cfg["n_milestones_p4"]))

    thresholds      = [round((j + 1) / n_ms, 4) for j in range(n_ms)]
    thresholds[-1]  = 1.0

    weights         = [round(1.0 / n_ms, 6)] * n_ms
    weights[-1]     = round(1.0 - sum(weights[:-1]), 6)

    # Sample once per project — not per milestone.
    earliest_t_fraction = max(0.0, min(1.0, sample(
        rng, cfg["earliest_t_fraction_dist"],
        cfg["earliest_t_fraction_p1"], cfg["earliest_t_fraction_p2"],
        cfg["earliest_t_fraction_p3"], cfg["earliest_t_fraction_p4"]
    )))

    milestones = []
    for j in range(n_ms):
        is_final = (j == n_ms - 1)
        if is_final:
            earliest_t = proj["finish"]
        else:
            earliest_t = proj["start"] + max(1, round(
                thresholds[j] * proj["duration"] * earliest_t_fraction
            ))

        milestones.append({
            "j":               j,
            "threshold":       thresholds[j],
            "earliest_t":      earliest_t,
            "payment_weight":  weights[j],
            "certified":       False,
            "certified_t":     None,
            "payment_released": None,
        })

    return milestones


def sample_efficiency(rng, cfg: dict) -> float:
    """
    Draw period efficiency η from config distribution.
    Clamped to a minimum of 0.01 to prevent zero-progress deadlock.
    """
    eta = sample(rng, cfg["efficiency_dist"],
                 cfg["efficiency_p1"], cfg["efficiency_p2"],
                 cfg["efficiency_p3"], cfg["efficiency_p4"])
    return max(0.01, float(eta))