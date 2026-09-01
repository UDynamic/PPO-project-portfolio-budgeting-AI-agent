# src/env/payments.py
#
# Payment mechanics for the Portfolio Budgeting environment.
# All functions are pure (no randomness, no I/O, no state mutation beyond
# what is explicitly documented).
#
# Design:
#   All payment amounts are precomputed in sampler.sample_milestones() and
#   stored in the milestone profile. This module reads those values — it
#   does not recompute gross, recovery, retention, or net.
#
#   The milestone profile j-index structure:
#       j = 0        advance payment (at planned_start)
#       j = 1..n-1   interim milestones
#       j = n        final milestone (includes retention_released)
#
#   Runtime milestone state lives in a SEPARATE list (milestone_state),
#   not in the profile dicts. Each entry:
#       ms_state["certified"]        bool   — has this milestone been certified
#       ms_state["certified_t"]      int    — period when certified; None if not yet
#       ms_state["payment_released"] float  — net payment released; 0.0 if not yet
#
#   Profile and runtime state share the same list index (ms and ms_state at
#   index k correspond to the same milestone j).
#
#   Project cashflow is tracked in ps["inflow"] and ps["outflow"] (cumulative)
#   matching projects_status schema exactly.

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE CERTIFICATION CHECK  (early phase)
# ─────────────────────────────────────────────────────────────────────────────

def check_certifications(t: int,
                         ps: dict,
                         milestones: list[dict],
                         milestone_state: list[dict]) -> list[int]:
    """
    Check which uncertified milestones qualify for certification this period.

    Certification conditions (both must hold):
        ps["progress_actual"] >= ms["progress_threshold"]
        t                     >= ms["timestep_threshold"]

    The advance (j=0) is excluded — handled directly in env.py at project
    start and does not go through this check.

    Does NOT mutate any state. Returns qualifying j indices so the caller
    (env.py early phase) can record which milestones are pending payment
    delivery in the late phase.

    Parameters
    ----------
    t               : current episode timestep
    ps              : project state dict (read-only)
    milestones      : list of milestone profile dicts (read-only)
    milestone_state : list of milestone runtime dicts (read-only here)

    Returns
    -------
    List of j indices certified this period (may be empty).
    """
    certified_this_period = []

    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] == 0:
            continue                                  # advance handled separately
        if ms_state["certified"]:
            continue                                  # already paid
        if ps["progress_actual"] < ms["progress_threshold"]:
            continue
        if t < ms["timestep_threshold"]:
            continue

        certified_this_period.append(ms["j"])

    return certified_this_period


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFIED PAYMENT DELIVERY  (late phase)
# ─────────────────────────────────────────────────────────────────────────────

def deliver_payments(certified_js: list[int],
                     t: int,
                     ps: dict,
                     milestones: list[dict],
                     milestone_state: list[dict]) -> float:
    """
    Deliver payments for all milestones certified this period.

    Reads net_payment from the milestone profile dict.
    Mutates milestone runtime state and ps["inflow"] in-place.

    Parameters
    ----------
    certified_js    : list of j indices returned by check_certifications()
    t               : current episode timestep (recorded as certified_t)
    ps              : project state dict (ps["inflow"] incremented)
    milestones      : list of milestone profile dicts (read-only)
    milestone_state : list of milestone runtime dicts (mutated)

    Returns
    -------
    total_net : total net inflow delivered this period
    """
    total_net = 0.0

    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] not in certified_js:
            continue

        net = ms["net_payment"]

        ms_state["certified"]        = True
        ms_state["certified_t"]      = t
        ms_state["payment_released"] = net

        ps["inflow"] += net
        total_net    += net

    return total_net


# ─────────────────────────────────────────────────────────────────────────────
# TERMINATION SETTLEMENT  (late phase, on termination only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_termination_settlement(proj: dict, ps: dict) -> float:
    """
    Compute the net settlement amount on contract termination.

    settlement = (progress_actual × price) − cumulative_inflow

    The contractor is entitled to price × progress for work delivered.
    Against that they have already received all certified payments (inflow).
    The settlement is the balancing amount:
        positive → contractor receives a final payment
        negative → contractor owes money back (rare; signals over-advance)

    Does NOT mutate ps. The caller (env.py) writes the result into
    ps["termination_settlement"] and updates ps["inflow"] accordingly.

    Parameters
    ----------
    proj : project parameter dict (read-only)
    ps   : project state dict (read-only)

    Returns
    -------
    settlement : float
    """
    entitlement = ps["progress_actual"] * proj["price"]
    return entitlement - ps["inflow"]