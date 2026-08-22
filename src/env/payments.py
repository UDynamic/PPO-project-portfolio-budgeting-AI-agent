# src/env/payments.py
#
# Payment mechanics for the Portfolio Budgeting environment.
# All functions are pure (no randomness, no I/O, no state mutation).
# They receive the relevant dicts and return cash amounts; the caller
# (env.py Phase 1 / Phase 5) is responsible for writing results back
# into ps and the cashflow accumulator.
#
# FIDIC 14.2-style mechanics — spec §Payment mechanics:
#
#   Advance       advance_percent × price  at project start
#   Milestone     gross = weight × price
#                 recovery = min(gross × advance_recovery, remaining_advance)
#                 retention = gross × retention_rate  (interim only; 0 for final)
#                 net = gross − recovery − retention
#   Retention     releases the period AFTER final milestone (one-period lag)
#   Settlement    progress × price − cumulative_inflows_received

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCE  (Phase 1a)
# ─────────────────────────────────────────────────────────────────────────────

def compute_advance(proj: dict) -> float:
    """
    Return the advance payment amount for *proj*.

    advance = advance_percent × price

    Called when the project starts (t == proj["start"]).
    """
    return proj["advance_percent"] * proj["price"]


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE CERTIFICATION  (Phase 1b)
# ─────────────────────────────────────────────────────────────────────────────

def certify_milestone(ms: dict, proj: dict, ps: dict) -> dict | None:
    """
    Attempt to certify one milestone.

    Returns a result dict if the milestone triggers this period, else None.

    Trigger condition (both must hold):
        ps["progress"] >= ms["threshold"]
        t             >= ms["earliest_t"]   ← checked by caller before calling

    The caller is responsible for:
        - checking that ms["certified"] is False
        - passing the current t and checking t >= ms["earliest_t"]
        - writing result values back into ps and the cashflow accumulator

    Returned dict
    -------------
    gross       : float   weight × price
    recovery    : float   advance recovered this milestone
    retention   : float   withheld (0 for final milestone)
    net         : float   gross − recovery − retention
    """
    gross             = ms["payment_weight"] * proj["price"]
    remaining_advance = max(0.0, ps["advance_received"] - ps["advance_recovered"])
    recovery          = min(gross * proj["advance_recovery"], remaining_advance)
    is_final          = ms["threshold"] >= 1.0
    retention         = 0.0 if is_final else gross * proj["retention_rate"]
    net               = gross - recovery - retention

    return {
        "gross":     gross,
        "recovery":  recovery,
        "retention": retention,
        "net":       net,
    }


def process_milestones(t: int, proj: dict, ps: dict,
                       milestones: list[dict]) -> tuple[float, float]:
    """
    Iterate all uncertified milestones for one project in one period.

    Mutates *ms* dicts in *milestones* (certified, certified_t,
    payment_released) and *ps* (advance_recovered, milestone_inflows,
    retention_held) in-place.

    Returns
    -------
    payment_net  : total net payment received this period (sum across certified)
    gross_total  : total gross (informational; used by db_logger)
    """
    payment_net  = 0.0
    gross_total  = 0.0

    for ms in milestones:
        if ms["certified"]:
            continue
        if ps["progress"] < ms["threshold"]:
            continue
        if t < ms["earliest_t"]:
            continue

        result = certify_milestone(ms, proj, ps)

        ms["certified"]        = True
        ms["certified_t"]      = t
        ms["payment_released"] = result["net"]

        ps["advance_recovered"] += result["recovery"]
        ps["milestone_inflows"] += result["net"]
        ps["retention_held"]    += result["retention"]

        payment_net += result["net"]
        gross_total += result["gross"]

    return payment_net, gross_total


# ─────────────────────────────────────────────────────────────────────────────
# RETENTION RELEASE  (Phase 1c)
# ─────────────────────────────────────────────────────────────────────────────

def release_retention(ps: dict) -> float:
    """
    Release accumulated retention held for one project.

    Only called when ps["_release_retention_next_period"] is True (set by
    Phase 5 of the prior step when progress reached 1.0).

    Mutates *ps* in-place: clears the flag, marks retention_released = True.

    Returns the retention amount released (0.0 if nothing held).
    """
    amount = ps["retention_held"]
    if amount > 0.0:
        ps["retention_released"]           = True
        ps["_release_retention_next_period"] = False
    return amount


# ─────────────────────────────────────────────────────────────────────────────
# TERMINATION SETTLEMENT  (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

def termination_settlement(proj: dict, ps: dict) -> float:
    """
    Compute the settlement amount on contract termination.

    R_i^term = (P_actual × price) − cumulative_inflows_received

    The contractor is entitled to price × progress for work delivered.
    Against that they have already received: advance + net milestone payments.
    The settlement is the balancing amount — positive means contractor
    receives a payment; negative means the contractor owes money back.
    """
    entitlement        = ps["progress"] * proj["price"]
    cumulative_inflows = ps["advance_received"] + ps["milestone_inflows"]
    return entitlement - cumulative_inflows