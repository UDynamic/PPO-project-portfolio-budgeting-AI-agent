# src/env/breaches.py
#
# Breach evaluation, tolerance management, and termination logic.
# All functions are pure (no randomness, no I/O, no state mutation beyond
# what is explicitly documented).
#
# Design:
#   evaluate_breaches() is called twice per timestep per active project:
#     - early phase (pre-allocation): alloc=None → abandoned always False
#     - late phase (post-allocation): alloc=float → abandoned checked
#
#   over_duration_window is NOT evaluated in evaluate_breaches().
#   It is evaluated once, inside check_termination(), to keep deadline
#   logic in one place and avoid redundant checks across both phases.
#
# Breach flag names match projects_status schema exactly:
#   over_progress_delay  1 if progress_delay_t > progress_delay_cap
#   over_finish_delay    1 if projected_finish_delay > finish_delay_cap
#   over_cost_overrun    1 if projected_cost_overrun > cost_overrun_cap
#   over_any             1 if any of the above fired
#   over_duration_window 1 if t >= planned_finish + finish_delay_cap
#                          (set by check_termination, not evaluate_breaches)
#
# Tolerance logic:
#   over_duration_window → instant termination, tolerance not decremented
#   over_any (excl. deadline) → tolerance_remain -= 1
#   fully healthy → tolerance_remain reset to termination_tolerance
#   termination fires when tolerance_remain <= 0

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# BREACH EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_breaches(ps: dict, proj: dict) -> dict:
    """
    Evaluate all breach conditions for one active project.

    Called twice per timestep:
      - Early phase:  pre-allocation state
      - Late phase:   post-allocation state

    over_duration_window is NOT computed here — see check_termination().

    Parameters
    ----------
    ps   : project state dict (read-only)
    proj : project parameter dict (read-only)

    Returns
    -------
    dict with keys:
        over_progress_delay, over_finish_delay,
        over_cost_overrun, over_any, over_duration_window
    """

    over_progress_delay = (
        ps["progress_delay_t"] > proj["progress_delay_cap"]
    )
    over_finish_delay = (
        ps["projected_finish_delay"] > proj["finish_delay_cap"]
    )
    over_cost_overrun = (
        ps["projected_cost_overrun"] > proj["cost_overrun_cap"]
    )

    over_any = (
        over_progress_delay
        or over_finish_delay
        or over_cost_overrun
    )

    return {
        "over_progress_delay":  over_progress_delay,
        "over_finish_delay":    over_finish_delay,
        "over_cost_overrun":    over_cost_overrun,
        "over_any":             over_any,
        "over_duration_window": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOLERANCE UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def update_tolerance(ps: dict, proj: dict, flags: dict) -> None:
    """
    Update ps["tolerance_remain"] based on breach flags.

    Rules:
      over_any → decrement by 1  (floor at 0)
      healthy  → reset to termination_tolerance

    over_duration_window bypasses tolerance entirely — it triggers
    instant termination in check_termination() without decrementing.

    Mutates ps["tolerance_remain"] in-place.

    Parameters
    ----------
    ps    : project state dict (mutated)
    proj  : project parameter dict (read-only)
    flags : dict returned by evaluate_breaches()
    """
    if flags["over_any"]:
        ps["tolerance_remain"] = max(0, ps["tolerance_remain"] - 1)
    else:
        ps["tolerance_remain"] = proj["termination_tolerance"]


# ─────────────────────────────────────────────────────────────────────────────
# TERMINATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_termination(t: int, ps: dict, proj: dict,
                      flags: dict) -> tuple[bool, bool]:
    """
    Decide whether the project terminates this period.

    Termination conditions (evaluated in order):
      1. Deadline breach: t >= planned_finish + finish_delay_cap
         → instant termination, tolerance not involved
      2. Tolerance exhausted: tolerance_remain <= 0
         → termination after cure period expires

    Parameters
    ----------
    t     : current episode timestep
    ps    : project state dict (read-only; tolerance already updated by caller)
    proj  : project parameter dict (read-only)
    flags : dict returned by evaluate_breaches() (mutated: over_duration_window set)

    Returns
    -------
    (terminated, over_duration_window)
        terminated           : True if project should be terminated this period
        over_duration_window : True if deadline breach caused termination
    """
    over_duration_window = (
        t >= proj["planned_finish"] + proj["finish_delay_cap"]
    )
    flags["over_duration_window"] = over_duration_window

    if over_duration_window:
        return True, True

    if ps["tolerance_remain"] <= 0:
        return True, False

    return False, False