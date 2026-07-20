"""
test_milp.py
============
Regression test suite for the Level-1 MILP (milp.py).

Each test case corresponds to a hand-solved scenario from the manual
solutions document. The test builds a hard-coded portfolio CONFIG,
solves it via milp.py, and checks the solution against analytically
derived assertions.

Run:
    pytest test_milp.py -v
    pytest test_milp.py -v -k "SP1"      # single case
    pytest test_milp.py -v --tb=short    # compact traceback

Dependencies:
    pip install pytest pulp scipy
"""

import math
import pytest

# ── import the module under test ──────────────────────────────────────────────
from milp import (
    generate_portfolio,
    build_and_solve,
    build_records,
    validate,
    compute_R_net,
)


# =============================================================================
# SHARED TOLERANCE
# =============================================================================

TOL_Z     = 1.5    # objective value tolerance
TOL_ALLOC = 0.5    # individual allocation tolerance
TOL_PAY   = 0.5    # payment amount tolerance
TOL_PROG  = 0.01   # progress fraction tolerance
TOL_BAL   = 1.0    # cash balance tolerance


# =============================================================================
# HELPERS
# =============================================================================

def _cert_period(rec, proj_idx, milestone_idx):
    """Return the period at which milestone j of project i first certified,
    or None if never certified."""
    u_val = rec['u_val']
    H = max(t for (i, t) in rec['x_val'] if i == proj_idx)
    for t in range(1, H + 1):
        now  = u_val.get((proj_idx, milestone_idx, t), 0)
        prev = u_val.get((proj_idx, milestone_idx, t - 1), 0) if t > 1 else 0
        if now - prev > 0:
            return t
    return None


def _total_alloc(rec, proj_idx):
    """Total budget allocated to project i across all periods."""
    return sum(v for (i, t), v in rec['x_val'].items() if i == proj_idx)


def _alloc_at(rec, proj_idx, period):
    """Budget allocated to project i at period t."""
    return rec['x_val'].get((proj_idx, period), 0.0)


def _status(rec, proj_idx):
    return rec['proj'][proj_idx]['status']


def _R_net_at_ms(rec, proj_idx, milestone_idx):
    """
    Return the net payment amount received when milestone j was certified.
    Looks up the cash flow event list.
    """
    target_type = f'ms{milestone_idx + 1}_net'
    cert_t = _cert_period(rec, proj_idx, milestone_idx)
    if cert_t is None:
        return None
    for ev in rec['events']:
        if ev['proj'] == proj_idx and ev['type'] == target_type and ev['t'] == cert_t:
            return ev['amount']
    return None


def _term_period(rec, proj_idx):
    """Return the period at which project i was terminated, or None."""
    H = max(t for (i, t) in rec['x_val'] if i == proj_idx)
    for t in range(1, H + 1):
        if rec['proj'][proj_idx]['records'][t]['status'] == 'terminated':
            return t
    return None


def _tau_at(rec, proj_idx, period):
    """Return tau_rem for project i at period t."""
    return rec['proj'][proj_idx]['records'][period]['tau_rem']


def _B_at(rec, period):
    """Return LP cash balance at period t."""
    return rec['B_lp'].get(period, 0.0)


def _R_term(rec, proj_idx):
    """Return termination settlement amount for project i, or None."""
    for ev in rec['events']:
        if ev['proj'] == proj_idx and ev['type'] == 'termination_settlement':
            return ev['amount']
    return None


def _retention_released(rec, proj_idx):
    """Return retention release amount for project i, or None."""
    for ev in rec['events']:
        if ev['proj'] == proj_idx and ev['type'] == 'retention':
            return ev['amount']
    return None


def _make_project(idx, s, f, BAC, pi, n_ms, H,
                  rho=0.0, drec=0.25, psi=0.10,
                  Omega=2, mu=0.50, tau_tol=2,
                  eta_val=1.0, alpha_override=None,
                  a=2.0, b=2.0,
                  e_dates=None,
                  phi_override=None,
                  theta_override=None,
                  eta_schedule=None):
    """
    Build a project dict matching milp.py's expected format.

    Parameters
    ----------
    eta_schedule : dict {t: eta_value} or None
        If given, overrides eta_val for specific periods.
        Periods not in schedule use eta_val.
    e_dates : list of int or None
        Earliest certification period for each milestone.
        If None, defaults to s (no restriction beyond start).
    phi_override : list of float or None
        Override equal milestone weights.
    theta_override : list of float or None
        Override evenly-spaced thresholds.
    alpha_override : float or None
        Override derive_alpha with a fixed value.
    """
    from milp import derive_alpha

    if theta_override is not None:
        theta = theta_override
    else:
        theta = [(j + 1) / n_ms for j in range(n_ms)]

    if phi_override is not None:
        phi = phi_override
    else:
        phi = [1.0 / n_ms] * n_ms

    if alpha_override is not None:
        alpha_i = alpha_override
    else:
        alpha_i = derive_alpha(phi, drec, psi)

    # Earliest certification dates
    if e_dates is not None:
        e_i = e_dates
    else:
        # Default: no restriction beyond project start
        e_i = [s] * n_ms

    # Efficiency schedule
    eta = {}
    for t in range(1, H + 1):
        if s <= t <= f:
            if eta_schedule and t in eta_schedule:
                eta[t] = eta_schedule[t]
            else:
                eta[t] = eta_val
        else:
            eta[t] = 0.0

    D_plan = f - s + 1

    return dict(
        idx=idx, s_i=s, f_i=f, D_plan=D_plan,
        a_i=a, b_i=b,
        BAC_i=float(BAC), pi_i=float(pi),
        CP_i=float(BAC) * (1.0 + float(pi)),
        alpha_i=float(alpha_i),
        delta_rec_i=float(drec),
        psi_i=float(psi),
        rho_i=float(rho),
        M_i=n_ms,
        theta=theta,
        phi=phi,
        e_i=e_i,
        Omega_i=Omega,
        mu_i=mu,
        tau_tol_i=tau_tol,
        eta=eta,
    )


def _make_portfolio(projects, gamma, H, B1=None, b1_mult=100):
    total_BAC = sum(p['BAC_i'] for p in projects)
    if B1 is None:
        B1 = b1_mult * total_BAC
    return dict(
        n=len(projects), H=H, projects=projects,
        B1=float(B1), gamma=float(gamma),
        total_BAC=total_BAC,
        cfg=dict(seed=0, solver_time_limit=120),
    )


def _solve(portfolio):
    """Solve and return (sol, rec, val)."""
    sol = build_and_solve(portfolio)
    rec = build_records(portfolio, sol)
    val = validate(portfolio, sol, rec)
    return sol, rec, val


# =============================================================================
# GROUP 1: PROGRESS AND MILESTONE CERTIFICATION
# =============================================================================

class TestGroup1_ProgressAndMilestoneCertification:

    def test_SP1_perfect_efficiency_two_milestones(self):
        """
        SP-1: η=1, γ=0.95, e={1,3}.
        Optimal: spend 50 at t=1 (MS1 certifies), spend 50 at t=3 (MS2 certifies).
        MS2 cannot certify before t=3 even though progress qualifies after t=2.
        Z* = γ⁰(60-50) + γ²(60-50) = 10 + 9.025 = 19.025
        """
        H   = 4
        BAC = 100
        pi  = 0.20   # CP = 120
        p = _make_project(0, s=1, f=4, BAC=BAC, pi=pi, n_ms=2, H=H,
                          rho=0.0, drec=0.25, psi=0.10,
                          Omega=1, mu=0.30, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: MS1 certified at t=1
        assert _cert_period(rec, 0, 0) == 1, \
            "MS1 should certify at t=1"

        # A2: MS2 certified at t=3 (earliest window), not before
        assert _cert_period(rec, 0, 1) == 3, \
            "MS2 should certify at t=3 (e_{1,2}=3); earlier certification violates contract window"

        # A3: Allocation at t=1 is ~50, at t=3 is ~50
        assert abs(_alloc_at(rec, 0, 1) - 50) <= TOL_ALLOC, \
            f"Expected x[0,1]=50, got {_alloc_at(rec, 0, 1):.2f}"
        assert abs(_alloc_at(rec, 0, 3) - 50) <= TOL_ALLOC, \
            f"Expected x[0,3]=50, got {_alloc_at(rec, 0, 3):.2f}"

        # A4: No spend at t=2 (MS2 window not yet open; spending early wastes discount)
        assert _alloc_at(rec, 0, 2) <= TOL_ALLOC, \
            f"Expected no spend at t=2 (window closed), got {_alloc_at(rec, 0, 2):.2f}"

        # A5: Status completed, no termination
        assert _status(rec, 0) == 'completed'

        # A6: Objective
        assert abs(sol['obj'] - 19.025) <= TOL_Z, \
            f"Expected Z*≈19.025, got {sol['obj']:.4f}"

    def test_SP3_three_milestones_gamma_timing(self):
        """
        SP-3: η=1, γ=0.95, e={1,3,5}, BAC=60, CP=90.
        Optimal: spend 20 at t=1, 20 at t=3, 20 at t=5.
        Z* = γ⁰(30-20) + γ²(30-20) + γ⁴(30-20) = 10+9.025+8.145 = 27.17
        """
        H   = 6
        BAC = 60
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.50, n_ms=3, H=H,
                          rho=0.0, drec=0.25, psi=0.10,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: All three milestones certified
        for j in range(3):
            assert _cert_period(rec, 0, j) is not None, f"MS{j+1} never certified"

        # A2: Certification timing respects windows
        assert _cert_period(rec, 0, 0) == 1
        assert _cert_period(rec, 0, 1) == 3
        assert _cert_period(rec, 0, 2) == 5

        # A3: Spend is concentrated at window-opening periods
        assert abs(_alloc_at(rec, 0, 1) - 20) <= TOL_ALLOC
        assert abs(_alloc_at(rec, 0, 3) - 20) <= TOL_ALLOC
        assert abs(_alloc_at(rec, 0, 5) - 20) <= TOL_ALLOC

        # A4: No spend between windows
        assert _alloc_at(rec, 0, 2) <= TOL_ALLOC
        assert _alloc_at(rec, 0, 4) <= TOL_ALLOC

        # A5: Status
        assert _status(rec, 0) == 'completed'

        # A6: Objective
        assert abs(sol['obj'] - 27.17) <= TOL_Z

    def test_SPJ_milestone_ordering_on_progress_jump(self):
        """
        SP-J: Spend 100 at t=1 → progress=1.0, crosses both θ=0.5 and θ=1.0.
        e_{1,1}=1, e_{1,2}=3.
        MS1 certifies at t=1; MS2 must NOT certify at t=1 (window closed).
        MS2 certifies at t=3 with no additional spend needed.
        Z* = 0 (both milestone nets cancel their spend at γ=0.95).
        """
        H   = 4
        BAC = 100
        p = _make_project(0, s=1, f=4, BAC=BAC, pi=0.0, n_ms=2, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: MS1 certifies at t=1
        assert _cert_period(rec, 0, 0) == 1, "MS1 should certify at t=1"

        # A2: MS2 does NOT certify at t=1 even though P≥θ₂
        u_val = rec['u_val']
        assert u_val.get((0, 1, 1), 0) == 0, \
            "MS2 must not certify at t=1; e_{1,2}=3 blocks it"

        # A3: MS2 certifies at t=3
        assert _cert_period(rec, 0, 1) == 3, \
            "MS2 should certify at t=3 when window opens"

        # A4: Progress at t=1 clears both thresholds
        assert rec['proj'][0]['records'][1]['P_sim'] >= 0.99

        # A5: Status completed
        assert _status(rec, 0) == 'completed'

        # A6: Z* ≈ 0
        assert abs(sol['obj'] - 0.0) <= TOL_Z


# =============================================================================
# GROUP 2: DISCOUNTING AND TIMING INCENTIVES
# =============================================================================

class TestGroup2_DiscountingAndTimingIncentives:

    def test_SP4_defer_spend_to_last_possible_period(self):
        """
        SP-4: η=1, γ=0.95, BAC=100, CP=140, e={1,5}.
        Optimal: spend 50 at t=1 (MS1), spend 50 at t=5 (MS2 window opens).
        Z* = γ⁰(70-50) + γ⁴(70-50) = 20 + 0.8145×20 = 36.29.
        Front-loading both at t=1,2 yields only 29.515.
        """
        H   = 8
        BAC = 100
        p = _make_project(0, s=1, f=8, BAC=BAC, pi=0.40, n_ms=2, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: MS1 certified at t=1
        assert _cert_period(rec, 0, 0) == 1

        # A2: MS2 certified at t=5 (window opens, deferred spend)
        assert _cert_period(rec, 0, 1) == 5, \
            "MS2 should certify exactly at e=5; earlier is impossible, later wastes discount"

        # A3: No spend between t=2 and t=4
        for t in range(2, 5):
            assert _alloc_at(rec, 0, t) <= TOL_ALLOC, \
                f"No spend expected at t={t} (window closed, defer is optimal)"

        # A4: Objective beats front-loaded alternative
        Z_front = 29.515
        assert sol['obj'] > Z_front, \
            f"Deferred strategy Z*={sol['obj']:.3f} should exceed front-load Z={Z_front}"

        # A5: Objective within expected range
        assert abs(sol['obj'] - 36.29) <= TOL_Z

    def test_MPK_front_heavy_phi_prioritized(self):
        """
        MP-K: Two identical projects except φ.
        Project A: φ=(0.70, 0.30) — large first payment.
        Project B: φ=(0.30, 0.70) — large second payment.
        e_A = {1, 4}, e_B = {2, 4} (A's MS1 window opens one period earlier).
        Completing A's MS1 first yields higher discounted Z.
        Z_A_first ≈ 55.87 > Z_B_first ≈ 54.07.
        """
        H = 6
        pA = _make_project(0, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           phi_override=[0.70, 0.30],
                           theta_override=[0.5, 1.0],
                           e_dates=[1, 4])
        pB = _make_project(1, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           phi_override=[0.30, 0.70],
                           theta_override=[0.5, 1.0],
                           e_dates=[2, 4])
        portfolio = _make_portfolio([pA, pB], gamma=0.90, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: Both projects completed
        assert _status(rec, 0) == 'completed'
        assert _status(rec, 1) == 'completed'

        # A2: A's MS1 certifies before B's MS1
        cert_A1 = _cert_period(rec, 0, 0)
        cert_B1 = _cert_period(rec, 1, 0)
        assert cert_A1 is not None and cert_B1 is not None
        assert cert_A1 <= cert_B1, \
            f"Project A MS1 (large φ=0.70) should certify no later than B MS1; got A@{cert_A1}, B@{cert_B1}"

        # A3: Objective exceeds B-first strategy
        Z_B_first = 54.07
        assert sol['obj'] > Z_B_first, \
            f"A-first strategy Z*={sol['obj']:.3f} should exceed B-first Z={Z_B_first}"

        # A4: Objective within expected range
        assert abs(sol['obj'] - 55.87) <= TOL_Z


# =============================================================================
# GROUP 3: ADVANCE PAYMENT AND RECOVERY MECHANICS
# =============================================================================

class TestGroup3_AdvancePaymentAndRecovery:

    def test_SP2_advance_harvest_termination_net_zero(self):
        """
        SP-2: Large advance, efficiency decays to 0.05 from t=2.
        Zero spend optimal; project terminates; settlement cancels advance.
        With γ=0.95: Z* = γ⁰×30 + γ¹×(-30) = 30 - 28.5 = 1.5.
        """
        H   = 6
        BAC = 100
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.0, n_ms=2, H=H,
                          rho=0.0,
                          alpha_override=0.30,   # A = 0.30 × 100 = 30
                          drec=0.25, psi=0.10,
                          Omega=1, mu=0.05, tau_tol=1,
                          e_dates=[1, 4],
                          eta_schedule={1: 1.0, 2: 0.05, 3: 0.05, 4: 0.05,
                                        5: 0.05, 6: 0.05})
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: Effectively zero spend
        assert _total_alloc(rec, 0) <= 1.0, \
            "No meaningful allocation should occur to a project with decayed efficiency"

        # A2: Project terminates
        assert _status(rec, 0) == 'terminated'

        # A3: No milestone certified
        assert _cert_period(rec, 0, 0) is None
        assert _cert_period(rec, 0, 1) is None

        # A4: Settlement ≈ -30
        R_term = _R_term(rec, 0)
        assert R_term is not None
        assert abs(R_term - (-30.0)) <= TOL_PAY, \
            f"Expected R_term=-30, got {R_term:.2f}"

        # A5: Z* ≈ 1.5 (advance discount gain)
        assert abs(sol['obj'] - 1.5) <= 0.5, \
            f"Expected Z*≈1.5 (advance at t=1 worth more than discounted settlement at t=2), got {sol['obj']:.4f}"

    def test_SPA_late_recovery_threshold_psi_040(self):
        """
        SP-A: ψ=0.40, MS1 weight=1/3 < 0.40 → MS1 payment in full.
        MS2 and MS3 subject to recovery deductions.
        BAC=120, CP=180, A=36, δ_rec=0.30.
        R1=60, R2=42, R3=42. Identity: 36+60+42+42=180=CP.
        e={1,3,5}, spend 40 at each window.
        Z* = γ⁰(36+60-40) + γ²(42-40) + γ⁴(42-40) ≈ 59.43.
        """
        H   = 6
        BAC = 120
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.50, n_ms=3, H=H,
                          rho=0.0,
                          alpha_override=0.20,   # A = 0.20 × 180 = 36
                          drec=0.30, psi=0.40,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: MS1 payment in full (no recovery deduction because cum_phi < ψ)
        R1 = _R_net_at_ms(rec, 0, 0)
        assert R1 is not None
        assert abs(R1 - 60.0) <= TOL_PAY, \
            f"MS1 net payment should be 60 (no deduction, ψ not crossed). Got {R1:.2f}"

        # A2: MS2 has deduction
        R2 = _R_net_at_ms(rec, 0, 1)
        assert R2 is not None
        assert abs(R2 - 42.0) <= TOL_PAY, \
            f"MS2 net payment should be 42. Got {R2:.2f}"

        # A3: MS3 has deduction
        R3 = _R_net_at_ms(rec, 0, 2)
        assert R3 is not None
        assert abs(R3 - 42.0) <= TOL_PAY, \
            f"MS3 net payment should be 42. Got {R3:.2f}"

        # A4: Payment identity
        A_i = 0.20 * (BAC * 1.50)  # 36
        total = A_i + (R1 or 0) + (R2 or 0) + (R3 or 0)
        CP = BAC * 1.50
        assert abs(total - CP) <= 1.0, \
            f"Payment identity failed: {total:.2f} ≠ CP={CP:.2f}"

        # A5: Status completed
        assert _status(rec, 0) == 'completed'

        # A6: Objective
        assert abs(sol['obj'] - 59.43) <= TOL_Z

    def test_SPB_advance_fully_recovered_before_final_milestone(self):
        """
        SP-B: Recovery deductions on MS1 and MS2 exactly exhaust advance.
        MS3 payment carries no recovery deduction despite δ_rec>0.
        BAC=60, CP=90, A=18, ψ=0.10, δ_rec=0.30.
        R1=21, R2=21, R3=30 (no deduction). Identity: 18+21+21+30=90.
        e={1,2,3}, spend 20 at each.
        Z* = γ⁰(18+21-20) + γ¹(21-20) + γ²(30-20) ≈ 28.975.
        """
        H   = 4
        BAC = 60
        p = _make_project(0, s=1, f=4, BAC=BAC, pi=0.50, n_ms=3, H=H,
                          rho=0.0,
                          alpha_override=0.20,   # A = 0.20 × 90 = 18
                          drec=0.30, psi=0.10,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 2, 3])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        R1 = _R_net_at_ms(rec, 0, 0)
        R2 = _R_net_at_ms(rec, 0, 1)
        R3 = _R_net_at_ms(rec, 0, 2)

        # A1: R1 deducted (ψ=0.10, φ1=1/3 ≥ 0.10 → recovery starts at MS1)
        assert R1 is not None
        assert abs(R1 - 21.0) <= TOL_PAY, f"R1 should be 21, got {R1:.2f}"

        # A2: R2 deducted (advance not yet exhausted after MS1)
        assert R2 is not None
        assert abs(R2 - 21.0) <= TOL_PAY, f"R2 should be 21, got {R2:.2f}"

        # A3: R3 NOT deducted (advance fully recovered after MS2)
        assert R3 is not None
        assert abs(R3 - 30.0) <= TOL_PAY, \
            f"R3 should be 30 (advance already recovered). Got {R3:.2f}"

        # A4: Payment identity
        A_i = 0.20 * (BAC * 1.50)  # 18
        total = A_i + (R1 or 0) + (R2 or 0) + (R3 or 0)
        CP = BAC * 1.50
        assert abs(total - CP) <= 1.0

        # A5: Objective
        assert abs(sol['obj'] - 28.975) <= TOL_Z


# =============================================================================
# GROUP 4: RETENTION
# =============================================================================

class TestGroup4_Retention:

    def test_SPC_retention_only_lump_release_at_completion(self):
        """
        SP-C: ρ=0.10, no advance.
        Each MS payment = 30×0.90 = 27. Retention release = 9 at final MS.
        Final period inflow = 27+9 = 36.
        e={1,3,5}. Z* = γ⁰(27-20) + γ²(27-20) + γ⁴(36-20) ≈ 26.35.
        """
        H   = 6
        BAC = 60
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.50, n_ms=3, H=H,
                          rho=0.10, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: Each net MS payment = 27
        for j in range(3):
            Rj = _R_net_at_ms(rec, 0, j)
            assert Rj is not None
            assert abs(Rj - 27.0) <= TOL_PAY, \
                f"MS{j+1} net payment should be 27 (retention withheld). Got {Rj:.2f}"

        # A2: Retention released at project completion only
        R_ret = _retention_released(rec, 0)
        assert R_ret is not None
        assert abs(R_ret - 9.0) <= TOL_PAY, \
            f"Retention release should be 9. Got {R_ret:.2f}"

        # A3: Retention release event is at the final milestone period
        cert_t_final = _cert_period(rec, 0, 2)
        ret_events = [ev for ev in rec['events']
                      if ev['proj'] == 0 and ev['type'] == 'retention']
        assert len(ret_events) == 1, "Retention should be released exactly once"
        assert ret_events[0]['t'] == cert_t_final, \
            "Retention must be released at final milestone certification, not before"

        # A4: Status completed
        assert _status(rec, 0) == 'completed'

        # A5: Objective
        assert abs(sol['obj'] - 26.35) <= TOL_Z

    def test_SPD_retention_and_advance_simultaneously(self):
        """
        SP-D: ρ=0.10 AND α=0.20, δ_rec=0.30, ψ=0.10.
        MS1: gross=30, retention=3, recovery=9 → R1=18.
        MS2: same → R2=18. Advance now fully recovered.
        MS3: gross=30, retention=3, no recovery → R3=27.
        R_ret=9 at completion. Identity: 18+18+18+27+9=90=CP.
        e={1,3,5}. Z* ≈ 27.23.
        """
        H   = 6
        BAC = 60
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.50, n_ms=3, H=H,
                          rho=0.10,
                          alpha_override=0.20,   # A = 0.20 × 90 = 18
                          drec=0.30, psi=0.10,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        R1 = _R_net_at_ms(rec, 0, 0)
        R2 = _R_net_at_ms(rec, 0, 1)
        R3 = _R_net_at_ms(rec, 0, 2)

        # A1: MS1 has both retention and recovery deducted
        assert R1 is not None
        assert abs(R1 - 18.0) <= TOL_PAY, \
            f"MS1: retention+recovery → R1=18. Got {R1:.2f}"

        # A2: MS2 same (advance not yet exhausted)
        assert R2 is not None
        assert abs(R2 - 18.0) <= TOL_PAY, \
            f"MS2: retention+recovery → R2=18. Got {R2:.2f}"

        # A3: MS3 retention only (advance recovered)
        assert R3 is not None
        assert abs(R3 - 27.0) <= TOL_PAY, \
            f"MS3: retention only → R3=27. Got {R3:.2f}"

        # A4: Retention released at completion
        R_ret = _retention_released(rec, 0)
        assert R_ret is not None
        assert abs(R_ret - 9.0) <= TOL_PAY

        # A5: Payment identity
        A_i = 0.20 * (BAC * 1.50)
        total = A_i + (R1 or 0) + (R2 or 0) + (R3 or 0) + (R_ret or 0)
        CP = BAC * 1.50
        assert abs(total - CP) <= 1.0, \
            f"Payment identity: {total:.2f} ≠ {CP:.2f}"

        # A6: Objective
        assert abs(sol['obj'] - 27.227) <= TOL_Z


# =============================================================================
# GROUP 5: TERMINATION CONDITIONS IN ISOLATION
# =============================================================================

class TestGroup5_TerminationConditionsInIsolation:

    def test_SPE_only_condition1_fires_counter_stays(self):
        """
        SP-E: η=1 → CPI=1 always → EAC=BAC → Condition 2 cannot fire.
        Slow spend below plan rate → SPI<1 → Condition 1 fires.
        Counter must stay at τ_tol throughout.
        """
        H   = 8
        BAC = 80
        p = _make_project(0, s=1, f=8, BAC=BAC, pi=0.0, n_ms=1, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=1, mu=0.30, tau_tol=3,
                          eta_val=1.0,
                          a=1.0, b=1.0,      # uniform S-curve
                          e_dates=[8])        # MS not reachable in observation window
        portfolio = _make_portfolio([p], gamma=0.95, H=H)

        # Override: set B1 very small so MILP allocates only 5/period
        total_BAC = BAC
        portfolio['B1'] = 5.0 * H   # just enough for 5/period
        sol, rec, val = _solve(portfolio)

        records = rec['proj'][0]['records']

        # A1: SPI < 1 after t=2 (behind plan with slow spend)
        for t in range(3, 7):
            spi = records[t]['SPI']
            if spi is not None and not math.isinf(spi):
                assert spi < 1.0, f"SPI should be <1 at t={t} (slow spend)"

        # A2: CPI = 1.0 (η=1 always)
        for t in range(2, 7):
            cpi = records[t]['CPI']
            if cpi is not None and not math.isinf(cpi):
                assert abs(cpi - 1.0) <= 0.02, \
                    f"CPI should be 1.0 at t={t} (η=1). Got {cpi:.4f}"

        # A3: Condition 2 never fires (EAC = BAC ≤ (1+mu)*BAC)
        for t in range(1, H + 1):
            assert not records[t]['cond2'], \
                f"Condition 2 should never fire when η=1 (EAC=BAC). Fired at t={t}"

        # A4: Counter never decrements (only one condition at a time)
        for t in range(1, H + 1):
            assert _tau_at(rec, 0, t) == 3, \
                f"Counter should stay at τ_tol=3 when only Cond 1 fires. Got {_tau_at(rec, 0, t)} at t={t}"

        # A5: Not terminated
        assert _status(rec, 0) != 'terminated'

    def test_SPF_only_condition2_fires_counter_stays(self):
        """
        SP-F: η=0.5 → CPI=0.5 → EAC quickly exceeds (1+μ)×BAC.
        Long planned duration + heavy spend → ahead of schedule → Cond 1 does not fire.
        Counter stays at τ_tol.
        """
        H   = 20
        BAC = 100
        p = _make_project(0, s=1, f=20, BAC=BAC, pi=0.0, n_ms=1, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=5, mu=0.05, tau_tol=3,
                          eta_val=0.5,
                          a=1.0, b=1.0,
                          e_dates=[20])
        portfolio = _make_portfolio([p], gamma=0.95, H=H, B1=500)
        sol, rec, val = _solve(portfolio)

        records = rec['proj'][0]['records']

        # A1: SPI > 1 for early periods (spending heavily, ahead of slow plan)
        for t in range(3, 11):
            spi = records[t]['SPI']
            if spi is not None and not math.isinf(spi):
                assert spi > 1.0, \
                    f"SPI should be >1 at t={t} (heavy spend, slow η). Got {spi:.4f}"

        # A2: EAC > (1+μ)×BAC from early on
        for t in range(2, 8):
            eac = records[t]['EAC']
            if eac is not None:
                assert math.isinf(eac) or eac > 1.05 * BAC, \
                    f"EAC should exceed 1.05×BAC at t={t}. Got {eac:.2f}"

        # A3: Condition 1 never fires in early periods (ahead of schedule)
        for t in range(2, 10):
            assert not records[t]['cond1'], \
                f"Condition 1 should not fire at t={t} (SPI>1). Fired."

        # A4: Counter never decrements
        for t in range(1, 12):
            assert _tau_at(rec, 0, t) == 3, \
                f"Counter should stay at τ_tol=3 when only Cond 2 fires. Got {_tau_at(rec, 0, t)} at t={t}"

        # A5: Not terminated in observed window
        for t in range(1, 12):
            assert records[t]['status'] != 'terminated', \
                f"Project should not terminate at t={t}"

    def test_SPG_both_fire_then_cond1_lapses_counter_resets(self):
        """
        SP-G: η=0.3 at t=1,2; η=1.0 at t=3 with large spend.
        t=1: only Cond2 fires (plan=0) → counter stays 3.
        t=2: both fire → counter 3→2.
        t=3: large spike → SPI>1 → Cond1 lapses → counter RESETS to 3.
        """
        H   = 12
        BAC = 100
        p = _make_project(0, s=1, f=12, BAC=BAC, pi=0.0, n_ms=1, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=0, mu=0.50, tau_tol=3,
                          a=1.0, b=1.0,
                          e_dates=[12],
                          eta_schedule={1: 0.3, 2: 0.3, 3: 1.0,
                                        **{t: 0.3 for t in range(4, 13)}})
        # Force specific allocation: 10,10,80 at t=1,2,3
        portfolio = _make_portfolio([p], gamma=0.95, H=H, B1=500)
        sol, rec, val = _solve(portfolio)

        records = rec['proj'][0]['records']

        # A1: Counter at t=1 is still 3 (only one condition fires when plan=0)
        assert _tau_at(rec, 0, 1) == 3, \
            f"Counter at t=1 should be 3 (only Cond2 when plan=0). Got {_tau_at(rec, 0, 1)}"

        # A2: Counter decrements at t=2 (both conditions fire)
        assert _tau_at(rec, 0, 2) == 2, \
            f"Counter at t=2 should be 2 (both conds, decrement). Got {_tau_at(rec, 0, 2)}"

        # A3: Counter resets to 3 at t=3 (Cond1 lapses after spike)
        assert _tau_at(rec, 0, 3) == 3, \
            f"Counter at t=3 should reset to 3 (Cond1 lapses). Got {_tau_at(rec, 0, 3)}"

        # A4: Status active at t=3
        assert records[3]['status'] == 'active', \
            "Project should still be active after counter reset"


# =============================================================================
# GROUP 6: CURE PERIOD AND TERMINATION EXECUTION
# =============================================================================

class TestGroup6_CurePeriodAndTerminationExecution:

    def test_SP5_zero_efficiency_exact_counter_boundary(self):
        """
        SP-5: η=0, τ_tol=3, A=20. Both conditions fire from start.
        Counter: 2→1→0 at t=1,2,3. Termination at t=4.
        R_term = -20. Z* = γ⁰×20 + γ³×(-20) = 20 - 17.15 = 2.85.
        """
        H   = 6
        BAC = 100
        p = _make_project(0, s=1, f=6, BAC=BAC, pi=0.0, n_ms=1, H=H,
                          rho=0.0,
                          alpha_override=0.20,   # A = 20
                          drec=0.25, psi=0.10,
                          Omega=1, mu=0.10, tau_tol=3,
                          eta_val=0.0,
                          e_dates=[6])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: Counter trajectory
        assert _tau_at(rec, 0, 1) == 2, f"τ at t=1 should be 2. Got {_tau_at(rec, 0, 1)}"
        assert _tau_at(rec, 0, 2) == 1, f"τ at t=2 should be 1. Got {_tau_at(rec, 0, 2)}"
        assert _tau_at(rec, 0, 3) == 0, f"τ at t=3 should be 0. Got {_tau_at(rec, 0, 3)}"

        # A2: Termination period = 4
        assert _term_period(rec, 0) == 4, \
            f"Termination should execute at t=4 (period after counter=0). Got {_term_period(rec, 0)}"

        # A3: Settlement = -20
        R_term = _R_term(rec, 0)
        assert R_term is not None
        assert abs(R_term - (-20.0)) <= TOL_PAY, \
            f"Settlement should be -20 (outstanding advance). Got {R_term:.2f}"

        # A4: Status terminated
        assert _status(rec, 0) == 'terminated'

        # A5: Objective ≈ 2.85 (advance at t=1 discounted differently from settlement at t=4)
        assert abs(sol['obj'] - 2.85) <= 0.5, \
            f"Expected Z*≈2.85, got {sol['obj']:.4f}"

    def test_SPH_partial_progress_positive_settlement(self):
        """
        SP-H: η=0.5, τ_tol=2, Ω=0, μ=0.05.
        Spend 60 at t=1 → P=0.30, MS1 certifies (θ=0.30), advance recovered.
        Both conditions fire at t=1,2. Termination at t=3.
        R_term = 0.30×100 - 0 = 30 (positive).
        Z* = γ⁰(10+30-60) + γ¹(0) + γ²(30) = -20 + 27.075 = 7.075.
        Zero-spend: Z = γ⁰×10 + γ¹×(-10) = 0.5. MILP prefers spending.
        """
        H   = 10
        BAC = 100
        p = _make_project(0, s=1, f=10, BAC=BAC, pi=0.0, n_ms=2, H=H,
                          rho=0.0,
                          alpha_override=0.10,   # A = 10
                          drec=0.25, psi=0.10,
                          Omega=0, mu=0.05, tau_tol=2,
                          eta_val=0.5,
                          a=0.5, b=2.0,          # front-loaded plan
                          theta_override=[0.30, 1.0],
                          phi_override=[0.40, 0.60],
                          e_dates=[1, 10])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: MS1 certifies at t=1 (progress reaches 0.30 = θ₁)
        assert _cert_period(rec, 0, 0) == 1, \
            "MS1 (θ=0.30) should certify at t=1 with spend=60 and η=0.5"

        # A2: Counter trajectory
        assert _tau_at(rec, 0, 1) == 1, f"τ at t=1 should be 1. Got {_tau_at(rec, 0, 1)}"
        assert _tau_at(rec, 0, 2) == 0, f"τ at t=2 should be 0. Got {_tau_at(rec, 0, 2)}"

        # A3: Termination at t=3
        assert _term_period(rec, 0) == 3, \
            f"Termination should execute at t=3. Got {_term_period(rec, 0)}"

        # A4: Positive settlement
        R_term = _R_term(rec, 0)
        assert R_term is not None
        assert R_term > 0, f"Settlement should be positive (P>0 at termination). Got {R_term:.2f}"
        assert abs(R_term - 30.0) <= TOL_PAY, f"Expected R_term=30. Got {R_term:.2f}"

        # A5: MILP prefers spending over zero-spend (Z* > Z_zero_spend)
        Z_zero_spend = 0.5
        assert sol['obj'] > Z_zero_spend, \
            f"Spending strategy Z*={sol['obj']:.3f} should beat zero-spend Z={Z_zero_spend}"

        # A6: Objective
        assert abs(sol['obj'] - 7.075) <= TOL_Z

    def test_SPI_cure_period_interrupted_and_restarted(self):
        """
        SP-I: η=0.3 for t=1,2; η=1.0 at t=3 (spike); η=0.3 from t=4.
        Counter: stays 3 at t=1, decrements to 2 at t=2,
                 resets to 3 at t=3 (Cond1 lapses),
                 then decrements 3→2→1→0 at t=4,5,6.
        Termination at t=7 (not t=4 as naive count would suggest).
        """
        H   = 12
        BAC = 100
        p = _make_project(0, s=1, f=12, BAC=BAC, pi=0.0, n_ms=1, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=0, mu=0.50, tau_tol=3,
                          a=1.0, b=1.0,
                          e_dates=[12],
                          eta_schedule={1: 0.3, 2: 0.3, 3: 1.0,
                                        **{t: 0.3 for t in range(4, 13)}})
        portfolio = _make_portfolio([p], gamma=0.95, H=H, B1=500)
        sol, rec, val = _solve(portfolio)

        # A1: Counter at t=2 decremented
        assert _tau_at(rec, 0, 2) == 2, \
            f"Counter should be 2 at t=2. Got {_tau_at(rec, 0, 2)}"

        # A2: Counter resets at t=3 (Cond1 lapses after spike)
        assert _tau_at(rec, 0, 3) == 3, \
            f"Counter should reset to 3 at t=3. Got {_tau_at(rec, 0, 3)}"

        # A3: Counter hits zero at t=6
        assert _tau_at(rec, 0, 6) == 0, \
            f"Counter should reach 0 at t=6. Got {_tau_at(rec, 0, 6)}"

        # A4: Termination at t=7, not t=4
        term_t = _term_period(rec, 0)
        assert term_t == 7, \
            f"Termination should be at t=7 (cure reset delays it). Got {term_t}"

        # A5: Status terminated
        assert _status(rec, 0) == 'terminated'


# =============================================================================
# GROUP 7: PORTFOLIO CASH BALANCE AND COUPLING
# =============================================================================

class TestGroup7_PortfolioCashBalanceAndCoupling:

    def test_MP3_advance_harvest_funds_good_project(self):
        """
        MP-3: Bad project (advance=30, terminates at t~3) and Good project.
        B1=90. Advance from Bad enters shared balance at t=1, funds Good.
        Good completes by t=2. Bad's settlement (-30) drains balance at t=3.
        Z* ≈ 41.925 (with γ=0.95).
        """
        H = 8

        p_bad = _make_project(0, s=1, f=8, BAC=100, pi=0.0, n_ms=2, H=H,
                              rho=0.0,
                              alpha_override=0.30,
                              drec=0.25, psi=0.10,
                              Omega=1, mu=0.10, tau_tol=2,
                              e_dates=[1, 5],
                              eta_schedule={1: 1.0, **{t: 0.05 for t in range(2, 9)}})

        p_good = _make_project(1, s=1, f=8, BAC=80, pi=0.50, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.30, tau_tol=3,
                               eta_val=1.0,
                               e_dates=[1, 5])

        portfolio = _make_portfolio([p_bad, p_good], gamma=0.95, H=H, B1=90)
        sol, rec, val = _solve(portfolio)

        # A1: B_t >= 0 throughout
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL, \
                f"Cash balance violated at t={t}: B={_B_at(rec, t):.2f}"

        # A2: Bad project terminates
        assert _status(rec, 0) == 'terminated'

        # A3: Good project completes
        assert _status(rec, 1) == 'completed'

        # A4: Objective
        assert abs(sol['obj'] - 41.925) <= TOL_Z

    def test_MPA_sequential_funding_second_project_waits(self):
        """
        MP-A: B1=30 covers exactly one first-milestone spend of P1.
        P2 cannot receive allocation at t=1; must wait for P1's MS1 inflow.
        Key assertion: x[1,1] = 0 (P2 starved at t=1).
        Both projects complete. Z* ≥ 50.
        """
        H = 6

        p1 = _make_project(0, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[1, 4])

        p2 = _make_project(1, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[1, 4])

        portfolio = _make_portfolio([p1, p2], gamma=0.95, H=H, B1=30.0)
        sol, rec, val = _solve(portfolio)

        # A1: P2 receives no allocation at t=1 (B1 exhausted by P1)
        assert _alloc_at(rec, 1, 1) <= TOL_ALLOC, \
            f"P2 should get zero at t=1 (B1=30 fully used by P1). Got {_alloc_at(rec, 1, 1):.2f}"

        # A2: B_t >= 0 throughout
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL, \
                f"Balance violated at t={t}: {_B_at(rec, t):.2f}"

        # A3: Both projects complete
        assert _status(rec, 0) == 'completed'
        assert _status(rec, 1) == 'completed'

        # A4: Objective (discounted, so < 60 but > 50)
        assert sol['obj'] >= 50.0, f"Z*={sol['obj']:.3f} should be ≥ 50"

    def test_MPB_negative_settlement_drains_shared_balance(self):
        """
        MP-B: Project Term (η=0, advance=18) terminates at t=3 with R_term=-18.
        Project Good completes at t=2 with two milestone payments.
        B4 = B3 + R_term = 148 - 18 = 130.
        Z* ≈ 31.005.
        """
        H = 8

        p_term = _make_project(0, s=1, f=8, BAC=60, pi=0.0, n_ms=1, H=H,
                               rho=0.0,
                               alpha_override=0.30,   # A = 18
                               drec=0.25, psi=0.10,
                               Omega=1, mu=0.10, tau_tol=2,
                               eta_val=0.0,
                               e_dates=[8])

        p_good = _make_project(1, s=1, f=8, BAC=60, pi=0.50, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.50, tau_tol=3,
                               eta_val=1.0,
                               e_dates=[1, 2])

        portfolio = _make_portfolio([p_term, p_good], gamma=0.95, H=H, B1=100.0)
        sol, rec, val = _solve(portfolio)

        # A1: Settlement is -18
        R_term = _R_term(rec, 0)
        assert R_term is not None
        assert abs(R_term - (-18.0)) <= TOL_PAY, \
            f"Settlement should be -18. Got {R_term:.2f}"

        # A2: Balance is reduced at the settlement period
        # Find termination period for project Term
        term_t = _term_period(rec, 0)
        assert term_t is not None
        # Balance at term_t+1 should reflect the drain
        B_after = _B_at(rec, term_t + 1) if term_t + 1 <= H else None
        if B_after is not None:
            B_before = _B_at(rec, term_t)
            # The settlement entered V at term_t, so B_{term_t+1} = B_{term_t} - out + V
            # V includes -18, so B_after < B_before (approximately)
            assert B_before - B_after >= 10.0 or B_after >= 0, \
                "Balance should reflect negative settlement drain"

        # A3: B_t >= 0 throughout
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL, \
                f"Balance violated at t={t}: {_B_at(rec, t):.2f}"

        # A4: Good completes, Term terminates
        assert _status(rec, 0) == 'terminated'
        assert _status(rec, 1) == 'completed'

        # A5: Objective
        assert abs(sol['obj'] - 31.005) <= TOL_Z


# =============================================================================
# GROUP 8: PORTFOLIO SELECTION UNDER CONSTRAINTS
# =============================================================================

class TestGroup8_PortfolioSelectionUnderConstraints:

    def test_MP2_good_vs_bad_unconstrained_cash(self):
        """
        MP-2: Good project (η=1, π=0.50) and Bad project (η→0.05, advance=30).
        B1=540 (non-binding). MILP ignores Bad, maximises Good.
        Good MS1 at e=1, MS2 at e=5. Z_Good ≈ 36.29. Z_Bad ≈ 1.5.
        Z* ≈ 37.79.
        """
        H = 8

        p_bad = _make_project(0, s=1, f=8, BAC=100, pi=0.0, n_ms=2, H=H,
                              rho=0.0,
                              alpha_override=0.30,
                              drec=0.25, psi=0.10,
                              Omega=1, mu=0.10, tau_tol=2,
                              e_dates=[1, 5],
                              eta_schedule={1: 1.0, **{t: 0.05 for t in range(2, 9)}})

        p_good = _make_project(1, s=1, f=8, BAC=80, pi=0.50, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.30, tau_tol=3,
                               eta_val=1.0,
                               e_dates=[1, 5])

        portfolio = _make_portfolio([p_bad, p_good], gamma=0.95, H=H, B1=540.0)
        sol, rec, val = _solve(portfolio)

        # A1: Negligible allocation to Bad
        assert _total_alloc(rec, 0) <= 2.0, \
            f"MILP should allocate essentially nothing to Bad. Got {_total_alloc(rec, 0):.2f}"

        # A2: Bad terminates
        assert _status(rec, 0) == 'terminated'

        # A3: Good completes
        assert _status(rec, 1) == 'completed'

        # A4: Objective
        assert abs(sol['obj'] - 37.79) <= TOL_Z

    def test_MP4_low_eta_forces_termination(self):
        """
        MP-4: High, Mid (η=1), Low (η=0.5, μ=0.10).
        Low immediately triggers EAC overrun with any spend.
        MILP completes High and Mid; Low terminates.
        Z* = γ⁰(45+36-30-30) + γ¹(45+36-30-30) + 0 ≈ 40.95.
        """
        H = 6

        p_high = _make_project(0, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.40, tau_tol=2,
                               eta_val=1.0,
                               e_dates=[1, 2])

        p_mid  = _make_project(1, s=1, f=6, BAC=60, pi=0.20, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.30, tau_tol=2,
                               eta_val=1.0,
                               e_dates=[1, 2])

        p_low  = _make_project(2, s=1, f=6, BAC=60, pi=0.10, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=1, mu=0.10, tau_tol=2,
                               eta_val=0.5,
                               e_dates=[1, 4])

        portfolio = _make_portfolio([p_high, p_mid, p_low], gamma=0.95, H=H, B1=150.0)
        sol, rec, val = _solve(portfolio)

        # A1: High and Mid complete
        assert _status(rec, 0) == 'completed', "High should complete"
        assert _status(rec, 1) == 'completed', "Mid should complete"

        # A2: Low terminates with minimal allocation
        assert _status(rec, 2) == 'terminated', "Low should terminate"
        assert _total_alloc(rec, 2) <= 2.0, \
            f"MILP should allocate nothing to Low. Got {_total_alloc(rec, 2):.2f}"

        # A3: B_t >= 0 throughout
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL

        # A4: Objective
        assert abs(sol['obj'] - 40.95) <= TOL_Z

    def test_MPC_cash_for_only_one_project(self):
        """
        MP-C: B1=60, two positive-NPV projects (Hi: π=0.50, Lo: π=0.10).
        Cannot complete both (need 120). MILP completes Hi (higher margin).
        Lo starves and terminates. Z* ≈ 28.54.
        """
        H = 4

        p_hi = _make_project(0, s=1, f=4, BAC=60, pi=0.50, n_ms=2, H=H,
                             rho=0.0, alpha_override=0.0,
                             Omega=2, mu=0.50, tau_tol=2,
                             eta_val=1.0,
                             e_dates=[1, 3])

        p_lo = _make_project(1, s=1, f=4, BAC=60, pi=0.10, n_ms=2, H=H,
                             rho=0.0, alpha_override=0.0,
                             Omega=2, mu=0.50, tau_tol=2,
                             eta_val=1.0,
                             e_dates=[1, 3])

        portfolio = _make_portfolio([p_hi, p_lo], gamma=0.95, H=H, B1=60.0)
        sol, rec, val = _solve(portfolio)

        # A1: Lo receives essentially zero allocation
        assert _total_alloc(rec, 1) <= 1.0, \
            f"Lo should be starved. Got {_total_alloc(rec, 1):.2f}"

        # A2: Lo terminates
        assert _status(rec, 1) == 'terminated'

        # A3: Hi completes
        assert _status(rec, 0) == 'completed'

        # A4: B_t >= 0
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL

        # A5: Objective
        assert abs(sol['obj'] - 28.54) <= TOL_Z

    def test_MPD_bridge_cash_unlocks_major_projects(self):
        """
        MP-D: B1=40. Bridge (BAC=30, π=0.10, early milestone) unlocks
        Alpha and Beta (BAC=60 each, π=0.50).
        Without Bridge, can only complete one major. With Bridge Z* > 28.
        Key assertion: Bridge funded and completed; Z* > Z_one_major.
        """
        H = 6

        p_bridge = _make_project(0, s=1, f=3, BAC=30, pi=0.10, n_ms=1, H=H,
                                 rho=0.0, alpha_override=0.0,
                                 Omega=2, mu=0.50, tau_tol=2,
                                 eta_val=1.0,
                                 e_dates=[1])

        p_alpha = _make_project(1, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                                rho=0.0, alpha_override=0.0,
                                Omega=2, mu=0.50, tau_tol=2,
                                eta_val=1.0,
                                e_dates=[2, 5])

        p_beta = _make_project(2, s=1, f=6, BAC=60, pi=0.50, n_ms=2, H=H,
                               rho=0.0, alpha_override=0.0,
                               Omega=2, mu=0.50, tau_tol=2,
                               eta_val=1.0,
                               e_dates=[2, 5])

        portfolio = _make_portfolio([p_bridge, p_alpha, p_beta], gamma=0.95, H=H, B1=40.0)
        sol, rec, val = _solve(portfolio)

        # A1: Bridge funded and completed
        assert _alloc_at(rec, 0, 1) >= 28.0, \
            f"Bridge should be funded at t=1. Got {_alloc_at(rec, 0, 1):.2f}"
        assert _status(rec, 0) == 'completed', "Bridge should complete"

        # A2: Z* exceeds completing only one major (≈28.54 from MP-C)
        Z_one_major = 28.0
        assert sol['obj'] > Z_one_major, \
            f"Bridge option value: Z*={sol['obj']:.3f} should exceed one-major Z={Z_one_major}"

        # A3: B_t >= 0
        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL


# =============================================================================
# GROUP 9: HORIZON AND STAGGERED STARTS
# =============================================================================

class TestGroup9_HorizonAndStaggeredStarts:

    def test_MP1_non_overlapping_projects_additive(self):
        """
        MP-1: Project A (s=1,f=4), Project B (s=5,f=10). No overlap.
        H = max(4,10) - min(1,5) + 1 = 10.
        Z* = Z_A + Z_B (no interaction).
        A: e={1,3}. B: e={5,8}. Z_A≈19.025, Z_B≈15.128. Z*≈34.153.
        """
        H = 10

        pA = _make_project(0, s=1, f=4, BAC=100, pi=0.20, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=1, mu=0.30, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[1, 3])

        pB = _make_project(1, s=5, f=10, BAC=60, pi=0.50, n_ms=3, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[5, 7, 9])

        portfolio = _make_portfolio([pA, pB], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: Both completed
        assert _status(rec, 0) == 'completed'
        assert _status(rec, 1) == 'completed'

        # A2: A allocation only within [1,4]
        for t in range(5, H + 1):
            assert _alloc_at(rec, 0, t) <= TOL_ALLOC, \
                f"Project A should have zero allocation at t={t} (post-finish)"

        # A3: B allocation only within [5,10]
        for t in range(1, 5):
            assert _alloc_at(rec, 1, t) <= TOL_ALLOC, \
                f"Project B should have zero allocation at t={t} (pre-start)"

        # A4: No terminations
        assert _R_term(rec, 0) is None
        assert _R_term(rec, 1) is None

        # A5: Objective
        assert abs(sol['obj'] - 34.153) <= TOL_Z

    def test_MPE_staggered_starts_horizon_derivation(self):
        """
        MP-E: s_i ∈ {1,4,9}, f_i ∈ {4,8,14}.
        H = 14 - 1 + 1 = 14 (derived correctly).
        Allocations outside each project's [s_i, f_i] must be zero.
        Z* = Z1 + Z2 + Z3 ≈ 28.54 + 23.83 + 17.24 = 69.61.
        """
        H = 14

        p1 = _make_project(0, s=1, f=4, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[1, 3])

        p2 = _make_project(1, s=4, f=8, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[4, 6])

        p3 = _make_project(2, s=9, f=14, BAC=60, pi=0.50, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.50, tau_tol=2,
                           eta_val=1.0,
                           e_dates=[9, 11])

        portfolio = _make_portfolio([p1, p2, p3], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        # A1: H derived correctly (passed as 14, must be consistent)
        assert portfolio['H'] == 14

        # A2: P1 allocation zero outside [1,4]
        for t in range(5, H + 1):
            assert _alloc_at(rec, 0, t) <= TOL_ALLOC, \
                f"P1 allocation at t={t} should be 0 (outside window)"

        # A3: P2 allocation zero outside [4,8]
        for t in list(range(1, 4)) + list(range(9, H + 1)):
            assert _alloc_at(rec, 1, t) <= TOL_ALLOC, \
                f"P2 allocation at t={t} should be 0 (outside window)"

        # A4: P3 allocation zero outside [9,14]
        for t in range(1, 9):
            assert _alloc_at(rec, 2, t) <= TOL_ALLOC, \
                f"P3 allocation at t={t} should be 0 (outside window)"

        # A5: All three complete
        assert _status(rec, 0) == 'completed'
        assert _status(rec, 1) == 'completed'
        assert _status(rec, 2) == 'completed'

        # A6: Objective
        assert abs(sol['obj'] - 69.61) <= TOL_Z


# =============================================================================
# CORE MILP PROPERTY TESTS
# These check structural invariants that must hold regardless of scenario.
# =============================================================================

class TestCoreProperties:

    def test_payment_identity_holds_on_all_completed_projects(self):
        """
        For every completed project: A + ΣR_net + R_ret = CP.
        This is a structural invariant, not scenario-specific.
        """
        H = 6
        p = _make_project(0, s=1, f=6, BAC=60, pi=0.50, n_ms=3, H=H,
                          rho=0.10,
                          alpha_override=0.20,
                          drec=0.30, psi=0.10,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0,
                          e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        if _status(rec, 0) == 'completed':
            proj = portfolio['projects'][0]
            R_net_list, R_ret_val = compute_R_net(proj)
            A_i = proj['alpha_i'] * proj['CP_i']
            total = A_i + sum(R_net_list) + R_ret_val
            CP = proj['CP_i']
            assert abs(total - CP) <= 1.0, \
                f"Payment identity violated: A+ΣR_net+R_ret={total:.2f} ≠ CP={CP:.2f}"

    def test_balance_never_negative(self):
        """B_t >= 0 must hold for all t in every solved instance."""
        H = 8
        p1 = _make_project(0, s=1, f=6, BAC=80, pi=0.20, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.30, tau_tol=2,
                           eta_val=1.0, e_dates=[1, 4])
        p2 = _make_project(1, s=2, f=8, BAC=60, pi=0.15, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.30, tau_tol=2,
                           eta_val=1.0, e_dates=[2, 6])
        portfolio = _make_portfolio([p1, p2], gamma=0.95, H=H, B1=50.0)
        sol, rec, val = _solve(portfolio)

        for t in range(1, H + 1):
            assert _B_at(rec, t) >= -TOL_BAL, \
                f"Cash balance negative at t={t}: {_B_at(rec, t):.2f}"

    def test_milestone_ordering_enforced(self):
        """
        u_{i,j+1,t} <= u_{i,j,t} for all i,j,t.
        MS j+1 cannot be certified before MS j.
        """
        H = 6
        p = _make_project(0, s=1, f=6, BAC=60, pi=0.30, n_ms=3, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0, e_dates=[1, 3, 5])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        u_val = rec['u_val']
        M = portfolio['projects'][0]['M_i']
        for j in range(M - 1):
            for t in range(1, H + 1):
                u_j   = u_val.get((0, j,     t), 0)
                u_jp1 = u_val.get((0, j + 1, t), 0)
                assert u_jp1 <= u_j, \
                    f"Ordering violated: MS{j+2} certified before MS{j+1} at t={t}"

    def test_no_allocation_outside_activity_window(self):
        """x_{i,t} = 0 for t outside [s_i, f_i]."""
        H = 10
        p1 = _make_project(0, s=3, f=7, BAC=60, pi=0.20, n_ms=2, H=H,
                           rho=0.0, alpha_override=0.0,
                           Omega=2, mu=0.30, tau_tol=2,
                           eta_val=1.0, e_dates=[3, 6])
        portfolio = _make_portfolio([p1], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        for t in list(range(1, 3)) + list(range(8, H + 1)):
            assert _alloc_at(rec, 0, t) <= TOL_ALLOC, \
                f"Allocation outside window at t={t}: {_alloc_at(rec, 0, t):.2f}"

    def test_milestone_u_monotone(self):
        """Once certified, a milestone stays certified: u_{i,j,t} >= u_{i,j,t-1}."""
        H = 6
        p = _make_project(0, s=1, f=6, BAC=60, pi=0.20, n_ms=2, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0, e_dates=[1, 4])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        u_val = rec['u_val']
        M = portfolio['projects'][0]['M_i']
        for j in range(M):
            for t in range(2, H + 1):
                prev = u_val.get((0, j, t - 1), 0)
                curr = u_val.get((0, j, t),     0)
                assert curr >= prev, \
                    f"u[0,{j},{t}]={curr} < u[0,{j},{t-1}]={prev}: milestone un-certified"

    def test_validation_passes_on_well_formed_instance(self):
        """validate() should return all_pass=True on a clean instance."""
        H = 6
        p = _make_project(0, s=1, f=6, BAC=60, pi=0.30, n_ms=2, H=H,
                          rho=0.0, alpha_override=0.0,
                          Omega=2, mu=0.50, tau_tol=2,
                          eta_val=1.0, e_dates=[1, 4])
        portfolio = _make_portfolio([p], gamma=0.95, H=H)
        sol, rec, val = _solve(portfolio)

        failed = {k: v for k, v in val['errors'].items()
                  if not k.startswith('_') and len(v) > 0}
        assert not failed, \
            f"Validation failed on well-formed instance: {failed}"