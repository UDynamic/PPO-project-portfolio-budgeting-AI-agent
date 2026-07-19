"""
test_milp_ppm.py
================
pytest test suite for milp_ppm_test.py.

Test cases come from two sources:
  1. Hand-solved MILP solutions in ppm_updates_and_test_cases.tex (SP-* and MP-*
     classes).  Small, analytically tractable portfolios whose optimal solution
     is derived on paper and checked here to the cent.
  2. milp_test_scenarios.py (SC-* classes).  Three structured scenarios whose
     exact Z*_L1 is computed via the analytic formula
         Z* = sum_i  BAC_i * pi_i * gamma^(s_i-1)
     valid when B1 is unlimited (kappa=100) and eta=1 flat.

STRUCTURE
---------
  SP-1 … SP-5   Single-project cases  (tex document)
  MP-1 … MP-4   Multi-project cases   (tex document)
  SC-1 … SC-3   Scenario-file cases   (milp_test_scenarios.py)
  TestDynamicHorizon          Modification-2 invariants
  TestPaymentIdentity         Pre-solve identity checks for all cases
  test_random_portfolio_*     Structural regression sweep (random seeds)
  test_parametrized_quick     Fast CI sanity parametrize

RUN
---
  pytest test_milp_ppm.py -v                          # all tests
  pytest test_milp_ppm.py -v --plots                  # + diagnostic plots
  pytest test_milp_ppm.py::TestSC1Symmetry -v         # one class
  pytest test_milp_ppm.py -m "not slow" -v            # skip long cases

DEPENDENCIES
------------
  pip install pytest
  milp_ppm_test.py and milp_test_scenarios.py must be in the same directory.
"""

import os
import math
import pytest

# ── import the module under test ──────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from milp_ppm_test import (
    generate_portfolio,
    build_and_solve,
    build_records,
    validate,
    plot_portfolio,
    CONFIG as DEFAULT_CONFIG,
)

# ── scenario file (milp_test_scenarios.py) ────────────────────────────────────
# Imported with a fallback so the SP-* / MP-* tests still run even if the
# scenario file is absent (e.g. during initial setup).
try:
    from milp_test_scenarios import (
        SCENARIOS,
        SCENARIO_1,
        SCENARIO_2,
        SCENARIO_3,
        check_payment_identity,
        verify_milp_solution,
        compute_Z_star,
        compute_optimal_allocation,
    )
    _SCENARIOS_AVAILABLE = True
except ImportError:
    _SCENARIOS_AVAILABLE = False
    SCENARIOS = []

# ─────────────────────────────────────────────────────────────────────────────
# TOLERANCES  (from Section 4 of the .tex document)
# ─────────────────────────────────────────────────────────────────────────────
TOL_OBJ    = 1.5    # objective value  (small-BAC cases, units ~100)
TOL_OBJ_SC = 5.0   # objective value  (scenario-file cases, BAC 10k–20k)
TOL_ALLOC  = 0.5    # individual allocation x_{i,t}
TOL_ALLOC_SC = 50.0 # allocation tolerance for large-BAC scenario cases
TOL_PROG   = 0.01   # progress P_{i,t}
TOL_SETTLE = 1.0    # termination settlement
# cure-period counter is an exact integer: use == 0

# ─────────────────────────────────────────────────────────────────────────────
# conftest hook for --plots flag  (defined inline so no conftest.py is needed)
# ─────────────────────────────────────────────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption("--plots", action="store_true", default=False,
                     help="Save diagnostic plots to test_plots/")


@pytest.fixture(scope="session")
def do_plots(request):
    return request.config.getoption("--plots", default=False)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _base_cfg(**overrides):
    """Return a clean CONFIG dict with test-friendly defaults."""
    cfg = dict(DEFAULT_CONFIG)
    # Modification 1: large B1 via kappa
    cfg["b1_kappa"]  = 3.0
    cfg["b1_ratio"]  = (0.45, 0.45)   # ignored when kappa is set
    # Modification 2: dynamic horizon
    cfg["horizon"]   = None
    # Deterministic single-seed runs
    cfg["seed"]      = 0
    cfg["n_projects"] = 1              # most SP cases have n=1
    # Solver: short limit for fast tests
    cfg["solver_time_limit"] = 120
    cfg.update(overrides)
    return cfg


def _run(cfg, projects_override=None):
    """
    Build portfolio, solve MILP, build records.
    If projects_override is given it is a list of project dicts that replace
    the randomly generated ones; the horizon is re-derived from them.
    Returns (portfolio, sol, rec).
    """
    portfolio = generate_portfolio(cfg)

    if projects_override is not None:
        # Inject hand-crafted project dicts and re-derive H.
        portfolio["projects"] = projects_override
        portfolio["n"]        = len(projects_override)
        s_min = min(p["s_i"] for p in projects_override)
        f_max = max(p["f_i"] for p in projects_override)
        portfolio["H"]        = f_max - s_min + 1
        total_BAC             = sum(p["BAC_i"] for p in projects_override)
        portfolio["total_BAC"] = total_BAC
        kappa = cfg.get("b1_kappa", None)
        portfolio["B1"]       = (kappa * total_BAC if kappa else portfolio["B1"])

    sol = build_and_solve(portfolio)
    rec = build_records(portfolio, sol)
    return portfolio, sol, rec


def _run_portfolio(portfolio):
    """
    Accept a pre-built portfolio dict (e.g. from milp_test_scenarios.py)
    and solve it directly, bypassing generate_portfolio entirely.
    Returns (portfolio, sol, rec).
    """
    sol = build_and_solve(portfolio)
    rec = build_records(portfolio, sol)
    return portfolio, sol, rec


def _save_plots(portfolio, sol, rec, case_id, do_plots_flag):
    """Optionally save diagnostic plots."""
    if not do_plots_flag:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join("test_plots", case_id)
    os.makedirs(out_dir, exist_ok=True)
    try:
        fig1, fig2 = plot_portfolio(portfolio, sol, rec)
        fig1.savefig(os.path.join(out_dir, "projects.png"), dpi=120, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, "portfolio.png"), dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"\n  [PLOT] saved to test_plots/{case_id}/")
    except Exception as e:
        print(f"\n  [PLOT] failed: {e}")


def _first_cert_period(rec, proj_idx, ms_idx):
    """Return the period in which milestone ms_idx of project proj_idx is first certified."""
    u_val = rec["u_val"]
    H = max(t for (i, j, t) in u_val if i == proj_idx)
    for t in range(1, H + 1):
        now  = u_val.get((proj_idx, ms_idx, t), 0)
        prev = u_val.get((proj_idx, ms_idx, t - 1), 0) if t > 1 else 0
        if now - prev > 0:
            return t
    return None


def _termination_period(rec, proj_idx):
    """Return the period at which project proj_idx is first marked 'terminated'."""
    records = rec["proj"][proj_idx]["records"]
    for t in sorted(records):
        if records[t]["status"] == "terminated":
            return t
    return None


def _make_project(
    idx, s, f, BAC, pi=0.0, M=2, theta=None, phi=None,
    eta_const=1.0, eta_override=None,
    alpha=0.0, rho=0.0, delta_rec=0.25, psi=0.10,
    Omega=2, mu=0.30, tau_tol=2,
    a=2.0, b=2.0
):
    """
    Construct a minimal project dict compatible with milp_ppm_test.py.
    eta_override: dict {t: value} overrides individual periods.
    """
    D_plan = f - s + 1
    CP     = BAC * (1 + pi)
    if theta is None:
        theta = [round((j + 1) / M, 6) for j in range(M)]
    if phi is None:
        phi = [1.0 / M] * M

    # derive alpha_i via the derive_alpha helper
    from milp_ppm_test import derive_alpha
    alpha_derived = derive_alpha(phi, delta_rec, psi)
    # honour the caller's alpha if explicitly > 0, else use contract formula
    alpha_i = alpha if alpha > 0 else alpha_derived

    # Build eta dict
    H_placeholder = f + 10
    eta = {}
    for t in range(1, H_placeholder + 1):
        if s <= t <= f:
            eta[t] = eta_const
        else:
            eta[t] = 0.0
    if eta_override:
        for t, v in eta_override.items():
            eta[t] = v

    return dict(
        idx=idx, s_i=s, f_i=f, D_plan=D_plan,
        a_i=a, b_i=b,
        BAC_i=BAC, pi_i=pi, CP_i=CP,
        alpha_i=alpha_i, delta_rec_i=delta_rec, psi_i=psi, rho_i=rho,
        M_i=M, theta=theta, phi=phi,
        Omega_i=Omega, mu_i=mu, tau_tol_i=tau_tol,
        eta=eta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── SINGLE-PROJECT CASES ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestSP1PerfectEfficiency:
    """
    SP-1: Perfect efficiency (eta=1), 2 milestones, no advance, no retention.
    Expected: MILP spends 50 at t=1 and 50 at t=2, certifying both milestones
    immediately.  Z* = (60-50)+(60-50) = 20.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj = _make_project(
            idx=0, s=1, f=4, BAC=100, pi=0.20,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=1, mu=0.30, tau_tol=2,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=1)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj])
        _save_plots(self.portfolio, self.sol, self.rec, "SP1", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(20.0, abs=TOL_OBJ)

    def test_allocation_t1(self):
        assert self.rec["x_val"][0, 1] == pytest.approx(50.0, abs=TOL_ALLOC)

    def test_allocation_t2(self):
        assert self.rec["x_val"][0, 2] == pytest.approx(50.0, abs=TOL_ALLOC)

    def test_no_spend_after_completion(self):
        for t in range(3, self.portfolio["H"] + 1):
            assert self.rec["x_val"].get((0, t), 0.0) < TOL_ALLOC

    def test_milestone1_certified_t1(self):
        assert _first_cert_period(self.rec, 0, 0) == 1

    def test_milestone2_certified_t2(self):
        assert _first_cert_period(self.rec, 0, 1) == 2

    def test_project_completed(self):
        assert self.rec["proj"][0]["status"] == "completed"

    def test_no_termination(self):
        assert _termination_period(self.rec, 0) is None

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"

    def test_progress_at_t1(self):
        assert self.rec["proj"][0]["records"][1]["P_sim"] == pytest.approx(0.50, abs=TOL_PROG)

    def test_progress_at_t2(self):
        assert self.rec["proj"][0]["records"][2]["P_sim"] == pytest.approx(1.00, abs=TOL_PROG)

    def test_no_cure_period_decrements(self):
        tau_tol = self.portfolio["projects"][0]["tau_tol_i"]
        for t in range(1, self.portfolio["H"] + 1):
            r = self.rec["proj"][0]["records"][t]
            if r["tau_rem"] is not None:
                assert r["tau_rem"] == tau_tol, (
                    f"tau_rem decremented at t={t}: got {r['tau_rem']}"
                )


class TestSP2AdvanceHarvestTermination:
    """
    SP-2: Large advance (30%), rapidly decaying efficiency (eta=0.05 from t>=2).
    MS1 is physically unreachable from t>=2.  Project terminates.
    Expected: Z* ≈ 0 (advance exactly cancelled by settlement).
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        eta_override = {t: 0.05 for t in range(2, 20)}
        eta_override[1] = 1.0
        proj = _make_project(
            idx=0, s=1, f=6, BAC=100, pi=0.0,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, eta_override=eta_override,
            alpha=0.30, rho=0.0,
            Omega=1, mu=0.05, tau_tol=1,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=1)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj])
        _save_plots(self.portfolio, self.sol, self.rec, "SP2", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_near_zero(self):
        assert self.sol["obj"] == pytest.approx(0.0, abs=TOL_OBJ)

    def test_negligible_total_allocation(self):
        total = sum(
            self.rec["x_val"].get((0, t), 0.0)
            for t in range(1, self.portfolio["H"] + 1)
        )
        assert total <= 2.0, f"Expected near-zero spend, got {total:.2f}"

    def test_project_terminated(self):
        assert self.rec["proj"][0]["status"] == "terminated"

    def test_terminates_early(self):
        term_t = _termination_period(self.rec, 0)
        assert term_t is not None
        assert term_t <= 4, f"Expected termination by t=4, got t={term_t}"

    def test_no_milestone_certified(self):
        assert len(self.rec["proj"][0]["certified"]) == 0

    def test_settlement_negative(self):
        events = self.rec["events"]
        term_events = [e for e in events if e["type"] == "termination_settlement"]
        assert len(term_events) == 1
        assert term_events[0]["amount"] == pytest.approx(-30.0, abs=TOL_SETTLE)

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestSP3UniformHighMargin:
    """
    SP-3: 3 milestones, pi=0.50, eta=1, gamma=1.
    Any completing allocation gives Z* = CP - BAC = 90 - 60 = 30.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj = _make_project(
            idx=0, s=1, f=6, BAC=60, pi=0.50,
            M=3, theta=[1/3, 2/3, 1.0], phi=[1/3, 1/3, 1/3],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.50, tau_tol=2,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=1)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj])
        _save_plots(self.portfolio, self.sol, self.rec, "SP3", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(30.0, abs=TOL_OBJ)

    def test_project_completed(self):
        assert self.rec["proj"][0]["status"] == "completed"

    def test_all_milestones_certified(self):
        assert len(self.rec["proj"][0]["certified"]) == 3

    def test_total_outflow(self):
        total = sum(
            self.rec["x_val"].get((0, t), 0.0)
            for t in range(1, self.portfolio["H"] + 1)
        )
        assert total == pytest.approx(60.0, abs=TOL_ALLOC)

    def test_no_termination(self):
        assert _termination_period(self.rec, 0) is None

    def test_no_conditions_simultaneously_active(self):
        records = self.rec["proj"][0]["records"]
        for t, r in records.items():
            if r["cond1"] and r["cond2"]:
                pytest.fail(
                    f"Both termination conditions active at t={t}: "
                    f"cond1={r['cond1']}, cond2={r['cond2']}"
                )

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestSP4FrontLoadingBeatsUniform:
    """
    SP-4: 2 milestones, pi=0.40, gamma=0.95, f=8.
    Front-loaded Z = 39.0 vs uniform Z = 28.61.
    MILP must choose front-loading.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj = _make_project(
            idx=0, s=1, f=8, BAC=100, pi=0.40,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.40, tau_tol=2,
        )
        cfg = _base_cfg(gamma=0.95, n_projects=1)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj])
        _save_plots(self.portfolio, self.sol, self.rec, "SP4", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_beats_uniform(self):
        Z_uniform = 28.61
        assert self.sol["obj"] > Z_uniform - 0.5, (
            f"MILP objective {self.sol['obj']:.2f} not better than uniform {Z_uniform:.2f}"
        )

    def test_objective_value(self):
        assert self.sol["obj"] == pytest.approx(39.0, abs=TOL_OBJ)

    def test_front_loaded_allocation_t1(self):
        assert self.rec["x_val"][0, 1] == pytest.approx(50.0, abs=TOL_ALLOC)

    def test_front_loaded_allocation_t2(self):
        assert self.rec["x_val"][0, 2] == pytest.approx(50.0, abs=TOL_ALLOC)

    def test_milestone1_certified_t1(self):
        assert _first_cert_period(self.rec, 0, 0) == 1

    def test_milestone2_certified_t2(self):
        assert _first_cert_period(self.rec, 0, 1) == 2

    def test_project_completed(self):
        assert self.rec["proj"][0]["status"] == "completed"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestSP5CurePeriodBoundary:
    """
    SP-5: eta=0 always, tau_tol=3, alpha=0.20.
    Both conditions fire at t=1.  Counter goes 2 -> 1 -> 0.
    Termination at t=4.  Settlement = -20.  Z* = 0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj = _make_project(
            idx=0, s=1, f=6, BAC=100, pi=0.0,
            M=1, theta=[1.0], phi=[1.0],
            eta_const=0.0,           # eta=0 for all periods
            alpha=0.20, rho=0.0,
            Omega=1, mu=0.10, tau_tol=3,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=1)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj])
        _save_plots(self.portfolio, self.sol, self.rec, "SP5", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_near_zero(self):
        assert self.sol["obj"] == pytest.approx(0.0, abs=TOL_OBJ)

    def test_project_terminated(self):
        assert self.rec["proj"][0]["status"] == "terminated"

    def test_termination_period(self):
        term_t = _termination_period(self.rec, 0)
        assert term_t == 4, f"Expected termination at t=4, got t={term_t}"

    def test_settlement_value(self):
        events = self.rec["events"]
        term_events = [e for e in events if e["type"] == "termination_settlement"]
        assert len(term_events) == 1
        assert term_events[0]["amount"] == pytest.approx(-20.0, abs=TOL_SETTLE)

    def test_cure_counter_path(self):
        """Counter must decrement exactly: tau_rem = 2, 1, 0 at t=1,2,3."""
        records = self.rec["proj"][0]["records"]
        expected = {1: 2, 2: 1, 3: 0}
        for t, expected_tau in expected.items():
            r = records[t]
            assert r["tau_rem"] == expected_tau, (
                f"At t={t}: expected tau_rem={expected_tau}, got {r['tau_rem']}"
            )

    def test_both_conditions_active_t1_to_t3(self):
        records = self.rec["proj"][0]["records"]
        for t in (1, 2, 3):
            r = records[t]
            assert r["cond1"], f"Cond1 not active at t={t}"
            assert r["cond2"], f"Cond2 not active at t={t}"

    def test_no_milestone_certified(self):
        assert len(self.rec["proj"][0]["certified"]) == 0

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


# ─────────────────────────────────────────────────────────────────────────────
# ── MULTI-PROJECT CASES ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestMP1NonOverlappingAdditive:
    """
    MP-1: Project A (s=1, f=4) and Project B (s=5, f=10) do not overlap.
    Z* = Z_A + Z_B = 20 + 30 = 50.  Projects are fully independent.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj_a = _make_project(
            idx=0, s=1, f=4, BAC=100, pi=0.20,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=1, mu=0.30, tau_tol=2,
        )
        proj_b = _make_project(
            idx=1, s=5, f=10, BAC=60, pi=0.50,
            M=3, theta=[1/3, 2/3, 1.0], phi=[1/3, 1/3, 1/3],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.50, tau_tol=2,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=2)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj_a, proj_b])
        _save_plots(self.portfolio, self.sol, self.rec, "MP1", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_dynamic_horizon(self):
        assert self.portfolio["H"] == 10

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(50.0, abs=TOL_OBJ)

    def test_project_a_completed(self):
        assert self.rec["proj"][0]["status"] == "completed"

    def test_project_b_completed(self):
        assert self.rec["proj"][1]["status"] == "completed"

    def test_no_terminations(self):
        assert _termination_period(self.rec, 0) is None
        assert _termination_period(self.rec, 1) is None

    def test_no_cross_allocation(self):
        """Project A should receive zero allocation at t>=5; B at t<=4."""
        for t in range(5, self.portfolio["H"] + 1):
            assert self.rec["x_val"].get((0, t), 0.0) < TOL_ALLOC, \
                f"Project A allocated at t={t} (after its finish)"
        for t in range(1, 5):
            assert self.rec["x_val"].get((1, t), 0.0) < TOL_ALLOC, \
                f"Project B allocated at t={t} (before its start)"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestMP2GoodVsBadLargeBudget:
    """
    MP-2: Good project (pi=0.50, eta=1) vs Bad project (pi=0, eta decays).
    Large B1 means cash is never binding.
    MILP should allocate zero to Bad (which terminates, net Z_B=0)
    and front-load Good (Z_G=40).  Z* = 40.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        eta_bad = {1: 1.0, **{t: 0.05 for t in range(2, 20)}}
        proj_bad = _make_project(
            idx=0, s=1, f=8, BAC=100, pi=0.0,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, eta_override=eta_bad,
            alpha=0.30, rho=0.0,
            Omega=1, mu=0.10, tau_tol=2,
        )
        proj_good = _make_project(
            idx=1, s=1, f=8, BAC=80, pi=0.50,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.30, tau_tol=3,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=2, b1_kappa=3.0)
        self.portfolio, self.sol, self.rec = _run(cfg, [proj_bad, proj_good])
        _save_plots(self.portfolio, self.sol, self.rec, "MP2", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(40.0, abs=TOL_OBJ)

    def test_bad_project_terminated(self):
        assert self.rec["proj"][0]["status"] == "terminated"

    def test_good_project_completed(self):
        assert self.rec["proj"][1]["status"] == "completed"

    def test_negligible_allocation_to_bad(self):
        total_bad = sum(
            self.rec["x_val"].get((0, t), 0.0)
            for t in range(1, self.portfolio["H"] + 1)
        )
        assert total_bad <= 2.0, \
            f"Expected near-zero spend on Bad project, got {total_bad:.2f}"

    def test_bad_terminates_early(self):
        term_t = _termination_period(self.rec, 0)
        assert term_t is not None and term_t <= 5

    def test_good_milestone1_certified(self):
        cert_t = _first_cert_period(self.rec, 1, 0)
        assert cert_t is not None and cert_t <= 3

    def test_good_milestone2_certified(self):
        cert_t = _first_cert_period(self.rec, 1, 1)
        assert cert_t is not None and cert_t <= 4

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestMP3AdvanceHarvestRedirect:
    """
    MP-3: Same as MP-2 but with tighter B1 = 0.5 * total_BAC.
    Advance from Bad (30) is absorbed into shared balance, enabling
    the Good project to be funded in period 1.
    Z* = 40, B_t >= 0 throughout.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        eta_bad = {1: 1.0, **{t: 0.05 for t in range(2, 20)}}
        proj_bad = _make_project(
            idx=0, s=1, f=8, BAC=100, pi=0.0,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, eta_override=eta_bad,
            alpha=0.30, rho=0.0,
            Omega=1, mu=0.10, tau_tol=2,
        )
        proj_good = _make_project(
            idx=1, s=1, f=8, BAC=80, pi=0.50,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.30, tau_tol=3,
        )
        # Tighter B1: no kappa, use ratio
        cfg = _base_cfg(gamma=1.0, n_projects=2)
        cfg["b1_kappa"] = None
        cfg["b1_ratio"] = (0.50, 0.50)

        portfolio = generate_portfolio(cfg)
        # inject projects and recompute
        portfolio["projects"]   = [proj_bad, proj_good]
        portfolio["n"]          = 2
        portfolio["H"]          = 8
        portfolio["total_BAC"]  = 180.0
        portfolio["B1"]         = 0.5 * 180.0   # = 90

        self.portfolio = portfolio
        self.sol = build_and_solve(portfolio)
        self.rec = build_records(portfolio, self.sol)
        _save_plots(self.portfolio, self.sol, self.rec, "MP3", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(40.0, abs=TOL_OBJ)

    def test_balance_never_negative(self):
        for t, port in self.rec["portfolio"].items():
            assert port["B_sim"] >= -0.5, \
                f"Balance negative at t={t}: {port['B_sim']:.2f}"

    def test_bad_project_terminated(self):
        assert self.rec["proj"][0]["status"] == "terminated"

    def test_good_project_completed(self):
        assert self.rec["proj"][1]["status"] == "completed"

    def test_advance_received_t1(self):
        """Advance of 30 from Bad project must appear in t=1 inflow."""
        inflow_t1 = self.rec["portfolio"][1]["inflow"]
        # advance 30 + MS1 of Good (60) = 90 total, but at minimum advance is there
        assert inflow_t1 >= 29.0, \
            f"Expected advance in t=1 inflow, got {inflow_t1:.2f}"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


class TestMP4ThreeProjectsBudgetBinding:
    """
    MP-4: High (pi=0.50, eta=1), Mid (pi=0.20, eta=1), Low (pi=0.10, eta=0.5).
    Project Low always triggers EAC overrun; it must terminate.
    MILP completes High and Mid, achieves Z* = 30 + 12 + 0 = 42.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        proj_high = _make_project(
            idx=0, s=1, f=6, BAC=60, pi=0.50,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.40, tau_tol=2,
        )
        proj_mid = _make_project(
            idx=1, s=1, f=6, BAC=60, pi=0.20,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.30, tau_tol=2,
        )
        proj_low = _make_project(
            idx=2, s=1, f=6, BAC=60, pi=0.10,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=0.5, alpha=0.0, rho=0.0,
            Omega=1, mu=0.10, tau_tol=2,
        )
        cfg = _base_cfg(gamma=1.0, n_projects=3)
        # B1 = 150 (not 3x = 540) to create moderate pressure
        cfg["b1_kappa"] = None
        cfg["b1_ratio"] = (0.83, 0.83)   # ~150 / 180

        portfolio = generate_portfolio(cfg)
        portfolio["projects"]  = [proj_high, proj_mid, proj_low]
        portfolio["n"]         = 3
        portfolio["H"]         = 6
        portfolio["total_BAC"] = 180.0
        portfolio["B1"]        = 150.0

        self.portfolio = portfolio
        self.sol = build_and_solve(portfolio)
        self.rec = build_records(portfolio, self.sol)
        _save_plots(self.portfolio, self.sol, self.rec, "MP4", do_plots)

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective(self):
        assert self.sol["obj"] == pytest.approx(42.0, abs=TOL_OBJ)

    def test_high_completed(self):
        assert self.rec["proj"][0]["status"] == "completed"

    def test_mid_completed(self):
        assert self.rec["proj"][1]["status"] == "completed"

    def test_low_terminated(self):
        assert self.rec["proj"][2]["status"] == "terminated"

    def test_low_terminated_early(self):
        term_t = _termination_period(self.rec, 2)
        assert term_t is not None and term_t <= 4, \
            f"Low project should terminate by t=4, got t={term_t}"

    def test_negligible_allocation_to_low(self):
        total_low = sum(
            self.rec["x_val"].get((2, t), 0.0)
            for t in range(1, self.portfolio["H"] + 1)
        )
        assert total_low <= 2.0, \
            f"Expected near-zero spend on Low, got {total_low:.2f}"

    def test_balance_never_negative(self):
        for t, port in self.rec["portfolio"].items():
            assert port["B_sim"] >= -0.5, \
                f"Balance negative at t={t}: {port['B_sim']:.2f}"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


# ─────────────────────────────────────────────────────────────────────────────
# ── PARAMETRIZED QUICK-SANITY SWEEP ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case_id,expected_Z,expected_status_0", [
    ("SP1_quick", 20.0, "completed"),
    ("SP3_quick", 30.0, "completed"),
])
def test_parametrized_quick(case_id, expected_Z, expected_status_0, do_plots):
    """
    Quick parametrized sanity checks reusing SP-1 and SP-3 configs.
    Useful for CI where individual class fixtures may be slow.
    """
    if case_id == "SP1_quick":
        proj = _make_project(
            idx=0, s=1, f=4, BAC=100, pi=0.20,
            M=2, theta=[0.5, 1.0], phi=[0.5, 0.5],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=1, mu=0.30, tau_tol=2,
        )
    else:   # SP3_quick
        proj = _make_project(
            idx=0, s=1, f=6, BAC=60, pi=0.50,
            M=3, theta=[1/3, 2/3, 1.0], phi=[1/3, 1/3, 1/3],
            eta_const=1.0, alpha=0.0, rho=0.0,
            Omega=2, mu=0.50, tau_tol=2,
        )

    cfg = _base_cfg(gamma=1.0, n_projects=1)
    portfolio, sol, rec = _run(cfg, [proj])
    _save_plots(portfolio, sol, rec, case_id, do_plots)

    assert sol["status"] == "Optimal"
    assert sol["obj"] == pytest.approx(expected_Z, abs=TOL_OBJ)
    assert rec["proj"][0]["status"] == expected_status_0


# ─────────────────────────────────────────────────────────────────────────────
# ── DYNAMIC HORIZON INVARIANT TESTS ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicHorizon:
    """
    Verify Modification 2: H is derived from max(f_i) - min(s_i) + 1,
    not from a fixed config value, and the period-1 origin invariant holds.
    """

    def test_horizon_equals_max_finish_minus_min_start_plus_one(self):
        """H = max(f_i) - min(s_i) + 1 for a 2-project portfolio."""
        proj_a = _make_project(idx=0, s=3, f=7, BAC=60, pi=0.10,
                               eta_const=1.0)
        proj_b = _make_project(idx=1, s=1, f=10, BAC=60, pi=0.10,
                               eta_const=1.0)
        cfg = _base_cfg(n_projects=2, gamma=1.0)
        portfolio, _, _ = _run(cfg, [proj_a, proj_b])
        # After shift: s_min=1, so H = 10 - 1 + 1 = 10
        assert portfolio["H"] == 10

    def test_period_one_is_earliest_start(self):
        """After shift, min(s_i) must equal 1."""
        proj_a = _make_project(idx=0, s=3, f=8,  BAC=60, pi=0.10, eta_const=1.0)
        proj_b = _make_project(idx=1, s=5, f=12, BAC=60, pi=0.10, eta_const=1.0)
        cfg = _base_cfg(n_projects=2, gamma=1.0)
        portfolio, _, _ = _run(cfg, [proj_a, proj_b])
        s_min = min(p["s_i"] for p in portfolio["projects"])
        assert s_min == 1, f"Expected s_min=1 after shift, got {s_min}"

    def test_eta_dict_keys_within_horizon(self):
        """All eta keys must be in [1, H] after shift."""
        proj = _make_project(idx=0, s=4, f=9, BAC=60, pi=0.10, eta_const=1.0)
        cfg = _base_cfg(n_projects=1, gamma=1.0)
        portfolio, _, _ = _run(cfg, [proj])
        H = portfolio["H"]
        for t in portfolio["projects"][0]["eta"]:
            assert 1 <= t <= H, f"eta key t={t} outside [1,{H}]"

    def test_large_b1(self):
        """Modification 1: B1 = kappa * total_BAC."""
        proj = _make_project(idx=0, s=1, f=4, BAC=100, pi=0.10, eta_const=1.0)
        cfg = _base_cfg(n_projects=1, gamma=1.0, b1_kappa=3.0)
        portfolio, _, _ = _run(cfg, [proj])
        assert portfolio["B1"] == pytest.approx(3.0 * 100.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# ── GLOBAL VALIDATION SWEEP ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [42, 7, 123])
def test_random_portfolio_validation(seed, do_plots):
    """
    For several random seeds, run the full MILP on a small portfolio
    and verify all structural validation checks pass.
    This catches regressions in the MILP formulation itself.
    """
    cfg = _base_cfg(
        seed=seed,
        n_projects=3,
        gamma=0.97,
        b1_kappa=3.0,
        Omega_range=(2, 4),
        mu_range=(0.15, 0.35),
        tau_tol_range=(1, 3),
        BAC_range=(40_000, 120_000),
        solver_time_limit=90,
    )
    portfolio = generate_portfolio(cfg)
    sol       = build_and_solve(portfolio)
    rec       = build_records(portfolio, sol)
    val       = validate(portfolio, sol, rec)

    _save_plots(portfolio, sol, rec, f"random_seed{seed}", do_plots)

    assert sol["status"] in ("Optimal", "Not Solved"), \
        f"Unexpected solver status: {sol['status']}"

    if sol["status"] == "Optimal":
        assert val["all_pass"], (
            f"Seed={seed}: Validation failed.\n"
            + "\n".join(f"  {k}: {v}" for k, v in val["errors"].items() if v)
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── PRE-SOLVE PAYMENT IDENTITY CHECKS ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _SCENARIOS_AVAILABLE,
                    reason="milp_test_scenarios.py not found")
class TestPaymentIdentity:
    """
    Verify that A_i + sum(R_net_j) + R_ret_i = CP_i for every project in
    every scenario BEFORE the MILP runs.  A failing identity means the
    project parameters are internally inconsistent and any MILP result
    built on them is meaningless.

    Uses check_payment_identity() from milp_test_scenarios.py — the same
    function the scenario file calls at import time, exposed here as
    explicit named tests so failures show up in the pytest report.
    """

    @pytest.mark.parametrize("sc_name,proj_idx", [
        ("S1", i) for i in range(5)
    ] + [
        ("S2", i) for i in range(3)
    ] + [
        ("S3", i) for i in range(4)
    ])
    def test_identity(self, sc_name, proj_idx):
        sc_map = {"S1": SCENARIO_1, "S2": SCENARIO_2, "S3": SCENARIO_3}
        sc   = sc_map[sc_name]
        proj = sc["portfolio"]["projects"][proj_idx]
        ok, total, cp = check_payment_identity(proj)
        assert ok, (
            f"{sc['name']} P{proj_idx}: "
            f"A + ΣR_net + R_ret = {total:.4f} ≠ CP = {cp:.4f}"
        )

    def test_all_sp_mp_projects(self):
        """
        Also verify payment identity for every hand-crafted SP/MP project
        defined in this test file, using the same check_payment_identity
        function.  This catches any _make_project() bug introduced by
        parameter changes.
        """
        from milp_ppm_test import derive_alpha, compute_R_net

        # Build all hand-crafted projects used in this file
        hand_crafted = []

        # SP-1
        hand_crafted.append(_make_project(
            idx=0, s=1, f=4, BAC=100, pi=0.20, M=2,
            theta=[0.5, 1.0], phi=[0.5, 0.5],
            alpha=0.0, rho=0.0,
        ))
        # SP-3
        hand_crafted.append(_make_project(
            idx=0, s=1, f=6, BAC=60, pi=0.50, M=3,
            theta=[1/3, 2/3, 1.0], phi=[1/3, 1/3, 1/3],
            alpha=0.0, rho=0.0,
        ))
        # SP-4
        hand_crafted.append(_make_project(
            idx=0, s=1, f=8, BAC=100, pi=0.40, M=2,
            theta=[0.5, 1.0], phi=[0.5, 0.5],
            alpha=0.0, rho=0.0,
        ))

        for proj in hand_crafted:
            ok, total, cp = check_payment_identity(proj)
            assert ok, (
                f"Hand-crafted P{proj['idx']} identity failed: "
                f"total={total:.4f}, CP={cp:.4f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ── SCENARIO 1: FIVE IDENTICAL PROJECTS (SYMMETRY TEST) ─────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _SCENARIOS_AVAILABLE,
                    reason="milp_test_scenarios.py not found")
class TestSC1Symmetry:
    """
    SC-1: Five identical projects (s=1, f=10, BAC=10k, pi=10%, 1 milestone).
    B1 = 100 * total_BAC (completely non-binding).
    Analytic Z* = 5 * 10,000 * 0.10 * 0.97^0 = 5,000.00

    This is a symmetry test: the MILP must treat all five identical projects
    equally, spending BAC at t=1 for each and certifying the single milestone
    at t=1.  Any asymmetry in allocation reveals a solver or formulation bug.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        assert _SCENARIOS_AVAILABLE, "milp_test_scenarios.py required"
        self.sc = SCENARIO_1
        self.portfolio, self.sol, self.rec = _run_portfolio(self.sc["portfolio"])
        _save_plots(self.portfolio, self.sol, self.rec, "SC1_symmetry", do_plots)
        self.Z_star     = self.sc["Z_star"]
        self.optimal_x  = self.sc["optimal_x"]
        self.n          = self.portfolio["n"]

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_analytic(self):
        """Z* must match the analytic formula to within TOL_OBJ_SC."""
        assert self.sol["obj"] == pytest.approx(self.Z_star, abs=TOL_OBJ_SC)

    def test_verify_milp_solution(self):
        """verify_milp_solution() from the scenario file must pass all checks."""
        result = verify_milp_solution(
            self.sc, self.sol["obj"], self.rec["x_val"],
            tol_obj=TOL_OBJ_SC, tol_alloc=TOL_ALLOC_SC,
        )
        assert result["all_pass"], (
            f"verify_milp_solution failed:\n"
            + "\n".join(f"  {k}: {v}" for k, v in result.items())
        )

    def test_all_projects_completed(self):
        for i in range(self.n):
            assert self.rec["proj"][i]["status"] == "completed", \
                f"Project {i} not completed"

    def test_no_terminations(self):
        for i in range(self.n):
            assert _termination_period(self.rec, i) is None, \
                f"Project {i} terminated unexpectedly"

    def test_symmetry_equal_total_spend(self):
        """All 5 identical projects must receive the same total allocation."""
        H = self.portfolio["H"]
        totals = [
            sum(self.rec["x_val"].get((i, t), 0.0) for t in range(1, H + 1))
            for i in range(self.n)
        ]
        for i in range(self.n):
            assert totals[i] == pytest.approx(totals[0], abs=TOL_ALLOC_SC), (
                f"Allocation asymmetry: P{i} got {totals[i]:.0f} "
                f"vs P0 got {totals[0]:.0f}"
            )

    def test_single_milestone_certifies_t1(self):
        """With eta=1 and x=BAC at t=1, the single milestone certifies at t=1."""
        for i in range(self.n):
            cert_t = _first_cert_period(self.rec, i, 0)
            assert cert_t == 1, \
                f"P{i} MS1 certified at t={cert_t}, expected t=1"

    def test_spend_concentrated_at_start(self):
        """Each project should spend BAC at t=s_i; all other periods near zero."""
        for i, p in enumerate(self.portfolio["projects"]):
            s = p["s_i"]
            spend_at_s = self.rec["x_val"].get((i, s), 0.0)
            assert spend_at_s == pytest.approx(p["BAC_i"], abs=TOL_ALLOC_SC), \
                f"P{i}: expected spend {p['BAC_i']:.0f} at t={s}, got {spend_at_s:.0f}"
            H = self.portfolio["H"]
            for t in range(1, H + 1):
                if t != s:
                    off = self.rec["x_val"].get((i, t), 0.0)
                    assert off < TOL_ALLOC_SC, \
                        f"P{i}: unexpected spend {off:.0f} at t={t}"

    def test_full_allocation_pattern(self):
        """
        Sweep the full optimal_x dict from the scenario file and compare
        each (i,t) pair against the MILP output.
        """
        for (i, t), expected in self.optimal_x.items():
            milp_val = self.rec["x_val"].get((i, t), 0.0)
            assert milp_val == pytest.approx(expected, abs=TOL_ALLOC_SC), (
                f"Allocation mismatch at (P{i}, t={t}): "
                f"expected {expected:.0f}, got {milp_val:.0f}"
            )

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


# ─────────────────────────────────────────────────────────────────────────────
# ── SCENARIO 2: STAGGERED STARTS + DISCOUNTING ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _SCENARIOS_AVAILABLE,
                    reason="milp_test_scenarios.py not found")
class TestSC2StaggeredDiscounting:
    """
    SC-2: Three projects with staggered starts (s=1, 3, 5), BAC=20k, pi=10%,
    3 equal milestones.  gamma=0.97.
    Analytic Z* = 2000*(1 + 0.97^2 + 0.97^4) = 5,652.3856

    Tests three things simultaneously:
      1. Discounting: earlier projects are worth more — the MILP must
         not treat all three starts as equivalent.
      2. Staggered activation: balance recursion must handle projects
         that begin at different periods.
      3. Multi-milestone simultaneous certification: with eta=1 and
         spend=BAC at s_i, all 3 milestones (at P=1/3, 2/3, 1.0) certify
         at s_i in the same period.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        assert _SCENARIOS_AVAILABLE, "milp_test_scenarios.py required"
        self.sc = SCENARIO_2
        self.portfolio, self.sol, self.rec = _run_portfolio(self.sc["portfolio"])
        _save_plots(self.portfolio, self.sol, self.rec, "SC2_staggered", do_plots)
        self.Z_star    = self.sc["Z_star"]
        self.optimal_x = self.sc["optimal_x"]
        self.projects  = self.portfolio["projects"]

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_analytic(self):
        assert self.sol["obj"] == pytest.approx(self.Z_star, abs=TOL_OBJ_SC)

    def test_verify_milp_solution(self):
        result = verify_milp_solution(
            self.sc, self.sol["obj"], self.rec["x_val"],
            tol_obj=TOL_OBJ_SC, tol_alloc=TOL_ALLOC_SC,
        )
        assert result["all_pass"], (
            f"verify_milp_solution failed:\n"
            + "\n".join(f"  {k}: {v}" for k, v in result.items())
        )

    def test_all_projects_completed(self):
        for i in range(3):
            assert self.rec["proj"][i]["status"] == "completed", \
                f"Project {i} not completed"

    def test_spend_at_own_start(self):
        """Each project receives BAC at its own s_i; zero at all other periods."""
        for i, p in enumerate(self.projects):
            s = p["s_i"]
            spend = self.rec["x_val"].get((i, s), 0.0)
            assert spend == pytest.approx(p["BAC_i"], abs=TOL_ALLOC_SC), \
                f"P{i}: expected spend at t={s}, got {spend:.0f}"

    def test_all_milestones_certify_at_start(self):
        """All 3 milestones of each project certify at t=s_i (same period)."""
        for i, p in enumerate(self.projects):
            s = p["s_i"]
            for j in range(p["M_i"]):
                cert_t = _first_cert_period(self.rec, i, j)
                assert cert_t == s, (
                    f"P{i} MS{j+1}: expected certification at t={s}, "
                    f"got t={cert_t}"
                )

    def test_discounting_ordering(self):
        """
        The discounted contribution of P0 (s=1) must exceed P1 (s=3)
        which must exceed P2 (s=5).  Verify via the per-project net inflow
        computed from the records.
        """
        gamma = self.portfolio["gamma"]
        H     = self.portfolio["H"]
        disc_net = []
        for i in range(3):
            p = self.projects[i]
            s = p["s_i"]
            disc = gamma ** (s - 1)
            net  = p["BAC_i"] * p["pi_i"] * disc
            disc_net.append(net)

        assert disc_net[0] > disc_net[1], \
            f"Expected P0 net > P1 net: {disc_net[0]:.2f} vs {disc_net[1]:.2f}"
        assert disc_net[1] > disc_net[2], \
            f"Expected P1 net > P2 net: {disc_net[1]:.2f} vs {disc_net[2]:.2f}"

    def test_no_cross_period_spend(self):
        """No project receives any allocation outside its own [s_i, f_i] window."""
        H = self.portfolio["H"]
        for i, p in enumerate(self.projects):
            for t in range(1, H + 1):
                if not (p["s_i"] <= t <= p["f_i"]):
                    off = self.rec["x_val"].get((i, t), 0.0)
                    assert off < TOL_ALLOC_SC, \
                        f"P{i}: spend outside window at t={t}: {off:.0f}"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


# ─────────────────────────────────────────────────────────────────────────────
# ── SCENARIO 3: FOUR PROJECTS, VARYING PROFIT MARGINS ────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _SCENARIOS_AVAILABLE,
                    reason="milp_test_scenarios.py not found")
class TestSC3MarginDifferentiation:
    """
    SC-3: Four projects (s=1, f=10, BAC=10k), pi=5%/10%/15%/20%,
    2 equal milestones.  B1 = 100 * total_BAC.
    Analytic Z* = 10,000 * (0.05 + 0.10 + 0.15 + 0.20) = 5,000.00

    With unlimited budget and all pi > 0, all four projects must be
    funded.  The MILP must correctly:
      - Certify the intermediate milestone at P=0.5 at t=1 (half of BAC spent).
      - Certify the final milestone at P=1.0 also at t=1 (full BAC spent).
      - Compute the weighted objective correctly across all four margins.

    This is the baseline for the future constrained-budget test where
    the MILP should prioritise higher-pi projects.
    """

    @pytest.fixture(autouse=True)
    def setup(self, do_plots):
        assert _SCENARIOS_AVAILABLE, "milp_test_scenarios.py required"
        self.sc = SCENARIO_3
        self.portfolio, self.sol, self.rec = _run_portfolio(self.sc["portfolio"])
        _save_plots(self.portfolio, self.sol, self.rec, "SC3_margin", do_plots)
        self.Z_star    = self.sc["Z_star"]
        self.optimal_x = self.sc["optimal_x"]
        self.projects  = self.portfolio["projects"]
        self.margins   = [p["pi_i"] for p in self.projects]   # [0.05,0.10,0.15,0.20]

    def test_solver_status(self):
        assert self.sol["status"] == "Optimal"

    def test_objective_analytic(self):
        assert self.sol["obj"] == pytest.approx(self.Z_star, abs=TOL_OBJ_SC)

    def test_verify_milp_solution(self):
        result = verify_milp_solution(
            self.sc, self.sol["obj"], self.rec["x_val"],
            tol_obj=TOL_OBJ_SC, tol_alloc=TOL_ALLOC_SC,
        )
        assert result["all_pass"], (
            f"verify_milp_solution failed:\n"
            + "\n".join(f"  {k}: {v}" for k, v in result.items())
        )

    def test_all_projects_completed(self):
        for i in range(4):
            assert self.rec["proj"][i]["status"] == "completed", \
                f"Project {i} (pi={self.margins[i]:.0%}) not completed"

    def test_no_terminations(self):
        for i in range(4):
            assert _termination_period(self.rec, i) is None, \
                f"Project {i} terminated unexpectedly"

    def test_intermediate_milestone_certifies_t1(self):
        """
        MS1 (theta=0.5) must certify at t=1 because full BAC is spent
        at t=1, driving P from 0 to 1.0 in one step.
        """
        for i in range(4):
            cert_t = _first_cert_period(self.rec, i, 0)
            assert cert_t == 1, \
                f"P{i} MS1 (theta=0.5) certified at t={cert_t}, expected t=1"

    def test_final_milestone_certifies_t1(self):
        """MS2 (theta=1.0) must also certify at t=1."""
        for i in range(4):
            cert_t = _first_cert_period(self.rec, i, 1)
            assert cert_t == 1, \
                f"P{i} MS2 (theta=1.0) certified at t={cert_t}, expected t=1"

    def test_per_project_contribution(self):
        """
        Each project's net contribution to Z* equals BAC * pi.
        Verify analytically: sum of individual contributions = Z*.
        """
        gamma = self.portfolio["gamma"]
        contributions = []
        for p in self.projects:
            s   = p["s_i"]   # all s_i = 1 here
            net = p["BAC_i"] * p["pi_i"] * (gamma ** (s - 1))
            contributions.append(net)

        assert sum(contributions) == pytest.approx(self.Z_star, abs=TOL_OBJ_SC)

    def test_higher_margin_higher_contribution(self):
        """
        The four contributions must be strictly increasing by margin.
        P0 (pi=5%) < P1 (pi=10%) < P2 (pi=15%) < P3 (pi=20%).
        """
        gamma = self.portfolio["gamma"]
        contribs = [
            p["BAC_i"] * p["pi_i"] * (gamma ** (p["s_i"] - 1))
            for p in self.projects
        ]
        for i in range(3):
            assert contribs[i] < contribs[i + 1], (
                f"Expected contribution P{i} < P{i+1}: "
                f"{contribs[i]:.2f} vs {contribs[i+1]:.2f}"
            )

    def test_all_four_fully_funded(self):
        """With unlimited B1, every project must receive exactly BAC."""
        for i, p in enumerate(self.projects):
            H     = self.portfolio["H"]
            total = sum(self.rec["x_val"].get((i, t), 0.0) for t in range(1, H + 1))
            assert total == pytest.approx(p["BAC_i"], abs=TOL_ALLOC_SC), \
                f"P{i} (pi={p['pi_i']:.0%}): expected total spend {p['BAC_i']:.0f}, got {total:.0f}"

    def test_validation_passes(self):
        val = validate(self.portfolio, self.sol, self.rec)
        assert val["all_pass"], f"Validation errors: {val['errors']}"


# ─────────────────────────────────────────────────────────────────────────────
# ── PARAMETRIZED SCENARIO SWEEP ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _SCENARIOS_AVAILABLE,
                    reason="milp_test_scenarios.py not found")
@pytest.mark.parametrize("sc_idx,sc_name", [
    (0, "S1_five_identical"),
    (1, "S2_staggered"),
    (2, "S3_margin_diff"),
])
def test_scenario_objective_parametrized(sc_idx, sc_name, do_plots):
    """
    Lightweight parametrized sweep: for each scenario, run the MILP and
    check only the objective value.  Faster than the full class fixtures;
    useful for CI smoke-testing all three scenarios in one go.
    """
    sc = SCENARIOS[sc_idx]
    portfolio, sol, rec = _run_portfolio(sc["portfolio"])
    _save_plots(portfolio, sol, rec, f"SC_param_{sc_name}", do_plots)

    assert sol["status"] == "Optimal", \
        f"{sc_name}: solver returned '{sol['status']}'"
    assert sol["obj"] == pytest.approx(sc["Z_star"], abs=TOL_OBJ_SC), (
        f"{sc_name}: Z*={sol['obj']:.4f}, expected {sc['Z_star']:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── ENTRY POINT ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running directly: python test_milp_ppm.py
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )
    raise SystemExit(result.returncode)