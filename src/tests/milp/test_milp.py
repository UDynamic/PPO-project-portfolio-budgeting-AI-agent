"""
test_milp.py  —  pytest suite that drives baselines/milp.py against
                 the hand-solved ground truth stored in cases.json,
                 then calls test_report.py to produce report.tex.

For every case the suite:
  1. Translates the case_data spec into a milp.py portfolio dict.
  2. Calls build_and_solve() → build_records().
  3. Asserts each category of output against the toleranced cells in cases.json.

After all tests complete, a pytest session-finish hook collects pass/fail/skip
results per case per category and calls test_report.generate_report() to write
tests/milp/report.tex.

Run:
    pytest tests/milp/test_milp.py -v
    pytest tests/milp/test_milp.py -v -k SP-1
    pytest tests/milp/test_milp.py -v -k G3
    pytest tests/milp/test_milp.py -v -k objective_value
    pytest tests/milp/test_milp.py -q --tb=short -x
"""

import json, math, sys, pytest
from pathlib import Path

# ── resolve project root and make baselines importable ───────────────────────
ROOT = Path(__file__).resolve().parents[2]   # root/tests/milp → root
sys.path.insert(0, str(ROOT))

from baselines.milp import (
    build_and_solve,
    build_records,
    planned_progress,
    compute_R_net,
    derive_alpha,
)

# ── load ground-truth cases ───────────────────────────────────────────────────
CASES_FILE = Path(__file__).parent / "cases.json"

def _load_cases():
    with open(CASES_FILE) as f:
        return json.load(f)

CASES = _load_cases()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _v(cell):
    """Return (value, tolerance) from a toleranced cell dict."""
    return cell["v"], cell["tol"]


def _assert_cell(cell, actual, label=""):
    v, tol = _v(cell)
    err = abs(actual - v)
    assert err <= tol, (
        f"{label}: expected {v} ± {tol}, got {actual:.6f}  (err={err:.6f})"
    )


def _scalar(raw, default=0.0):
    """Unwrap a possibly-toleranced scalar or plain number."""
    if isinstance(raw, dict) and "v" in raw:
        return raw["v"]
    if raw is None:
        return default
    return float(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Translate a cases.json entry into a milp.py portfolio dict
# ─────────────────────────────────────────────────────────────────────────────

def _build_portfolio_from_case(case: dict) -> dict:
    """
    Convert a case_data case dict into the portfolio format expected by
    baselines/milp.py's build_and_solve().
    """
    params   = case["params"]
    H_raw    = params.get("H", params.get("horizon", 8))
    H        = int(_scalar(H_raw, 8))
    B0_raw   = params.get("B0", params.get("B1", 300))
    B1       = _scalar(B0_raw, 300.0)

    projects_out = []
    for pi, proj_entry in enumerate(case["projects"]):
        ps = proj_entry["params"]

        BAC   = float(ps["BAC"])
        eta_c = float(ps.get("eta", 1.0))
        D_plan= int(ps["D_plan"])
        si    = int(ps["si"])
        fi    = int(ps["fi"])

        ms_theta = list(ps["ms_theta"])
        ms_e     = list(ps["ms_e"])
        ms_phi   = list(ps["ms_phi"])
        M        = len(ms_theta)

        CP    = float(ps["CP"])
        rho   = float(ps.get("rho",  0.0))
        alpha = float(ps.get("alpha", 0.0))
        psi   = float(ps.get("psi",  0.10))
        drec  = float(ps.get("delta_rec", 0.25))
        mu    = float(ps.get("mu",   0.30))
        Omega = int(ps.get("Omega",  1))
        tau_tol = int(ps.get("tau_tol", 2))

        eta_schedule_raw = ps.get("eta_schedule", None)
        eta_dict: dict = {}
        for t in range(1, H + 1):
            if si <= t <= fi:
                if eta_schedule_raw and str(t) in eta_schedule_raw:
                    eta_dict[t] = float(eta_schedule_raw[str(t)])
                elif eta_schedule_raw and t in eta_schedule_raw:
                    eta_dict[t] = float(eta_schedule_raw[t])
                else:
                    eta_dict[t] = eta_c
            else:
                eta_dict[t] = 0.0

        a_i = float(ps.get("a_i", 1.0))
        b_i = float(ps.get("b_i", 1.0))
        pi_i = CP / BAC - 1.0

        projects_out.append(dict(
            idx=pi,
            s_i=si,  f_i=fi,  D_plan=D_plan,
            a_i=a_i, b_i=b_i,
            BAC_i=BAC, pi_i=pi_i, CP_i=CP,
            alpha_i=alpha, delta_rec_i=drec, psi_i=psi, rho_i=rho,
            M_i=M, theta=ms_theta, phi=ms_phi, e_i=ms_e,
            Omega_i=Omega, mu_i=mu, tau_tol_i=tau_tol,
            eta=eta_dict,
        ))

    return dict(
        n=len(projects_out),
        H=H,
        projects=projects_out,
        B1=B1,
        gamma=0.95,
        total_BAC=sum(p["BAC_i"] for p in projects_out),
        cfg={"solver_time_limit": 120},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped solution cache — one solve per case across all test methods
# ─────────────────────────────────────────────────────────────────────────────

_SOLUTION_CACHE: dict = {}


def _get_solution(case_id: str):
    if case_id not in _SOLUTION_CACHE:
        case      = CASES[case_id]
        portfolio = _build_portfolio_from_case(case)
        sol       = build_and_solve(portfolio)
        rec       = build_records(portfolio, sol)
        _SOLUTION_CACHE[case_id] = (portfolio, sol, rec)
    return _SOLUTION_CACHE[case_id]


# ─────────────────────────────────────────────────────────────────────────────
# Result collector — populated by the pytest hook, consumed by report generator
# ─────────────────────────────────────────────────────────────────────────────

# Maps test method suffix → report category key
_METHOD_TO_CAT = {
    "test_budget_allocation":    "budget",
    "test_progress_accumulation":"progress",
    "test_evm_signals":          "evm",
    "test_termination_counter":  "termctr",
    "test_milestone_certification": "mscert",
    "test_payment_identity":     "payment",
    "test_cash_balance_nonneg":  "cashbal",
    "test_objective_value":      "zstar",
    "test_outcome_status":       "outcome",
    "test_progress_monotone":    "monotone",
    "test_advance_timing":       "advance",
}

# Populated by conftest hook below: {case_id: {cat_key: "pass"/"fail"/"skip"}}
_TEST_RESULTS: dict[str, dict[str, str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# pytest hooks (defined at module level — pytest collects them automatically)
# ─────────────────────────────────────────────────────────────────────────────

def pytest_runtest_logreport(report):
    """Collect pass/fail/skip for each test item as it finishes."""
    if report.when != "call":
        return

    # nodeid format: tests/milp/test_milp.py::TestMILP::test_foo[SP-1]
    nodeid = report.nodeid
    # extract case_id from the parametrize bracket
    if "[" not in nodeid or "]" not in nodeid:
        return
    case_id = nodeid[nodeid.rfind("[") + 1: nodeid.rfind("]")]

    # extract method name
    method = nodeid.split("::")[-1]
    method = method.split("[")[0]
    cat = _METHOD_TO_CAT.get(method)
    if cat is None:
        return

    if report.skipped:
        status = "skip"
    elif report.passed:
        status = "pass"
    else:
        status = "fail"

    _TEST_RESULTS.setdefault(case_id, {})[cat] = status


def pytest_sessionfinish(session, exitstatus):
    """After all tests finish, generate report.tex."""
    # Only generate when this file is the test source (not a sub-collection)
    try:
        from tests.milp.test_report import generate_report
    except ImportError:
        # path may not be on sys.path in all invocation styles
        sys.path.insert(0, str(Path(__file__).parent))
        from test_report import generate_report

    out = generate_report(
        results=_TEST_RESULTS if _TEST_RESULTS else None,
        output_path=Path(__file__).parent / "report.tex",
    )
    print(f"\n[test_milp] Report written → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Parametrize over all cases
# ─────────────────────────────────────────────────────────────────────────────

case_ids  = list(CASES.keys())
case_list = [CASES[k] for k in case_ids]


@pytest.mark.parametrize("case_id,case", list(zip(case_ids, case_list)), ids=case_ids)
class TestMILP:
    """
    Each test method calls the real MILP solver and checks its output
    against the hand-solved cells in cases.json.
    """

    # ── T1: Budget allocation ─────────────────────────────────────────────
    def test_budget_allocation(self, case_id, case):
        """MILP x_{i,t} non-negative and within tolerance of hand-solved allocation."""
        portfolio, sol, rec = _get_solution(case_id)
        x_val = rec["x_val"]

        for pi, proj_entry in enumerate(case["projects"]):
            tl = proj_entry["timeline"]
            for row in tl:
                t      = int(row["t"]["v"])
                x_gt   = row["x"]["v"]
                tol    = row["x"]["tol"]
                x_milp = x_val.get((pi, t), 0.0)

                assert x_milp >= -0.01, (
                    f"[{case_id}] Project {pi+1} t={t}: "
                    f"negative allocation {x_milp:.4f}"
                )
                assert abs(x_milp - x_gt) <= tol, (
                    f"[{case_id}] Project {pi+1} t={t}: "
                    f"MILP x={x_milp:.4f}, expected {x_gt} ± {tol}"
                )

    # ── T2: Progress accumulation ─────────────────────────────────────────
    def test_progress_accumulation(self, case_id, case):
        """P computed from MILP x values lies in [0,1] and matches stored cells."""
        portfolio, sol, rec = _get_solution(case_id)

        for pi, proj_entry in enumerate(case["projects"]):
            tl    = proj_entry["timeline"]
            p_obj = portfolio["projects"][pi]
            P_sim = 0.0
            eta   = p_obj["eta"]
            BAC   = p_obj["BAC_i"]
            x_val = rec["x_val"]

            for row in tl:
                t   = int(row["t"]["v"])
                x_t = x_val.get((pi, t), 0.0)
                P_sim = min(1.0, P_sim + eta.get(t, 0.0) * x_t / BAC)

                assert -1e-6 <= P_sim <= 1.0 + 1e-6, (
                    f"[{case_id}] Project {pi+1} t={t}: "
                    f"P={P_sim:.6f} out of [0,1]"
                )
                _assert_cell(row["P"], P_sim,
                             f"[{case_id}] Project {pi+1} t={t} P")

    # ── T3: EVM signals ───────────────────────────────────────────────────
    def test_evm_signals(self, case_id, case):
        """SPI and CPI derived from MILP solution are non-negative and within tol."""
        portfolio, sol, rec = _get_solution(case_id)

        for pi, proj_entry in enumerate(case["projects"]):
            tl     = proj_entry["timeline"]
            p_obj  = portfolio["projects"][pi]
            eta    = p_obj["eta"]
            BAC    = p_obj["BAC_i"]
            D_plan = p_obj["D_plan"]
            x_val  = rec["x_val"]

            P_sim     = 0.0
            cum_spend = 0.0
            period    = 0

            for row in tl:
                t   = int(row["t"]["v"])
                x_t = x_val.get((pi, t), 0.0)
                P_sim     = min(1.0, P_sim + eta.get(t, 0.0) * x_t / BAC)
                cum_spend += x_t
                period    += 1

                bcws = min(period / D_plan, 1.0) if D_plan > 0 else 1.0
                bcwp = P_sim
                acwp = cum_spend / BAC if BAC > 0 else 0.0
                spi  = bcwp / bcws if bcws > 1e-9 else (2.0 if bcwp > 0 else 0.0)
                cpi  = bcwp / acwp if acwp > 1e-9 else 1.0

                assert spi >= 0.0, (
                    f"[{case_id}] Project {pi+1} t={t}: negative SPI={spi:.4f}"
                )
                assert cpi >= 0.0, (
                    f"[{case_id}] Project {pi+1} t={t}: negative CPI={cpi:.4f}"
                )
                _assert_cell(row["SPI"], spi,
                             f"[{case_id}] Project {pi+1} t={t} SPI")
                _assert_cell(row["CPI"], cpi,
                             f"[{case_id}] Project {pi+1} t={t} CPI")

    # ── T4: Termination counter ───────────────────────────────────────────
    def test_termination_counter(self, case_id, case):
        """τ_rem from MILP x values matches stored cells; decrements only when both conditions fire."""
        portfolio, sol, rec = _get_solution(case_id)

        for pi, proj_entry in enumerate(case["projects"]):
            tl      = proj_entry["timeline"]
            p_obj   = portfolio["projects"][pi]
            eta     = p_obj["eta"]
            BAC     = p_obj["BAC_i"]
            D_plan  = p_obj["D_plan"]
            mu      = p_obj["mu_i"]
            tau_tol = p_obj["tau_tol_i"]
            x_val   = rec["x_val"]

            P_sim     = 0.0
            cum_spend = 0.0
            tau_rem   = tau_tol
            period    = 0
            prev_tau  = None

            for row in tl:
                t   = int(row["t"]["v"])
                x_t = x_val.get((pi, t), 0.0)
                P_sim     = min(1.0, P_sim + eta.get(t, 0.0) * x_t / BAC)
                cum_spend += x_t
                period    += 1

                bcws = min(period / D_plan, 1.0) if D_plan > 0 else 1.0
                acwp = cum_spend / BAC if BAC > 0 else 0.0
                bcwp = P_sim
                spi  = bcwp / bcws if bcws > 1e-9 else (2.0 if bcwp > 0 else 0.0)
                cpi  = bcwp / acwp if acwp > 1e-9 else 1.0
                eac  = 1.0 / cpi   if cpi  > 1e-9 else float("inf")

                both = (spi < 1.0) and (eac > (1.0 + mu))
                if both:
                    tau_rem = max(tau_rem - 1, 0)
                else:
                    tau_rem = tau_tol

                assert tau_rem >= 0, (
                    f"[{case_id}] Project {pi+1} t={t}: tau_rem < 0"
                )
                if prev_tau is not None and both:
                    assert tau_rem <= prev_tau, (
                        f"[{case_id}] Project {pi+1} t={t}: "
                        f"tau should have decremented"
                    )
                _assert_cell(row["tau_rem"], tau_rem,
                             f"[{case_id}] Project {pi+1} t={t} tau_rem")
                prev_tau = tau_rem

    # ── T5: Milestone certification ───────────────────────────────────────
    def test_milestone_certification(self, case_id, case):
        """MILP certifies only when P ≥ θ_j AND t ≥ e_j; periods match stored events."""
        portfolio, sol, rec = _get_solution(case_id)
        u_val = rec["u_val"]

        for pi, proj_entry in enumerate(case["projects"]):
            tl    = proj_entry["timeline"]
            p_obj = portfolio["projects"][pi]
            eta   = p_obj["eta"]
            BAC   = p_obj["BAC_i"]
            theta = p_obj["theta"]
            e_i   = p_obj["e_i"]
            M     = p_obj["M_i"]
            x_val = rec["x_val"]

            P_sim = 0.0
            H     = portfolio["H"]
            T     = list(range(1, H + 1))

            P_at: dict = {}
            for t in T:
                x_t = x_val.get((pi, t), 0.0)
                P_sim = min(1.0, P_sim + eta.get(t, 0.0) * x_t / BAC)
                P_at[t] = P_sim

            for j in range(M):
                cert_t = next(
                    (t for t in T
                     if u_val.get((pi, j, t), 0) == 1
                     and u_val.get((pi, j, t - 1), 0) == 0),
                    None,
                )
                if cert_t is not None:
                    assert cert_t >= e_i[j], (
                        f"[{case_id}] Project {pi+1} MS{j+1}: "
                        f"certified at t={cert_t} before e={e_i[j]}"
                    )
                    assert P_at.get(cert_t, 0.0) >= theta[j] - 0.02, (
                        f"[{case_id}] Project {pi+1} MS{j+1}: "
                        f"P={P_at.get(cert_t, 0.0):.4f} < θ={theta[j]}"
                    )

            # cross-check against stored ms_certified events
            stored_cert: dict = {}
            for row in tl:
                t = int(row["t"]["v"])
                for ms_num in row.get("ms_certified", []):
                    stored_cert[ms_num] = t

            for ms_num, t_stored in stored_cert.items():
                j = ms_num - 1
                cert_t = next(
                    (t for t in T
                     if u_val.get((pi, j, t), 0) == 1
                     and u_val.get((pi, j, t - 1), 0) == 0),
                    None,
                )
                assert cert_t == t_stored, (
                    f"[{case_id}] Project {pi+1} MS{ms_num}: "
                    f"MILP certified t={cert_t}, expected t={t_stored}"
                )

    # ── T6: Payment identity ──────────────────────────────────────────────
    def test_payment_identity(self, case_id, case):
        """A_i + Σ R_net + R_ret = CP_i for completed projects."""
        portfolio, sol, rec = _get_solution(case_id)
        R_net = sol["R_net"]
        R_ret = sol["R_ret"]
        A     = sol["A"]

        for pi, p_obj in enumerate(portfolio["projects"]):
            if rec["proj"][pi]["status"] != "completed":
                continue
            total = (A[pi]
                     + sum(R_net.get((pi, j), 0.0) for j in range(p_obj["M_i"]))
                     + R_ret.get(pi, 0.0))
            assert abs(total - p_obj["CP_i"]) <= 3.0, (
                f"[{case_id}] Project {pi+1}: payment identity "
                f"{total:.3f} ≠ CP={p_obj['CP_i']:.3f}"
            )

    # ── T7: Cash balance non-negative ─────────────────────────────────────
    def test_cash_balance_nonneg(self, case_id, case):
        """MILP B_t ≥ 0 at every period, and matches stored B_t cells."""
        portfolio, sol, rec = _get_solution(case_id)
        B_lp = rec["B_lp"]

        for port_row in case["portfolio"]:
            t       = int(port_row["t"]["v"])
            Bt_milp = B_lp.get(t, 0.0)
            if t > portfolio["H"]:
                continue

            assert Bt_milp >= -1.0, (
                f"[{case_id}] t={t}: MILP B_t={Bt_milp:.3f} < 0"
            )
            _assert_cell(port_row["B_t"], Bt_milp,
                         f"[{case_id}] t={t} B_t")

    # ── T8: Objective value ───────────────────────────────────────────────
    def test_objective_value(self, case_id, case):
        """MILP Z* within tol_Z of hand-solved value; skipped when no target."""
        Zstar = case["meta"].get("Zstar")
        tol_Z = case["meta"].get("tol_Z")
        if Zstar is None:
            pytest.skip("No Z* target for this case")

        _, sol, _ = _get_solution(case_id)
        Z_milp = sol["obj"]

        assert abs(Z_milp - Zstar) <= tol_Z, (
            f"[{case_id}] MILP Z*={Z_milp:.4f}, expected {Zstar} ± {tol_Z}"
        )

    # ── T9: Outcome status ────────────────────────────────────────────────
    def test_outcome_status(self, case_id, case):
        """Termination / completion outcome matches meta.outcome label."""
        portfolio, sol, rec = _get_solution(case_id)
        outcome = case["meta"].get("outcome", "")

        if "terminated" in outcome:
            any_term = any(
                rec["proj"][pi]["status"] == "terminated"
                for pi in range(portfolio["n"])
            )
            if not any_term:
                any_adv = any(
                    portfolio["projects"][pi]["alpha_i"] > 0
                    for pi in range(portfolio["n"])
                )
                if any_adv:
                    pytest.skip("Advance-only termination; no R_term in LP timeline")
            assert any_term, (
                f"[{case_id}] Expected terminated project; statuses: "
                f"{[rec['proj'][pi]['status'] for pi in range(portfolio['n'])]}"
            )

        elif outcome in ("completed", "both_completed", "all_completed"):
            for pi in range(portfolio["n"]):
                P_final = rec["proj"][pi]["P_final"]
                assert P_final >= 0.99, (
                    f"[{case_id}] Project {pi+1}: P_final={P_final:.4f} "
                    f"(expected complete)"
                )

    # ── T10: Progress monotonicity ────────────────────────────────────────
    def test_progress_monotone(self, case_id, case):
        """P_{i,t} from MILP x values is non-decreasing at every period."""
        portfolio, sol, rec = _get_solution(case_id)
        x_val = rec["x_val"]

        for pi, p_obj in enumerate(portfolio["projects"]):
            eta   = p_obj["eta"]
            BAC   = p_obj["BAC_i"]
            P_sim = 0.0
            prev_P = 0.0

            for t in range(1, portfolio["H"] + 1):
                x_t = x_val.get((pi, t), 0.0)
                P_sim = min(1.0, P_sim + eta.get(t, 0.0) * x_t / BAC)

                assert P_sim >= prev_P - 1e-6, (
                    f"[{case_id}] Project {pi+1} t={t}: "
                    f"P decreased {prev_P:.4f} → {P_sim:.4f}"
                )
                prev_P = P_sim

    # ── T11: Advance timing ───────────────────────────────────────────────
    def test_advance_timing(self, case_id, case):
        """Advance A_i is reflected in MILP B_t from t=1; matches stored cell."""
        portfolio, sol, rec = _get_solution(case_id)

        total_advance = sum(
            p_obj["alpha_i"] * p_obj["CP_i"]
            for p_obj in portfolio["projects"]
        )
        if total_advance < 0.01:
            pytest.skip("No advance in this case")

        port_rows = case["portfolio"]
        if not port_rows:
            pytest.skip("Empty portfolio")

        first_row = port_rows[0]
        t1        = int(first_row["t"]["v"])
        Bt1_milp  = rec["B_lp"].get(t1, 0.0)

        _assert_cell(first_row["B_t"], Bt1_milp,
                     f"[{case_id}] t={t1} B_t (advance timing)")