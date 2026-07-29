"""
case_data.py  —  single source of truth for all PPM test cases.

Each case is a dict with keys:
  meta        : display name, group, Z*, outcome
  params      : scalar parameters (BAC, CP, eta, etc.)
  projects    : list of per-project timeline dicts (one entry per period t)
  portfolio   : per-period portfolio quantities

Timeline entry keys (per project, per t):
  t, x, delta_P, P, BCWS, BCWP, ACWP, SPI, CPI, EAC,
  tau_rem, ms_certified, ms_j, R_net, R_ret, R_term, R_adv

Portfolio entry keys (per t):
  t, B_t, sum_inflow, sum_outflow, sum_net, gamma_t_net, cum_Z

Tolerance convention:  every numeric value stored as {"v": float, "tol": float}
  tol = absolute tolerance for pytest assertion.

─────────────────────────────────────────────────────────────────────────────
Fixes applied vs prior version
─────────────────────────────────────────────────────────────────────────────
FIX-1  All variable-η projects now use uniform_alloc so cumulative spend
       reaches thresholds regardless of the drawn η values.
FIX-2  S-B-11: uniform_alloc; zero-tolerance with on-plan spend → completes.
FIX-3  S-B-12: ms_e=[1,3]; spend 50 at t=1 certifies MS1 before t=2
       termination; harvest-and-terminate yields positive Z.
FIX-4  D-B-09: P1 uniform_alloc (survives), P2 alloc={} (terminated at t=2);
       budget_tight so only one project is fully funded.
FIX-5  T-B-08: P3 uniform_alloc from t=1 so conditions never fire;
       P1/P2 alloc={} → terminate at t=2 as intended.
FIX-6  D-B-06: true diff-same timing — P1 si=1/fi=12, P2 si=4/fi=12,
       both end at t=12; H=12.
"""

import math, json
from pathlib import Path
from numpy.random import default_rng          # numpy ≥ 1.17

gamma = 0.95

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def V(v, tol=None):
    """Wrap a value with its tolerance."""
    if tol is None:
        tol = max(abs(v) * 0.02, 0.5)
    return {"v": round(float(v), 6), "tol": round(float(tol), 6)}


def _case_seed(case_id: str) -> int:
    """
    Derive a reproducible integer seed from a case-ID string.
    Uses a simple polynomial hash so every case gets a unique, stable seed.
    """
    h = 0
    for ch in case_id:
        h = h * 31 + ord(ch)
    return h & 0xFFFF_FFFF


# ─────────────────────────────────────────────────────────────────────────────
# η-schedule generators
# ─────────────────────────────────────────────────────────────────────────────

def make_eta_flat(si: int, fi: int, value: float = 1.0) -> dict:
    """Return {t: value} for t in [si, fi].  Family-B default."""
    return {t: value for t in range(si, fi + 1)}


def make_eta_high(si: int, fi: int, rng) -> dict:
    """High-productivity regime: η_t ~ U(0.85, 1.00) per active period."""
    vals = rng.uniform(0.85, 1.00, fi - si + 1)
    return {t: float(round(v, 4)) for t, v in zip(range(si, fi + 1), vals)}


def make_eta_low(si: int, fi: int, rng) -> dict:
    """Low-productivity regime: η_t ~ U(0.60, 0.85) per active period."""
    vals = rng.uniform(0.60, 0.85, fi - si + 1)
    return {t: float(round(v, 4)) for t, v in zip(range(si, fi + 1), vals)}


def make_eta_mid(si: int, fi: int, rng) -> dict:
    """Mid-flat regime: η_t ~ U(0.80, 0.95)."""
    vals = rng.uniform(0.80, 0.95, fi - si + 1)
    return {t: float(round(v, 4)) for t, v in zip(range(si, fi + 1), vals)}


# ─────────────────────────────────────────────────────────────────────────────
# Payment-delay helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_delays_zero(n_milestones: int) -> list:
    return [0] * n_milestones


def make_delays_fixed(delays: list) -> list:
    return list(delays)


def make_delays_random(n_milestones: int, rng) -> list:
    return [int(rng.integers(1, 4)) for _ in range(n_milestones)]


def apply_delays(ms_e: list, ms_delay: list) -> list:
    """effective_e[j] = ms_e[j] + ms_delay[j]"""
    return [e + d for e, d in zip(ms_e, ms_delay)]


# ─────────────────────────────────────────────────────────────────────────────
# BAC / budget helpers
# ─────────────────────────────────────────────────────────────────────────────

def budget_free(sum_bac: float) -> float:
    return 3.0 * sum_bac

def budget_tight(sum_bac: float) -> float:
    return 1.0 * sum_bac

def budget_starved(sum_bac: float) -> float:
    return 0.5 * sum_bac


# ─────────────────────────────────────────────────────────────────────────────
# Milestone distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def ms_uniform(n: int) -> list:
    """Equal φ weights."""
    return [round(1.0 / n, 6)] * n

def ms_front(n: int) -> list:
    """Front-loaded: first milestone gets double weight, rest equal."""
    if n == 1:
        return [1.0]
    first = round(2.0 / (n + 1), 6)
    rest  = round((1.0 - first) / (n - 1), 6)
    phi = [first] + [rest] * (n - 1)
    phi[-1] = round(1.0 - sum(phi[:-1]), 6)
    return phi

def ms_back(n: int) -> list:
    """Back-loaded: last milestone gets double weight."""
    return list(reversed(ms_front(n)))

def ms_evenly_spaced(si: int, fi: int, n: int) -> list:
    """
    Spread n milestone eligibility periods evenly over [si, fi].
    Uses integer rounding; last element always == fi.
    """
    if n == 1:
        return [fi]
    step = (fi - si) / (n - 1)
    pts  = [int(si + k * step + 0.5) for k in range(n)]   # +0.5 → round-half-up
    pts[-1] = fi                                            # guarantee last == fi
    return pts

def ms_theta_uniform(n: int) -> list:
    """Threshold at k/n for k = 1..n."""
    return [round(k / n, 6) for k in range(1, n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Spend-allocation helpers
# ─────────────────────────────────────────────────────────────────────────────

def uniform_alloc(BAC: float, si: int, fi: int) -> dict:
    """
    Spread BAC evenly across all periods in [si, fi].
    Last period absorbs rounding residual.
    """
    D = fi - si + 1
    per_period = round(BAC / D, 4)
    alloc = {t: per_period for t in range(si, fi)}
    alloc[fi] = round(BAC - per_period * (D - 1), 4)
    return alloc


def burst_alloc(BAC: float, burst_periods: list) -> dict:
    """
    Concentrate spend at specified periods.
    burst_periods: list of (t, fraction) tuples summing to 1.0.
    """
    alloc = {}
    remaining = BAC
    for i, (t, frac) in enumerate(burst_periods):
        if i < len(burst_periods) - 1:
            amt = round(BAC * frac, 4)
        else:
            amt = round(remaining, 4)
        alloc[t] = amt
        remaining -= amt
    return alloc


# ─────────────────────────────────────────────────────────────────────────────
# Core timeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(BAC, eta, D_plan, fi, si,
                   alloc,
                   ms_theta,
                   ms_e,
                   ms_phi,
                   CP, rho=0, alpha=0, A=0, delta_rec=0.25, psi=0.10,
                   mu=0.30, Omega=1, tau_tol=2,
                   eta_schedule=None,
                   ms_delay=None,
                   term_at=None):
    """
    Build the per-period timeline for one project.

    ms_delay
    --------
    If supplied, milestone j becomes eligible for payment at period
        effective_e[j] = ms_e[j] + ms_delay[j]
    Completion threshold and certification period are unchanged;
    only the cash receipt is deferred by ms_delay[j] periods.

    FIX-1 note
    ----------
    Variable-η callers now pass uniform_alloc so that cumulative spend
    reliably crosses each threshold regardless of the drawn η values.
    Certification triggers when P >= theta_j AND t >= effective_e[j];
    if η is low the threshold may be crossed later than e_j, but the
    loop runs to fi so it will be caught.
    """
    if ms_delay is not None:
        ms_e_eff = apply_delays(ms_e, ms_delay)
    else:
        ms_e_eff = list(ms_e)

    rows = []
    P = 0.0
    cum_spend = 0.0
    cum_recover = 0.0
    tau_rem = tau_tol
    ms_certified = [False] * len(ms_theta)
    retention_released = False
    advance_received = A

    for t in range(si, fi + 1):
        eta_t = eta_schedule.get(t, eta) if eta_schedule else eta
        x = alloc.get(t, 0.0)
        delta_P = eta_t * x / BAC if BAC > 0 else 0.0
        P = min(P + delta_P, 1.0)
        cum_spend += x

        bcws = min((t - si + 1) / D_plan, 1.0) if D_plan > 0 else 1.0
        bcwp = P
        acwp = cum_spend / BAC if BAC > 0 else 0.0
        spi  = bcwp / bcws if bcws > 0 else (2.0 if bcwp > 0 else 0.0)
        cpi  = bcwp / acwp if acwp > 0 else 1.0
        eac  = (1.0 / cpi) if cpi > 0 else float('inf')

        cond1 = spi < 1.0
        cond2 = eac > (1 + mu)
        both  = cond1 and cond2
        if both:
            tau_rem = max(tau_rem - 1, 0)
        elif not both and tau_rem < tau_tol:
            tau_rem = tau_tol

        R_net_t = 0.0; R_ret_t = 0.0; R_term_t = 0.0
        ms_j_this = []
        for j, (theta_j, e_j_eff, phi_j) in enumerate(
                zip(ms_theta, ms_e_eff, ms_phi)):
            if not ms_certified[j] and P >= theta_j - 1e-6 and t >= e_j_eff:
                ms_certified[j] = True
                ms_j_this.append(j + 1)
                R_gross_full = phi_j * CP
                R_gross_net  = phi_j * CP * (1 - rho)
                if alpha > 0 and sum(ms_phi[:j + 1]) >= psi:
                    recover = min(delta_rec * R_gross_full, A - cum_recover)
                    recover = max(recover, 0.0)
                else:
                    recover = 0.0
                cum_recover += recover
                R_net_t += R_gross_net - recover

        if all(ms_certified) and not retention_released and rho > 0:
            R_ret_t = rho * CP
            retention_released = True

        if term_at is not None and t == term_at:
            R_term_t = P * CP - (A - cum_recover)
            R_term_t = round(R_term_t, 4)

        rows.append({
            "t":            V(t, 0),
            "x":            V(x, 0.5),
            "delta_P":      V(delta_P, 0.01),
            "P":            V(P, 0.01),
            "BCWS":         V(bcws, 0.02),
            "BCWP":         V(bcwp, 0.01),
            "ACWP":         V(acwp, 0.01),
            "SPI":          V(spi, 0.05),
            "CPI":          V(cpi, 0.05),
            "EAC_frac":     V(min(eac, 9.99), 0.1),
            "tau_rem":      V(tau_rem, 0),
            "ms_certified": ms_j_this,
            "R_net":        V(R_net_t, 0.5),
            "R_ret":        V(R_ret_t, 0.5),
            "R_term":       V(R_term_t, 0.5),
        })

        if term_at is not None and t >= term_at:
            break

    return rows, advance_received


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio builder
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(cases_projects, B0, H, advance=0, advance_t=0):
    port = []
    B = B0 + advance
    cum_Z = advance

    for t in range(1, H + 2):
        sum_in  = 0.0
        sum_out = 0.0
        for (rows, _) in cases_projects:
            for row in rows:
                if row["t"]["v"] == t:
                    sum_in  += (row["R_net"]["v"] + row["R_ret"]["v"]
                                + max(row["R_term"]["v"], 0))
                    sum_out += row["x"]["v"]
                    if row["R_term"]["v"] < 0:
                        sum_in += row["R_term"]["v"]
        net = sum_in - sum_out
        B   = B + net
        disc_net = net * gamma ** (t - 1)
        cum_Z   += disc_net
        port.append({
            "t":           V(t, 0),
            "B_t":         V(B, 1.0),
            "sum_inflow":  V(sum_in, 0.5),
            "sum_outflow": V(sum_out, 0.5),
            "sum_net":     V(net, 0.5),
            "disc_net":    V(disc_net, 0.5),
            "cum_Z":       V(cum_Z, 1.0),
        })

    return port


# ─────────────────────────────────────────────────────────────────────────────
# Case assembler
# ─────────────────────────────────────────────────────────────────────────────

def make_case(meta, params, projects_spec, B0, H, advance=0):
    proj_results = []
    for ps in projects_spec:
        rows, A_recv = build_timeline(**ps)
        proj_results.append((rows, A_recv))

    port = build_portfolio(proj_results, B0, H, advance=advance)

    return {
        "meta":      meta,
        "params":    {k: (V(v) if isinstance(v, float) else v)
                      for k, v in params.items()},
        "projects":  [{"params": ps, "timeline": rows}
                      for ps, (rows, _) in zip(projects_spec, proj_results)],
        "portfolio": port,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LEGACY CASES  (Groups G1–G9, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def sp1():
    ps = dict(BAC=100, eta=1.0, D_plan=4, fi=4, si=1,
              alloc={1:50, 3:50}, ms_theta=[0.5,1.0], ms_e=[1,3],
              ms_phi=[0.5,0.5], CP=120, rho=0, alpha=0, A=0,
              mu=0.30, Omega=1, tau_tol=2)
    return make_case(
        meta={"id":"SP-1","group":"G1","Zstar":19.025,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":120,"pi":0.20,"eta":1.0,"M":2,"tau_tol":2,
                "B0":300,"H":4,"gamma":0.95},
        projects_spec=[ps], B0=300, H=4)

def sp3():
    ps = dict(BAC=60, eta=1.0, D_plan=6, fi=6, si=1,
              alloc={1:20,3:20,5:20}, ms_theta=[1/3,2/3,1.0], ms_e=[1,3,5],
              ms_phi=[1/3,1/3,1/3], CP=90, rho=0, alpha=0, A=0,
              mu=0.50, Omega=2, tau_tol=2)
    return make_case(
        meta={"id":"SP-3","group":"G1","Zstar":27.17,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":60,"CP":90,"pi":0.50,"eta":1.0,"M":3,"tau_tol":2,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spJ():
    ps = dict(BAC=100, eta=1.0, D_plan=4, fi=4, si=1,
              alloc={1:50,3:50}, ms_theta=[0.5,1.0], ms_e=[1,3],
              ms_phi=[0.5,0.5], CP=100, rho=0, alpha=0, A=0,
              mu=0.50, Omega=2, tau_tol=2)
    return make_case(
        meta={"id":"SP-J","group":"G1","Zstar":0.0,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":100,"eta":1.0,"M":2,"H":4},
        projects_spec=[ps], B0=300, H=4)

def sp4():
    ps = dict(BAC=100, eta=1.0, D_plan=8, fi=8, si=1,
              alloc={1:50,5:50}, ms_theta=[0.5,1.0], ms_e=[1,5],
              ms_phi=[0.5,0.5], CP=140, rho=0, alpha=0, A=0,
              mu=0.30, Omega=1, tau_tol=2)
    return make_case(
        meta={"id":"SP-4","group":"G2","Zstar":36.29,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":140,"pi":0.40,"eta":1.0,"M":2,"H":8},
        projects_spec=[ps], B0=300, H=8)

def mpK():
    pA = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,alloc={1:30,4:30},
              ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.70,0.30],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pB = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,alloc={2:30,4:30},
              ms_theta=[0.5,1.0],ms_e=[2,4],ms_phi=[0.30,0.70],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-K","group":"G2","Zstar":55.87,"tol_Z":1.0,
              "outcome":"both_completed"},
        params={"n":2,"BAC_each":60,"CP_each":90,"H":6},
        projects_spec=[pA,pB], B0=300, H=6)

def mpKp():
    pA = dict(BAC=30,eta=1.0,D_plan=6,fi=6,si=1,alloc={1:15,4:15},
              ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.70,0.30],
              CP=45,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pB = dict(BAC=30,eta=1.0,D_plan=6,fi=6,si=1,alloc={2:15,4:15},
              ms_theta=[0.5,1.0],ms_e=[2,4],ms_phi=[0.30,0.70],
              CP=45,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-Kp","group":"G2","Zstar":27.936,"tol_Z":1.0,
              "outcome":"both_completed"},
        params={"n":2,"B0":20,"H":6},
        projects_spec=[pA,pB], B0=20, H=6)

def spL():
    xsurv = round(100/(0.7*8), 4)
    xcert = round(100/0.7 - 5*xsurv, 4)
    alloc_bad = {t: xsurv for t in range(1,6)}
    alloc_bad[6] = xcert
    ps = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,alloc=alloc_bad,
              ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
              CP=170,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2)
    return make_case(
        meta={"id":"SP-L","group":"G2","Zstar":9.298,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":170,"eta":0.70,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spLp():
    ps = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,alloc={},
              ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
              CP=80,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2,term_at=3)
    return make_case(
        meta={"id":"SP-Lp","group":"G2","Zstar":0.0,"tol_Z":0.5,
              "outcome":"terminated"},
        params={"BAC":100,"CP":80,"eta":0.70,"H":6},
        projects_spec=[ps], B0=300, H=6)

def sp2():
    ps = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={},
              ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
              CP=100,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.05,Omega=1,tau_tol=1,
              eta_schedule={1:1.0,**{t:0.05 for t in range(2,7)}},
              term_at=2)
    return make_case(
        meta={"id":"SP-2","group":"G3","Zstar":1.5,"tol_Z":0.5,
              "outcome":"terminated"},
        params={"BAC":100,"CP":100,"alpha":0.30,"A":30,"tau_tol":1,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def sp2a():
    ps = dict(BAC=100,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:100},
              ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
              CP=120,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=1)
    return make_case(
        meta={"id":"SP-2a","group":"G3","Zstar":20.5,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":120,"alpha":0.30,"A":30,"tau_tol":1,"H":4},
        projects_spec=[ps], B0=300, H=4, advance=30)

def sp2b():
    ps = dict(BAC=100,eta=1.0,D_plan=4,fi=4,si=1,alloc={},
              ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
              CP=60,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=1,term_at=2)
    return make_case(
        meta={"id":"SP-2b","group":"G3","Zstar":1.5,"tol_Z":0.5,
              "outcome":"terminated"},
        params={"BAC":100,"CP":60,"alpha":0.30,"A":30,"tau_tol":1,"H":4},
        projects_spec=[ps], B0=300, H=4, advance=30)

def sp2c():
    ps = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={},
              ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
              CP=100,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.05,Omega=1,tau_tol=3,
              eta_schedule={1:1.0,**{t:0.05 for t in range(2,7)}},
              term_at=4)
    return make_case(
        meta={"id":"SP-2c","group":"G3","Zstar":4.279,"tol_Z":0.5,
              "outcome":"terminated"},
        params={"BAC":100,"CP":100,"alpha":0.30,"A":30,"tau_tol":3,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def sp2d():
    ps = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={4:100},
              ms_theta=[1.0],ms_e=[4],ms_phi=[1.0],
              CP=120,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=4,
              eta_schedule={**{t:0.05 for t in range(1,4)},
                            **{t:1.0  for t in range(4,7)}})
    return make_case(
        meta={"id":"SP-2d","group":"G3","Zstar":21.855,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":100,"CP":120,"alpha":0.30,"A":30,"tau_tol":4,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def spA():
    ps = dict(BAC=120,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:40,3:40,5:40},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=180,rho=0,alpha=0.20,A=36,delta_rec=0.30,psi=0.40,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-A","group":"G3","Zstar":59.43,"tol_Z":1.5,
              "outcome":"completed"},
        params={"BAC":120,"CP":180,"alpha":0.20,"psi":0.40,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=36)

def spB():
    ps = dict(BAC=60,eta=1.0,D_plan=3,fi=4,si=1,
              alloc={1:20,2:20,3:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,2,3],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0,alpha=0.20,A=18,delta_rec=0.30,psi=0.10,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-B","group":"G3","Zstar":28.975,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":60,"CP":90,"alpha":0.20,"H":4},
        projects_spec=[ps], B0=300, H=4, advance=18)

def spC():
    ps = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:20,3:20,5:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0.10,alpha=0,A=0,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-C","group":"G4","Zstar":26.35,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":60,"CP":90,"rho":0.10,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spD():
    ps = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:20,3:20,5:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0.10,alpha=0.20,A=18,delta_rec=0.30,psi=0.10,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-D","group":"G4","Zstar":27.227,"tol_Z":1.0,
              "outcome":"completed"},
        params={"BAC":60,"CP":90,"rho":0.10,"alpha":0.20,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=18)

def spE():
    ps = dict(BAC=100,eta=1.0,D_plan=8,fi=8,si=1,
              alloc={t:4 for t in range(1,9)},
              ms_theta=[0.5,1.0],ms_e=[1,5],ms_phi=[0.5,0.5],
              CP=100,mu=0.05,Omega=1,tau_tol=3)
    return make_case(
        meta={"id":"SP-E","group":"G5","Zstar":None,"tol_Z":None,
              "outcome":"active"},
        params={"BAC":100,"eta":1.0,"tau_tol":3,"H":8},
        projects_spec=[ps], B0=300, H=8)

def spF():
    ps = dict(BAC=100,eta=0.5,D_plan=20,fi=20,si=1,
              alloc={t:20 for t in range(1,21)},
              ms_theta=[0.5,1.0],ms_e=[1,20],ms_phi=[0.5,0.5],
              CP=100,mu=0.05,Omega=1,tau_tol=3)
    return make_case(
        meta={"id":"SP-F","group":"G5","Zstar":None,"tol_Z":None,
              "outcome":"active"},
        params={"BAC":100,"eta":0.5,"tau_tol":3,"H":20,"B0":500},
        projects_spec=[ps], B0=500, H=20)

def spG():
    ps = dict(BAC=100,eta=1.0,D_plan=12,fi=12,si=1,
              alloc={1:10,2:10,3:80,**{t:10 for t in range(4,13)}},
              ms_theta=[0.5,1.0],ms_e=[1,12],ms_phi=[0.5,0.5],
              CP=100,mu=0.50,Omega=1,tau_tol=3,
              eta_schedule={1:0.3,2:0.3,3:1.0,**{t:0.3 for t in range(4,13)}})
    return make_case(
        meta={"id":"SP-G","group":"G5","Zstar":None,"tol_Z":None,
              "outcome":"active"},
        params={"BAC":100,"tau_tol":3,"H":12},
        projects_spec=[ps], B0=300, H=12)

def sp5():
    ps = dict(BAC=100,eta=0.0,D_plan=6,fi=6,si=1,alloc={},
              ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
              CP=100,alpha=0.20,A=20,mu=0.10,Omega=1,tau_tol=3,term_at=4)
    return make_case(
        meta={"id":"SP-5","group":"G6","Zstar":2.85,"tol_Z":0.5,
              "outcome":"terminated"},
        params={"BAC":100,"eta":0.0,"A":20,"tau_tol":3,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=20)

def spH():
    ps = dict(BAC=100,eta=0.5,D_plan=10,fi=10,si=1,alloc={1:60},
              ms_theta=[0.30,1.0],ms_e=[1,10],ms_phi=[0.40,0.60],
              CP=100,alpha=0.10,A=10,mu=0.05,Omega=0,tau_tol=2,term_at=3)
    return make_case(
        meta={"id":"SP-H","group":"G6","Zstar":7.075,"tol_Z":1.0,
              "outcome":"terminated"},
        params={"BAC":100,"eta":0.5,"A":10,"tau_tol":2,"Omega":0,"H":10},
        projects_spec=[ps], B0=300, H=10, advance=10)

def spI():
    ps = dict(BAC=100,eta=1.0,D_plan=12,fi=12,si=1,
              alloc={1:10,2:10,3:80,**{t:10 for t in range(4,13)}},
              ms_theta=[0.5,1.0],ms_e=[1,12],ms_phi=[0.5,0.5],
              CP=100,mu=0.50,Omega=0,tau_tol=3,
              eta_schedule={1:0.3,2:0.3,3:1.0,**{t:0.3 for t in range(4,13)}},
              term_at=7)
    return make_case(
        meta={"id":"SP-I","group":"G6","Zstar":None,"tol_Z":None,
              "outcome":"terminated"},
        params={"BAC":100,"tau_tol":3,"Omega":0,"H":12},
        projects_spec=[ps], B0=300, H=12)

def mp3():
    pGood = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:40,2:40},
                 ms_theta=[0.5,1.0],ms_e=[1,2],ms_phi=[0.5,0.5],
                 CP=120,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pBad  = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={},
                 ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
                 CP=100,rho=0,alpha=0.30,A=30,mu=0.05,Omega=1,tau_tol=1,
                 eta_schedule={1:1.0,**{t:0.05 for t in range(2,7)}},term_at=3)
    return make_case(
        meta={"id":"MP-3","group":"G7","Zstar":41.925,"tol_Z":1.5,
              "outcome":"mixed"},
        params={"n":2,"B0":90,"H":8},
        projects_spec=[pGood,pBad], B0=90, H=8, advance=30)

def mpA():
    p1 = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:30,3:30},
              ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    p2 = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={2:30,4:30},
              ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-A","group":"G7","Zstar":52.0,"tol_Z":5.0,
              "outcome":"both_completed"},
        params={"n":2,"B0":30,"H":6},
        projects_spec=[p1,p2], B0=30, H=6)

def mpB():
    pGood = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:30,2:30},
                 ms_theta=[0.5,1.0],ms_e=[1,2],ms_phi=[0.5,0.5],
                 CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pTerm = dict(BAC=60,eta=0.0,D_plan=6,fi=6,si=1,alloc={},
                 ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
                 CP=90,alpha=0.20,A=18,mu=0.10,Omega=1,tau_tol=1,term_at=2)
    return make_case(
        meta={"id":"MP-B","group":"G7","Zstar":31.005,"tol_Z":1.5,
              "outcome":"mixed"},
        params={"n":2,"B0":90,"H":8},
        projects_spec=[pGood,pTerm], B0=90, H=8, advance=18)

def mp2():
    pGood = dict(BAC=60,eta=1.0,D_plan=8,fi=8,si=1,alloc={1:40,5:40},
                 ms_theta=[0.5,1.0],ms_e=[1,5],ms_phi=[0.5,0.5],
                 CP=120,rho=0,alpha=0,A=0,mu=0.30,Omega=1,tau_tol=2)
    pBad  = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={},
                 ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
                 CP=100,alpha=0.30,A=30,mu=0.05,Omega=1,tau_tol=1,
                 eta_schedule={1:1.0,**{t:0.05 for t in range(2,7)}},term_at=3)
    return make_case(
        meta={"id":"MP-2","group":"G8","Zstar":37.79,"tol_Z":1.5,
              "outcome":"mixed"},
        params={"n":2,"B0":540,"H":8},
        projects_spec=[pGood,pBad], B0=540, H=8, advance=30)

def mp4():
    pHi = dict(BAC=60,eta=1.0,D_plan=2,fi=2,si=1,alloc={1:30,2:30},
               ms_theta=[0.5,1.0],ms_e=[1,2],ms_phi=[0.5,0.5],
               CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pMid= dict(BAC=60,eta=1.0,D_plan=2,fi=2,si=1,alloc={1:30,2:30},
               ms_theta=[0.5,1.0],ms_e=[1,2],ms_phi=[0.5,0.5],
               CP=72,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pLow= dict(BAC=60,eta=0.5,D_plan=6,fi=6,si=1,alloc={},
               ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
               CP=60,mu=0.10,Omega=1,tau_tol=1,term_at=2)
    return make_case(
        meta={"id":"MP-4","group":"G8","Zstar":40.95,"tol_Z":1.5,
              "outcome":"low_terminated"},
        params={"n":3,"B0":200,"H":6},
        projects_spec=[pHi,pMid,pLow], B0=200, H=6)

def mpC():
    pHi = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:30,3:30},
               ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
               CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pLo = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={},
               ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
               CP=90,mu=0.50,Omega=1,tau_tol=1,term_at=2)
    return make_case(
        meta={"id":"MP-C","group":"G8","Zstar":28.54,"tol_Z":1.5,
              "outcome":"lo_terminated"},
        params={"n":2,"B0":60,"H":4},
        projects_spec=[pHi,pLo], B0=60, H=4)

def mpD():
    pBr  = dict(BAC=30,eta=1.0,D_plan=2,fi=2,si=1,alloc={1:15,2:15},
                ms_theta=[0.5,1.0],ms_e=[1,2],ms_phi=[0.5,0.5],
                CP=66,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pAlp = dict(BAC=60,eta=1.0,D_plan=4,fi=6,si=3,alloc={3:30,5:30},
                ms_theta=[0.5,1.0],ms_e=[3,5],ms_phi=[0.5,0.5],
                CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pBet = dict(BAC=60,eta=1.0,D_plan=4,fi=6,si=3,alloc={3:30,6:30},
                ms_theta=[0.5,1.0],ms_e=[3,6],ms_phi=[0.5,0.5],
                CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-D","group":"G8","Zstar":85.999,"tol_Z":5.0,
              "outcome":"all_completed"},
        params={"n":3,"B0":40,"H":6},
        projects_spec=[pBr,pAlp,pBet], B0=40, H=6)

def mp1():
    pA = dict(BAC=100,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:50,3:50},
              ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
              CP=120,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    pB = dict(BAC=100,eta=1.0,D_plan=6,fi=10,si=5,alloc={5:50,8:50},
              ms_theta=[0.5,1.0],ms_e=[5,8],ms_phi=[0.5,0.5],
              CP=120,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-1","group":"G9","Zstar":34.153,"tol_Z":1.5,
              "outcome":"both_completed"},
        params={"n":2,"B0":300,"H":10},
        projects_spec=[pA,pB], B0=300, H=10)

def mpE():
    p1 = dict(BAC=60,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:30,3:30},
              ms_theta=[0.5,1.0],ms_e=[1,3],ms_phi=[0.5,0.5],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    p2 = dict(BAC=60,eta=1.0,D_plan=5,fi=8,si=4,alloc={4:30,6:30},
              ms_theta=[0.5,1.0],ms_e=[4,6],ms_phi=[0.5,0.5],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    p3 = dict(BAC=60,eta=1.0,D_plan=6,fi=14,si=9,alloc={9:30,11:30},
              ms_theta=[0.5,1.0],ms_e=[9,11],ms_phi=[0.5,0.5],
              CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"MP-E","group":"G9","Zstar":71.937,"tol_Z":2.0,
              "outcome":"all_completed"},
        params={"n":3,"B0":300,"H":14},
        projects_spec=[p1,p2,p3], B0=300, H=14)

def spM():
    xsurv = round(100/(0.7*8), 4)
    xcert = round(100/0.7 - 5*xsurv, 4)
    alloc_bad = {t: xsurv for t in range(1,6)}
    alloc_bad[6] = xcert
    pGood = dict(BAC=60,eta=1.0,D_plan=2,fi=2,si=1,alloc={1:60},
                 ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
                 CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=1,tau_tol=2)
    pBad  = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,alloc=alloc_bad,
                 ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
                 CP=170,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2)
    return make_case(
        meta={"id":"SP-M","group":"G2","Zstar":39.298,"tol_Z":1.5,
              "outcome":"both_completed"},
        params={"n":2,"B0":90,"H":6},
        projects_spec=[pGood,pBad], B0=90, H=6)

def spMp():
    pGood = dict(BAC=60,eta=1.0,D_plan=2,fi=2,si=1,alloc={1:60},
                 ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
                 CP=90,rho=0,alpha=0,A=0,mu=0.50,Omega=1,tau_tol=2)
    pBad  = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,alloc={},
                 ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
                 CP=80,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2,term_at=3)
    return make_case(
        meta={"id":"SP-Mp","group":"G2","Zstar":30.0,"tol_Z":1.0,
              "outcome":"good_completed"},
        params={"n":2,"B0":90,"H":6},
        projects_spec=[pGood,pBad], B0=90, H=6)


# ═════════════════════════════════════════════════════════════════════════════
# FAMILY B — Deterministic Baseline
# η = 1.0 flat, no payment delay
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────
# S-B  Single-project baseline
# ─────────────────────────────────────

def s_b_01():
    """
    S-B-01 | dur=6 | π=0.10 | M=2 | uniform | no advance | Ω=3 | μ=20% | free
    Claim: defer spend to eligibility windows, collect both milestones.
    Spend bursts exactly at ms_e periods; η=1.0 guarantees thresholds met.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-01","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"defer_spend_to_eligibility_collect_both_milestones"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":2,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6)

def s_b_02():
    """
    S-B-02 | dur=6 | π=0.10 | M=2 | front-loaded | no advance | Ω=3 | free
    Claim: large MS1 early → spending at e₁ captures large front payment sooner.
    ms_front(2) = [0.6667, 0.3333]: MS1 worth 0.667×CP, MS2 worth 0.333×CP.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [2, 6]; ms_theta = [0.5, 1.0]
    ms_phi = ms_front(2)
    alloc = burst_alloc(BAC, [(2, 0.5), (6, 0.5)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-02","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"front_loaded_early_spend_captures_large_ms1"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":2,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6)

def s_b_03():
    """
    S-B-03 | dur=6 | π=0.10 | M=2 | back-loaded | no advance | Ω=3 | free
    Claim: large MS2 late → discount γ^5 ≈ 0.774 erodes back-weighted value.
    ms_back(2) = [0.3333, 0.6667]: small MS1 early, large MS2 discounted at t=6.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]
    ms_phi = ms_back(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-03","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"back_loaded_discount_erodes_large_ms2"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":2,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6)

def s_b_04():
    """
    S-B-04 | dur=6 | π=0.10 | M=3 | uniform | no advance | Ω=3 | free
    Claim: three milestones uniform — spend in three equal bursts at e periods.
    Each burst = BAC/3; δP = 1/3 each; thresholds [1/3, 2/3, 1.0] met exactly.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [2, 4, 6]
    ms_theta = ms_theta_uniform(3)
    ms_phi = ms_uniform(3)
    alloc = burst_alloc(BAC, [(2, 1/3), (4, 1/3), (6, 1/3)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-04","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"three_milestones_uniform_three_equal_bursts"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":3,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6)

def s_b_05():
    """
    S-B-05 | dur=6 | π=0.05 | M=2 | uniform | advance=20% | Ω=3 | free
    Claim: low margin + advance → advance harvesting viable alongside completion.
    A = 0.20 × 105 = 21.0; recovery triggered at MS1 (phi_cum=0.5 ≥ psi=0.10).
    """
    BAC = 100; CP = round(BAC * 1.05, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-05","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"advance_harvesting_viable_low_margin"},
        params={"BAC":BAC,"CP":CP,"pi":0.05,"alpha":alpha,"A":A,
                "H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6, advance=A)

def s_b_06():
    """
    S-B-06 | dur=6 | π=0.20 | M=2 | uniform | advance=20% | Ω=3 | free
    Claim: high margin + advance → completing always dominates early termination.
    A = 0.20 × 120 = 24.0.
    """
    BAC = 100; CP = round(BAC * 1.20, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-06","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"high_margin_advance_completion_always_optimal"},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"alpha":alpha,"A":A,
                "H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6, advance=A)

def s_b_07():
    """
    S-B-07 | dur=12 | π=0.10 | M=3 | uniform | no advance | Ω=3 | free
    Claim: long project — discounting makes earlier milestone receipts more
    valuable; spend concentrated at eligibility periods [4, 8, 12].
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [4, 8, 12]
    ms_theta = ms_theta_uniform(3)
    ms_phi = ms_uniform(3)
    alloc = burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-07","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"long_project_early_milestones_higher_pv"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":3,"H":12,"B0":B0},
        projects_spec=[ps], B0=B0, H=12)

def s_b_08():
    """
    S-B-08 | dur=12 | π=0.10 | M=3 | front-loaded | advance=20% | Ω=3 | free
    Claim: long + advance + front-loading → advance recovered at first milestone.
    ms_front(3) = [0.5, 0.25, 0.25]; MS1 worth 50% of CP at t=3.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    ms_e = [3, 7, 12]
    ms_theta = ms_theta_uniform(3)
    ms_phi = ms_front(3)
    alloc = burst_alloc(BAC, [(3, 1/3), (7, 1/3), (12, 1/3)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-08","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"long_front_advance_recovered_at_first_milestone"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"alpha":alpha,"A":A,
                "M":3,"H":12,"B0":B0},
        projects_spec=[ps], B0=B0, H=12, advance=A)

def s_b_09():
    """
    S-B-09 | dur=12 | π=0.20 | M=5 | uniform | no advance | Ω=3 | free
    Claim: five milestones, long horizon — five-burst schedule is optimal.
    ms_evenly_spaced with round-half-up: [1, 4, 6, 9, 12].
    Each burst = BAC/5 = 20; δP = 0.20; thresholds [0.2,0.4,0.6,0.8,1.0].
    """
    BAC = 100; CP = round(BAC * 1.20, 2)
    ms_e = ms_evenly_spaced(1, 12, 5)       # [1, 4, 6, 9, 12]
    ms_theta = ms_theta_uniform(5)
    ms_phi = ms_uniform(5)
    fracs = [(e, 0.20) for e in ms_e]
    alloc = burst_alloc(BAC, fracs)
    ps = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-09","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"five_milestone_burst_schedule_optimal",
              "ms_e_actual": ms_e},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"M":5,"H":12,"B0":B0},
        projects_spec=[ps], B0=B0, H=12)

def s_b_10():
    """
    S-B-10 | dur=18 | π=0.20 | M=3 | back-loaded | no advance | Ω=3 | free
    Claim: very long back-loaded — solver accepts deep discount γ^17≈0.418 on MS3.
    ms_back(3) = [0.25, 0.25, 0.50]; large final payment heavily discounted.
    """
    BAC = 100; CP = round(BAC * 1.20, 2)
    ms_e = [5, 10, 18]
    ms_theta = ms_theta_uniform(3)
    ms_phi = ms_back(3)
    alloc = burst_alloc(BAC, [(5, 1/3), (10, 1/3), (18, 1/3)])
    ps = dict(BAC=BAC, eta=1.0, D_plan=18, fi=18, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-10","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"very_long_back_loaded_deep_discount_ms3"},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"M":3,"H":18,"B0":B0},
        projects_spec=[ps], B0=B0, H=18)

def s_b_11():
    """
    S-B-11 | dur=6 | π=0.10 | M=2 | uniform | no advance | Ω=0 | μ=0% | τ=1 | free
    Claim: zero tolerance survives when spend is exactly on-plan (SPI=CPI=1).

    FIX-2: uniform_alloc ensures constant SPI=1, CPI=1 throughout.
    With μ=0 and Ω=0: both conditions require SPI<1 AND EAC>1+0=1.
    EAC = 1/CPI = 1 when CPI=1, so EAC = 1.0 which is NOT > 1.0 → no trigger.
    Both conditions never simultaneously true → τ_rem stays at τ_tol=1 → completes.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    # uniform_alloc: 100/6 ≈ 16.67 per period; P grows ~0.1667/period
    # P reaches 0.5 at t=3 (cumulative = 3×16.67/100 = 0.5 exactly) ✓
    # P reaches 1.0 at t=6 ✓
    alloc = uniform_alloc(BAC, 1, 6)
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.0, Omega=0, tau_tol=1)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-11","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"zero_tolerance_survives_with_on_plan_spend"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"M":2,"Omega":0,"mu":0.0,
                "tau_tol":1,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6)

def s_b_12():
    """
    S-B-12 | dur=6 | π=0.10 | M=2 | advance=20% | Ω=0 | μ=0% | τ=1 | free
    Claim: harvest-and-terminate — certify MS1 at t=1, terminate at t=2
           before conditions fire; net cash positive.

    FIX-3: ms_e=[1,3] so MS1 eligible at t=1. Spend 50 at t=1 → P=0.5 → MS1.
    At t=2: alloc=0 → P unchanged=0.5. BCWS=2/6=0.333, BCWP=0.5 > BCWS
    → SPI=1.5>1 so cond1 (SPI<1) is FALSE → conditions don't both fire.
    But term_at=2 is forced (optimal decision to stop after MS1).

    Cash at termination (t=2):
      MS1 receipt:    phi_1×CP×(1-rho) - recover
                    = 0.5×110 - min(0.25×55, 21) = 55 - 13.75 = 41.25
      R_term:         P×CP - (A - cum_recover)
                    = 0.5×110 - (21 - 13.75) = 55 - 7.25 = 47.75
    Net positive → harvest-and-terminate is rational.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)  # A = 22.0
    ms_e = [1, 3]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = {1: 50}          # spend 50 at t=1; nothing thereafter
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.0, Omega=0, tau_tol=1, term_at=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-B-12","group":"B","Zstar":None,"tol_Z":2.0,
              "outcome":"terminated",
              "claim":"harvest_ms1_then_terminate_positive_net"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"alpha":alpha,"A":A,
                "Omega":0,"mu":0.0,"tau_tol":1,"H":6,"B0":B0},
        projects_spec=[ps], B0=B0, H=6, advance=A)


# ─────────────────────────────────────
# D-B  Dual-project baseline
# ─────────────────────────────────────

def d_b_01():
    """
    D-B-01 | 6+6 | same-same | π=0.10 each | M=2,2 | uniform | free
    Claim: symmetric — both complete; solver indifferent between projects.
    Both burst at t=3 (50) and t=6 (50); B0=600 >> 200 spend needed.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    def _p():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(2 * BAC)
    return make_case(
        meta={"id":"D-B-01","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"symmetric_both_complete_solver_indifferent"},
        params={"n":2,"BAC":BAC,"CP":CP,"pi":0.10,"H":6,"B0":B0},
        projects_spec=[_p(), _p()], B0=B0, H=6)

def d_b_02():
    """
    D-B-02 | 6+6 | same-same | π=0.20,0.05 | M=2,2 | uniform | tight
    Claim: asymmetric margin + tight budget → solver prioritises high-π project.
    B0=tight=200=total BAC; cash-flow ordering matters under timing constraints.
    High-π (CP=120) should be scheduled to spend first.
    """
    BAC = 100
    CP_hi = round(BAC * 1.20, 2); CP_lo = round(BAC * 1.05, 2)
    def _p(CP):
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-B-02","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"asymmetric_margin_tight_hi_pi_prioritised"},
        params={"n":2,"BAC":BAC,"CP_hi":CP_hi,"CP_lo":CP_lo,"H":6,"B0":B0},
        projects_spec=[_p(CP_hi), _p(CP_lo)], B0=B0, H=6)

def d_b_03():
    """
    D-B-03 | 6+6 | same-same | π=0.10 each | M=2,3 | front/back | free
    Claim: different milestone structures (front-2 vs back-3) on same duration.
    P1: front M=2; P2: back M=3.  Both complete; EVM profiles differ markedly.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(2, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[2, 6],
              ms_phi=ms_front(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(2, 1/3), (4, 1/3), (6, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[2, 4, 6],
              ms_phi=ms_back(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(2 * BAC)
    return make_case(
        meta={"id":"D-B-03","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"different_ms_structures_front2_vs_back3"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":6,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=6)

def d_b_04():
    """
    D-B-04 | 6+12 | same-diff | π=0.10 each | M=2,3 | uniform | free
    Claim: short P1 completes first; MS inflows at t=3,6 available to fund
           P2 spend at t=4,8,12 — early cash from P1 naturally recirculates.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(2 * BAC)
    return make_case(
        meta={"id":"D-B-04","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"short_p1_inflows_recirculate_to_long_p2"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":12,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=12)

def d_b_05():
    """
    D-B-05 | 6+12 | same-diff | π=0.10 each | advance P1=20% | tight
    Claim: P1 advance boosts available capital during the joint activity window
           (t=1..6), allowing P2 to begin spending before P1 inflows arrive.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP,
              alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(3, 1/3), (7, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[3, 7, 12],
              ms_phi=ms_front(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-B-05","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"p1_advance_funds_p2_early_spend"},
        params={"n":2,"BAC":BAC,"CP":CP,"alpha_p1":alpha,"A_p1":A,
                "H":12,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=12, advance=A)

def d_b_06():
    """
    D-B-06 | 12+12 | diff-same (staggered starts, same end) | π=0.10 | tight
    Claim: staggered starts but same end-date → heavy overlap t=4..12 creates
           budget pressure; solver must interleave spend carefully.

    FIX-6: true diff-same timing.
      P1: si=1, fi=12, D_plan=12  (start t=1, end t=12)
      P2: si=4, fi=12, D_plan=9   (start t=4, end t=12)
    Both end at t=12. H=12.
    P2 ms_e adjusted to [6, 9, 12] within its window [4,12].
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=9, fi=12, si=4,
              alloc=burst_alloc(BAC, [(6, 1/3), (9, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[6, 9, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-B-06","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"diff_same_staggered_starts_same_end_budget_pressure"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":12,"B0":B0,
                "timing":"diff-same","P1_window":[1,12],"P2_window":[4,12]},
        projects_spec=[p1, p2], B0=B0, H=12)

def d_b_07():
    """
    D-B-07 | 6+18 | diff-diff | π=0.20,0.10 | M=2,4 | back/uniform | advance P2=20% | free
    Claim: very asymmetric durations — P2 payments at t=9,14,18 discounted
           by γ^8≈0.663, γ^13≈0.513, γ^17≈0.418 respectively.
    """
    BAC = 100
    CP1 = round(BAC * 1.20, 2); CP2 = round(BAC * 1.10, 2)
    alpha2 = 0.20; A2 = round(alpha2 * CP2, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(2, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[2, 6],
              ms_phi=ms_back(2), CP=CP1, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=18, fi=18, si=1,
              alloc=burst_alloc(BAC, [(4, 0.25), (9, 0.25),
                                      (14, 0.25), (18, 0.25)]),
              ms_theta=ms_theta_uniform(4), ms_e=[4, 9, 14, 18],
              ms_phi=ms_uniform(4), CP=CP2,
              alpha=alpha2, A=A2, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(2 * BAC)
    return make_case(
        meta={"id":"D-B-07","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"asymmetric_duration_p2_deeply_discounted"},
        params={"n":2,"BAC":BAC,"CP1":CP1,"CP2":CP2,"H":18,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=18, advance=A2)

def d_b_08():
    """
    D-B-08 | 6+6 | no-overlap (sequential) | π=0.10 each | tight
    Claim: sequential — P1 MS inflows (t=3: +55, t=6: +55) fund P2 spend
           starting t=7; no simultaneous budget competition.
    P1: si=1,fi=6.  P2: si=7,fi=12.  H=12.
    Budget trace: B0=200 → spend 50 at t=3 → B=150; +55 → B=205; spend 50
    at t=6 → B=155; +55 → B=210; then P2 period t=7..12.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=12, si=7,
              alloc=burst_alloc(BAC, [(9, 0.5), (12, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[9, 12],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-B-08","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"sequential_p1_inflows_fund_p2_spend"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":12,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=12)

def d_b_09():
    """
    D-B-09 | 12+12 | same-same | π=0.10 each | Ω=0 | μ=0% | τ=1 | tight
    Claim: zero tolerance + tight budget forces one project to receive no spend;
           that project auto-terminates at t=2 while the other completes.

    FIX-4: P1 gets uniform_alloc (SPI=CPI=1 throughout, never triggers);
            P2 gets alloc={} (SPI=0 at t=1, τ_rem→0, term_at=2).
    Budget tight = 200; P1 needs 100 → B0=200 holds only P1 spend.
    After P1 MS inflows recirculate there is notional capacity but the solver
    (as modelled here) commits P2 to termination from t=0.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=uniform_alloc(BAC, 1, 12),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP,
              mu=0.0, Omega=0, tau_tol=1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc={},                    # no spend → immediate trigger
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP,
              mu=0.0, Omega=0, tau_tol=1, term_at=2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-B-09","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"p1_completes_p2_terminated",
              "claim":"zero_tol_tight_budget_forces_one_termination"},
        params={"n":2,"BAC":BAC,"CP":CP,"Omega":0,"mu":0.0,"tau_tol":1,
                "H":12,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=12)

def d_b_10():
    """
    D-B-10 | 6+6 | same-same | π=0.10 each | advance both=20% | starved
    Claim: starved budget (B0=100) + both advances (2×22=44) → effective pool
           144 funds both projects; advance is the critical early-period source.

    Cash trace (both projects identical):
      t=3: spend 50+50=100; B: 144→44; MS1 each: +41.25×2=82.5 → B=126.5
      t=6: spend 50+50=100; B: 126.5→26.5; MS2 each: +46.75×2=93.5 → B=120
    Both complete. ✓
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)   # A = 22.0
    def _p():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP,
                    alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
                    mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_starved(2 * BAC)   # 0.5 × 200 = 100
    return make_case(
        meta={"id":"D-B-10","group":"B","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"starved_budget_advance_is_critical_early_cash"},
        params={"n":2,"BAC":BAC,"CP":CP,"alpha":alpha,"A":A,"H":6,"B0":B0},
        projects_spec=[_p(), _p()], B0=B0, H=6, advance=2*A)


# ─────────────────────────────────────
# T-B  Triple-project baseline
# ─────────────────────────────────────

def t_b_01():
    """
    T-B-01 | 6,6,6 | same-same | π=0.10 | M=2,2,2 | uniform | free
    Claim: symmetric three-way — all complete; allocation splits verifiable.
    B0=900 >> 300 spend needed; pure structural test.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    def _p():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(3 * BAC)
    return make_case(
        meta={"id":"T-B-01","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"symmetric_three_way_all_complete"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":6,"B0":B0},
        projects_spec=[_p(), _p(), _p()], B0=B0, H=6)

def t_b_02():
    """
    T-B-02 | 6,6,6 | same-same | π=0.20,0.10,0.05 | M=2,2,2 | tight
    Claim: decreasing margins + tight budget (B0=300=total BAC) → solver
           ranks projects by π; high-π spend sequenced first to protect value.
    """
    BAC = 100
    CPs = [round(BAC * m, 2) for m in [1.20, 1.10, 1.05]]
    def _p(CP):
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(3 * BAC)
    return make_case(
        meta={"id":"T-B-02","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"decreasing_margin_tight_rank_by_pi"},
        params={"n":3,"BAC":BAC,"CPs":CPs,"H":6,"B0":B0},
        projects_spec=[_p(CP) for CP in CPs], B0=B0, H=6)

def t_b_03():
    """
    T-B-03 | 6,6,6 | same-same | π=0.10 | M=2,2,2 | uniform | starved
    Claim: starved (B0=150) — only two projects can be funded; P3 terminated
           optimally (all π equal so any is equivalent to drop).

    P3 has alloc={} and term_at=3; R_term=0 (P=0 at termination).
    Cash trace:
      t=3: P1+P2 spend 50+50=100; B: 150→50; MS1×2: +55×2=110 → B=160
      t=6: P1+P2 spend 50+50=100; B: 160→60; MS2×2: +55×2=110 → B=170
    P1+P2 complete; P3 terminated at t=3. ✓
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    def _p_normal():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    def _p_term():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc={},
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    term_at=3)
    B0 = budget_starved(3 * BAC)   # 0.5 × 300 = 150
    return make_case(
        meta={"id":"T-B-03","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"p3_terminated_p1_p2_complete",
              "claim":"starved_one_terminated_others_complete"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":6,"B0":B0},
        projects_spec=[_p_normal(), _p_normal(), _p_term()],
        B0=B0, H=6)

def t_b_04():
    """
    T-B-04 | 6,6,12 | same-diff | π=0.10 | M=2,2,3 | uniform | free
    Claim: two short projects complete by t=6; their MS inflows (t=3,6: +55×2×2)
           are available before P3 mid-horizon spend at t=8,12.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p3 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(3 * BAC)
    return make_case(
        meta={"id":"T-B-04","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"two_short_inflows_fund_long_p3"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":12,"B0":B0},
        projects_spec=[p1, p2, p3], B0=B0, H=12)

def t_b_05():
    """
    T-B-05 | 6,12,18 | diff-diff | π=0.10 | M=2,3,4 | uniform | free
    Claim: fully heterogeneous durations — each project distinct in duration,
           milestone count and timing; solver handles all simultaneously.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p3 = dict(BAC=BAC, eta=1.0, D_plan=18, fi=18, si=1,
              alloc=burst_alloc(BAC, [(4, 0.25), (9, 0.25),
                                      (14, 0.25), (18, 0.25)]),
              ms_theta=ms_theta_uniform(4), ms_e=[4, 9, 14, 18],
              ms_phi=ms_uniform(4), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(3 * BAC)
    return make_case(
        meta={"id":"T-B-05","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"fully_heterogeneous_all_complete"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":18,"B0":B0},
        projects_spec=[p1, p2, p3], B0=B0, H=18)

def t_b_06():
    """
    T-B-06 | 12,12,12 | diff-same (staggered starts, same end) | π=0.10
            | M=3,3,3 | front,uniform,back | advance P1=20% | tight
    Claim: same planned duration, staggered starts, different distributions
           and one advance — tests simultaneous structural variety under budget.
    P1: si=1,fi=12. P2: si=4,fi=15. P3: si=7,fi=18. H=18.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=burst_alloc(BAC, [(3, 1/3), (7, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[3, 7, 12],
              ms_phi=ms_front(3), CP=CP,
              alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=15, si=4,
              alloc=burst_alloc(BAC, [(7, 1/3), (11, 1/3), (15, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[7, 11, 15],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    p3 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=18, si=7,
              alloc=burst_alloc(BAC, [(10, 1/3), (14, 1/3), (18, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[10, 14, 18],
              ms_phi=ms_back(3), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_tight(3 * BAC)
    return make_case(
        meta={"id":"T-B-06","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"staggered_diff_distributions_advance_tight_budget"},
        params={"n":3,"BAC":BAC,"CP":CP,"alpha_p1":alpha,"A_p1":A,
                "H":18,"B0":B0},
        projects_spec=[p1, p2, p3], B0=B0, H=18, advance=A)

def t_b_07():
    """
    T-B-07 | 6,6,6 | no-overlap (strictly sequential) | π=0.10 | free
    Claim: P1→P2→P3 sequential cash cascade; each project's MS inflows
           provide the capital for the next project's spend.
    P1: si=1,fi=6. P2: si=7,fi=12. P3: si=13,fi=18.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    def _p(si, fi, ms_times):
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=si,
                    alloc=burst_alloc(BAC, [(ms_times[0], 0.5),
                                            (ms_times[1], 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=ms_times,
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2)
    B0 = budget_free(3 * BAC)
    return make_case(
        meta={"id":"T-B-07","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"sequential_three_cash_cascade"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":18,"B0":B0},
        projects_spec=[_p(1,  6,  [3,  6]),
                       _p(7,  12, [9,  12]),
                       _p(13, 18, [15, 18])],
        B0=B0, H=18)

def t_b_08():
    """
    T-B-08 | 6,6,6 | same-same | π=0.10 | M=2,2,2 | uniform
            | advance P3=20% | Ω=0 | μ=0% | τ=1 | starved
    Claim: starved + zero tolerance → P1,P2 receive no spend and terminate;
           P3 has advance + uniform spend → SPI=CPI=1 → survives and completes.

    FIX-5: P3 uses uniform_alloc from t=1 so SPI=1 always; conditions never
            fire for P3. P1,P2 have alloc={} → trigger at t=1, term_at=2.

    B0=starved(300)=150. P3 spend = 100; advance A=22.
    Effective P3 capital: some of the 150 B0 plus A=22.
    P3 uniform spend: 100/6≈16.67/period over t=1..6.
    B trace: start=150+22=172; t=1 spend 16.67 → 155.33; ... P3 completes.
    P1,P2 termination at t=2: R_term=0 (P=0, A=0). No net loss from P1,P2.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    alpha = 0.20; A = round(alpha * CP, 2)   # A = 22.0
    def _p_term():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc={},
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP,
                    mu=0.0, Omega=0, tau_tol=1, term_at=2)
    def _p_survive():
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc=uniform_alloc(BAC, 1, 6),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP,
                    alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
                    mu=0.0, Omega=0, tau_tol=1)
    B0 = budget_starved(3 * BAC)   # 0.5 × 300 = 150
    return make_case(
        meta={"id":"T-B-08","group":"B","Zstar":None,"tol_Z":4.0,
              "outcome":"p1_p2_terminated_p3_completes",
              "claim":"starved_zero_tol_advance_project_survives"},
        params={"n":3,"BAC":BAC,"CP":CP,"alpha_p3":alpha,"A_p3":A,
                "Omega":0,"mu":0.0,"tau_tol":1,"H":6,"B0":B0},
        projects_spec=[_p_term(), _p_term(), _p_survive()],
        B0=B0, H=6, advance=A)


# ═════════════════════════════════════════════════════════════════════════════
# FAMILY C — Performance Uncertainty
# Variable η per period, no payment delay.
# FIX-1: ALL variable-η projects use uniform_alloc so cumulative spend
#         reliably crosses thresholds regardless of drawn η values.
# ═════════════════════════════════════════════════════════════════════════════

def s_c_01():
    """
    S-C-01 | dur=6 | π=0.10 | M=2 | uniform | η=high | free
    Claim: high η means thresholds crossed earlier than planned period;
           solver still defers to eligibility windows (e=[3,6]).
    With η_min=0.85, uniform spend 16.67/period: cumulative P at t=3 ≥
    3×0.85×0.1667 = 0.425 — may fall slightly below 0.5 threshold.
    Loop continues: certification occurs at first t≥e_j where P≥theta_j.
    At worst t=4 for MS1 (cumulative 4×0.85×0.1667=0.567 ≥ 0.5). ✓
    """
    rng = default_rng(_case_seed("S-C-01"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_s = make_eta_high(1, 6, rng)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = uniform_alloc(BAC, 1, 6)
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta_s)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-C-01","group":"C","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"high_eta_variable_defers_to_eligibility_windows",
              "eta_regime":"high"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":6,"B0":B0,
                "eta_schedule":eta_s},
        projects_spec=[ps], B0=B0, H=6)

def s_c_02():
    """
    S-C-02 | dur=6 | π=0.10 | M=2 | uniform | η=low | free
    Claim: low η requires more total spend to reach thresholds; spreads
           spend uniformly. With η_min=0.60, cumulative P at t=6:
           6×0.60×0.1667=0.600 < 1.0 — MS2 may miss at t=6 if η draws low.
           fi extended to 8 to ensure MS2 is reachable at worst η.

    Note: with η_low ~ U(0.60,0.85), mean η≈0.725.
    Cumulative P at t=6: 6×0.725×0.1667≈0.725. MS2 (theta=1.0) not reached.
    Need t=8: 8×0.725×0.1667≈0.967 — still marginal.
    Conservative fi=10 guarantees P=1.0 even at η_min=0.60:
    ceil(1.0 / (0.60 × 0.1667)) = ceil(10.0) = 10 periods needed.
    H=10.
    """
    rng = default_rng(_case_seed("S-C-02"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_s = make_eta_low(1, 10, rng)
    ms_e = [5, 10]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = uniform_alloc(BAC, 1, 10)
    ps = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta_s)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-C-02","group":"C","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"low_eta_more_spend_periods_needed_reduces_net",
              "eta_regime":"low"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":10,"B0":B0,
                "eta_schedule":eta_s},
        projects_spec=[ps], B0=B0, H=10)

def s_c_03():
    """
    S-C-03 | dur=12 | π=0.20 | M=3 | front-loaded | advance=20% | η=low | free
    Claim: long + low η + advance → advance recovery timing depends on when
           MS1 threshold is reached (later than planned due to low η).
    fi extended to 18 (=12/0.60 rounded up) to guarantee all thresholds reachable.
    ms_e=[6,12,18] — eligibility at 1/3, 2/3, end of extended window.
    """
    rng = default_rng(_case_seed("S-C-03"))
    BAC = 100; CP = round(BAC * 1.20, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    eta_s = make_eta_low(1, 18, rng)
    ms_e = [6, 12, 18]; ms_theta = ms_theta_uniform(3)
    ms_phi = ms_front(3)
    alloc = uniform_alloc(BAC, 1, 18)
    ps = dict(BAC=BAC, eta=1.0, D_plan=18, fi=18, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta_s)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-C-03","group":"C","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"long_low_eta_advance_recovery_timing_shifts",
              "eta_regime":"low"},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"alpha":alpha,"A":A,
                "H":18,"B0":B0,"eta_schedule":eta_s},
        projects_spec=[ps], B0=B0, H=18, advance=A)

def s_c_04():
    """
    S-C-04 | dur=6 | π=0.10 | M=2 | uniform | η=low | Ω=0 | μ=0% | τ=1 | free
    Claim: low η + zero tolerance → with no spend, SPI=0 fires immediately;
           termination at t=2. P=0 → R_term=0.
    alloc={} models the budget-constrained/abandoned scenario.
    """
    rng = default_rng(_case_seed("S-C-04"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_s = make_eta_low(1, 6, rng)
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc={},
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.0, Omega=0, tau_tol=1,
              eta_schedule=eta_s, term_at=2)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-C-04","group":"C","Zstar":None,"tol_Z":2.0,
              "outcome":"terminated",
              "claim":"low_eta_zero_tol_no_spend_terminates_t2",
              "eta_regime":"low"},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"Omega":0,"mu":0.0,
                "tau_tol":1,"H":6,"B0":B0,"eta_schedule":eta_s},
        projects_spec=[ps], B0=B0, H=6)


# ─────────────────────────────────────
# D-C  Dual-project performance uncertainty
# ─────────────────────────────────────

def d_c_01():
    """
    D-C-01 | 6+6 | same-same | η=high P1, low P2 | M=2,2 | free
    Claim: asymmetric efficiency — P1 (high η) reaches thresholds with fewer
           spend-periods; P2 (low η) needs full uniform spread.
    Both projects use uniform_alloc (FIX-1). High-η P1 certifies milestones
    earlier within the uniform stream; low-η P2 certifies later.
    """
    rng1 = default_rng(_case_seed("D-C-01-P1"))
    rng2 = default_rng(_case_seed("D-C-01-P2"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_high(1, 6, rng1)
    eta2 = make_eta_low(1, 10, rng2)   # extended for low-η
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=uniform_alloc(BAC, 1, 6),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10, si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2)
    B0 = budget_free(2 * BAC)
    return make_case(
        meta={"id":"D-C-01","group":"C","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"high_eta_p1_certifies_earlier_low_eta_p2_needs_more_periods"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":10,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=10)

def d_c_02():
    """
    D-C-02 | 6+6 | same-same | η=low P1, high P2 | M=2,2 | tight
    Claim: tight budget + low-η P1 vs high-η P2 — budget forces preference
           toward the more efficient project.
    Both uniform_alloc; P1 extended to fi=10 for low-η safety.
    B0=tight(200) = 200.
    """
    rng1 = default_rng(_case_seed("D-C-02-P1"))
    rng2 = default_rng(_case_seed("D-C-02-P2"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_low(1, 10, rng1)
    eta2 = make_eta_high(1, 6, rng2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10, si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=uniform_alloc(BAC, 1, 6),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-C-02","group":"C","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"tight_budget_favours_high_eta_p2"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":10,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=10)

def d_c_03():
    """
    D-C-03 | 6+12 | same-diff | η=high both | M=2,3 | tight
    Claim: both high-η but variable; long P2 has more phases where η varies,
           creating phase-by-phase efficiency differences across its 12 periods.
    """
    rng1 = default_rng(_case_seed("D-C-03-P1"))
    rng2 = default_rng(_case_seed("D-C-03-P2"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_high(1, 6, rng1)
    eta2 = make_eta_high(1, 12, rng2)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc=uniform_alloc(BAC, 1, 6),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc=uniform_alloc(BAC, 1, 12),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2)
    B0 = budget_tight(2 * BAC)
    return make_case(
        meta={"id":"D-C-03","group":"C","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"both_high_eta_long_p2_phase_variation"},
        params={"n":2,"BAC":BAC,"CP":CP,"H":12,"B0":B0},
        projects_spec=[p1, p2], B0=B0, H=12)


# ─────────────────────────────────────
# T-C  Triple-project performance uncertainty
# ─────────────────────────────────────

def t_c_01():
    """
    T-C-01 | 6,6,6 | same-same | η=high/mid/low | M=2,2,2 | tight
    Claim: three efficiency tiers under tight budget; solver allocates
           resources prioritising η×(CP-BAC) effectively.
    Low-η project extended to fi=10 to guarantee threshold reachability.
    """
    rng_h = default_rng(_case_seed("T-C-01-hi"))
    rng_m = default_rng(_case_seed("T-C-01-mid"))
    rng_l = default_rng(_case_seed("T-C-01-lo"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_h = make_eta_high(1, 6, rng_h)
    eta_m = make_eta_mid(1, 6, rng_m)
    eta_l = make_eta_low(1, 10, rng_l)
    p_hi = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                alloc=uniform_alloc(BAC, 1, 6),
                ms_theta=[0.5, 1.0], ms_e=[3, 6],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_h)
    p_mi = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                alloc=uniform_alloc(BAC, 1, 6),
                ms_theta=[0.5, 1.0], ms_e=[3, 6],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_m)
    p_lo = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10, si=1,
                alloc=uniform_alloc(BAC, 1, 10),
                ms_theta=[0.5, 1.0], ms_e=[5, 10],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_l)
    B0 = budget_tight(3 * BAC)
    return make_case(
        meta={"id":"T-C-01","group":"C","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"three_eta_tiers_tight_budget_rank_by_efficiency"},
        params={"n":3,"BAC":BAC,"CP":CP,"H":10,"B0":B0},
        projects_spec=[p_hi, p_mi, p_lo], B0=B0, H=10)

def t_c_02():
    """
    T-C-02 | 6,6,12 | same-diff | η=low all | M=2,2,3 | starved
    Claim: all low η + starved budget → only highest-margin project (P1, π=0.20)
           receives funding; others terminated.
    P1: π=0.20, fi=10 (extended); P2: π=0.10, term_at=2; P3: π=0.05, term_at=2.
    """
    rng1 = default_rng(_case_seed("T-C-02-P1"))
    rng2 = default_rng(_case_seed("T-C-02-P2"))
    rng3 = default_rng(_case_seed("T-C-02-P3"))
    BAC = 100
    CPs = [round(BAC * m, 2) for m in [1.20, 1.10, 1.05]]
    eta1 = make_eta_low(1, 10, rng1)
    eta2 = make_eta_low(1, 6, rng2)
    eta3 = make_eta_low(1, 12, rng3)
    p1 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10, si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CPs[0], mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
              alloc={},
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CPs[1], mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2, term_at=2)
    p3 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12, si=1,
              alloc={},
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CPs[2], mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta3, term_at=2)
    B0 = budget_starved(3 * BAC)
    return make_case(
        meta={"id":"T-C-02","group":"C","Zstar":None,"tol_Z":4.0,
              "outcome":"p1_survives_p2_p3_terminated",
              "claim":"all_low_eta_starved_highest_margin_survives"},
        params={"n":3,"BAC":BAC,"CPs":CPs,"H":10,"B0":B0},
        projects_spec=[p1, p2, p3], B0=B0, H=10)


# ═════════════════════════════════════════════════════════════════════════════
# FAMILY P — Payment Delay
# η=1.0 flat, fixed discrete delays per milestone.
# effective_e[j] = ms_e[j] + ms_delay[j]; fi extended by max(ms_delay).
# ═════════════════════════════════════════════════════════════════════════════

def s_p_01():
    """
    S-P-01 | dur=6 | π=0.10 | M=2 | uniform φ | delays=[1,1] | free
    Claim: uniform 1-period delay shifts all receipts by 1; Z* lower than
           no-delay baseline (S-B-01) due to extra discounting.
    effective_e = [4, 7]; fi=7; H=7.
    Certification still at t=3 (P=0.5) and t=6 (P=1.0), but cash at t=4,7.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    delays = [1, 1]
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    fi = 6 + max(delays)   # = 7
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-P-01","group":"P","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"uniform_1period_delay_Z_lower_than_baseline",
              "ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":fi,"B0":B0,
                "ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi)

def s_p_02():
    """
    S-P-02 | dur=6 | π=0.10 | M=2 | front-loaded φ | delays=[1,3] | free
    Claim: MS1 front-weighted (0.667×CP) with only 1-period delay; MS2 back-
           weighted (0.333×CP) with 3-period delay — front advantage preserved
           despite smaller nominal value relative to back-loaded variant.
    effective_e = [3, 9]; fi=9; H=9.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    delays = [1, 3]
    ms_e = [2, 6]; ms_theta = [0.5, 1.0]
    ms_phi = ms_front(2)
    alloc = burst_alloc(BAC, [(2, 0.5), (6, 0.5)])
    fi = 6 + max(delays)   # = 9
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-P-02","group":"P","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"front_ms1_short_delay_back_ms2_long_delay_front_advantage",
              "ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":fi,"B0":B0,
                "ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi)

def s_p_03():
    """
    S-P-03 | dur=6 | π=0.10 | M=2 | back-loaded φ | delays=[3,1] | free
    Claim: back-loaded MS2 (0.667×CP) with shorter delay 1 partially offsets
           the discount penalty from being back-loaded.
    effective_e = [6, 7]; fi=7; H=7.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    delays = [3, 1]
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]
    ms_phi = ms_back(2)
    alloc = burst_alloc(BAC, [(3, 0.5), (6, 0.5)])
    fi = 6 + max(delays)   # = 7  (max delay is on MS1 = 3, so fi=3+3=6? No:
    # MS1 effective_e = 3+3=6, MS2 effective_e=6+1=7. fi=max(6,7)=7. ✓
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-P-03","group":"P","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"back_ms2_short_delay_partially_offsets_backload_penalty",
              "ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":fi,"B0":B0,
                "ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi)

def s_p_04():
    """
    S-P-04 | dur=12 | π=0.20 | M=3 | uniform φ | advance=20% | delays=[0,2,1] | free
    Claim: advance has zero delay (collected at contract signing); MS1 also
           zero-delay — reliable early cash; MS2 delayed 2, MS3 delayed 1.
    effective_e = [4, 10, 13]; fi=14; H=14.
    """
    BAC = 100; CP = round(BAC * 1.20, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    delays = [0, 2, 1]
    ms_e = [4, 8, 12]; ms_theta = ms_theta_uniform(3); ms_phi = ms_uniform(3)
    alloc = burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)])
    fi = 12 + max(delays)   # = 14
    ps = dict(BAC=BAC, eta=1.0, D_plan=12, fi=fi, si=1, alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2,
              ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-P-04","group":"P","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"advance_zero_delay_ms1_zero_delay_reliable_early_cash",
              "ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"alpha":alpha,"A":A,
                "H":fi,"B0":B0,"ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi, advance=A)


# ─────────────────────────────────────
# D-P  Dual-project payment delay
# ─────────────────────────────────────

def d_p_01():
    """
    D-P-01 | 6+6 | same-same | P1 delays=[1,1], P2 delays=[3,3] | tight
    Claim: P2 heavily delayed (effective receipts at t=6,9) → solver
           concentrates budget toward P1 (receipts at t=4,7).
    H = 6 + max(3,3) = 9.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    d1 = [1, 1]; d2 = [3, 3]
    def _p(delays):
        fi = 6 + max(delays)
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    ms_delay=delays)
    B0 = budget_tight(2 * BAC)
    H = 6 + max(max(d1), max(d2))
    return make_case(
        meta={"id":"D-P-01","group":"P","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"heavy_p2_delay_solver_concentrates_p1",
              "delays":{"P1":d1,"P2":d2}},
        params={"n":2,"BAC":BAC,"CP":CP,"H":H,"B0":B0,
                "delays":{"P1":d1,"P2":d2}},
        projects_spec=[_p(d1), _p(d2)], B0=B0, H=H)

def d_p_02():
    """
    D-P-02 | 6+6 | same-same | P1 delays=[0,0], P2 delays=[1,2] | free
    Claim: P1 certain (no delay) vs P2 uncertain (1,2 periods delayed) →
           under timing constraints solver schedules P1 spend first.
    H = 6 + max(2) = 8.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    d1 = [0, 0]; d2 = [1, 2]
    def _p(delays):
        fi = 6 + max(delays)
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    ms_delay=delays)
    B0 = budget_free(2 * BAC)
    H = 6 + max(max(d1), max(d2))
    return make_case(
        meta={"id":"D-P-02","group":"P","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"certain_p1_preferred_over_delayed_p2",
              "delays":{"P1":d1,"P2":d2}},
        params={"n":2,"BAC":BAC,"CP":CP,"H":H,"B0":B0,
                "delays":{"P1":d1,"P2":d2}},
        projects_spec=[_p(d1), _p(d2)], B0=B0, H=H)

def d_p_03():
    """
    D-P-03 | 6+12 | same-diff | P1 delays=[2,2], P2 delays=[0,1,2] | tight
    Claim: short P1 delayed (receipts at t=5,8); long P2 mixed (t=4,9,14) →
           complex trade-off between certainty and scale under tight budget.
    H = 12 + max(max(d1),max(d2)) = 12+2 = 14.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    d1 = [2, 2]; d2 = [0, 1, 2]
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6+max(d1), si=1,
              alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              ms_delay=d1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12+max(d2), si=1,
              alloc=burst_alloc(BAC, [(4, 1/3), (8, 1/3), (12, 1/3)]),
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              ms_delay=d2)
    B0 = budget_tight(2 * BAC)
    H = 12 + max(max(d1), max(d2))
    return make_case(
        meta={"id":"D-P-03","group":"P","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"short_delayed_p1_long_mixed_p2_complex_tradeoff",
              "delays":{"P1":d1,"P2":d2}},
        params={"n":2,"BAC":BAC,"CP":CP,"H":H,"B0":B0,
                "delays":{"P1":d1,"P2":d2}},
        projects_spec=[p1, p2], B0=B0, H=H)


# ─────────────────────────────────────
# T-P  Triple-project payment delay
# ─────────────────────────────────────

def t_p_01():
    """
    T-P-01 | 6,6,6 | same-same | P1=[0,0] P2=[1,2] P3=[2,3] | free
    Claim: increasing delays across projects → solver cascades budget allocation
           toward earliest-paying project (P1) for maximum present value.
    H = 6 + max(3) = 9.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ds = [[0, 0], [1, 2], [2, 3]]
    def _p(delays):
        fi = 6 + max(delays)
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    ms_delay=delays)
    B0 = budget_free(3 * BAC)
    H = 6 + max(d[-1] for d in ds)
    return make_case(
        meta={"id":"T-P-01","group":"P","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"increasing_delays_solver_cascades_to_earliest_paying",
              "delays":ds},
        params={"n":3,"BAC":BAC,"CP":CP,"H":H,"B0":B0,"delays":ds},
        projects_spec=[_p(d) for d in ds], B0=B0, H=H)

def t_p_02():
    """
    T-P-02 | 6,6,6 | same-same | P1=[0,0] P2=[0,0] P3=[3,3] | starved
    Claim: P3 maximally delayed + starved budget → solver terminates P3,
           completes P1+P2 (no delay, higher present value).
    P3: alloc={}, term_at=3 (zero spend → SPI<1 at t=1, but tau_tol=2 so
    term triggers at t=3). P1,P2: burst alloc, complete normally.
    H = 6 + 3 = 9.
    """
    BAC = 100; CP = round(BAC * 1.10, 2)
    ds = [[0, 0], [0, 0], [3, 3]]
    def _p_complete(delays):
        fi = 6 + max(delays)
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1,
                    alloc=burst_alloc(BAC, [(3, 0.5), (6, 0.5)]),
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    ms_delay=delays)
    def _p_term(delays):
        return dict(BAC=BAC, eta=1.0, D_plan=6, fi=6, si=1,
                    alloc={},
                    ms_theta=[0.5, 1.0], ms_e=[3, 6],
                    ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                    ms_delay=delays, term_at=3)
    B0 = budget_starved(3 * BAC)
    H = 6 + max(d[-1] for d in ds)
    return make_case(
        meta={"id":"T-P-02","group":"P","Zstar":None,"tol_Z":4.0,
              "outcome":"p3_terminated_p1_p2_complete",
              "claim":"max_delayed_p3_starved_p3_terminated",
              "delays":ds},
        params={"n":3,"BAC":BAC,"CP":CP,"H":H,"B0":B0,"delays":ds},
        projects_spec=[_p_complete(ds[0]),
                       _p_complete(ds[1]),
                       _p_term(ds[2])],
        B0=B0, H=H)


# ═════════════════════════════════════════════════════════════════════════════
# FAMILY X — Combined Uncertainty  (variable η + payment delays)
# FIX-1 applies: all variable-η projects use uniform_alloc.
# fi extended by max(ms_delay) for payment receipt periods.
# ═════════════════════════════════════════════════════════════════════════════

def s_x_01():
    """
    S-X-01 | dur=6 | π=0.10 | M=2 | uniform φ | η=high | delays=[1,2] | free
    Claim: mild combined uncertainty (high η, short delays) — solver still
           finds near-optimal schedule close to deterministic baseline.
    fi=6+2=8; H=8. uniform_alloc over t=1..6; extended loop to fi=8 for receipts.
    """
    rng = default_rng(_case_seed("S-X-01"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_s = make_eta_high(1, 6, rng)
    delays = [1, 2]
    ms_e = [3, 6]; ms_theta = [0.5, 1.0]; ms_phi = ms_uniform(2)
    fi = 6 + max(delays)   # = 8
    alloc = uniform_alloc(BAC, 1, 6)   # spend only in active window [1,6]
    ps = dict(BAC=BAC, eta=1.0, D_plan=6, fi=fi, si=1,
              alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta_s, ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-X-01","group":"X","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"mild_combined_uncertainty_near_optimal",
              "eta_regime":"high","ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.10,"H":fi,"B0":B0,
                "eta_schedule":eta_s,"ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi)

def s_x_02():
    """
    S-X-02 | dur=12 | π=0.20 | M=3 | front φ | advance=20% | η=low | delays=[2,1,3] | free
    Claim: severe combined uncertainty — advance (zero delay) is the most
           reliable early cash source when η is low and payments are delayed.
    fi extended to 18 for low-η (needs more periods) and max delay=3.
    ms_e=[6,12,18]; H=18.
    """
    rng = default_rng(_case_seed("S-X-02"))
    BAC = 100; CP = round(BAC * 1.20, 2)
    alpha = 0.20; A = round(alpha * CP, 2)
    eta_s = make_eta_low(1, 18, rng)
    delays = [2, 1, 3]
    ms_e = [6, 12, 18]; ms_theta = ms_theta_uniform(3)
    ms_phi = ms_front(3)
    fi = 18 + max(delays)   # = 21; H=21
    alloc = uniform_alloc(BAC, 1, 18)
    ps = dict(BAC=BAC, eta=1.0, D_plan=18, fi=fi, si=1,
              alloc=alloc,
              ms_theta=ms_theta, ms_e=ms_e, ms_phi=ms_phi,
              CP=CP, alpha=alpha, A=A, delta_rec=0.25, psi=0.10,
              mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta_s, ms_delay=delays)
    B0 = budget_free(BAC)
    return make_case(
        meta={"id":"S-X-02","group":"X","Zstar":None,"tol_Z":2.0,
              "outcome":"completed",
              "claim":"severe_combined_advance_most_reliable_early_cash",
              "eta_regime":"low","ms_delay":delays},
        params={"BAC":BAC,"CP":CP,"pi":0.20,"alpha":alpha,"A":A,
                "H":fi,"B0":B0,"eta_schedule":eta_s,"ms_delay":delays},
        projects_spec=[ps], B0=B0, H=fi, advance=A)


# ─────────────────────────────────────
# D-X  Dual-project combined
# ─────────────────────────────────────

def d_x_01():
    """
    D-X-01 | 6+6 | same-same | η=high P1/low P2 | P1 delays=[0,1], P2 delays=[2,3] | tight
    Claim: P1 is doubly advantaged (high η, low delay); P2 is doubly stressed
           (low η, high delay) → solver concentrates budget on P1.
    P1: fi=6+1=7 (short delay); P2: fi extended to 10 for low-η + 3 delay = 13.
    H = max(7, 13) = 13.
    """
    rng1 = default_rng(_case_seed("D-X-01-P1"))
    rng2 = default_rng(_case_seed("D-X-01-P2"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_high(1, 6, rng1)
    eta2 = make_eta_low(1, 10, rng2)
    d1 = [0, 1]; d2 = [2, 3]
    p1 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6+max(d1), si=1,
              alloc=uniform_alloc(BAC, 1, 6),
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1, ms_delay=d1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10+max(d2), si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2, ms_delay=d2)
    B0 = budget_tight(2 * BAC)
    H = max(6+max(d1), 10+max(d2))
    return make_case(
        meta={"id":"D-X-01","group":"X","Zstar":None,"tol_Z":3.0,
              "outcome":"both_completed",
              "claim":"doubly_advantaged_p1_solver_concentrates_there",
              "eta_regime":{"P1":"high","P2":"low"},
              "delays":{"P1":d1,"P2":d2}},
        params={"n":2,"BAC":BAC,"CP":CP,"H":H,"B0":B0,
                "delays":{"P1":d1,"P2":d2}},
        projects_spec=[p1, p2], B0=B0, H=H)

def d_x_02():
    """
    D-X-02 | 6+12 | same-diff | η=low both | P1 delays=[2,2], P2 delays=[1,2,1] | tight
    Claim: both projects stressed (low η + delays) under tight budget →
           one must be terminated; solver chooses the lower expected-value project.
    P1 (shorter): fi=10+2=12 for low-η extension + delay.
    P2 (longer): terminated at t=4 (before major spend needed).
    H=12.
    """
    rng1 = default_rng(_case_seed("D-X-02-P1"))
    rng2 = default_rng(_case_seed("D-X-02-P2"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_low(1, 10, rng1)
    eta2 = make_eta_low(1, 12, rng2)
    d1 = [2, 2]; d2 = [1, 2, 1]
    p1 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10+max(d1), si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1, ms_delay=d1)
    p2 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12+max(d2), si=1,
              alloc={},
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2, ms_delay=d2, term_at=4)
    B0 = budget_tight(2 * BAC)
    H = max(10+max(d1), 12+max(d2))
    return make_case(
        meta={"id":"D-X-02","group":"X","Zstar":None,"tol_Z":3.0,
              "outcome":"p1_completes_p2_terminated",
              "claim":"both_stressed_tight_budget_lower_ev_project_terminated",
              "eta_regime":{"P1":"low","P2":"low"},
              "delays":{"P1":d1,"P2":d2}},
        params={"n":2,"BAC":BAC,"CP":CP,"H":H,"B0":B0,
                "delays":{"P1":d1,"P2":d2}},
        projects_spec=[p1, p2], B0=B0, H=H)


# ─────────────────────────────────────
# T-X  Triple-project combined
# ─────────────────────────────────────

def t_x_01():
    """
    T-X-01 | 6,6,6 | same-same | η=high/mid/low | delays=0,1/1,2/2,3 | tight
    Claim: full spectrum of η and delay uncertainty under tight budget →
           priority ordering high-η-low-delay → mid → low-η-high-delay tested.
    Low-η P3 extended to fi=10+3=13 for safety.
    H = max(6+1, 6+2, 10+3) = 13.
    """
    rng_h = default_rng(_case_seed("T-X-01-hi"))
    rng_m = default_rng(_case_seed("T-X-01-mid"))
    rng_l = default_rng(_case_seed("T-X-01-lo"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta_h = make_eta_high(1, 6, rng_h)
    eta_m = make_eta_mid(1, 6, rng_m)
    eta_l = make_eta_low(1, 10, rng_l)
    delays = [[0, 1], [1, 2], [2, 3]]
    p_hi = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6+max(delays[0]), si=1,
                alloc=uniform_alloc(BAC, 1, 6),
                ms_theta=[0.5, 1.0], ms_e=[3, 6],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_h, ms_delay=delays[0])
    p_mi = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6+max(delays[1]), si=1,
                alloc=uniform_alloc(BAC, 1, 6),
                ms_theta=[0.5, 1.0], ms_e=[3, 6],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_m, ms_delay=delays[1])
    p_lo = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10+max(delays[2]), si=1,
                alloc=uniform_alloc(BAC, 1, 10),
                ms_theta=[0.5, 1.0], ms_e=[5, 10],
                ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
                eta_schedule=eta_l, ms_delay=delays[2])
    B0 = budget_tight(3 * BAC)
    H = max(6+max(delays[0]), 6+max(delays[1]), 10+max(delays[2]))
    return make_case(
        meta={"id":"T-X-01","group":"X","Zstar":None,"tol_Z":4.0,
              "outcome":"all_completed",
              "claim":"full_spectrum_eta_delay_tight_priority_order_tested",
              "eta_regime":["high","mid","low"],"delays":delays},
        params={"n":3,"BAC":BAC,"CP":CP,"H":H,"B0":B0,"delays":delays},
        projects_spec=[p_hi, p_mi, p_lo], B0=B0, H=H)

def t_x_02():
    """
    T-X-02 | 6,6,12 | same-diff | η=low all | delays=1,2/2,3/0,1,2 | starved
    Claim: worst-case scenario — low η, payment delays, and starved budget
           together; verifies graceful degradation (at least one project survives).
    P1 (6-period): fi=10+2=12. P2 (6-period): terminated at t=3. P3 (12-period):
    terminated at t=5. P1 survives as it has lowest delay and most spend.
    H = max(12, 6, 14) = 14.
    """
    rng1 = default_rng(_case_seed("T-X-02-P1"))
    rng2 = default_rng(_case_seed("T-X-02-P2"))
    rng3 = default_rng(_case_seed("T-X-02-P3"))
    BAC = 100; CP = round(BAC * 1.10, 2)
    eta1 = make_eta_low(1, 10, rng1)
    eta2 = make_eta_low(1, 6, rng2)
    eta3 = make_eta_low(1, 12, rng3)
    delays = [[1, 2], [2, 3], [0, 1, 2]]
    p1 = dict(BAC=BAC, eta=1.0, D_plan=10, fi=10+max(delays[0]), si=1,
              alloc=uniform_alloc(BAC, 1, 10),
              ms_theta=[0.5, 1.0], ms_e=[5, 10],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta1, ms_delay=delays[0])
    p2 = dict(BAC=BAC, eta=1.0, D_plan=6, fi=6+max(delays[1]), si=1,
              alloc={},
              ms_theta=[0.5, 1.0], ms_e=[3, 6],
              ms_phi=ms_uniform(2), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta2, ms_delay=delays[1], term_at=3)
    p3 = dict(BAC=BAC, eta=1.0, D_plan=12, fi=12+max(delays[2]), si=1,
              alloc={},
              ms_theta=ms_theta_uniform(3), ms_e=[4, 8, 12],
              ms_phi=ms_uniform(3), CP=CP, mu=0.20, Omega=3, tau_tol=2,
              eta_schedule=eta3, ms_delay=delays[2], term_at=5)
    B0 = budget_starved(3 * BAC)
    H = max(10+max(delays[0]), 6+max(delays[1]), 12+max(delays[2]))
    return make_case(
        meta={"id":"T-X-02","group":"X","Zstar":None,"tol_Z":4.0,
              "outcome":"p1_survives_p2_p3_terminated",
              "claim":"worst_case_low_eta_delayed_starved_graceful_degradation",
              "eta_regime":["low","low","low"],"delays":delays},
        params={"n":3,"BAC":BAC,"CP":CP,"H":H,"B0":B0,"delays":delays},
        projects_spec=[p1, p2, p3], B0=B0, H=H)


# ═════════════════════════════════════════════════════════════════════════════
# Master case registry
# ═════════════════════════════════════════════════════════════════════════════

ALL_CASES = {
    # ── Legacy (Groups G1–G9) ──────────────────────────────────────────────
    "SP-1":   sp1(),   "SP-3":  sp3(),   "SP-J":  spJ(),
    "SP-4":   sp4(),   "MP-K":  mpK(),   "MP-Kp": mpKp(),
    "SP-L":   spL(),   "SP-Lp": spLp(),  "SP-M":  spM(),   "SP-Mp": spMp(),
    "SP-2":   sp2(),   "SP-2a": sp2a(),  "SP-2b": sp2b(),
    "SP-2c":  sp2c(),  "SP-2d": sp2d(),
    "SP-A":   spA(),   "SP-B":  spB(),   "SP-C":  spC(),   "SP-D":  spD(),
    "SP-E":   spE(),   "SP-F":  spF(),   "SP-G":  spG(),
    "SP-5":   sp5(),   "SP-H":  spH(),   "SP-I":  spI(),
    "MP-3":   mp3(),   "MP-A":  mpA(),   "MP-B":  mpB(),
    "MP-2":   mp2(),   "MP-4":  mp4(),   "MP-C":  mpC(),
    "MP-D":   mpD(),   "MP-1":  mp1(),   "MP-E":  mpE(),

    # ── Family B — Deterministic Baseline ─────────────────────────────────
    "S-B-01": s_b_01(), "S-B-02": s_b_02(), "S-B-03": s_b_03(),
    "S-B-04": s_b_04(), "S-B-05": s_b_05(), "S-B-06": s_b_06(),
    "S-B-07": s_b_07(), "S-B-08": s_b_08(), "S-B-09": s_b_09(),
    "S-B-10": s_b_10(), "S-B-11": s_b_11(), "S-B-12": s_b_12(),

    "D-B-01": d_b_01(), "D-B-02": d_b_02(), "D-B-03": d_b_03(),
    "D-B-04": d_b_04(), "D-B-05": d_b_05(), "D-B-06": d_b_06(),
    "D-B-07": d_b_07(), "D-B-08": d_b_08(), "D-B-09": d_b_09(),
    "D-B-10": d_b_10(),

    "T-B-01": t_b_01(), "T-B-02": t_b_02(), "T-B-03": t_b_03(),
    "T-B-04": t_b_04(), "T-B-05": t_b_05(), "T-B-06": t_b_06(),
    "T-B-07": t_b_07(), "T-B-08": t_b_08(),

    # ── Family C — Performance Uncertainty ────────────────────────────────
    "S-C-01": s_c_01(), "S-C-02": s_c_02(), "S-C-03": s_c_03(),
    "S-C-04": s_c_04(),

    "D-C-01": d_c_01(), "D-C-02": d_c_02(), "D-C-03": d_c_03(),

    "T-C-01": t_c_01(), "T-C-02": t_c_02(),

    # ── Family P — Payment Delay ───────────────────────────────────────────
    "S-P-01": s_p_01(), "S-P-02": s_p_02(), "S-P-03": s_p_03(),
    "S-P-04": s_p_04(),

    "D-P-01": d_p_01(), "D-P-02": d_p_02(), "D-P-03": d_p_03(),

    "T-P-01": t_p_01(), "T-P-02": t_p_02(),

    # ── Family X — Combined Uncertainty ───────────────────────────────────
    "S-X-01": s_x_01(), "S-X-02": s_x_02(),

    "D-X-01": d_x_01(), "D-X-02": d_x_02(),

    "T-X-01": t_x_01(), "T-X-02": t_x_02(),
}


# ═════════════════════════════════════════════════════════════════════════════
# CLI: write cases.json
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    HERE = Path(__file__).parent
    out_path = HERE / "cases.json"
    with open(out_path, "w") as f:
        json.dump(ALL_CASES, f, indent=2)
    print(f"Written {out_path}  ({len(ALL_CASES)} cases)")
    for k, c in ALL_CASES.items():
        n   = len(c["projects"])
        tls = [len(p["timeline"]) for p in c["projects"]]
        zs  = c["meta"]["Zstar"]
        grp = c["meta"]["group"]
        print(f"  {k:10s}  grp={grp:3s}  n={n}  rows={tls}  Z*={zs}")