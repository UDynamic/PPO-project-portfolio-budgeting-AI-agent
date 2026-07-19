"""
milp_ppm_test.py
================
Level-1 Deterministic Full-Foresight MILP — PPM Budget Allocation
Implements Section 3 of the paper exactly.

The MILP is the paper's Level-1 formulation:
  - Deterministic eta (known efficiency)
  - No termination inside the MILP (termination is post-solve validation only)
  - Variables: x, P, B, u
  - Constraints: balance, progress recursion, budget, certification big-M

Termination is evaluated post-solve in build_records() using the
corrected conditions (inf SPI/EAC = condition active).

Usage:
    python milp_ppm_test.py [--seed N] [--n_projects N] [--horizon N]
                            [--b1_ratio F] [--no_csv] [--no_plot]

    # Run on a known-solution test scenario:
    python milp_ppm_test.py --scenario 1   # or 2 or 3
"""

import argparse
import random
import math
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import betainc
import pulp
from pulp import (LpProblem, LpMaximize, LpVariable, LpBinary,
                  lpSum, value, PULP_CBC_CMD)


# =============================================================================
# CONFIG  —  edit to control random portfolio generation
# =============================================================================

CONFIG = dict(
    seed              = 42,
    n_projects        = 5,       # set None for random in [n_min, n_max]
    n_min             = 5,
    n_max             = 10,
    horizon           = 24,
    gamma             = 0.97,
    b1_ratio          = (0.35, 0.55),   # B1 as fraction of total BAC

    # Schedule
    s_max_frac        = 0.20,
    dur_min_frac      = 0.33,
    dur_max_frac      = 0.80,

    # S-curve shape
    a_range           = (1.5, 3.5),
    b_range           = (2.0, 5.0),

    # Contract
    BAC_range         = (50_000, 300_000),
    pi_range          = (0.05, 0.20),
    rho_range         = (0.05, 0.10),
    delta_rec         = 0.25,
    psi               = 0.10,

    # Milestones
    n_ms_range        = (3, 5),
    ms_min_gap        = 0.15,
    ms_lo             = 0.20,
    ms_hi             = 0.85,

    # Termination parameters
    Omega_range       = (2, 5),
    mu_range          = (0.10, 0.30),
    tau_tol_range     = (1, 3),

    # Efficiency eta_{i,t} — bell-shaped over [s_i, f_i]
    eta_base_lo       = 0.65,
    eta_base_hi       = 1.00,
    eta_noise         = 0.03,
    eta_min           = 0.50,
    eta_max           = 1.10,

    solver_time_limit = 300,
)


# =============================================================================
# 1.  HELPERS
# =============================================================================

def scurve(tau, a, b):
    """Regularised incomplete Beta — P^plan_{i,t} = I_{tau^c}(a,b)."""
    tc = max(0.0, min(1.0, tau))
    if tc <= 0.0: return 0.0
    if tc >= 1.0: return 1.0
    return float(betainc(a, b, tc))


def derive_alpha(phi, drec, psi):
    """
    alpha_i = drec * sum(phi_j : cumulative phi_j >= psi)
    Ensures payment identity A + sum(R_net) + R_ret = CP holds exactly.
    """
    cum = rec_frac = 0.0
    for phi_j in phi:
        cum += phi_j
        if cum >= psi:
            rec_frac += phi_j
    return drec * rec_frac


def compute_R_net(proj):
    """
    Returns (R_net_list, R_ret) where:
      R_net[j] = phi_j*CP*(1-rho) - drec*phi_j*CP*1{cum_phi_j >= psi}
      R_ret    = rho * CP
    """
    CP, rho   = proj['CP_i'], proj['rho_i']
    drec, psi = proj['delta_rec_i'], proj['psi_i']
    cum, out  = 0.0, []
    for phi_j in proj['phi']:
        cum   += phi_j
        gross  = phi_j * CP
        out.append(gross - rho * gross - (drec * gross if cum >= psi else 0.0))
    return out, rho * CP


def planned_progress(proj, H):
    """P^plan_{i,t} for t = 1..H via S-curve (eq. 2)."""
    s, f, D = proj['s_i'], proj['f_i'], proj['D_plan']
    a, b    = proj['a_i'], proj['b_i']
    out = {}
    for t in range(1, H + 1):
        if   t < s: out[t] = 0.0
        elif t > f: out[t] = 1.0
        else:       out[t] = scurve((t - s + 1) / D, a, b)
    return out


def verify_payment_identity(projects, A, R_net, R_ret):
    """Assert A + sum(R_net) + R_ret = CP for every project."""
    for i, p in enumerate(projects):
        total = A[i] + sum(R_net[i, j] for j in range(p['M_i'])) + R_ret[i]
        assert abs(total - p['CP_i']) < 1.0, \
            f"Payment identity failed P{i}: {total:.4f} vs CP={p['CP_i']:.4f}"


# =============================================================================
# 2.  PORTFOLIO GENERATION
# =============================================================================

def generate_project(idx, H, cfg, rng):
    s_i    = rng.randint(1, max(1, int(H * cfg['s_max_frac'])))
    dur    = rng.randint(int(H * cfg['dur_min_frac']), int(H * cfg['dur_max_frac']))
    f_i    = min(s_i + dur - 1, H)
    D_plan = f_i - s_i + 1
    a_i    = rng.uniform(*cfg['a_range'])
    b_i    = rng.uniform(*cfg['b_range'])

    BAC_i  = rng.uniform(*cfg['BAC_range'])
    pi_i   = rng.uniform(*cfg['pi_range'])
    CP_i   = BAC_i * (1 + pi_i)
    rho_i  = rng.uniform(*cfg['rho_range'])
    drec_i = cfg['delta_rec']
    psi_i  = cfg['psi']

    n_ms   = rng.randint(*cfg['n_ms_range'])
    pts    = sorted(rng.uniform(cfg['ms_lo'], cfg['ms_hi']) for _ in range(n_ms - 1))
    spaced = [pts[0]]
    for v in pts[1:]:
        spaced.append(max(v, spaced[-1] + cfg['ms_min_gap']))
    theta   = [min(v, 0.95) for v in spaced] + [1.0]
    M_i     = len(theta)
    phi     = [1.0 / M_i] * M_i
    alpha_i = derive_alpha(phi, drec_i, psi_i)

    Omega_i   = rng.randint(*cfg['Omega_range'])
    mu_i      = rng.uniform(*cfg['mu_range'])
    tau_tol_i = rng.randint(*cfg['tau_tol_range'])

    lo, hi, noise = cfg['eta_base_lo'], cfg['eta_base_hi'], cfg['eta_noise']
    eta = {}
    for t in range(1, H + 1):
        if s_i <= t <= f_i:
            tau_n  = (t - s_i) / max(D_plan - 1, 1)
            base   = lo + (hi - lo) * math.sin(math.pi * tau_n)
            eta[t] = max(cfg['eta_min'], min(cfg['eta_max'],
                         base + rng.uniform(-noise, noise)))
        else:
            eta[t] = 0.0

    return dict(
        idx=idx, s_i=s_i, f_i=f_i, D_plan=D_plan,
        a_i=a_i, b_i=b_i,
        BAC_i=BAC_i, pi_i=pi_i, CP_i=CP_i,
        alpha_i=alpha_i, delta_rec_i=drec_i, psi_i=psi_i, rho_i=rho_i,
        M_i=M_i, theta=theta, phi=phi,
        Omega_i=Omega_i, mu_i=mu_i, tau_tol_i=tau_tol_i,
        eta=eta,
    )


def generate_portfolio(cfg):
    rng       = random.Random(cfg['seed'])
    n         = cfg['n_projects'] or rng.randint(cfg['n_min'], cfg['n_max'])
    H         = cfg['horizon']
    projects  = [generate_project(i, H, cfg, rng) for i in range(n)]
    total_BAC = sum(p['BAC_i'] for p in projects)
    lo, hi    = cfg['b1_ratio']
    B1        = rng.uniform(lo, hi) * total_BAC
    return dict(n=n, H=H, projects=projects, B1=B1,
                gamma=cfg['gamma'], total_BAC=total_BAC, cfg=cfg)


# =============================================================================
# 3.  KNOWN-SOLUTION TEST SCENARIOS
#     Three hand-crafted portfolios with analytically known optimal solutions.
#     B1 = 100 * total_BAC (unlimited), eta = 1.0 (flat).
#     Optimal policy: spend BAC_i at t = s_i for every project.
#     Z*_L1 = sum_i BAC_i * pi_i * gamma^(s_i - 1)
# =============================================================================

def _make_test_project(idx, s, f, BAC, pi, n_ms, H,
                       rho=0.05, drec=0.25, psi=0.10,
                       Omega=5, mu=0.30, tau_tol=3):
    phi   = [1.0 / n_ms] * n_ms
    theta = [(j + 1) / n_ms for j in range(n_ms)]
    alpha = derive_alpha(phi, drec, psi)
    eta   = {t: (1.0 if s <= t <= f else 0.0) for t in range(1, H + 1)}
    return dict(
        idx=idx, s_i=s, f_i=f, D_plan=f - s + 1,
        a_i=2.0, b_i=3.0,
        BAC_i=float(BAC), pi_i=float(pi), CP_i=float(BAC) * (1 + float(pi)),
        alpha_i=alpha, delta_rec_i=drec, psi_i=psi, rho_i=rho,
        M_i=n_ms, theta=theta, phi=phi,
        Omega_i=Omega, mu_i=mu, tau_tol_i=tau_tol,
        eta=eta,
    )


def _make_test_portfolio(projects, gamma, H=20, b1_mult=100):
    total_BAC = sum(p['BAC_i'] for p in projects)
    return dict(
        n=len(projects), H=H, projects=projects,
        B1=b1_mult * total_BAC,
        gamma=gamma, total_BAC=total_BAC,
        cfg=dict(seed=0, solver_time_limit=300),
    )


def _Z_star(portfolio):
    """Z*_L1 = sum_i BAC_i * pi_i * gamma^(s_i-1)  (unlimited B1, eta=1)."""
    return sum(p['BAC_i'] * p['pi_i'] * portfolio['gamma'] ** (p['s_i'] - 1)
               for p in portfolio['projects'])


def build_test_scenarios(gamma=0.97):
    H = 20

    # Scenario 1: 5 identical projects, 1 milestone
    # Z* = 5 * 10000 * 0.10 * 1.0 = 5000.0
    sc1_projs = [_make_test_project(i, 1, 10, 10_000, 0.10, 1, H) for i in range(5)]
    sc1_port  = _make_test_portfolio(sc1_projs, gamma, H)

    # Scenario 2: 3 staggered projects, 3 milestones each
    # Z* = 2000*1.0 + 2000*0.97^2 + 2000*0.97^4 = 5652.3856
    sc2_projs = [
        _make_test_project(0,  1,  8, 20_000, 0.10, 3, H),
        _make_test_project(1,  3, 12, 20_000, 0.10, 3, H),
        _make_test_project(2,  5, 15, 20_000, 0.10, 3, H),
    ]
    sc2_port = _make_test_portfolio(sc2_projs, gamma, H)

    # Scenario 3: 4 same-start projects, varying pi, 2 milestones each
    # Z* = 10000*(0.05+0.10+0.15+0.20)*1.0 = 5000.0
    sc3_projs = [
        _make_test_project(0, 1, 10, 10_000, 0.05, 2, H),
        _make_test_project(1, 1, 10, 10_000, 0.10, 2, H),
        _make_test_project(2, 1, 10, 10_000, 0.15, 2, H),
        _make_test_project(3, 1, 10, 10_000, 0.20, 2, H),
    ]
    sc3_port = _make_test_portfolio(sc3_projs, gamma, H)

    return [
        dict(
            name      = "S1: 5 identical single-milestone projects",
            portfolio = sc1_port,
            Z_star    = _Z_star(sc1_port),
            tol       = 0.01,
            note      = "Symmetry + no discounting (all s=1). Z*=5000.",
        ),
        dict(
            name      = "S2: 3 staggered projects, 3 milestones",
            portfolio = sc2_port,
            Z_star    = _Z_star(sc2_port),
            tol       = 0.01,
            note      = "Discounting test: earlier start = higher contribution. Z*=5652.39.",
        ),
        dict(
            name      = "S3: 4 same-start projects, varying pi, 2 milestones",
            portfolio = sc3_port,
            Z_star    = _Z_star(sc3_port),
            tol       = 0.01,
            note      = "Profit margin test: all funded (B1 unlimited). Z*=5000.",
        ),
    ]


# =============================================================================
# 4.  MILP  (eq. L1 — Section 3 of the paper)
#
# Variables:
#   x[i,t]     continuous >= 0      budget allocated to project i at period t
#   P[i,t]     continuous [0,1]     cumulative actual progress
#   B[t]       continuous >= 0      portfolio cash balance at start of t
#   u[i,j,t]   binary               1 if milestone j of project i certified BY t
#
# Objective:
#   max  sum_t  gamma^(t-1) * ( V_t  -  sum_i act[i,t]*x[i,t] )
#   V_t = advance payments at t=s_i
#       + milestone net payments when first certified
#       + retention at final milestone
#
# Constraints (labelled to match paper equations):
#   B_init        B[1] = B1
#   bal_{t}       B[t+1] = B[t] - outflow_t + V_t
#   bud_{t}       sum_i act[i,t]*x[i,t] <= B[t]
#   prg_{i,t}     P[i,t] = P[i,t-1] + eta[i,t]*x[i,t]/BAC_i
#   off_{i,t}     x[i,t] = 0  if t outside [s_i, f_i]
#   clb_{i,j,t}   P[i,t] >= theta[j] - BIG_M*(1 - u[i,j,t])
#   cub_{i,j,t}   P[i,t] <= (theta[j]-EPS) + BIG_M*u[i,j,t]
#   mon_{i,j,t}   u[i,j,t] >= u[i,j,t-1]
#   ord_{i,j,t}   u[i,j+1,t] <= u[i,j,t]
#   pre_{i,j,t}   u[i,j,t] = 0  for t < s_i
# =============================================================================

def build_and_solve(portfolio):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    B1       = portfolio['B1']
    gamma    = portfolio['gamma']
    T        = list(range(1, H + 1))

    plan_prog = {i: planned_progress(p, H) for i, p in enumerate(projects)}

    # Fixed parameters (pre-computed before building LP)
    act   = {(i, t): int(projects[i]['s_i'] <= t <= projects[i]['f_i'])
             for i in range(n) for t in T}
    A     = {i: projects[i]['alpha_i'] * projects[i]['CP_i'] for i in range(n)}
    R_net = {}
    R_ret = {}
    for i, p in enumerate(projects):
        rn, rr = compute_R_net(p)
        for j, v in enumerate(rn):
            R_net[i, j] = v
        R_ret[i] = rr

    verify_payment_identity(projects, A, R_net, R_ret)

    BIG_M = 1.0    # all theta in (0,1]
    EPS   = 1e-4   # strict inequality tolerance for certification UB

    prob = LpProblem("PPM_L1", LpMaximize)

    # ── Variables ─────────────────────────────────────────────────────────
    x = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0)
         for i in range(n) for t in T}

    u = {(i, j, t): LpVariable(f"u_{i}_{j}_{t}", cat=LpBinary)
         for i in range(n)
         for j in range(projects[i]['M_i'])
         for t in T}

    P = {(i, t): LpVariable(f"P_{i}_{t}", lowBound=0, upBound=1.0)
         for i in range(n) for t in T}

    B = {t: LpVariable(f"B_{t}", lowBound=0) for t in T}

    # ── Inflow expression V_t ─────────────────────────────────────────────
    # V_t = A_i * 1{t=s_i}
    #     + sum_j (R_net[i,j] + R_ret[i]*1{j=last}) * (u[i,j,t] - u[i,j,t-1])
    def V(t):
        terms = []
        for i, p in enumerate(projects):
            # Advance payment at project start
            if t == p['s_i']:
                terms.append(A[i])
            # Milestone payments (triggered when first certified)
            for j in range(p['M_i']):
                u_prev  = u[i, j, t - 1] if t > 1 else 0
                pay     = R_net[i, j] + (R_ret[i] if j == p['M_i'] - 1 else 0)
                terms.append(pay * (u[i, j, t] - u_prev))
        return lpSum(terms)

    # ── Objective ─────────────────────────────────────────────────────────
    prob += lpSum(
        gamma ** (t - 1) * (V(t) - lpSum(act[i, t] * x[i, t] for i in range(n)))
        for t in T
    ), "obj"

    # ── B_init: initial balance ───────────────────────────────────────────
    prob += B[1] == B1, "B_init"

    # ── bal_{t}: balance recursion ────────────────────────────────────────
    for t in T[:-1]:
        prob += (B[t + 1] == B[t]
                 - lpSum(act[i, t] * x[i, t] for i in range(n))
                 + V(t),
                 f"bal_{t}")

    # ── bud_{t}: budget feasibility ───────────────────────────────────────
    for t in T:
        prob += (lpSum(act[i, t] * x[i, t] for i in range(n)) <= B[t],
                 f"bud_{t}")

    # ── prg_{i,t}: progress recursion  /  off_{i,t}: inactive lock ───────
    for i, p in enumerate(projects):
        for t in T:
            P_prev = P[i, t - 1] if t > 1 else 0.0
            prob += (P[i, t] == P_prev + p['eta'][t] * x[i, t] / p['BAC_i'],
                     f"prg_{i}_{t}")
            if act[i, t] == 0:
                prob += (x[i, t] == 0, f"off_{i}_{t}")

    # ── Certification big-M ───────────────────────────────────────────────
    for i, p in enumerate(projects):
        for j in range(p['M_i']):
            th = p['theta'][j]
            for t in T:
                # clb: u=1 forces P >= theta
                prob += (P[i, t] >= th - BIG_M * (1 - u[i, j, t]),
                         f"clb_{i}_{j}_{t}")
                # cub: u=0 forces P < theta
                prob += (P[i, t] <= (th - EPS) + BIG_M * u[i, j, t],
                         f"cub_{i}_{j}_{t}")
                # mon: once certified stays certified
                if t > 1:
                    prob += (u[i, j, t] >= u[i, j, t - 1],
                             f"mon_{i}_{j}_{t}")
                # ord: milestone j+1 requires milestone j
                if j < p['M_i'] - 1:
                    prob += (u[i, j + 1, t] <= u[i, j, t],
                             f"ord_{i}_{j}_{t}")
                # pre: no certification before project start
                if t < p['s_i']:
                    prob += (u[i, j, t] == 0, f"pre_{i}_{j}_{t}")

    # ── Solve ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"SOLVING MILP  (CBC, limit={portfolio['cfg']['solver_time_limit']}s)...")
    print("=" * 72)
    solver = PULP_CBC_CMD(msg=0, timeLimit=portfolio['cfg']['solver_time_limit'])
    prob.solve(solver)

    return dict(
        status    = pulp.LpStatus[prob.status],
        obj       = value(prob.objective),
        prob      = prob,
        x=x, u=u, P=P, B=B,
        act=act, A=A, R_net=R_net, R_ret=R_ret,
        plan_prog = plan_prog,
    )


# =============================================================================
# 5.  STATE RECORD SYSTEM
#     Captures every variable at every (project, period) post-solve.
#     Termination is evaluated here using the solved x_{i,t}.
# =============================================================================

def build_records(portfolio, sol):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    B1       = portfolio['B1']
    T        = list(range(1, H + 1))

    x_val = {k: max(0.0, value(v) or 0.0) for k, v in sol['x'].items()}
    u_val = {k: int(round(value(v) or 0))  for k, v in sol['u'].items()}
    P_lp  = {k: (value(v) or 0.0)          for k, v in sol['P'].items()}
    B_lp  = {t: (value(v) or 0.0)          for t, v in sol['B'].items()}

    plan_prog       = sol['plan_prog']
    R_net, R_ret, A = sol['R_net'], sol['R_ret'], sol['A']
    act             = sol['act']

    proj_records = {}
    all_events   = []

    for i, p in enumerate(projects):
        s, f       = p['s_i'], p['f_i']
        BAC, CP    = p['BAC_i'], p['CP_i']
        alpha      = p['alpha_i']
        drec, psi  = p['delta_rec_i'], p['psi_i']
        rho        = p['rho_i']
        theta, phi = p['theta'], p['phi']
        M          = p['M_i']
        eta        = p['eta']
        Omega      = p['Omega_i']
        mu         = p['mu_i']
        tau_tol    = p['tau_tol_i']
        D_plan     = p['D_plan']

        P_sim        = 0.0
        ACWP_sim     = 0.0
        certified    = set()
        cum_phi_cert = 0.0
        adv_recov    = 0.0
        status       = 'pre_start'
        tau_rem      = tau_tol
        records      = {}

        # Advance payment event at project start
        if s <= H:
            all_events.append(dict(t=s, proj=i, type='advance', amount=A[i]))

        for t in T:
            x_t = x_val[i, t]

            # Window
            if   t < s: window = 'pre_start'
            elif t > f: window = 'post_plan'
            else:       window = 'active'

            # Progress update (eq. 1)
            if window == 'active' and status not in ('terminated', 'completed'):
                P_sim    = min(1.0, P_sim + eta[t] * x_t / BAC)
                ACWP_sim += x_t
                status   = 'active'

            EV_sim = P_sim * BAC

            # Milestone certification (sim)
            ms_new_sim = []
            if window == 'active' and status == 'active':
                for j in range(M):
                    if j not in certified and P_sim >= theta[j] - 1e-6:
                        certified.add(j)
                        cum_phi_cert += phi[j]
                        gross     = phi[j] * CP
                        ret_ded   = rho * gross
                        rec_ded   = drec * gross if cum_phi_cert >= psi else 0.0
                        adv_recov += rec_ded
                        R_net_j   = gross - ret_ded - rec_ded
                        ms_new_sim.append(j)
                        all_events.append(dict(t=t, proj=i,
                                               type=f'ms{j+1}_net', amount=R_net_j))
                        if j == M - 1:
                            all_events.append(dict(t=t, proj=i,
                                                   type='retention', amount=rho * CP))

            # EVM signals (eqs. 4, 5, 6)
            # Corrected: inf SPI/EAC counts as condition active, not missing data
            if window == 'active' and status == 'active':
                P_plan_t = plan_prog[i][t]

                # SPI = P_actual / P_plan
                if P_plan_t > 1e-9:
                    SPI = P_sim / P_plan_t
                elif P_sim < 1e-9:
                    SPI = 1.0          # both zero: on schedule
                else:
                    SPI = float('inf') # ahead of a zero plan

                # CPI = EV / ACWP
                # If ACWP=0 and t>s: project started but spent nothing → EAC=inf
                if ACWP_sim > 1e-9:
                    CPI = EV_sim / ACWP_sim
                    EAC = ACWP_sim + (BAC - EV_sim) / CPI if CPI > 1e-9 else float('inf')
                elif t > s:
                    CPI = float('inf')
                    EAC = float('inf')
                else:
                    CPI = 1.0
                    EAC = BAC

                # Forecast finish (eq. 7)
                elapsed = t - s
                rem_per = D_plan - elapsed
                if SPI > 1e-9 and not math.isinf(SPI):
                    f_hat   = t + rem_per / SPI
                    Delta_f = f_hat - f
                else:
                    f_hat   = float('inf')
                    Delta_f = float('inf')

                # Termination conditions (eqs. 8, 9)
                # inf = worst case: condition always active
                cond1 = math.isinf(Delta_f) or (Delta_f > Omega)
                cond2 = math.isinf(EAC)     or (EAC > (1 + mu) * BAC)
            else:
                SPI = CPI = EAC = Delta_f = f_hat = None
                cond1 = cond2 = False

            # Cure-period counter (eq. counter)
            if window == 'active' and status == 'active':
                tau_rem = tau_rem - 1 if (cond1 and cond2) else tau_tol

            # LP u values at this period
            u_now  = {j: u_val.get((i, j, t), 0) for j in range(M)}
            u_prev = ({j: u_val.get((i, j, t - 1), 0) for j in range(M)}
                      if t > 1 else {j: 0 for j in range(M)})
            ms_new_lp = [j for j in range(M) if u_now[j] - u_prev.get(j, 0) > 0]

            records[t] = dict(
                # Indices
                proj=i, t=t,
                # Window / status
                window=window, status=status,
                # LP solution values
                x_lp=x_t, P_lp=P_lp.get((i, t)),
                u_lp=u_now, u_prev_lp=u_prev, ms_new_lp=ms_new_lp,
                # Simulated state (forward sim with solved x)
                P_sim=P_sim, ACWP_sim=ACWP_sim, EV_sim=EV_sim,
                # Planned
                P_plan=plan_prog[i][t], eta=eta.get(t, 0.0),
                # EVM signals
                SPI=SPI, CPI=CPI, EAC=EAC, f_hat=f_hat, Delta_f=Delta_f,
                # Termination
                cond1=cond1, cond2=cond2, tau_rem=tau_rem,
                ms_new_sim=ms_new_sim,
                # Parameters (for traceability)
                theta=list(theta), Omega=Omega, mu=mu, BAC=BAC, CP=CP,
            )

            # Termination execution
            if window == 'active' and status == 'active' and tau_rem <= 0:
                status = 'terminated'
                A_i    = alpha * CP
                R_term = P_sim * CP * (1 - rho) - (A_i - adv_recov)
                all_events.append(dict(t=t + 1, proj=i,
                                       type='termination_settlement', amount=R_term))
                records[t]['status'] = 'terminated'
            elif window == 'active' and status == 'active' and P_sim >= 1.0 - 1e-6:
                status = 'completed'
                records[t]['status'] = 'completed'

        proj_records[i] = dict(
            records=records, status=status,
            P_final=P_sim, ACWP_final=ACWP_sim, certified=certified,
        )

    # Portfolio-level records
    events_by_t = {}
    for ev in all_events:
        events_by_t.setdefault(ev['t'], []).append(ev)

    port_records = {}
    running_B = B1
    for t in T:
        outflow = sum(x_val[i, t] for i in range(n))
        inflow  = sum(ev['amount'] for ev in events_by_t.get(t, []))
        active  = [i for i in range(n)
                   if proj_records[i]['records'][t]['window'] == 'active'
                   and proj_records[i]['records'][t]['status'] == 'active']
        port_records[t] = dict(
            t=t, B_lp=B_lp.get(t), B_sim=running_B,
            outflow=outflow, inflow=inflow, net=inflow - outflow,
            n_active=len(active), active_proj=active,
            events=events_by_t.get(t, []),
        )
        running_B = running_B - outflow + inflow

    run = 0.0
    for ev in sorted(all_events, key=lambda e: (e['t'], e['proj'])):
        run += ev['amount']
        ev['running_total'] = run

    return dict(
        proj=proj_records, portfolio=port_records,
        events=all_events,
        x_val=x_val, u_val=u_val, P_lp=P_lp, B_lp=B_lp,
    )


# =============================================================================
# 6.  VALIDATION
# =============================================================================

def validate(portfolio, sol, rec, tol=1e-3):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    T        = list(range(1, H + 1))
    x_val, u_val, P_lp, B_lp = rec['x_val'], rec['u_val'], rec['P_lp'], rec['B_lp']
    R_net, R_ret, A, act = sol['R_net'], sol['R_ret'], sol['A'], sol['act']
    errors = {}

    # C1: B_t >= 0
    errors['C1_B_nonneg'] = [
        (t, round(B_lp[t], 2)) for t in T if B_lp.get(t, 0) < -tol
    ]

    # C2: sum x_{i,t} <= B_t
    errors['C2_budget'] = [
        dict(t=t, spend=round(sum(x_val[i, t] for i in range(n)), 2),
             bal=round(B_lp.get(t, 0), 2))
        for t in T if sum(x_val[i, t] for i in range(n)) > B_lp.get(t, 0) + tol
    ]

    # C3: P recursion
    prog_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc  = min(1.0, Pc + p['eta'][t] * x_val[i, t] / p['BAC_i'])
            lp  = P_lp.get((i, t), 0.0)
            if abs(lp - Pc) > tol * 5:
                prog_errs.append(dict(proj=i, t=t, lp=round(lp, 6), sim=round(Pc, 6)))
    errors['C3_progress'] = prog_errs[:10]

    # C4: balance recursion
    bal_errs = []
    Bc = portfolio['B1']
    for t in T[:-1]:
        out = sum(act[i, t] * x_val[i, t] for i in range(n))
        inf = sum(A[i] for i in range(n) if t == projects[i]['s_i'])
        for i, p in enumerate(projects):
            for j in range(p['M_i']):
                u_now  = u_val.get((i, j, t), 0)
                u_prev = u_val.get((i, j, t - 1), 0) if t > 1 else 0
                if u_now - u_prev > 0:
                    inf += R_net[i, j] + (R_ret[i] if j == p['M_i'] - 1 else 0)
        Bc_next = Bc - out + inf
        lp_next = B_lp.get(t + 1, 0.0)
        if abs(lp_next - Bc_next) > tol * 10:
            bal_errs.append(dict(t=t, lp=round(lp_next, 2), comp=round(Bc_next, 2)))
        Bc = Bc_next
    errors['C4_balance'] = bal_errs[:5]

    # C5: u=1 iff P >= theta
    cert_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc = min(1.0, Pc + p['eta'][t] * x_val[i, t] / p['BAC_i'])
            for j in range(p['M_i']):
                u_ij    = u_val.get((i, j, t), 0)
                reached = Pc >= p['theta'][j] - tol
                if reached and u_ij == 0:
                    cert_errs.append(dict(issue='missed', proj=i, ms=j, t=t, P=round(Pc, 5)))
                if not reached and u_ij == 1:
                    cert_errs.append(dict(issue='premature', proj=i, ms=j, t=t, P=round(Pc, 5)))
    errors['C5_cert'] = cert_errs[:10]

    # C6: u monotone
    errors['C6_mono'] = [
        (i, j, t) for i, p in enumerate(projects)
        for j in range(p['M_i']) for t in range(2, H + 1)
        if u_val.get((i, j, t), 0) < u_val.get((i, j, t - 1), 0)
    ][:10]

    # C7: u ordered
    errors['C7_order'] = [
        (i, j, t) for i, p in enumerate(projects)
        for j in range(p['M_i'] - 1) for t in T
        if u_val.get((i, j + 1, t), 0) > u_val.get((i, j, t), 0)
    ][:10]

    # C8: no allocation outside window
    errors['C8_inactive'] = [
        (i, t, round(x_val[i, t], 2)) for i in range(n) for t in T
        if act[i, t] == 0 and x_val[i, t] > tol
    ][:10]

    # C9: payment identity (completed projects)
    pay_errs = []
    for i, p in enumerate(projects):
        if rec['proj'][i]['status'] == 'completed':
            total = A[i] + sum(R_net[i, j] for j in range(p['M_i'])) + R_ret[i]
            if abs(total - p['CP_i']) > tol * p['CP_i']:
                pay_errs.append(dict(proj=i, total=round(total, 2), CP=round(p['CP_i'], 2)))
    errors['C9_payment_id'] = pay_errs

    # C10: x >= 0
    errors['C10_nonneg'] = [
        (i, t, round(x_val[i, t], 4)) for (i, t) in x_val if x_val[i, t] < -tol
    ]

    # Informational: terminations
    errors['_terminations'] = {
        i: next((t for t in T
                 if rec['proj'][i]['records'][t]['status'] == 'terminated'), None)
        for i in range(n) if rec['proj'][i]['status'] == 'terminated'
    }

    all_pass = all(len(v) == 0 for k, v in errors.items() if not k.startswith('_'))
    return dict(errors=errors, all_pass=all_pass)


def check_known_solution(sc, sol, rec, tol_alloc=1.0):
    """
    Additional check for known-solution scenarios.
    Verifies objective value and that each project was spent in full at s_i.
    """
    portfolio = sc['portfolio']
    Z_star    = sc['Z_star']
    H         = portfolio['H']
    T         = list(range(1, H + 1))
    x_val     = rec['x_val']

    obj_err  = abs(sol['obj'] - Z_star)
    obj_pass = obj_err <= sc['tol']

    alloc_errs = []
    for i, p in enumerate(portfolio['projects']):
        total    = sum(x_val.get((i, t), 0.0) for t in T)
        at_start = x_val.get((i, p['s_i']), 0.0)
        if abs(total - p['BAC_i']) > tol_alloc:
            alloc_errs.append(dict(proj=i, check='total_spend',
                                   milp=round(total, 2), expected=round(p['BAC_i'], 2)))
        if abs(at_start - p['BAC_i']) > tol_alloc:
            alloc_errs.append(dict(proj=i, check=f'spend_at_s={p["s_i"]}',
                                   milp=round(at_start, 2), expected=round(p['BAC_i'], 2)))

    return dict(
        obj_pass   = obj_pass,
        obj_milp   = round(sol['obj'], 4),
        obj_expect = round(Z_star, 4),
        obj_err    = round(obj_err, 4),
        alloc_pass = len(alloc_errs) == 0,
        alloc_errs = alloc_errs,
        all_pass   = obj_pass and len(alloc_errs) == 0,
    )


# =============================================================================
# 7.  VISUALISATION
# =============================================================================

COLORS = dict(
    bcws    = '#1f77b4',
    bcwp    = '#2ca02c',
    acwp    = '#ff7f0e',
    eac     = '#d62728',
    alloc   = '#9467bd',
    balance = '#17becf',
    inflow  = '#2ca02c',
    outflow = '#d62728',
    net     = '#1f77b4',
    spi     = '#2ca02c',
    cpi     = '#ff7f0e',
    ms      = '#e377c2',
)


def _safe(v, fallback=None):
    if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
        return fallback
    return v


def plot_project(proj_idx, proj, rec, H, ax_evm, ax_alloc, ax_indices):
    T      = list(range(1, H + 1))
    pr     = rec['proj'][proj_idx]
    recs   = pr['records']
    s, f   = proj['s_i'], proj['f_i']
    BAC    = proj['BAC_i']
    mu     = proj['mu_i']
    Omega  = proj['Omega_i']
    status = pr['status']

    ts       = list(range(0, H + 1))
    bcws     = [0.0] + [recs[t]['P_plan'] for t in T]
    bcwp     = [0.0] + [recs[t]['P_sim']  for t in T]
    acwp_n   = [0.0] + [recs[t]['ACWP_sim'] / BAC for t in T]
    eac_t    = [t for t in T if _safe(recs[t]['EAC']) is not None and recs[t]['EAC'] / BAC <= 2.0]
    eac_v    = [recs[t]['EAC'] / BAC for t in eac_t]

    # EVM curves
    ax_evm.plot(ts, bcws,   color=COLORS['bcws'], lw=2, label='BCWS')
    ax_evm.plot(ts, bcwp,   color=COLORS['bcwp'], lw=2, label='BCWP')
    ax_evm.plot(ts, acwp_n, color=COLORS['acwp'], lw=2, label='ACWP/BAC')
    if eac_t:
        ax_evm.plot(eac_t, eac_v, color=COLORS['eac'], lw=1.5, ls='--', label='EAC/BAC')

    # Cost overrun cap
    cap = 1.0 + mu
    ax_evm.axhspan(1.0, min(cap + 0.05, 1.40),
                   color=COLORS['eac'], alpha=0.08, label=f'Cost cap ({cap:.2f}xBAC)')
    ax_evm.axhline(cap, color=COLORS['eac'], lw=1.0, ls=':', alpha=0.6)

    # Schedule overrun cap
    if f + Omega <= H + 2:
        ax_evm.axvspan(f, min(f + Omega, H + 1),
                       color=COLORS['eac'], alpha=0.08, label=f'Sched cap (Omega={Omega})')

    ax_evm.axvline(f, color=COLORS['bcws'], lw=1.2, ls='--', alpha=0.6)
    ax_evm.text(f + 0.1, 1.28, f'f={f}', color=COLORS['bcws'], fontsize=8)

    # Milestone thresholds
    for j, th in enumerate(proj['theta']):
        ax_evm.axhline(th, color=COLORS['ms'], lw=0.8, ls=':', alpha=0.5)
        ax_evm.text(H + 0.2, th, f'th{j+1}', color=COLORS['ms'], fontsize=7, va='center')

    # Termination marker
    if status == 'terminated':
        term_t = next((t for t in T if recs[t]['status'] == 'terminated'), None)
        if term_t:
            ax_evm.axvline(term_t, color='red', lw=2, alpha=0.8)
            ax_evm.text(term_t + 0.1, 0.05, 'TERM', color='red', fontsize=8, fontweight='bold')

    ax_evm.set_xlim(0, H + 1)
    ax_evm.set_ylim(0, 1.40)
    ax_evm.set_ylabel('Fraction of BAC', fontsize=9)
    ax_evm.set_title(
        f'P{proj_idx}  EVM  s={s} f={f} D={proj["D_plan"]}  '
        f'BAC={BAC/1000:.0f}k  CP={proj["CP_i"]/1000:.0f}k  [{status}]',
        fontsize=9, fontweight='bold')
    ax_evm.legend(fontsize=7, loc='upper left', ncol=2)
    ax_evm.grid(True, alpha=0.3)

    # Allocation bars
    alloc_v = [recs[t]['x_lp'] / 1000 for t in T]
    bar_c   = ['#9467bd' if recs[t]['window'] == 'active' else '#cccccc' for t in T]
    ax_alloc.bar(T, alloc_v, color=bar_c, alpha=0.8, width=0.8)
    for j in range(proj['M_i']):
        ct = next((t for t in T
                   if rec['u_val'].get((proj_idx, j, t), 0) == 1
                   and rec['u_val'].get((proj_idx, j, t - 1), 0) == 0), None)
        if ct:
            ymax = max(alloc_v) * 0.85 if max(alloc_v) > 0 else 0.1
            ax_alloc.axvline(ct, color=COLORS['ms'], lw=1.5, ls='--', alpha=0.8)
            ax_alloc.text(ct, ymax, f'm{j+1}', color=COLORS['ms'], fontsize=7, ha='center')
    ax_alloc.set_xlim(0, H + 1)
    ax_alloc.set_ylabel('Alloc (k)', fontsize=9)
    ax_alloc.set_title(f'P{proj_idx}  Budget allocation x_{{i,t}}', fontsize=9)
    ax_alloc.grid(True, alpha=0.3, axis='y')

    # SPI / CPI
    spi_t = [t for t in T if _safe(recs[t]['SPI']) is not None]
    cpi_t = [t for t in T if _safe(recs[t]['CPI']) is not None]
    if spi_t:
        ax_indices.plot(spi_t, [min(recs[t]['SPI'], 2.0) for t in spi_t],
                        color=COLORS['spi'], lw=2, label='SPI', marker='o', ms=3)
    if cpi_t:
        ax_indices.plot(cpi_t, [min(recs[t]['CPI'], 2.0) for t in cpi_t],
                        color=COLORS['cpi'], lw=2, label='CPI', marker='s', ms=3)
    ax_indices.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.7)
    for t in T:
        if recs[t]['cond1'] and recs[t]['cond2']:
            ax_indices.axvspan(t - 0.5, t + 0.5, color='red', alpha=0.12)
    ax_indices.set_xlim(0, H + 1)
    ax_indices.set_ylim(0, 2.1)
    ax_indices.set_xlabel('Period t', fontsize=9)
    ax_indices.set_ylabel('Index', fontsize=9)
    ax_indices.set_title(f'P{proj_idx}  SPI / CPI  (red = both term. conds active)', fontsize=9)
    ax_indices.legend(fontsize=8, loc='lower right')
    ax_indices.grid(True, alpha=0.3)


def plot_portfolio(portfolio, sol, rec):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    T        = list(range(1, H + 1))
    port     = rec['portfolio']
    x_val    = rec['x_val']
    B_lp     = rec['B_lp']

    # Figure 1: per-project (3 panels each)
    fig1, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    fig1.suptitle(
        f'Per-Project State  |  seed={portfolio["cfg"]["seed"]}  '
        f'n={n}  H={H}  Z*_L1={sol["obj"]:,.0f}',
        fontsize=11, fontweight='bold')
    if n == 1:
        axes = [axes]
    for i, p in enumerate(projects):
        plot_project(i, p, rec, H, axes[i][0], axes[i][1], axes[i][2])
    fig1.tight_layout()

    # Figure 2: portfolio aggregate (6 panels)
    fig2 = plt.figure(figsize=(16, 14))
    fig2.suptitle(
        f'Portfolio Aggregate  |  n={n}  H={H}  '
        f'B1={portfolio["B1"]:,.0f}  totalBAC={portfolio["total_BAC"]:,.0f}',
        fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

    # (a) Cash balance
    ax = fig2.add_subplot(gs[0, 0])
    ax.plot(T, [B_lp.get(t, 0) / 1000 for t in T],
            color=COLORS['balance'], lw=2, label='B_lp')
    ax.plot(T, [port[t]['B_sim'] / 1000 for t in T],
            color=COLORS['balance'], lw=1.5, ls='--', alpha=0.7, label='B_sim')
    ax.set_title('Cash Balance B_t (thousands)', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('k', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.5)

    # (b) Inflow / outflow / net
    ax = fig2.add_subplot(gs[0, 1])
    inflows  = [port[t]['inflow']  / 1000 for t in T]
    outflows = [port[t]['outflow'] / 1000 for t in T]
    nets     = [port[t]['net']     / 1000 for t in T]
    ax.bar(T, inflows,  color=COLORS['inflow'],  alpha=0.7, label='Inflow',  width=0.4, align='edge')
    ax.bar([t + 0.4 for t in T], outflows, color=COLORS['outflow'], alpha=0.7,
           label='Outflow', width=0.4, align='edge')
    ax.plot(T, nets, color=COLORS['net'], lw=2, marker='o', ms=3, label='Net')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Period Cash Flows (thousands)', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('k', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # (c) Allocation stacked bar
    ax = fig2.add_subplot(gs[1, 0])
    cmap   = matplotlib.colormaps['tab10']
    bottom = [0.0] * H
    for i in range(n):
        vals = [x_val[i, t] / 1000 for t in T]
        ax.bar(T, vals, bottom=bottom, color=cmap(i), alpha=0.85, label=f'P{i}', width=0.8)
        bottom = [bottom[t - 1] + vals[t - 1] for t in T]
    ax.set_title('Budget Allocation by Project (stacked, thousands)', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('k', fontsize=8)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    # (d) Portfolio aggregate progress
    ax    = fig2.add_subplot(gs[1, 1])
    tBAC  = portfolio['total_BAC']
    a_plan = [sum(rec['proj'][i]['records'][t]['P_plan'] * projects[i]['BAC_i']
                  for i in range(n)) / tBAC for t in T]
    a_sim  = [sum(rec['proj'][i]['records'][t]['P_sim']  * projects[i]['BAC_i']
                  for i in range(n)) / tBAC for t in T]
    a_acwp = [sum(rec['proj'][i]['records'][t]['ACWP_sim']
                  for i in range(n)) / tBAC for t in T]
    ax.plot(T, a_plan, color=COLORS['bcws'], lw=2, label='BCWS agg.')
    ax.plot(T, a_sim,  color=COLORS['bcwp'], lw=2, label='BCWP agg.')
    ax.plot(T, a_acwp, color=COLORS['acwp'], lw=2, label='ACWP/totalBAC')
    ax.set_title('Portfolio Aggregate Progress (fraction of total BAC)', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('Fraction', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (e) Active project count
    ax = fig2.add_subplot(gs[2, 0])
    n_act = [port[t]['n_active'] for t in T]
    ax.step(T, n_act, color='steelblue', lw=2, where='mid')
    ax.fill_between(T, n_act, step='mid', alpha=0.2, color='steelblue')
    ax.set_title('Active Projects per Period', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('Count', fontsize=8)
    ax.set_ylim(0, n + 1); ax.grid(True, alpha=0.3)

    # (f) Cumulative discounted net cash flow
    ax    = fig2.add_subplot(gs[2, 1])
    gamma = portfolio['gamma']
    cum   = []
    run   = 0.0
    for t in T:
        run += gamma ** (t - 1) * port[t]['net']
        cum.append(run)
    ax.plot(T, [v / 1000 for v in cum], color='#1f77b4', lw=2, marker='o', ms=3)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'Cumulative Discounted Net Cash Flow (thousands, gamma={gamma})', fontsize=9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('k', fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig1, fig2


# =============================================================================
# 8.  PRINT
# =============================================================================

def _f(v, fmt='.4f', width=8, none='--'):
    if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
        return f'{none:>{width}}'
    return f'{v:{width}{fmt}}'


def print_portfolio(pf):
    print("\n" + "=" * 72)
    print("PORTFOLIO PARAMETERS")
    print("=" * 72)
    print(f"  n={pf['n']}  H={pf['H']}  B1={pf['B1']:,.0f}  "
          f"totalBAC={pf['total_BAC']:,.0f}  gamma={pf['gamma']}")
    print(f"\n  {'i':>2}  {'s':>3}  {'f':>3}  {'D':>3}  "
          f"{'BAC':>9}  {'CP':>9}  {'pi%':>5}  {'al%':>5}  "
          f"{'rho%':>5}  {'M':>2}  {'Om':>3}  {'mu%':>5}  {'ttol':>4}")
    print("  " + "─" * 73)
    for p in pf['projects']:
        print(f"  {p['idx']:>2}  {p['s_i']:>3}  {p['f_i']:>3}  {p['D_plan']:>3}  "
              f"{p['BAC_i']:>9,.0f}  {p['CP_i']:>9,.0f}  "
              f"{p['pi_i']*100:>5.1f}  {p['alpha_i']*100:>5.1f}  "
              f"{p['rho_i']*100:>5.1f}  {p['M_i']:>2}  "
              f"{p['Omega_i']:>3}  {p['mu_i']*100:>5.1f}  {p['tau_tol_i']:>4}")
    print()
    for p in pf['projects']:
        ths = "  ".join(f"th{j+1}={v:.3f}" for j, v in enumerate(p['theta']))
        ets = "  ".join(f"t{t}:{p['eta'][t]:.2f}"
                        for t in range(p['s_i'], p['f_i'] + 1))
        print(f"  P{p['idx']}: [{ths}]  alpha={p['alpha_i']:.4f}")
        print(f"       eta: [{ets}]")


def print_solution(pf, sol, rec):
    n, H  = pf['n'], pf['H']
    T     = list(range(1, H + 1))
    x_val = rec['x_val']
    B_lp  = rec['B_lp']
    u_val = rec['u_val']

    print("\n" + "=" * 72)
    print("MILP SOLUTION")
    print("=" * 72)
    print(f"  Solver : {sol['status']}")
    print(f"  Z*_L1  : {sol['obj']:,.4f}  (discounted net cash flow)")

    print("\n── ALLOCATIONS x_{i,t} (thousands) ──")
    for blk in range(1, H + 1, 12):
        cols = list(range(blk, min(blk + 12, H + 1)))
        hdr  = "      " + "".join(f" {f't{c}':>5}" for c in cols)
        print(f"\n  t={blk}..{cols[-1]}")
        print("  " + hdr)
        print("  " + "  " + "─" * len(hdr))
        for i in range(n):
            row = "".join(
                f" {x_val[i,t]/1000:>5.1f}" if x_val[i, t] > 0.5 else f" {'--':>5}"
                for t in cols)
            print(f"  P{i:<2}  {row}")
        brow = "".join(f" {B_lp.get(t,0)/1000:>5.1f}" for t in cols)
        print(f"  Bt   {brow}   <- balance")

    print("\n── PER-PROJECT OUTCOME ──")
    print(f"  {'i':>2}  {'total_x':>10}  {'P_final':>8}  "
          f"{'status':>11}  ms  cert_periods")
    print("  " + "─" * 65)
    for i, p in enumerate(pf['projects']):
        pr  = rec['proj'][i]
        tx  = sum(x_val[i, t] for t in T)
        ms_str = []
        for j in range(p['M_i']):
            ct = next((t for t in T
                       if u_val.get((i, j, t), 0) == 1
                       and u_val.get((i, j, t - 1), 0) == 0), None)
            ms_str.append(f"m{j+1}@{ct}" if ct else f"m{j+1}:--")
        print(f"  {i:>2}  {tx:>10,.0f}  {pr['P_final']:>8.4f}  "
              f"{pr['status']:>11}  {len(pr['certified'])}/{p['M_i']}  "
              + "  ".join(ms_str))


def print_state_table(pf, rec):
    print("\n" + "=" * 72)
    print("FULL STATE RECORDS")
    print("=" * 72)
    T = list(range(1, pf['H'] + 1))

    for i, p in enumerate(pf['projects']):
        pr = rec['proj'][i]
        M  = p['M_i']
        print(f"\n{'─'*72}")
        print(f"  P{i}  s={p['s_i']} f={p['f_i']} D={p['D_plan']}  "
              f"BAC={p['BAC_i']:,.0f}  CP={p['CP_i']:,.0f}  M={M}")
        print(f"  theta={[round(v,3) for v in p['theta']]}  "
              f"alpha={p['alpha_i']:.4f}  rho={p['rho_i']:.4f}  "
              f"Omega={p['Omega_i']}  mu={p['mu_i']:.3f}  tau_tol={p['tau_tol_i']}")
        print(f"{'─'*72}")
        print(f"  {'t':>3}  {'x_lp':>8}  {'eta':>5}  "
              f"{'P_plan':>7}  {'P_sim':>7}  {'P_lp':>7}  "
              f"{'ACWP':>8}  {'EV':>8}  "
              f"{'SPI':>6}  {'CPI':>6}  {'EAC':>9}  "
              f"{'Df':>6}  {'C1':>3}  {'C2':>3}  {'tau':>3}  "
              f"{'u':>{M}}  {'cert_lp':>8}  {'status':>11}")
        print("  " + "─" * 115)
        for t in T:
            r   = pr['records'][t]
            xv  = f"{r['x_lp']:>8.1f}" if r['x_lp'] > 0.5 else f"{'--':>8}"
            etv = f"{r['eta']:>5.2f}"   if r['eta']  > 0.0 else f"{'--':>5}"
            spi = _f(r['SPI'],     '.3f', 6)
            cpi = _f(r['CPI'],     '.3f', 6)
            eac = _f(r['EAC'],     '.0f', 9)
            df  = _f(r['Delta_f'], '.2f', 6)
            c1  = ("YES" if r['cond1'] else "no ") if r['SPI'] is not None else "  -"
            c2  = ("YES" if r['cond2'] else "no ") if r['CPI'] is not None else "  -"
            uv  = "".join(str(r['u_lp'].get(j, 0)) for j in range(M))
            cert = ",".join(f"m{j+1}" for j in r['ms_new_lp']) or "--"
            print(f"  {t:>3}  {xv}  {etv}  "
                  f"{r['P_plan']:>7.4f}  {r['P_sim']:>7.4f}  "
                  f"{_f(r['P_lp'],'.4f',7)}  "
                  f"{r['ACWP_sim']:>8.0f}  {r['EV_sim']:>8.0f}  "
                  f"{spi}  {cpi}  {eac}  "
                  f"{df}  {c1}  {c2}  {r['tau_rem']:>3}  "
                  f"{uv:>{M}}  {cert:>8}  {r['status']:>11}")
        print(f"  -> final: {pr['status']}  P={pr['P_final']:.4f}  "
              f"ACWP={pr['ACWP_final']:,.0f}  certified={sorted(pr['certified'])}")

    print(f"\n{'─'*72}")
    print("  PORTFOLIO (per period)")
    print(f"{'─'*72}")
    print(f"  {'t':>3}  {'B_lp':>9}  {'B_sim':>9}  "
          f"{'outflow':>9}  {'inflow':>9}  {'net':>9}  {'nact':>4}  events")
    print("  " + "─" * 72)
    for t in T:
        r  = rec['portfolio'][t]
        ev = "; ".join(f"P{e['proj']}:{e['type']}={e['amount']:,.0f}"
                       for e in r['events']) or "--"
        print(f"  {t:>3}  {r['B_lp']:>9,.0f}  {r['B_sim']:>9,.0f}  "
              f"{r['outflow']:>9,.0f}  {r['inflow']:>9,.0f}  "
              f"{r['net']:>9,.0f}  {r['n_active']:>4}  {ev}")


def print_cashflow(rec, pf):
    T = list(range(1, pf['H'] + 1))
    print(f"\n{'─'*72}")
    print("  CASH FLOW EVENTS")
    print(f"{'─'*72}")
    evs = sorted(rec['events'], key=lambda e: (e['t'], e['proj']))
    print(f"  {'t':>3}  {'proj':>4}  {'type':>24}  {'amount':>10}  {'running':>10}")
    print("  " + "─" * 57)
    for ev in evs:
        print(f"  {ev['t']:>3}  P{ev['proj']:<3}  {ev['type']:>24}  "
              f"{ev['amount']:>10,.0f}  {ev.get('running_total',0):>10,.0f}")
    tin  = sum(ev['amount'] for ev in evs)
    tout = sum(rec['x_val'][i, t] for i in range(pf['n']) for t in T)
    print(f"\n  Total inflows     : {tin:>12,.0f}")
    print(f"  Total outflows    : {tout:>12,.0f}")
    print(f"  Net (undiscounted): {tin-tout:>12,.0f}")


def print_validation(val):
    labels = {
        'C1_B_nonneg':   'B_t >= 0',
        'C2_budget':     'sum x_{i,t} <= B_t',
        'C3_progress':   'P recursion P=P_prev+eta*x/BAC',
        'C4_balance':    'Balance recursion B_{t+1}=B_t-out+V',
        'C5_cert':       'u=1 iff P>=theta',
        'C6_mono':       'u monotone',
        'C7_order':      'u ordered',
        'C8_inactive':   'x=0 outside window',
        'C9_payment_id': 'Payment identity (completed)',
        'C10_nonneg':    'x >= 0',
    }
    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    for k, desc in labels.items():
        errs = val['errors'].get(k, [])
        flag = "PASS v" if not errs else "FAIL x"
        print(f"  {flag}  {desc}")
        for e in (errs if isinstance(errs, list) else []):
            print(f"         -> {e}")
    terms = val['errors'].get('_terminations', {})
    if terms:
        print(f"\n  [INFO] Terminations: {terms}")
    else:
        print("\n  [INFO] No terminations triggered.")
    print(f"\n  Overall: {'ALL PASS' if val['all_pass'] else 'SOME FAILED'}")


# =============================================================================
# 9.  CSV EXPORT
# =============================================================================

def export_csv(pf, rec, prefix="milp_out"):
    H, n = pf['H'], pf['n']
    T    = list(range(1, H + 1))
    rows = []
    for i in range(n):
        for t in T:
            r   = rec['proj'][i]['records'][t]
            row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            row['u_lp']       = "".join(str(r['u_lp'].get(j, 0))
                                        for j in range(pf['projects'][i]['M_i']))
            row['ms_new_lp']  = str(r['ms_new_lp'])
            row['ms_new_sim'] = str(r['ms_new_sim'])
            row['theta']      = str(r['theta'])
            for k in row:
                if isinstance(row[k], float) and (math.isinf(row[k]) or math.isnan(row[k])):
                    row[k] = 'inf'
            rows.append(row)
    fn_proj = f"{prefix}_project_states.csv"
    with open(fn_proj, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    port_rows = [
        dict(t=t, B_lp=rec['portfolio'][t]['B_lp'],
             B_sim=rec['portfolio'][t]['B_sim'],
             outflow=rec['portfolio'][t]['outflow'],
             inflow=rec['portfolio'][t]['inflow'],
             net=rec['portfolio'][t]['net'],
             n_active=rec['portfolio'][t]['n_active'],
             active=str(rec['portfolio'][t]['active_proj']))
        for t in T
    ]
    fn_port = f"{prefix}_portfolio_states.csv"
    with open(fn_port, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(port_rows[0].keys()))
        w.writeheader(); w.writerows(port_rows)

    print(f"\n  [CSV] {fn_proj}")
    print(f"  [CSV] {fn_port}")


# =============================================================================
# 10.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Level-1 MILP for PPM Budget Allocation")
    parser.add_argument('--seed',       type=int,   default=None)
    parser.add_argument('--n_projects', type=int,   default=None)
    parser.add_argument('--horizon',    type=int,   default=None)
    parser.add_argument('--b1_ratio',   type=float, default=None,
                        help="Fixed B1/totalBAC ratio")
    parser.add_argument('--scenario',   type=int,   default=None,
                        choices=[1, 2, 3],
                        help="Run a known-solution test scenario instead of random")
    parser.add_argument('--no_csv',     action='store_true')
    parser.add_argument('--no_plot',    action='store_true')
    args = parser.parse_args()

    # ── Choose portfolio ───────────────────────────────────────────────────
    if args.scenario is not None:
        scenarios = build_test_scenarios()
        sc        = scenarios[args.scenario - 1]
        portfolio = sc['portfolio']
        print(f"\nTEST SCENARIO {args.scenario}: {sc['name']}")
        print(f"  {sc['note']}")
        print(f"  Expected Z*_L1 = {sc['Z_star']:.4f}")
    else:
        cfg = dict(CONFIG)
        if args.seed       is not None: cfg['seed']       = args.seed
        if args.n_projects is not None: cfg['n_projects'] = args.n_projects
        if args.horizon    is not None: cfg['horizon']    = args.horizon
        if args.b1_ratio   is not None: cfg['b1_ratio']   = (args.b1_ratio, args.b1_ratio)
        portfolio = generate_portfolio(cfg)
        sc        = None

    print_portfolio(portfolio)

    # ── Solve ──────────────────────────────────────────────────────────────
    sol = build_and_solve(portfolio)
    rec = build_records(portfolio, sol)
    val = validate(portfolio, sol, rec)

    # ── Output ────────────────────────────────────────────────────────────
    print_solution(portfolio, sol, rec)
    print_state_table(portfolio, rec)
    print_cashflow(rec, portfolio)
    print_validation(val)

    # ── Known-solution check ───────────────────────────────────────────────
    if sc is not None:
        ks = check_known_solution(sc, sol, rec)
        print("\n" + "=" * 72)
        print("KNOWN-SOLUTION CHECK")
        print("=" * 72)
        flag_obj   = "PASS v" if ks['obj_pass']   else "FAIL x"
        flag_alloc = "PASS v" if ks['alloc_pass'] else "FAIL x"
        print(f"  {flag_obj}    Objective: MILP={ks['obj_milp']}  "
              f"Expected={ks['obj_expect']}  Error={ks['obj_err']}")
        print(f"  {flag_alloc}  Allocation pattern (spend BAC at s_i)")
        for e in ks['alloc_errs']:
            print(f"           -> {e}")
        print(f"\n  Overall: {'ALL PASS' if ks['all_pass'] else 'SOME FAILED'}")

    if not args.no_csv:
        export_csv(portfolio, rec)

    if not args.no_plot:
        try:
            fig1, fig2 = plot_portfolio(portfolio, sol, rec)
            seed_tag = portfolio['cfg']['seed']
            f1 = f"milp_plot_projects_seed{seed_tag}.png"
            f2 = f"milp_plot_portfolio_seed{seed_tag}.png"
            fig1.savefig(f1, dpi=150, bbox_inches='tight')
            fig2.savefig(f2, dpi=150, bbox_inches='tight')
            print(f"\n  [PLOT] {f1}")
            print(f"  [PLOT] {f2}")
            plt.close('all')
        except Exception as e:
            print(f"\n  [PLOT] Failed: {e}")

    print("\n" + "=" * 72)
    print(f"  {sol['status']}  ->  Z*_L1 = {sol['obj']:,.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()