"""
milp_ppm_test.py
================
Level-1 Deterministic Full-Foresight MILP — PPM Budget Allocation
Section 3 of the paper.

Usage:
    python milp_ppm_test.py [--seed N] [--n_projects N] [--horizon N]
                            [--b1_ratio F] [--no_csv] [--no_plot]

CONFIG block below controls all generation parameters.
Command-line args override relevant CONFIG entries.
"""

import argparse
import random
import math
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.special import betainc
import pulp
from pulp import (LpProblem, LpMaximize, LpVariable, LpBinary,
                  lpSum, value, PULP_CBC_CMD)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    seed          = 42,
    n_projects    = 5,
    n_min         = 5,
    n_max         = 10,
    horizon       = 24,
    gamma         = 0.97,
    b1_ratio      = (0.35, 0.55),

    s_max_frac    = 0.20,
    dur_min_frac  = 0.33,
    dur_max_frac  = 0.80,

    a_range       = (1.5, 3.5),
    b_range       = (2.0, 5.0),

    BAC_range     = (50_000, 300_000),
    pi_range      = (0.05, 0.20),
    rho_range     = (0.05, 0.10),
    delta_rec     = 0.25,
    psi           = 0.10,

    n_ms_range    = (3, 5),
    ms_min_gap    = 0.15,
    ms_lo         = 0.20,
    ms_hi         = 0.85,

    Omega_range   = (2, 5),
    mu_range      = (0.10, 0.30),
    tau_tol_range = (1, 3),

    eta_base_lo   = 0.65,
    eta_base_hi   = 1.00,
    eta_noise     = 0.03,
    eta_min       = 0.50,
    eta_max       = 1.10,

    solver_time_limit = 300,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def scurve(tau, a, b):
    tc = max(0.0, min(1.0, tau))
    if tc <= 0.0: return 0.0
    if tc >= 1.0: return 1.0
    return float(betainc(a, b, tc))


def derive_alpha(phi, drec, psi):
    cum = rec_frac = 0.0
    for phi_j in phi:
        cum += phi_j
        if cum >= psi:
            rec_frac += phi_j
    return drec * rec_frac


def compute_R_net(proj):
    CP, rho, drec, psi = proj['CP_i'], proj['rho_i'], proj['delta_rec_i'], proj['psi_i']
    cum, R_net = 0.0, []
    for phi_j in proj['phi']:
        cum   += phi_j
        gross  = phi_j * CP
        R_net.append(gross - rho * gross - (drec * gross if cum >= psi else 0.0))
    return R_net, rho * CP


def planned_progress(proj, H):
    s, f, D = proj['s_i'], proj['f_i'], proj['D_plan']
    a, b    = proj['a_i'], proj['b_i']
    out = {}
    for t in range(1, H + 1):
        if   t < s: out[t] = 0.0
        elif t > f: out[t] = 1.0
        else:       out[t] = scurve((t - s + 1) / D, a, b)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PORTFOLIO GENERATION
# ═════════════════════════════════════════════════════════════════════════════

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
    theta  = [min(v, 0.95) for v in spaced] + [1.0]
    M_i    = len(theta)
    phi    = [1.0 / M_i] * M_i
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


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PRE-COMPUTE ZERO-ALLOCATION TERMINATION
#     With the inf-guard fix, zero allocation triggers both conditions
#     from t=s_i onward. Pre-compute t^term_i under x=0 so the MILP
#     sees the financial consequence of not spending.
# ═════════════════════════════════════════════════════════════════════════════

def precompute_zero_alloc_termination(portfolio):
    """
    Simulate each project with x=0 using corrected termination conditions.

    With zero spend:
      - SPI = P_actual/P_plan = 0/P_plan = 0 → Delta_f = inf → cond1 = True
      - ACWP = 0 with t > s_i → project started but spent nothing
        → EAC = inf (at zero spend rate it will never complete) → cond2 = True
      - Both conditions fire from t = s_i+1 onward
      - Counter decrements every period from s_i+1
      - Termination at t = s_i + tau_tol_i

    Returns dict i -> t^term_i (period when counter hits 0).
    """
    projects = portfolio['projects']
    H        = portfolio['H']
    T        = list(range(1, H + 1))
    result   = {}

    for i, p in enumerate(projects):
        s        = p['s_i']
        tau_tol  = p['tau_tol_i']
        # Under x=0: both conds fire from t=s+1 (first active period with plan>0)
        # Counter starts at tau_tol, decrements each period both conds hold
        # Hits 0 after tau_tol periods → termination at s + tau_tol
        t_term = s + tau_tol
        result[i] = t_term if t_term <= H else None

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MILP
# ═════════════════════════════════════════════════════════════════════════════

def build_and_solve(portfolio):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    B1       = portfolio['B1']
    gamma    = portfolio['gamma']
    T        = list(range(1, H + 1))

    plan_prog   = {i: planned_progress(p, H) for i, p in enumerate(projects)}
    zero_term   = precompute_zero_alloc_termination(portfolio)

    act = {(i, t): int(projects[i]['s_i'] <= t <= projects[i]['f_i'])
           for i in range(n) for t in T}
    A   = {i: projects[i]['alpha_i'] * projects[i]['CP_i'] for i in range(n)}

    R_net, R_ret = {}, {}
    for i, p in enumerate(projects):
        rn, rr = compute_R_net(p)
        for j, v in enumerate(rn): R_net[i, j] = v
        R_ret[i] = rr

    # Payment identity check
    for i, p in enumerate(projects):
        total = A[i] + sum(R_net[i, j] for j in range(p['M_i'])) + R_ret[i]
        assert abs(total - p['CP_i']) < 1.0, \
            f"Payment identity failed P{i}: {total:.2f} vs {p['CP_i']:.2f}"

    term_settlements = {}  # unused in endogenous formulation; kept for compat

    BIG_M  = 1.0
    BIG_MX = max(p['BAC_i'] for p in projects) * 2   # budget big-M
    EPS    = 1e-4

    prob = LpProblem("PPM_L1", LpMaximize)

    # x_{i,t}: budget allocated
    x = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0)
         for i in range(n) for t in T}

    # u_{i,j,t}: milestone j of project i certified by period t
    u = {(i, j, t): LpVariable(f"u_{i}_{j}_{t}", cat=LpBinary)
         for i in range(n) for j in range(projects[i]['M_i']) for t in T}

    # P_{i,t}: cumulative actual progress
    P = {(i, t): LpVariable(f"P_{i}_{t}", lowBound=0, upBound=1.0)
         for i in range(n) for t in T}

    # B_t: cash balance
    B = {t: LpVariable(f"B_{t}", lowBound=0) for t in T}

    # ── Endogenous termination variables ──────────────────────────────────
    # alive_{i,t}: 1 if project i is still active at period t
    # terminated early means alive flips 0 and stays 0
    alive = {(i, t): LpVariable(f"alive_{i}_{t}", cat=LpBinary)
             for i in range(n) for t in T}

    # q_{i,t}: 1 if both termination conditions hold at period t
    # (linearised: SPI=0 iff P_actual < P_plan, EAC > (1+mu)*BAC iff CPI < 1/(1+mu))
    # We approximate via: q=1 iff x_{i,t}=0 AND project is active AND t > s_i
    # Precise cond1: P_actual < P_plan - schedule_slack (approximated)
    # For the MILP we use a simpler sufficient condition:
    #   if x_{i,t} = 0 for tau_tol_i consecutive periods → project terminates
    #   implemented via: counter c_{i,t} counts consecutive zero-spend periods
    #
    # c_{i,t} in {0,..,tau_tol}: consecutive zero-spend periods up to t
    # c_{i,t} = 0             if x_{i,t} > 0 or t < s_i
    # c_{i,t} = c_{i,t-1} + 1 if x_{i,t} = 0 and alive_{i,t}
    # alive_{i,t} = 0 if c_{i,t-1} >= tau_tol_i

    # Binary: z_{i,t} = 1 iff x_{i,t} = 0 (and project active window)
    z = {(i, t): LpVariable(f"z_{i}_{t}", cat=LpBinary)
         for i in range(n) for t in T}

    # Counter: c_{i,t} = consecutive periods of zero spend up to t (integer)
    tau_max = max(p['tau_tol_i'] for p in projects) + 1
    c = {(i, t): LpVariable(f"c_{i}_{t}", lowBound=0, upBound=tau_max, cat='Integer')
         for i in range(n) for t in T}

    def inflow_lp(t):
        terms = []
        for i, p in enumerate(projects):
            if t == p['s_i']:
                terms.append(A[i])
            for j in range(p['M_i']):
                u_prev = u[i, j, t - 1] if t > 1 else 0
                pay    = R_net[i, j] + (R_ret[i] if j == p['M_i'] - 1 else 0)
                terms.append(pay * (u[i, j, t] - u_prev))
            # Termination settlement: R_term = P_{i,t}*CP_i*(1-rho_i) - A_i
            # enters V_t when alive flips from 1 to 0
            # alive_{i,t} - alive_{i,t-1} = -1 at termination period
            # R_term at that period: approximate as -A_i (conservative lower bound)
            # because P is unknown at termination without nonlinear terms.
            # We use: -A_i * (alive_{i,t-1} - alive_{i,t})  for t>1
            if t > 1:
                terms.append(-A[i] * (alive[i, t - 1] - alive[i, t]))
        return lpSum(terms)

    # Objective
    prob += lpSum(
        gamma ** (t - 1) * (inflow_lp(t) - lpSum(act[i, t] * x[i, t] for i in range(n)))
        for t in T
    ), "obj"

    # Initial balance
    prob += B[1] == B1, "B_init"

    # Balance recursion
    for t in T[:-1]:
        prob += (B[t + 1] == B[t]
                 - lpSum(act[i, t] * x[i, t] for i in range(n))
                 + inflow_lp(t),
                 f"bal_{t}")

    # Budget feasibility
    for t in T:
        prob += (lpSum(act[i, t] * x[i, t] for i in range(n)) <= B[t],
                 f"bud_{t}")

    # Progress recursion
    for i, p in enumerate(projects):
        for t in T:
            P_prev = P[i, t - 1] if t > 1 else 0.0
            # Allocation only allowed when alive
            prob += (P[i, t] == P_prev + p['eta'][t] * x[i, t] / p['BAC_i'],
                     f"prg_{i}_{t}")
            # No allocation outside activity window
            if act[i, t] == 0:
                prob += (x[i, t] == 0, f"off_{i}_{t}")
            # No allocation when terminated
            prob += (x[i, t] <= BIG_MX * alive[i, t], f"xalive_{i}_{t}")

    # Alive initialisation: all active at start
    for i, p in enumerate(projects):
        for t in T:
            if t < p['s_i']:
                prob += (alive[i, t] == 1, f"alive_pre_{i}_{t}")
        prob += (alive[i, p['s_i']] == 1, f"alive_start_{i}")

    # Alive monotone: once terminated stays terminated
    for i in range(n):
        for t in range(2, H + 1):
            prob += (alive[i, t] <= alive[i, t - 1], f"alive_mono_{i}_{t}")

    # z_{i,t} = 1 iff x_{i,t} = 0 (within activity window)
    for i, p in enumerate(projects):
        for t in T:
            if act[i, t] == 1:
                # z=1 → x=0
                prob += (x[i, t] <= BIG_MX * (1 - z[i, t]), f"z_xub_{i}_{t}")
                # z=0 → x can be positive (no lower bound needed beyond x>=0)
                # force z=1 when x=0: x >= eps*(1-z) — use small epsilon
                prob += (x[i, t] >= EPS * (1 - z[i, t]), f"z_xlb_{i}_{t}")
            else:
                prob += (z[i, t] == 1, f"z_off_{i}_{t}")

    # Consecutive zero-spend counter
    for i, p in enumerate(projects):
        tau_tol = p['tau_tol_i']
        for t in T:
            if t < p['s_i']:
                prob += (c[i, t] == 0, f"c_pre_{i}_{t}")
            elif t == p['s_i']:
                prob += (c[i, t] == 0, f"c_start_{i}_{t}")
            else:
                # c_{i,t} = (c_{i,t-1} + 1) * z_{i,t}  [nonlinear — linearise]
                # Let c_{i,t} = c_prev + z - (c_prev + z)*(1-z)... use big-M:
                # if z=1: c = c_prev + 1
                # if z=0: c = 0
                # Linearise:
                #   c_{i,t} <= c_{i,t-1} + 1
                #   c_{i,t} <= tau_max * z_{i,t}
                #   c_{i,t} >= c_{i,t-1} + 1 - tau_max*(1-z_{i,t})
                #   c_{i,t} >= 0
                prob += (c[i, t] <= c[i, t - 1] + 1,                              f"c_ub1_{i}_{t}")
                prob += (c[i, t] <= tau_max * z[i, t],                             f"c_ub2_{i}_{t}")
                prob += (c[i, t] >= c[i, t - 1] + 1 - tau_max * (1 - z[i, t]),   f"c_lb_{i}_{t}")

            # Termination trigger: if c_{i,t} >= tau_tol → alive_{i,t+1} = 0
            # alive_{i,t+1} <= 1 - (c_{i,t} - tau_tol + 1) / tau_max  [approximate]
            # Exact big-M: alive_{i,t+1} <= 1 - (c_{i,t} >= tau_tol)
            # Binary trigger: introduce b_term_{i,t} = 1 iff c_{i,t} >= tau_tol
            # For simplicity: alive_{i,t+1} * tau_tol <= tau_tol - c_{i,t} + tau_max*(1-alive_flag)
            # Cleaner: if c_{i,t} >= tau_tol then alive_{i,t+1} = 0
            if t < H:
                # alive_{i,t+1} <= (tau_tol - c_{i,t}) / tau_tol  → not LP-clean
                # Use: c_{i,t} + alive_{i,t+1} * tau_tol <= tau_tol + tau_max*(1-alive_{i,t})
                # Simplified sufficient: c_{i,t} >= tau_tol → alive_{i,t+1} = 0
                # Big-M formulation:
                # alive_{i,t+1} * tau_tol <= tau_tol * alive_{i,t} - c_{i,t} + tau_max*(1-alive_{i,t+1})
                # Rearranged:
                # c_{i,t} <= tau_tol - 1 + tau_max * (1 - alive[i, t+1])
                prob += (c[i, t] <= (tau_tol - 1) + tau_max * alive[i, t + 1],
                         f"term_trigger_{i}_{t}")

    # Certification big-M
    for i, p in enumerate(projects):
        for j in range(p['M_i']):
            th = p['theta'][j]
            for t in T:
                prob += (P[i, t] >= th - BIG_M * (1 - u[i, j, t]),    f"clb_{i}_{j}_{t}")
                prob += (P[i, t] <= (th - EPS) + BIG_M * u[i, j, t],  f"cub_{i}_{j}_{t}")
                if t > 1:
                    prob += (u[i, j, t] >= u[i, j, t - 1],             f"mon_{i}_{j}_{t}")
                if j < p['M_i'] - 1:
                    prob += (u[i, j + 1, t] <= u[i, j, t],             f"ord_{i}_{j}_{t}")
                if t < p['s_i']:
                    prob += (u[i, j, t] == 0,                           f"pre_{i}_{j}_{t}")
                # No milestone certification after termination
                prob += (u[i, j, t] <= alive[i, t],                    f"ucert_alive_{i}_{j}_{t}")

    print("\n" + "=" * 72)
    print("SOLVING MILP  (CBC, limit={}s)...".format(portfolio['cfg']['solver_time_limit']))
    print("=" * 72)
    solver = PULP_CBC_CMD(msg=0, timeLimit=portfolio['cfg']['solver_time_limit'])
    prob.solve(solver)

    return dict(
        status=pulp.LpStatus[prob.status],
        obj=value(prob.objective),
        prob=prob,
        x=x, u=u, P=P, B=B,
        act=act, A=A, R_net=R_net, R_ret=R_ret,
        plan_prog=plan_prog,
        zero_term=zero_term,
        term_settlements=term_settlements,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5.  STATE RECORD SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

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

        if s <= H:
            all_events.append(dict(t=s, proj=i, type='advance', amount=A[i]))

        for t in T:
            x_t = x_val[i, t]

            if   t < s: window = 'pre_start'
            elif t > f: window = 'post_plan'
            else:       window = 'active'

            if window == 'active' and status not in ('terminated', 'completed'):
                P_sim    = min(1.0, P_sim + eta[t] * x_t / BAC)
                ACWP_sim += x_t
                status   = 'active'

            EV_sim = P_sim * BAC

            # Milestone certification
            ms_new_sim = []
            if window == 'active' and status == 'active':
                for j in range(M):
                    if j not in certified and P_sim >= theta[j] - 1e-6:
                        certified.add(j)
                        cum_phi_cert += phi[j]
                        gross    = phi[j] * CP
                        ret_ded  = rho * gross
                        rec_ded  = drec * gross if cum_phi_cert >= psi else 0.0
                        adv_recov += rec_ded
                        R_net_j  = gross - ret_ded - rec_ded
                        ms_new_sim.append(j)
                        all_events.append(dict(t=t, proj=i,
                                               type=f'ms{j+1}_net', amount=R_net_j))
                        if j == M - 1:
                            all_events.append(dict(t=t, proj=i,
                                                   type='retention', amount=rho * CP))

            # EVM signals with CORRECTED inf handling
            if window == 'active' and status == 'active':
                P_plan_t = plan_prog[i][t]
                SPI = (P_sim / P_plan_t) if P_plan_t > 1e-9 else (1.0 if P_sim < 1e-9 else float('inf'))
                # CPI: if ACWP=0 and t>s, project has started but spent nothing → EAC=inf
                if ACWP_sim > 1e-9:
                    CPI = EV_sim / ACWP_sim
                    EAC = ACWP_sim + (BAC - EV_sim) / CPI if CPI > 1e-9 else float('inf')
                elif t > s:
                    CPI = float('inf')   # undefined but treated as worst case
                    EAC = float('inf')
                else:
                    CPI = 1.0
                    EAC = BAC

                elapsed = t - s
                rem_per = D_plan - elapsed
                if SPI > 1e-9 and not math.isinf(SPI):
                    f_hat   = t + rem_per / SPI
                    Delta_f = f_hat - f
                else:
                    f_hat   = float('inf')
                    Delta_f = float('inf')

                # CORRECTED: inf counts as condition active
                cond1 = math.isinf(Delta_f) or (Delta_f > Omega)
                cond2 = math.isinf(EAC)     or (EAC > (1 + mu) * BAC)
            else:
                SPI = CPI = EAC = Delta_f = f_hat = None
                cond1 = cond2 = False

            # Cure counter
            if window == 'active' and status == 'active':
                tau_rem = tau_rem - 1 if (cond1 and cond2) else tau_tol

            u_now  = {j: u_val.get((i, j, t), 0) for j in range(M)}
            u_prev = {j: u_val.get((i, j, t - 1), 0) for j in range(M)} if t > 1 else {j: 0 for j in range(M)}
            ms_new_lp = [j for j in range(M) if u_now[j] - u_prev.get(j, 0) > 0]

            records[t] = dict(
                proj=i, t=t,
                window=window, status=status,
                x_lp=x_t, P_lp=P_lp.get((i, t)), u_lp=u_now, u_prev_lp=u_prev,
                ms_new_lp=ms_new_lp,
                P_sim=P_sim, ACWP_sim=ACWP_sim, EV_sim=EV_sim,
                P_plan=plan_prog[i][t], eta=eta.get(t, 0.0),
                SPI=SPI, CPI=CPI, EAC=EAC, f_hat=f_hat, Delta_f=Delta_f,
                cond1=cond1, cond2=cond2, tau_rem=tau_rem,
                ms_new_sim=ms_new_sim,
                theta=list(theta), Omega=Omega, mu=mu, BAC=BAC, CP=CP,
            )

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

    # Portfolio records
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


# ═════════════════════════════════════════════════════════════════════════════
# 6.  VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate(portfolio, sol, rec, tol=1e-3):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    T        = list(range(1, H + 1))
    x_val, u_val, P_lp, B_lp = rec['x_val'], rec['u_val'], rec['P_lp'], rec['B_lp']
    R_net, R_ret, A, act = sol['R_net'], sol['R_ret'], sol['A'], sol['act']
    errors = {}

    errors['C1_B_nonneg'] = [(t, round(B_lp[t], 2)) for t in T if B_lp.get(t, 0) < -tol]

    errors['C2_budget'] = [
        dict(t=t, spend=round(sum(x_val[i, t] for i in range(n)), 2), bal=round(B_lp.get(t, 0), 2))
        for t in T if sum(x_val[i, t] for i in range(n)) > B_lp.get(t, 0) + tol
    ]

    prog_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc  = min(1.0, Pc + p['eta'][t] * x_val[i, t] / p['BAC_i'])
            lp  = P_lp.get((i, t), 0.0)
            if abs(lp - Pc) > tol * 5:
                prog_errs.append(dict(proj=i, t=t, lp=round(lp, 6), sim=round(Pc, 6)))
    errors['C3_progress'] = prog_errs[:10]

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
        # add fixed termination settlements
        for i, ts in sol['term_settlements'].items():
            if ts['t_term'] + 1 == t:
                inf += ts['R_term']
        Bc_next = Bc - out + inf
        lp_next = B_lp.get(t + 1, 0.0)
        if abs(lp_next - Bc_next) > tol * 10:
            bal_errs.append(dict(t=t, lp=round(lp_next, 2), comp=round(Bc_next, 2)))
        Bc = Bc_next
    errors['C4_balance'] = bal_errs[:5]

    cert_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc = min(1.0, Pc + p['eta'][t] * x_val[i, t] / p['BAC_i'])
            for j in range(p['M_i']):
                u_ij    = u_val.get((i, j, t), 0)
                reached = Pc >= p['theta'][j] - tol
                if reached and u_ij == 0:
                    cert_errs.append(dict(issue='missed',    proj=i, ms=j, t=t, P=round(Pc, 5)))
                if not reached and u_ij == 1:
                    cert_errs.append(dict(issue='premature', proj=i, ms=j, t=t, P=round(Pc, 5)))
    errors['C5_cert'] = cert_errs[:10]

    errors['C6_mono'] = [
        (i, j, t) for i, p in enumerate(projects)
        for j in range(p['M_i']) for t in range(2, H + 1)
        if u_val.get((i, j, t), 0) < u_val.get((i, j, t - 1), 0)
    ][:10]

    errors['C7_order'] = [
        (i, j, t) for i, p in enumerate(projects)
        for j in range(p['M_i'] - 1) for t in T
        if u_val.get((i, j + 1, t), 0) > u_val.get((i, j, t), 0)
    ][:10]

    errors['C8_inactive'] = [
        (i, t, round(x_val[i, t], 2)) for i in range(n) for t in T
        if act[i, t] == 0 and x_val[i, t] > tol
    ][:10]

    pay_errs = []
    for i, p in enumerate(projects):
        if rec['proj'][i]['status'] == 'completed':
            total = A[i] + sum(R_net[i, j] for j in range(p['M_i'])) + R_ret[i]
            if abs(total - p['CP_i']) > tol * p['CP_i']:
                pay_errs.append(dict(proj=i, total=round(total, 2), CP=round(p['CP_i'], 2)))
    errors['C9_payment_id'] = pay_errs

    errors['C10_nonneg'] = [(i, t, round(x_val[i, t], 4)) for (i, t) in x_val if x_val[i, t] < -tol]

    errors['_terminations'] = {
        i: next((t for t in range(1, H + 1)
                 if rec['proj'][i]['records'][t]['status'] == 'terminated'), None)
        for i in range(n) if rec['proj'][i]['status'] == 'terminated'
    }

    all_pass = all(len(v) == 0 for k, v in errors.items() if not k.startswith('_'))
    return dict(errors=errors, all_pass=all_pass)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

COLORS = dict(
    bcws      = '#1f77b4',   # blue
    bcwp      = '#2ca02c',   # green
    acwp      = '#ff7f0e',   # orange
    eac       = '#d62728',   # red
    alloc     = '#9467bd',   # purple
    balance   = '#17becf',   # teal
    inflow    = '#2ca02c',
    outflow   = '#d62728',
    net       = '#1f77b4',
    spi       = '#2ca02c',
    cpi       = '#ff7f0e',
    ms        = '#e377c2',   # pink
    overrun   = '#d62728',
)


def _safe(v, fallback=None):
    if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
        return fallback
    return v


def plot_project(proj_idx, proj, records, H, ax_evm, ax_alloc, ax_indices):
    """
    Three-panel project chart:
      ax_evm    — BCWS / BCWP / ACWP / EAC  (mirrors paper figure)
      ax_alloc  — budget allocation bar chart
      ax_indices — SPI / CPI over time
    """
    T   = list(range(1, H + 1))
    rec = records['proj'][proj_idx]['records']
    s, f = proj['s_i'], proj['f_i']
    BAC  = proj['BAC_i']
    mu   = proj['mu_i']
    Omega = proj['Omega_i']
    status = records['proj'][proj_idx]['status']

    ts       = list(range(0, H + 1))
    bcws     = [0.0] + [rec[t]['P_plan'] for t in T]
    bcwp_sim = [0.0] + [rec[t]['P_sim']  for t in T]
    acwp_sim = [0.0] + [rec[t]['ACWP_sim'] / BAC for t in T]
    eac_vals = [_safe(rec[t]['EAC'], None) for t in T]
    eac_norm = [v / BAC if v is not None else None for v in eac_vals]

    # ── EVM panel ─────────────────────────────────────────────────────────
    ax = ax_evm
    ax.plot(ts, bcws,     color=COLORS['bcws'], lw=2,   label='BCWS (plan)')
    ax.plot(ts, bcwp_sim, color=COLORS['bcwp'], lw=2,   label='BCWP (actual)')
    ax.plot(ts, acwp_sim, color=COLORS['acwp'], lw=2,   label='ACWP / BAC')

    # EAC as scatter (only finite values)
    eac_t  = [t for t in T if eac_norm[t - 1] is not None and eac_norm[t - 1] <= 2.0]
    eac_v  = [eac_norm[t - 1] for t in eac_t]
    if eac_t:
        ax.plot(eac_t, eac_v, color=COLORS['eac'], lw=1.5, ls='--', label='EAC / BAC')

    # Cost overrun cap: (1+mu)*BAC normalised = (1+mu)
    cap_cost = 1.0 + mu
    ax.axhspan(1.0, min(cap_cost + 0.05, 1.35),
               color=COLORS['overrun'], alpha=0.10, label=f'Cost overrun cap ({cap_cost:.2f}×BAC)')
    ax.axhline(cap_cost, color=COLORS['overrun'], lw=1.0, ls=':', alpha=0.7)

    # Schedule overrun cap: planned finish + Omega
    sched_cap_t = f + Omega
    if sched_cap_t <= H + 2:
        ax.axvspan(f, min(sched_cap_t, H + 1),
                   color=COLORS['overrun'], alpha=0.10, label=f'Sched. overrun cap (Ω={Omega})')

    # Planned finish
    ax.axvline(f, color=COLORS['bcws'], lw=1.2, ls='--', alpha=0.7)
    ax.text(f + 0.1, 1.25, f'$f_i={f}$', color=COLORS['bcws'], fontsize=8)

    # Milestone thresholds
    for j, th in enumerate(proj['theta']):
        ax.axhline(th, color=COLORS['ms'], lw=0.8, ls=':', alpha=0.5)
        ax.text(H + 0.3, th, f'θ{j+1}', color=COLORS['ms'], fontsize=7, va='center')

    # Termination marker
    if status == 'terminated':
        term_t = next((t for t in T if rec[t]['status'] == 'terminated'), None)
        if term_t:
            ax.axvline(term_t, color='red', lw=2.0, ls='-', alpha=0.8)
            ax.text(term_t + 0.1, 0.05, 'TERM', color='red', fontsize=8, fontweight='bold')

    ax.set_xlim(0, H + 1)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel('Fraction of BAC', fontsize=9)
    ax.set_title(
        f'P{proj_idx}  EVM  |  s={s} f={f} D={proj["D_plan"]}  '
        f'BAC={BAC/1000:.0f}k  CP={proj["CP_i"]/1000:.0f}k  [{status}]',
        fontsize=9, fontweight='bold'
    )
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Allocation panel ──────────────────────────────────────────────────
    ax = ax_alloc
    alloc_vals = [rec[t]['x_lp'] / 1000 for t in T]
    colors_bar = ['#9467bd' if rec[t]['window'] == 'active' else '#cccccc' for t in T]
    ax.bar(T, alloc_vals, color=colors_bar, alpha=0.8, width=0.8)

    # Mark milestone certification periods
    for j in range(proj['M_i']):
        cert_t = next((t for t in T
                       if records['u_val'].get((proj_idx, j, t), 0) == 1
                       and records['u_val'].get((proj_idx, j, t - 1), 0) == 0), None)
        if cert_t:
            ax.axvline(cert_t, color=COLORS['ms'], lw=1.5, ls='--', alpha=0.8)
            ax.text(cert_t, max(alloc_vals) * 0.85 if max(alloc_vals) > 0 else 0.1,
                    f'm{j+1}', color=COLORS['ms'], fontsize=7, ha='center')

    ax.set_xlim(0, H + 1)
    ax.set_ylabel('Alloc (k)', fontsize=9)
    ax.set_title(f'P{proj_idx}  Budget allocation  x_{{i,t}}', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── SPI / CPI panel ───────────────────────────────────────────────────
    ax = ax_indices
    spi_vals = [_safe(rec[t]['SPI'], None) for t in T]
    cpi_vals = [_safe(rec[t]['CPI'], None) for t in T]

    # Clip to [0, 2] for readability
    spi_plot = [min(v, 2.0) if v is not None else None for v in spi_vals]
    cpi_plot = [min(v, 2.0) if v is not None else None for v in cpi_vals]

    spi_t = [t for t in T if spi_plot[t - 1] is not None]
    cpi_t = [t for t in T if cpi_plot[t - 1] is not None]

    if spi_t:
        ax.plot(spi_t, [spi_plot[t - 1] for t in spi_t],
                color=COLORS['spi'], lw=2, label='SPI', marker='o', ms=3)
    if cpi_t:
        ax.plot(cpi_t, [cpi_plot[t - 1] for t in cpi_t],
                color=COLORS['cpi'], lw=2, label='CPI', marker='s', ms=3)

    ax.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.7)
    ax.axhline(0.0, color='black', lw=0.5)

    # Shade termination zone (both conds active)
    for t in T:
        if rec[t]['cond1'] and rec[t]['cond2']:
            ax.axvspan(t - 0.5, t + 0.5, color='red', alpha=0.12)

    ax.set_xlim(0, H + 1)
    ax.set_ylim(0, 2.1)
    ax.set_xlabel('Period t', fontsize=9)
    ax.set_ylabel('Index', fontsize=9)
    ax.set_title(f'P{proj_idx}  SPI / CPI  (red shading = both term. conds active)', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)


def plot_portfolio(portfolio, sol, rec):
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    T        = list(range(1, H + 1))
    port     = rec['portfolio']
    x_val    = rec['x_val']
    B_lp     = rec['B_lp']

    # ── Figure 1: per-project EVM + allocation + SPI/CPI ─────────────────
    fig1, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    fig1.suptitle(
        f'Per-Project State  |  seed={portfolio["cfg"]["seed"]}  '
        f'n={n}  H={H}  Z*_L1={sol["obj"]:,.0f}',
        fontsize=11, fontweight='bold'
    )
    if n == 1:
        axes = [axes]

    for i, p in enumerate(projects):
        plot_project(i, p, rec, H, axes[i][0], axes[i][1], axes[i][2])

    fig1.tight_layout()

    # ── Figure 2: portfolio aggregate ────────────────────────────────────
    fig2 = plt.figure(figsize=(16, 14))
    fig2.suptitle(
        f'Portfolio Aggregate  |  n={n}  H={H}  B1={portfolio["B1"]:,.0f}  '
        f'totalBAC={portfolio["total_BAC"]:,.0f}',
        fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

    # (a) Cash balance: LP vs sim
    ax_bal = fig2.add_subplot(gs[0, 0])
    ax_bal.plot(T, [B_lp.get(t, 0) / 1000 for t in T],
                color=COLORS['balance'], lw=2, label='B_lp (LP)')
    ax_bal.plot(T, [port[t]['B_sim'] / 1000 for t in T],
                color=COLORS['balance'], lw=1.5, ls='--', alpha=0.7, label='B_sim (forward)')
    ax_bal.set_title('Cash Balance B_t  (thousands)', fontsize=9)
    ax_bal.set_xlabel('Period t', fontsize=8)
    ax_bal.set_ylabel('k', fontsize=8)
    ax_bal.legend(fontsize=8)
    ax_bal.grid(True, alpha=0.3)
    ax_bal.axhline(0, color='black', lw=0.5)

    # (b) Inflow / outflow / net per period
    ax_cf = fig2.add_subplot(gs[0, 1])
    inflows  = [port[t]['inflow']  / 1000 for t in T]
    outflows = [port[t]['outflow'] / 1000 for t in T]
    nets     = [port[t]['net']     / 1000 for t in T]
    ax_cf.bar(T, inflows,  color=COLORS['inflow'],  alpha=0.7, label='Inflow',  width=0.4,
              align='edge')
    ax_cf.bar([t + 0.4 for t in T], outflows, color=COLORS['outflow'], alpha=0.7,
              label='Outflow', width=0.4, align='edge')
    ax_cf.plot(T, nets, color=COLORS['net'], lw=2, marker='o', ms=3, label='Net')
    ax_cf.axhline(0, color='black', lw=0.5)
    ax_cf.set_title('Period Cash Flows  (thousands)', fontsize=9)
    ax_cf.set_xlabel('Period t', fontsize=8)
    ax_cf.set_ylabel('k', fontsize=8)
    ax_cf.legend(fontsize=8)
    ax_cf.grid(True, alpha=0.3, axis='y')

    # (c) Allocation stacked bar by project
    ax_alloc = fig2.add_subplot(gs[1, 0])
    cmap  = matplotlib.colormaps['tab10']
    bottom = [0.0] * H
    for i in range(n):
        vals = [x_val[i, t] / 1000 for t in T]
        ax_alloc.bar(T, vals, bottom=bottom, color=cmap(i), alpha=0.85,
                     label=f'P{i}', width=0.8)
        bottom = [bottom[t - 1] + vals[t - 1] for t in T]
    ax_alloc.set_title('Budget Allocation by Project  (stacked, thousands)', fontsize=9)
    ax_alloc.set_xlabel('Period t', fontsize=8)
    ax_alloc.set_ylabel('k', fontsize=8)
    ax_alloc.legend(fontsize=8, loc='upper right')
    ax_alloc.grid(True, alpha=0.3, axis='y')

    # (d) Portfolio-level BCWP aggregate (sum across projects, normalised to total BAC)
    ax_prog = fig2.add_subplot(gs[1, 1])
    total_BAC = portfolio['total_BAC']
    agg_plan = [sum(rec['proj'][i]['records'][t]['P_plan'] * projects[i]['BAC_i']
                    for i in range(n)) / total_BAC for t in T]
    agg_sim  = [sum(rec['proj'][i]['records'][t]['P_sim']  * projects[i]['BAC_i']
                    for i in range(n)) / total_BAC for t in T]
    agg_acwp = [sum(rec['proj'][i]['records'][t]['ACWP_sim']
                    for i in range(n)) / total_BAC for t in T]
    ax_prog.plot(T, agg_plan, color=COLORS['bcws'], lw=2, label='BCWS agg.')
    ax_prog.plot(T, agg_sim,  color=COLORS['bcwp'], lw=2, label='BCWP agg.')
    ax_prog.plot(T, agg_acwp, color=COLORS['acwp'], lw=2, label='ACWP/totalBAC')
    ax_prog.set_title('Portfolio Aggregate Progress  (fraction of total BAC)', fontsize=9)
    ax_prog.set_xlabel('Period t', fontsize=8)
    ax_prog.set_ylabel('Fraction', fontsize=8)
    ax_prog.legend(fontsize=8)
    ax_prog.grid(True, alpha=0.3)

    # (e) n_active projects per period
    ax_nact = fig2.add_subplot(gs[2, 0])
    n_active = [port[t]['n_active'] for t in T]
    ax_nact.step(T, n_active, color='steelblue', lw=2, where='mid')
    ax_nact.fill_between(T, n_active, step='mid', alpha=0.2, color='steelblue')
    ax_nact.set_title('Number of Active Projects per Period', fontsize=9)
    ax_nact.set_xlabel('Period t', fontsize=8)
    ax_nact.set_ylabel('Count', fontsize=8)
    ax_nact.set_ylim(0, n + 1)
    ax_nact.grid(True, alpha=0.3)

    # (f) Cumulative discounted net cash flow
    ax_dcf = fig2.add_subplot(gs[2, 1])
    gamma   = portfolio['gamma']
    cum_dcf = []
    running = 0.0
    for t in T:
        disc     = gamma ** (t - 1)
        net_t    = port[t]['inflow'] - port[t]['outflow']
        running += disc * net_t
        cum_dcf.append(running)
    ax_dcf.plot(T, [v / 1000 for v in cum_dcf],
                color='#1f77b4', lw=2, marker='o', ms=3)
    ax_dcf.axhline(0, color='black', lw=0.5)
    ax_dcf.set_title(f'Cumulative Discounted Net Cash Flow  (thousands, γ={gamma})', fontsize=9)
    ax_dcf.set_xlabel('Period t', fontsize=8)
    ax_dcf.set_ylabel('k', fontsize=8)
    ax_dcf.grid(True, alpha=0.3)

    return fig1, fig2


# ═════════════════════════════════════════════════════════════════════════════
# 8.  PRINT
# ═════════════════════════════════════════════════════════════════════════════

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
          f"{'BAC':>9}  {'CP':>9}  {'π%':>5}  {'α%':>5}  "
          f"{'ρ%':>5}  {'M':>2}  {'Ω':>3}  {'μ%':>5}  {'τtol':>4}")
    print("  " + "─" * 73)
    for p in pf['projects']:
        print(f"  {p['idx']:>2}  {p['s_i']:>3}  {p['f_i']:>3}  {p['D_plan']:>3}  "
              f"{p['BAC_i']:>9,.0f}  {p['CP_i']:>9,.0f}  "
              f"{p['pi_i']*100:>5.1f}  {p['alpha_i']*100:>5.1f}  "
              f"{p['rho_i']*100:>5.1f}  {p['M_i']:>2}  "
              f"{p['Omega_i']:>3}  {p['mu_i']*100:>5.1f}  {p['tau_tol_i']:>4}")
    print()
    for p in pf['projects']:
        ths = "  ".join(f"θ{j+1}={v:.3f}" for j, v in enumerate(p['theta']))
        ets = "  ".join(f"t{t}:{p['eta'][t]:.2f}" for t in range(p['s_i'], p['f_i'] + 1))
        print(f"  P{p['idx']}: [{ths}]  α={p['alpha_i']:.4f}")
        print(f"       η: [{ets}]")


def print_zero_term(zero_term, portfolio):
    print("\n── ZERO-ALLOCATION TERMINATION (pre-computed) ──")
    for i, tt in zero_term.items():
        p = portfolio['projects'][i]
        R_term_val = -p['alpha_i'] * p['CP_i']
        print(f"  P{i}: would terminate at t={tt}  "
              f"(s={p['s_i']} tau_tol={p['tau_tol_i']}  -> "
              f"R^term = -A_i = {R_term_val:,.0f})")


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
    print(f"  Z*_L1  : {sol['obj']:,.2f}  (discounted net cash flow — upper bound)")

    print("\n── ALLOCATIONS  x_{i,t}  (thousands) ──")
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
        print(f"  Bt   {brow}   ← balance")

    print("\n── PER-PROJECT OUTCOME ──")
    print(f"  {'i':>2}  {'total_x':>10}  {'P_final':>8}  {'status':>11}  ms  cert_periods")
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
        print(f"\n{'─'*72}")
        print(f"  P{i}  s={p['s_i']} f={p['f_i']} D={p['D_plan']}  "
              f"BAC={p['BAC_i']:,.0f}  CP={p['CP_i']:,.0f}  M={p['M_i']}")
        print(f"  θ={[round(v,3) for v in p['theta']]}  "
              f"α={p['alpha_i']:.4f}  ρ={p['rho_i']:.4f}  "
              f"Ω={p['Omega_i']}  μ={p['mu_i']:.3f}  τtol={p['tau_tol_i']}")
        print(f"{'─'*72}")
        M = p['M_i']
        print(f"  {'t':>3}  {'x_lp':>8}  {'η':>5}  "
              f"{'P_plan':>7}  {'P_sim':>7}  {'P_lp':>7}  "
              f"{'ACWP':>8}  {'EV':>8}  "
              f"{'SPI':>6}  {'CPI':>6}  {'EAC':>9}  "
              f"{'Δf':>6}  {'C1':>3}  {'C2':>3}  {'τ':>3}  "
              f"{'u':>{M}}  {'cert':>8}  {'status':>11}")
        print("  " + "─" * 115)
        for t in T:
            r   = pr['records'][t]
            xv  = f"{r['x_lp']:>8.1f}" if r['x_lp'] > 0.5 else f"{'--':>8}"
            etv = f"{r['eta']:>5.2f}"  if r['eta']  > 0.0 else f"{'--':>5}"
            spi = _f(r['SPI'],    '.3f', 6)
            cpi = _f(r['CPI'],    '.3f', 6)
            eac = _f(r['EAC'],    '.0f', 9)
            df  = _f(r['Delta_f'],'.2f', 6)
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
        print(f"  → final: {pr['status']}  P={pr['P_final']:.4f}  "
              f"ACWP={pr['ACWP_final']:,.0f}  certified={sorted(pr['certified'])}")

    print(f"\n{'─'*72}")
    print("  PORTFOLIO  (per period)")
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
        'C1_B_nonneg':  'B_t >= 0',
        'C2_budget':    'sum x_{i,t} <= B_t',
        'C3_progress':  'P recursion',
        'C4_balance':   'Balance recursion',
        'C5_cert':      'u=1 iff P>=θ',
        'C6_mono':      'u monotone',
        'C7_order':     'u ordered',
        'C8_inactive':  'x=0 outside window',
        'C9_payment_id':'Payment identity (completed)',
        'C10_nonneg':   'x >= 0',
    }
    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    for k, desc in labels.items():
        errs = val['errors'].get(k, [])
        flag = "PASS ✓" if not errs else "FAIL ✗"
        print(f"  {flag}  {desc}")
        for e in (errs if isinstance(errs, list) else []):
            print(f"         → {e}")
    terms = val['errors'].get('_terminations', {})
    print(f"\n  [INFO] Terminations: {terms}" if terms else "\n  [INFO] No terminations.")
    print(f"\n  Overall: {'ALL PASS ✓' if val['all_pass'] else 'SOME FAILED ✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  CSV EXPORT
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   default=None)
    parser.add_argument('--n_projects', type=int,   default=None)
    parser.add_argument('--horizon',    type=int,   default=None)
    parser.add_argument('--b1_ratio',   type=float, default=None)
    parser.add_argument('--no_csv',     action='store_true')
    parser.add_argument('--no_plot',    action='store_true')
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.seed       is not None: cfg['seed']       = args.seed
    if args.n_projects is not None: cfg['n_projects'] = args.n_projects
    if args.horizon    is not None: cfg['horizon']    = args.horizon
    if args.b1_ratio   is not None: cfg['b1_ratio']   = (args.b1_ratio, args.b1_ratio)

    portfolio = generate_portfolio(cfg)
    print_portfolio(portfolio)

    sol = build_and_solve(portfolio)
    print_zero_term(sol['zero_term'], portfolio)

    rec = build_records(portfolio, sol)
    val = validate(portfolio, sol, rec)

    print_solution(portfolio, sol, rec)
    print_state_table(portfolio, rec)
    print_cashflow(rec, portfolio)
    print_validation(val)

    if not args.no_csv:
        export_csv(portfolio, rec)

    if not args.no_plot:
        try:
            fig1, fig2 = plot_portfolio(portfolio, sol, rec)
            seed_tag = portfolio['cfg']['seed']
            f1_path  = f"milp_plot_projects_seed{seed_tag}.png"
            f2_path  = f"milp_plot_portfolio_seed{seed_tag}.png"
            fig1.savefig(f1_path, dpi=150, bbox_inches='tight')
            fig2.savefig(f2_path, dpi=150, bbox_inches='tight')
            print(f"\n  [PLOT] {f1_path}")
            print(f"  [PLOT] {f2_path}")
            plt.close('all')
        except Exception as e:
            print(f"\n  [PLOT] Failed: {e}")

    print("\n" + "=" * 72)
    print(f"  {sol['status']}  →  Z*_L1 = {sol['obj']:,.2f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()