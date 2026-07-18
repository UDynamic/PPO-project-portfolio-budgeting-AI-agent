"""
milp_ppm_test.py
================
Level-1 Deterministic Full-Foresight MILP — PPM Budget Allocation
Section 3 of the paper.

Usage:
    python milp_ppm_test.py [--seed N] [--n_projects N] [--horizon N]
                            [--b1_ratio F] [--no_csv]

All portfolio parameters are configurable at the top of the file
under CONFIG. Command-line args override the relevant CONFIG entries.
"""

import argparse
import random
import math
import csv
from scipy.special import betainc
import pulp
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, value, PULP_CBC_CMD

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG  —  edit these to control portfolio generation
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    seed          = 42,
    n_projects    = 5,       # or None → random in [n_min, n_max]
    n_min         = 5,
    n_max         = 10,
    horizon       = 24,      # H: number of periods
    gamma         = 0.97,    # per-period discount factor
    b1_ratio      = (0.35, 0.55),   # B1 as fraction of total BAC (uniform range)

    # Project schedule
    s_max_frac    = 0.20,    # s_i <= H * s_max_frac
    dur_min_frac  = 0.33,    # duration >= H * dur_min_frac
    dur_max_frac  = 0.80,    # duration <= H * dur_max_frac

    # S-curve shape (Beta CDF parameters)
    a_range       = (1.5, 3.5),
    b_range       = (2.0, 5.0),

    # Contract
    BAC_range     = (50_000, 300_000),
    pi_range      = (0.05, 0.20),    # profit margin
    rho_range     = (0.05, 0.10),    # retention ratio
    delta_rec     = 0.25,            # FIDIC default advance recovery rate
    psi           = 0.10,            # FIDIC default recovery threshold

    # Milestones
    n_ms_range    = (3, 5),          # number of milestones (last always = 1.0)
    ms_min_gap    = 0.15,            # minimum spacing between milestone thresholds
    ms_lo         = 0.20,            # lower bound for interior milestone thresholds
    ms_hi         = 0.85,            # upper bound for interior milestone thresholds

    # Termination
    Omega_range   = (2, 5),          # schedule tolerance (periods)
    mu_range      = (0.10, 0.30),    # max cost overrun ratio
    tau_tol_range = (1, 3),          # cure-period tolerance (periods)

    # Efficiency eta_{i,t} — bell-shaped, peaks mid-project
    eta_base_lo   = 0.65,
    eta_base_hi   = 1.00,   # peak at mid-project
    eta_noise     = 0.03,
    eta_min       = 0.50,
    eta_max       = 1.10,

    # MILP solver
    solver_time_limit = 300,  # seconds
)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def scurve(tau: float, a: float, b: float) -> float:
    """P^plan_{i,t} = I_{tau^c}(a_i, b_i)  — eq. (2)"""
    tc = max(0.0, min(1.0, tau))
    if tc <= 0.0: return 0.0
    if tc >= 1.0: return 1.0
    return float(betainc(a, b, tc))


def derive_alpha(phi: list, drec: float, psi: float) -> float:
    """
    Derive alpha_i so payment identity A + sum(R_net) + R_ret = CP holds exactly.
    Requires alpha_i = drec * sum(phi_j : cumulative phi_j >= psi).
    """
    cum, rec_frac = 0.0, 0.0
    for phi_j in phi:
        cum += phi_j
        if cum >= psi:
            rec_frac += phi_j
    return drec * rec_frac


def compute_R_net(proj: dict):
    """
    Returns (R_net_list, R_ret) where:
      R_net[j] = phi_j*CP*(1-rho) - drec*phi_j*CP*1{cum_phi_j >= psi}
      R_ret    = rho * CP
    """
    CP, rho   = proj['CP_i'], proj['rho_i']
    drec, psi = proj['delta_rec_i'], proj['psi_i']
    phi       = proj['phi']
    cum       = 0.0
    R_net     = []
    for phi_j in phi:
        cum    += phi_j
        gross   = phi_j * CP
        ret_ded = rho * gross
        rec_ded = drec * gross if cum >= psi else 0.0
        R_net.append(gross - ret_ded - rec_ded)
    return R_net, rho * CP


def planned_progress(proj: dict, H: int) -> dict:
    """P^plan_{i,t} for t = 1..H."""
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

def generate_project(idx: int, H: int, cfg: dict, rng: random.Random) -> dict:
    # Schedule
    s_max  = max(1, int(H * cfg['s_max_frac']))
    s_i    = rng.randint(1, s_max)
    dur    = rng.randint(int(H * cfg['dur_min_frac']),
                         int(H * cfg['dur_max_frac']))
    f_i    = min(s_i + dur - 1, H)
    D_plan = f_i - s_i + 1

    a_i = rng.uniform(*cfg['a_range'])
    b_i = rng.uniform(*cfg['b_range'])

    # Contract
    BAC_i  = rng.uniform(*cfg['BAC_range'])
    pi_i   = rng.uniform(*cfg['pi_range'])
    CP_i   = BAC_i * (1 + pi_i)
    rho_i  = rng.uniform(*cfg['rho_range'])
    drec_i = cfg['delta_rec']
    psi_i  = cfg['psi']

    # Milestones
    n_ms   = rng.randint(*cfg['n_ms_range'])
    pts    = sorted(rng.uniform(cfg['ms_lo'], cfg['ms_hi'])
                    for _ in range(n_ms - 1))
    spaced = [pts[0]]
    for v in pts[1:]:
        spaced.append(max(v, spaced[-1] + cfg['ms_min_gap']))
    theta  = [min(v, 0.95) for v in spaced] + [1.0]
    M_i    = len(theta)
    phi    = [1.0 / M_i] * M_i

    # Advance payment: derived to satisfy payment identity exactly
    alpha_i = derive_alpha(phi, drec_i, psi_i)

    # Termination
    Omega_i   = rng.randint(*cfg['Omega_range'])
    mu_i      = rng.uniform(*cfg['mu_range'])
    tau_tol_i = rng.randint(*cfg['tau_tol_range'])

    # Deterministic efficiency eta_{i,t}
    eta = {}
    lo, hi = cfg['eta_base_lo'], cfg['eta_base_hi']
    noise  = cfg['eta_noise']
    for t in range(1, H + 1):
        if s_i <= t <= f_i:
            tau_n  = (t - s_i) / max(D_plan - 1, 1)
            base   = lo + (hi - lo) * math.sin(math.pi * tau_n)
            eta[t] = max(cfg['eta_min'],
                         min(cfg['eta_max'], base + rng.uniform(-noise, noise)))
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


def generate_portfolio(cfg: dict) -> dict:
    rng = random.Random(cfg['seed'])
    n   = cfg['n_projects'] or rng.randint(cfg['n_min'], cfg['n_max'])
    H   = cfg['horizon']

    projects  = [generate_project(i, H, cfg, rng) for i in range(n)]
    total_BAC = sum(p['BAC_i'] for p in projects)
    lo, hi    = cfg['b1_ratio']
    B1        = rng.uniform(lo, hi) * total_BAC

    return dict(n=n, H=H, projects=projects, B1=B1,
                gamma=cfg['gamma'], total_BAC=total_BAC, cfg=cfg)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MILP  (eq. L1)
# ═════════════════════════════════════════════════════════════════════════════

def build_and_solve(portfolio: dict) -> dict:
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    B1       = portfolio['B1']
    gamma    = portfolio['gamma']
    T        = list(range(1, H + 1))

    plan_prog = {i: planned_progress(p, H) for i, p in enumerate(projects)}

    act   = {(i, t): int(projects[i]['s_i'] <= t <= projects[i]['f_i'])
             for i in range(n) for t in T}
    A     = {i: projects[i]['alpha_i'] * projects[i]['CP_i'] for i in range(n)}
    R_net = {}
    R_ret = {}
    for i, p in enumerate(projects):
        rn, rr = compute_R_net(p)
        for j, v in enumerate(rn): R_net[i, j] = v
        R_ret[i] = rr

    # Verify payment identity
    for i, p in enumerate(projects):
        total = A[i] + sum(R_net[i, j] for j in range(p['M_i'])) + R_ret[i]
        assert abs(total - p['CP_i']) < 1.0, \
            f"Payment identity failed P{i}: {total:.2f} vs {p['CP_i']:.2f}"

    BIG_M = 1.0
    EPS   = 1e-4

    prob = LpProblem("PPM_L1", LpMaximize)

    x = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0)
         for i in range(n) for t in T}
    u = {(i, j, t): LpVariable(f"u_{i}_{j}_{t}", cat=LpBinary)
         for i in range(n) for j in range(projects[i]['M_i']) for t in T}
    P = {(i, t): LpVariable(f"P_{i}_{t}", lowBound=0, upBound=1.0)
         for i in range(n) for t in T}
    B = {t: LpVariable(f"B_{t}", lowBound=0) for t in T}

    # Objective
    def inflow_expr(t):
        terms = []
        for i, p in enumerate(projects):
            if t == p['s_i']:
                terms.append(A[i])
            for j in range(p['M_i']):
                u_prev  = u[i, j, t-1] if t > 1 else 0
                pay     = R_net[i, j] + (R_ret[i] if j == p['M_i']-1 else 0)
                terms.append(pay * (u[i, j, t] - u_prev))
        return lpSum(terms)

    prob += lpSum(
        gamma**(t-1) * (inflow_expr(t) - lpSum(act[i,t]*x[i,t] for i in range(n)))
        for t in T
    ), "obj"

    # Initial balance
    prob += B[1] == B1, "B_init"

    # Balance recursion
    for t in T[:-1]:
        prob += (B[t+1] == B[t]
                 - lpSum(act[i,t]*x[i,t] for i in range(n))
                 + inflow_expr(t),
                 f"bal_{t}")

    # Budget feasibility
    for t in T:
        prob += (lpSum(act[i,t]*x[i,t] for i in range(n)) <= B[t], f"bud_{t}")

    # Progress recursion + inactive lock
    for i, p in enumerate(projects):
        for t in T:
            P_prev = P[i, t-1] if t > 1 else 0.0
            prob += (P[i,t] == P_prev + p['eta'][t]*x[i,t]/p['BAC_i'], f"prg_{i}_{t}")
            if act[i, t] == 0:
                prob += (x[i,t] == 0, f"off_{i}_{t}")

    # Certification big-M
    for i, p in enumerate(projects):
        for j in range(p['M_i']):
            th = p['theta'][j]
            for t in T:
                prob += (P[i,t] >= th - BIG_M*(1 - u[i,j,t]),    f"clb_{i}_{j}_{t}")
                prob += (P[i,t] <= (th-EPS) + BIG_M*u[i,j,t],    f"cub_{i}_{j}_{t}")
                if t > 1:
                    prob += (u[i,j,t] >= u[i,j,t-1],              f"mon_{i}_{j}_{t}")
                if j < p['M_i']-1:
                    prob += (u[i,j+1,t] <= u[i,j,t],              f"ord_{i}_{j}_{t}")
                if t < p['s_i']:
                    prob += (u[i,j,t] == 0,                        f"pre_{i}_{j}_{t}")

    solver = PULP_CBC_CMD(msg=0, timeLimit=portfolio['cfg']['solver_time_limit'])
    prob.solve(solver)

    return dict(
        status    = pulp.LpStatus[prob.status],
        obj       = value(prob.objective),
        x=x, u=u, P=P, B=B,
        act=act, A=A, R_net=R_net, R_ret=R_ret,
        plan_prog = plan_prog,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  STATE RECORD SYSTEM
#     Captures every variable at every (project, period).
# ═════════════════════════════════════════════════════════════════════════════

def build_records(portfolio: dict, sol: dict) -> dict:
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    B1       = portfolio['B1']
    T        = list(range(1, H + 1))

    # Raw LP values
    x_val = {k: max(0.0, value(v) or 0.0) for k, v in sol['x'].items()}
    u_val = {k: int(round(value(v) or 0))  for k, v in sol['u'].items()}
    P_lp  = {k: (value(v) or 0.0)          for k, v in sol['P'].items()}
    B_lp  = {t: (value(v) or 0.0)          for t, v in sol['B'].items()}

    plan_prog = sol['plan_prog']
    R_net, R_ret, A, act = sol['R_net'], sol['R_ret'], sol['A'], sol['act']

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

        # Advance payment event
        if s <= H:
            all_events.append(dict(t=s, proj=i, type='advance', amount=A[i]))

        for t in T:
            x_t = x_val[i, t]

            # Window
            if t < s:   window = 'pre_start'
            elif t > f: window = 'post_plan'
            else:       window = 'active'

            # Progress update
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
                                                   type='retention', amount=rho*CP))

            # EVM signals
            if window == 'active' and status == 'active':
                P_plan_t = plan_prog[i][t]
                SPI = (P_sim / P_plan_t) if P_plan_t > 1e-9 else (1.0 if P_sim < 1e-9 else float('inf'))
                CPI = (EV_sim / ACWP_sim) if ACWP_sim > 1e-9 else 1.0
                EAC = ACWP_sim + (BAC - EV_sim) / CPI if CPI > 1e-9 else float('inf')
                elapsed = t - s
                rem_per = D_plan - elapsed
                if SPI > 1e-9 and not math.isinf(SPI):
                    f_hat   = t + rem_per / SPI
                    Delta_f = f_hat - f
                else:
                    f_hat   = float('inf')
                    Delta_f = float('inf')
                cond1 = (not math.isinf(Delta_f)) and Delta_f > Omega
                cond2 = (not math.isinf(EAC))     and EAC > (1 + mu) * BAC
            else:
                SPI = CPI = EAC = Delta_f = f_hat = None
                cond1 = cond2 = False

            # Cure counter
            if window == 'active' and status == 'active':
                tau_rem = tau_rem - 1 if (cond1 and cond2) else tau_tol

            # LP u values at this period
            u_now  = {j: u_val.get((i,j,t), 0) for j in range(M)}
            u_prev = {j: u_val.get((i,j,t-1), 0) for j in range(M)} if t > 1 else {j: 0 for j in range(M)}
            ms_new_lp = [j for j in range(M) if u_now[j] - u_prev.get(j, 0) > 0]

            records[t] = dict(
                # Index
                proj=i, t=t,
                # Status
                window=window, status=status,
                # LP values
                x_lp=x_t, P_lp=P_lp.get((i,t)), u_lp=u_now, u_prev_lp=u_prev,
                ms_new_lp=ms_new_lp,
                # Simulated state
                P_sim=P_sim, ACWP_sim=ACWP_sim, EV_sim=EV_sim,
                # Planned
                P_plan=plan_prog[i][t], eta=eta.get(t, 0.0),
                # EVM
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
                all_events.append(dict(t=t+1, proj=i,
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

    # Annotate events with running total
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
# 5.  VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate(portfolio: dict, sol: dict, rec: dict, tol: float = 1e-3) -> dict:
    n, H     = portfolio['n'], portfolio['H']
    projects = portfolio['projects']
    T        = list(range(1, H + 1))
    x_val, u_val, P_lp, B_lp = rec['x_val'], rec['u_val'], rec['P_lp'], rec['B_lp']
    R_net, R_ret, A, act = sol['R_net'], sol['R_ret'], sol['A'], sol['act']
    errors = {}

    # C1: B_t >= 0
    errors['C1_B_nonneg'] = [(t, round(B_lp[t],2)) for t in T if B_lp.get(t,0) < -tol]

    # C2: sum x <= B_t
    errors['C2_budget'] = [dict(t=t, spend=round(sum(x_val[i,t] for i in range(n)),2),
                                bal=round(B_lp.get(t,0),2))
                           for t in T
                           if sum(x_val[i,t] for i in range(n)) > B_lp.get(t,0) + tol]

    # C3: progress recursion
    prog_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc = min(1.0, Pc + p['eta'][t] * x_val[i,t] / p['BAC_i'])
            lp = P_lp.get((i,t), 0.0)
            if abs(lp - Pc) > tol*5:
                prog_errs.append(dict(proj=i, t=t, lp=round(lp,6), sim=round(Pc,6)))
    errors['C3_progress'] = prog_errs[:10]

    # C4: balance recursion
    bal_errs = []
    Bc = portfolio['B1']
    for t in T[:-1]:
        out = sum(act[i,t]*x_val[i,t] for i in range(n))
        inf = sum(A[i] for i in range(n) if t == projects[i]['s_i'])
        for i, p in enumerate(projects):
            for j in range(p['M_i']):
                u_now  = u_val.get((i,j,t), 0)
                u_prev = u_val.get((i,j,t-1), 0) if t > 1 else 0
                if u_now - u_prev > 0:
                    inf += R_net[i,j] + (R_ret[i] if j==p['M_i']-1 else 0)
        Bc_next = Bc - out + inf
        lp_next = B_lp.get(t+1, 0.0)
        if abs(lp_next - Bc_next) > tol*10:
            bal_errs.append(dict(t=t, lp=round(lp_next,2), comp=round(Bc_next,2)))
        Bc = Bc_next
    errors['C4_balance'] = bal_errs[:5]

    # C5: certification u=1 iff P>=theta
    cert_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc = min(1.0, Pc + p['eta'][t]*x_val[i,t]/p['BAC_i'])
            for j in range(p['M_i']):
                u_ij    = u_val.get((i,j,t), 0)
                reached = Pc >= p['theta'][j] - tol
                if reached and u_ij == 0:
                    cert_errs.append(dict(issue='missed',    proj=i, ms=j, t=t, P=round(Pc,5)))
                if not reached and u_ij == 1:
                    cert_errs.append(dict(issue='premature', proj=i, ms=j, t=t, P=round(Pc,5)))
    errors['C5_cert'] = cert_errs[:10]

    # C6: monotonicity
    errors['C6_mono'] = [(i,j,t) for i,p in enumerate(projects)
                         for j in range(p['M_i']) for t in range(2,H+1)
                         if u_val.get((i,j,t),0) < u_val.get((i,j,t-1),0)][:10]

    # C7: ordering
    errors['C7_order'] = [(i,j,t) for i,p in enumerate(projects)
                          for j in range(p['M_i']-1) for t in T
                          if u_val.get((i,j+1,t),0) > u_val.get((i,j,t),0)][:10]

    # C8: no inactive allocation
    errors['C8_inactive'] = [(i,t,round(x_val[i,t],2)) for i in range(n)
                              for t in T if act[i,t]==0 and x_val[i,t]>tol][:10]

    # C9: payment identity (completed only)
    pay_errs = []
    for i, p in enumerate(projects):
        if rec['proj'][i]['status'] == 'completed':
            total = A[i] + sum(R_net[i,j] for j in range(p['M_i'])) + R_ret[i]
            if abs(total - p['CP_i']) > tol * p['CP_i']:
                pay_errs.append(dict(proj=i, total=round(total,2), CP=round(p['CP_i'],2)))
    errors['C9_payment_id'] = pay_errs

    # C10: x >= 0
    errors['C10_nonneg'] = [(i,t,round(x_val[i,t],4)) for (i,t) in x_val
                             if x_val[i,t] < -tol]

    # Termination info (informational)
    errors['_terminations'] = {i: next((t for t in range(1,H+1)
                                        if rec['proj'][i]['records'][t]['status']=='terminated'), None)
                                for i in range(n)
                                if rec['proj'][i]['status'] == 'terminated'}

    all_pass = all(len(v)==0 for k,v in errors.items() if not k.startswith('_'))
    return dict(errors=errors, all_pass=all_pass)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PRINT
# ═════════════════════════════════════════════════════════════════════════════

def _f(v, fmt='.4f', width=8, none='--'):
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return f'{none:>{width}}'
    return f'{v:{width}{fmt}}'


def print_portfolio(pf: dict):
    print("\n" + "="*72)
    print("PORTFOLIO PARAMETERS")
    print("="*72)
    print(f"  n={pf['n']}  H={pf['H']}  B1={pf['B1']:,.0f}  "
          f"totalBAC={pf['total_BAC']:,.0f}  gamma={pf['gamma']}")
    print(f"\n  {'i':>2}  {'s':>3}  {'f':>3}  {'D':>3}  "
          f"{'BAC':>9}  {'CP':>9}  {'π%':>5}  {'α%':>5}  "
          f"{'ρ%':>5}  {'M':>2}  {'Ω':>3}  {'μ%':>5}  {'τtol':>4}")
    print("  " + "─"*73)
    for p in pf['projects']:
        print(f"  {p['idx']:>2}  {p['s_i']:>3}  {p['f_i']:>3}  {p['D_plan']:>3}  "
              f"{p['BAC_i']:>9,.0f}  {p['CP_i']:>9,.0f}  "
              f"{p['pi_i']*100:>5.1f}  {p['alpha_i']*100:>5.1f}  "
              f"{p['rho_i']*100:>5.1f}  {p['M_i']:>2}  "
              f"{p['Omega_i']:>3}  {p['mu_i']*100:>5.1f}  {p['tau_tol_i']:>4}")
    print()
    for p in pf['projects']:
        ths = "  ".join(f"θ{j+1}={v:.3f}" for j,v in enumerate(p['theta']))
        ets = "  ".join(f"t{t}:{p['eta'][t]:.2f}"
                        for t in range(p['s_i'], p['f_i']+1))
        print(f"  P{p['idx']}: [{ths}]  α={p['alpha_i']:.4f}")
        print(f"       η: [{ets}]")


def print_solution(pf: dict, sol: dict, rec: dict):
    n, H = pf['n'], pf['H']
    T    = list(range(1, H+1))
    x_val, B_lp = rec['x_val'], rec['B_lp']
    u_val = rec['u_val']

    print("\n" + "="*72)
    print("MILP SOLUTION")
    print("="*72)
    print(f"  Solver : {sol['status']}")
    print(f"  Z*_L1  : {sol['obj']:,.2f}  (discounted net cash flow — upper bound)")

    # Allocation matrix
    print("\n── ALLOCATIONS  x_{i,t}  (thousands) ──")
    for blk in range(1, H+1, 12):
        cols = list(range(blk, min(blk+12, H+1)))
        hdr  = "      " + "".join(f" {f't{c}':>5}" for c in cols)
        print(f"\n  t={blk}..{cols[-1]}")
        print("  " + hdr)
        print("  " + "  " + "─"*len(hdr))
        for i in range(n):
            row = "".join(
                f" {x_val[i,t]/1000:>5.1f}" if x_val[i,t] > 0.5 else f" {'--':>5}"
                for t in cols)
            print(f"  P{i:<2}  {row}")
        brow = "".join(f" {B_lp.get(t,0)/1000:>5.1f}" for t in cols)
        print(f"  Bt   {brow}   ← balance")

    # Per-project outcome
    print("\n── PER-PROJECT OUTCOME ──")
    print(f"  {'i':>2}  {'total_x':>10}  {'P_final':>8}  "
          f"{'status':>11}  ms  cert_periods")
    print("  " + "─"*65)
    for i, p in enumerate(pf['projects']):
        pr     = rec['proj'][i]
        tx     = sum(x_val[i,t] for t in T)
        ms_str = []
        for j in range(p['M_i']):
            ct = next((t for t in T
                       if u_val.get((i,j,t),0)==1
                       and u_val.get((i,j,t-1),0)==0), None)
            ms_str.append(f"m{j+1}@{ct}" if ct else f"m{j+1}:--")
        print(f"  {i:>2}  {tx:>10,.0f}  {pr['P_final']:>8.4f}  "
              f"{pr['status']:>11}  {len(pr['certified'])}/{p['M_i']}  "
              + "  ".join(ms_str))


def print_state_table(pf: dict, rec: dict):
    print("\n" + "="*72)
    print("FULL STATE RECORDS  (all timesteps, all projects)")
    print("="*72)
    T = list(range(1, pf['H']+1))

    for i, p in enumerate(pf['projects']):
        pr = rec['proj'][i]
        print(f"\n{'─'*72}")
        print(f"  P{i}  s={p['s_i']} f={p['f_i']} D={p['D_plan']}  "
              f"BAC={p['BAC_i']:,.0f}  CP={p['CP_i']:,.0f}  M={p['M_i']}")
        print(f"  θ={[round(v,3) for v in p['theta']]}  "
              f"α={p['alpha_i']:.4f}  ρ={p['rho_i']:.4f}  "
              f"Ω={p['Omega_i']}  μ={p['mu_i']:.3f}  τtol={p['tau_tol_i']}")
        print(f"{'─'*72}")
        print(f"  {'t':>3}  {'x_lp':>8}  {'η':>5}  "
              f"{'P_plan':>7}  {'P_sim':>7}  {'P_lp':>7}  "
              f"{'ACWP':>8}  {'EV':>8}  "
              f"{'SPI':>6}  {'CPI':>6}  {'EAC':>9}  "
              f"{'Δf':>6}  {'C1':>3}  {'C2':>3}  {'τ':>3}  "
              f"{'u':>{p['M_i']}}  {'cert_lp':>8}  {'status':>11}")
        print("  " + "─"*110)
        for t in T:
            r = pr['records'][t]
            xv  = f"{r['x_lp']:>8.1f}" if r['x_lp'] > 0.5 else f"{'--':>8}"
            etv = f"{r['eta']:>5.2f}"   if r['eta']  > 0.0 else f"{'--':>5}"
            spi = _f(r['SPI'],  '.3f', 6)
            cpi = _f(r['CPI'],  '.3f', 6)
            eac = _f(r['EAC'],  '.0f', 9)
            df  = _f(r['Delta_f'], '.2f', 6)
            c1  = ("YES" if r['cond1'] else "no ") if r['SPI'] is not None else "  -"
            c2  = ("YES" if r['cond2'] else "no ") if r['CPI'] is not None else "  -"
            uv  = "".join(str(r['u_lp'].get(j,0)) for j in range(p['M_i']))
            cert = ",".join(f"m{j+1}" for j in r['ms_new_lp']) or "--"
            print(f"  {t:>3}  {xv}  {etv}  "
                  f"{r['P_plan']:>7.4f}  {r['P_sim']:>7.4f}  "
                  f"{_f(r['P_lp'],'.4f',7)}  "
                  f"{r['ACWP_sim']:>8.0f}  {r['EV_sim']:>8.0f}  "
                  f"{spi}  {cpi}  {eac}  "
                  f"{df}  {c1}  {c2}  {r['tau_rem']:>3}  "
                  f"{uv:>{p['M_i']}}  {cert:>8}  {r['status']:>11}")
        print(f"  → final: {pr['status']}  P={pr['P_final']:.4f}  "
              f"ACWP={pr['ACWP_final']:,.0f}  certified={sorted(pr['certified'])}")

    # Portfolio table
    print(f"\n{'─'*72}")
    print("  PORTFOLIO  (per period)")
    print(f"{'─'*72}")
    print(f"  {'t':>3}  {'B_lp':>9}  {'B_sim':>9}  "
          f"{'outflow':>9}  {'inflow':>9}  {'net':>9}  {'nact':>4}  events")
    print("  " + "─"*72)
    for t in T:
        r  = rec['portfolio'][t]
        ev = "; ".join(f"P{e['proj']}:{e['type']}={e['amount']:,.0f}"
                       for e in r['events']) or "--"
        print(f"  {t:>3}  {r['B_lp']:>9,.0f}  {r['B_sim']:>9,.0f}  "
              f"{r['outflow']:>9,.0f}  {r['inflow']:>9,.0f}  "
              f"{r['net']:>9,.0f}  {r['n_active']:>4}  {ev}")


def print_cashflow(rec: dict, pf: dict):
    T = list(range(1, pf['H']+1))
    print(f"\n{'─'*72}")
    print("  CASH FLOW EVENTS")
    print(f"{'─'*72}")
    evs = sorted(rec['events'], key=lambda e: (e['t'], e['proj']))
    print(f"  {'t':>3}  {'proj':>4}  {'type':>24}  {'amount':>10}  {'running':>10}")
    print("  " + "─"*57)
    for ev in evs:
        print(f"  {ev['t']:>3}  P{ev['proj']:<3}  {ev['type']:>24}  "
              f"{ev['amount']:>10,.0f}  {ev.get('running_total',0):>10,.0f}")
    tin  = sum(ev['amount'] for ev in evs)
    tout = sum(rec['x_val'][i,t] for i in range(pf['n']) for t in T)
    print(f"\n  Total inflows     : {tin:>12,.0f}")
    print(f"  Total outflows    : {tout:>12,.0f}")
    print(f"  Net (undiscounted): {tin-tout:>12,.0f}")


def print_validation(val: dict):
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
    print("\n" + "="*72)
    print("VALIDATION")
    print("="*72)
    for k, desc in labels.items():
        errs = val['errors'].get(k, [])
        flag = "PASS ✓" if not errs else "FAIL ✗"
        print(f"  {flag}  {desc}")
        for e in (errs if isinstance(errs, list) else []):
            print(f"         → {e}")
    terms = val['errors'].get('_terminations', {})
    if terms:
        print(f"\n  [INFO] Terminations: {terms}")
    else:
        print("\n  [INFO] No terminations.")
    print(f"\n  Overall: {'ALL PASS ✓' if val['all_pass'] else 'SOME FAILED ✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CSV EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def export_csv(pf: dict, rec: dict, prefix: str = "milp_out"):
    H, n = pf['H'], pf['n']
    T    = list(range(1, H+1))

    rows = []
    for i in range(n):
        for t in T:
            r = rec['proj'][i]['records'][t]
            row = {k: v for k, v in r.items()
                   if not isinstance(v, (dict, list))}
            row['u_lp']      = "".join(str(r['u_lp'].get(j,0))
                                       for j in range(pf['projects'][i]['M_i']))
            row['ms_new_lp'] = str(r['ms_new_lp'])
            row['ms_new_sim']= str(r['ms_new_sim'])
            row['theta']     = str(r['theta'])
            for k in row:
                if isinstance(row[k], float) and math.isinf(row[k]):
                    row[k] = 'inf'
            rows.append(row)

    fn_proj = f"{prefix}_project_states.csv"
    with open(fn_proj, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    port_rows = []
    for t in T:
        r = rec['portfolio'][t]
        port_rows.append(dict(
            t=t, B_lp=r['B_lp'], B_sim=r['B_sim'],
            outflow=r['outflow'], inflow=r['inflow'], net=r['net'],
            n_active=r['n_active'], active=str(r['active_proj'])))

    fn_port = f"{prefix}_portfolio_states.csv"
    with open(fn_port, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(port_rows[0].keys()))
        w.writeheader(); w.writerows(port_rows)

    print(f"\n  [CSV] {fn_proj}")
    print(f"  [CSV] {fn_port}")


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   default=None)
    parser.add_argument('--n_projects', type=int,   default=None)
    parser.add_argument('--horizon',    type=int,   default=None)
    parser.add_argument('--b1_ratio',   type=float, default=None,
                        help="Fixed B1/totalBAC ratio (overrides range)")
    parser.add_argument('--no_csv',     action='store_true')
    args = parser.parse_args()

    cfg = dict(CONFIG)   # copy so we don't mutate the module-level dict
    if args.seed       is not None: cfg['seed']       = args.seed
    if args.n_projects is not None: cfg['n_projects'] = args.n_projects
    if args.horizon    is not None: cfg['horizon']    = args.horizon
    if args.b1_ratio   is not None: cfg['b1_ratio']   = (args.b1_ratio, args.b1_ratio)

    portfolio = generate_portfolio(cfg)
    print_portfolio(portfolio)

    sol = build_and_solve(portfolio)
    rec = build_records(portfolio, sol)
    val = validate(portfolio, sol, rec)

    print_solution(portfolio, sol, rec)
    print_state_table(portfolio, rec)
    print_cashflow(rec, portfolio)
    print_validation(val)

    if not args.no_csv:
        export_csv(portfolio, rec)

    print("\n" + "="*72)
    print(f"  {sol['status']}  →  Z*_L1 = {sol['obj']:,.2f}")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()