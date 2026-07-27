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
"""

import math, json
from pathlib import Path
gamma = 0.95

def V(v, tol=None):
    """Wrap a value with its tolerance."""
    if tol is None:
        # default tolerances by magnitude
        tol = max(abs(v) * 0.02, 0.5)
    return {"v": round(float(v), 6), "tol": round(float(tol), 6)}

def build_timeline(BAC, eta, D_plan, fi, si,
                   alloc,        # {t: x_it}
                   ms_theta,     # [theta_j, ...]
                   ms_e,         # [e_j, ...]
                   ms_phi,       # [phi_j, ...]
                   CP, rho=0, alpha=0, A=0, delta_rec=0.25, psi=0.10,
                   mu=0.30, Omega=1, tau_tol=2,
                   eta_schedule=None,  # {t: eta} overrides constant eta
                   term_at=None):      # forced termination period
    """
    Simulate one project and return list of per-period dicts.
    All output values are wrapped with V().
    """
    rows = []
    P = 0.0
    cum_spend = 0.0
    cum_recover = 0.0
    tau_rem = tau_tol
    ms_certified = [False] * len(ms_theta)
    retention_released = False
    advance_received = A  # received at t=si-1 (t=0 for si=1)

    for t in range(si, fi + 1):
        eta_t = eta_schedule.get(t, eta) if eta_schedule else eta
        x = alloc.get(t, 0.0)
        delta_P = eta_t * x / BAC if BAC > 0 else 0.0
        P = min(P + delta_P, 1.0)
        cum_spend += x

        # BCWS
        bcws = min((t - si + 1) / D_plan, 1.0) if D_plan > 0 else 1.0
        bcwp = P
        acwp = cum_spend / BAC if BAC > 0 else 0.0
        spi  = bcwp / bcws if bcws > 0 else (2.0 if bcwp > 0 else 0.0)
        cpi  = bcwp / acwp if acwp > 0 else 1.0
        eac  = (1.0 / cpi) if cpi > 0 else float('inf')

        # termination conditions
        cond1 = spi < 1.0
        cond2 = eac > (1 + mu)
        both  = cond1 and cond2
        if both:
            tau_rem = max(tau_rem - 1, 0)
        elif not both and tau_rem < tau_tol:
            tau_rem = tau_tol  # reset on lapse

        # milestone certification
        R_net_t = 0.0; R_ret_t = 0.0; R_term_t = 0.0
        ms_j_this = []
        for j, (theta_j, e_j, phi_j) in enumerate(zip(ms_theta, ms_e, ms_phi)):
            if not ms_certified[j] and P >= theta_j - 1e-6 and t >= e_j:
                ms_certified[j] = True
                ms_j_this.append(j + 1)
                R_gross_full = phi_j * CP           # pre-retention base for recovery
                R_gross_net  = phi_j * CP * (1-rho)  # what gets paid (net of retention)
                # advance recovery (deducted from R_gross_full per FIDIC)
                if alpha > 0 and sum(ms_phi[:j+1]) >= psi:
                    recover = min(delta_rec * R_gross_full, A - cum_recover)
                    recover = max(recover, 0.0)
                else:
                    recover = 0.0
                cum_recover += recover
                R_net_t += R_gross_net - recover

        # Retention: lump release when all milestones certified
        if all(ms_certified) and not retention_released and rho > 0:
            R_ret_t = rho * CP
            retention_released = True

        # termination
        if term_at is not None and t == term_at:
            R_term_t = P * CP - (A - cum_recover)
            R_term_t = round(R_term_t, 4)

        rows.append({
            "t":             V(t, 0),
            "x":             V(x, 0.5),
            "delta_P":       V(delta_P, 0.01),
            "P":             V(P, 0.01),
            "BCWS":          V(bcws, 0.02),
            "BCWP":          V(bcwp, 0.01),
            "ACWP":          V(acwp, 0.01),
            "SPI":           V(spi, 0.05),
            "CPI":           V(cpi, 0.05),
            "EAC_frac":      V(min(eac, 9.99), 0.1),
            "tau_rem":       V(tau_rem, 0),
            "ms_certified":  ms_j_this,
            "R_net":         V(R_net_t, 0.5),
            "R_ret":         V(R_ret_t, 0.5),
            "R_term":        V(R_term_t, 0.5),
        })

        if term_at is not None and t >= term_at:
            break

    return rows, advance_received


def build_portfolio(cases_projects, B0, H, advance=0, advance_t=0):
    """
    cases_projects: list of (rows, A_received) per project.
    Returns per-period portfolio rows.
    """
    port = []
    B = B0 + advance  # advance at t=0
    cum_Z = advance   # advance discounted at gamma^0

    for t in range(1, H + 2):
        sum_in  = 0.0
        sum_out = 0.0
        for (rows, _) in cases_projects:
            for row in rows:
                if row["t"]["v"] == t:
                    sum_in  += row["R_net"]["v"] + row["R_ret"]["v"] + max(row["R_term"]["v"], 0)
                    sum_out += row["x"]["v"]
                    if row["R_term"]["v"] < 0:
                        sum_in += row["R_term"]["v"]  # negative settlement is outflow
        net = sum_in - sum_out
        B   = B + net
        disc_net = net * gamma**(t - 1)
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


# ── Build all cases ─────────────────────────────────────────────────────────
def make_case(meta, params, projects_spec, B0, H, advance=0):
    """
    projects_spec: list of dicts with keys matching build_timeline args.
    Returns complete case dict.
    """
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


def sp1():
    ps = dict(BAC=100, eta=1.0, D_plan=4, fi=4, si=1,
              alloc={1:50, 3:50}, ms_theta=[0.5,1.0], ms_e=[1,3],
              ms_phi=[0.5,0.5], CP=120, rho=0, alpha=0, A=0,
              mu=0.30, Omega=1, tau_tol=2)
    return make_case(
        meta={"id":"SP-1","group":"G1","Zstar":19.025,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":100,"CP":120,"pi":0.20,"eta":1.0,"M":2,"tau_tol":2,
                "B0":300,"H":4,"gamma":0.95},
        projects_spec=[ps], B0=300, H=4)

def sp3():
    ps = dict(BAC=60, eta=1.0, D_plan=6, fi=6, si=1,
              alloc={1:20,3:20,5:20}, ms_theta=[1/3,2/3,1.0], ms_e=[1,3,5],
              ms_phi=[1/3,1/3,1/3], CP=90, rho=0, alpha=0, A=0,
              mu=0.50, Omega=2, tau_tol=2)
    return make_case(
        meta={"id":"SP-3","group":"G1","Zstar":27.17,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":60,"CP":90,"pi":0.50,"eta":1.0,"M":3,"tau_tol":2,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spJ():
    ps = dict(BAC=100, eta=1.0, D_plan=4, fi=4, si=1,
              alloc={1:50,3:50}, ms_theta=[0.5,1.0], ms_e=[1,3],
              ms_phi=[0.5,0.5], CP=100, rho=0, alpha=0, A=0,
              mu=0.50, Omega=2, tau_tol=2)
    return make_case(
        meta={"id":"SP-J","group":"G1","Zstar":0.0,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":100,"CP":100,"eta":1.0,"M":2,"H":4},
        projects_spec=[ps], B0=300, H=4)

def sp4():
    ps = dict(BAC=100, eta=1.0, D_plan=8, fi=8, si=1,
              alloc={1:50,5:50}, ms_theta=[0.5,1.0], ms_e=[1,5],
              ms_phi=[0.5,0.5], CP=140, rho=0, alpha=0, A=0,
              mu=0.30, Omega=1, tau_tol=2)
    return make_case(
        meta={"id":"SP-4","group":"G2","Zstar":36.29,"tol_Z":1.0,"outcome":"completed"},
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
        meta={"id":"MP-K","group":"G2","Zstar":55.87,"tol_Z":1.0,"outcome":"both_completed"},
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
        meta={"id":"MP-Kp","group":"G2","Zstar":27.936,"tol_Z":1.0,"outcome":"both_completed"},
        params={"n":2,"B0":20,"H":6},
        projects_spec=[pA,pB], B0=20, H=6)

def spL():
    ps = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,
              alloc={**{t:100/(0.7*8) for t in range(1,6)}, 6:100/0.7-5*(100/(0.7*8))},
              ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
              CP=170,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2)
    return make_case(
        meta={"id":"SP-L","group":"G2","Zstar":9.298,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":100,"CP":170,"eta":0.70,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spLp():
    ps = dict(BAC=100,eta=0.70,D_plan=8,fi=6,si=1,alloc={},
              ms_theta=[1.0],ms_e=[6],ms_phi=[1.0],
              CP=80,rho=0,alpha=0,A=0,mu=0.10,Omega=1,tau_tol=2,term_at=3)
    return make_case(
        meta={"id":"SP-Lp","group":"G2","Zstar":0.0,"tol_Z":0.5,"outcome":"terminated"},
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
        meta={"id":"SP-2","group":"G3","Zstar":1.5,"tol_Z":0.5,"outcome":"terminated"},
        params={"BAC":100,"CP":100,"alpha":0.30,"A":30,"tau_tol":1,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def sp2a():
    ps = dict(BAC=100,eta=1.0,D_plan=4,fi=4,si=1,alloc={1:100},
              ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
              CP=120,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=1)
    return make_case(
        meta={"id":"SP-2a","group":"G3","Zstar":20.5,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":100,"CP":120,"alpha":0.30,"A":30,"tau_tol":1,"H":4},
        projects_spec=[ps], B0=300, H=4, advance=30)

def sp2b():
    ps = dict(BAC=100,eta=1.0,D_plan=4,fi=4,si=1,alloc={},
              ms_theta=[1.0],ms_e=[1],ms_phi=[1.0],
              CP=60,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=1,term_at=2)
    return make_case(
        meta={"id":"SP-2b","group":"G3","Zstar":1.5,"tol_Z":0.5,"outcome":"terminated"},
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
        meta={"id":"SP-2c","group":"G3","Zstar":4.279,"tol_Z":0.5,"outcome":"terminated"},
        params={"BAC":100,"CP":100,"alpha":0.30,"A":30,"tau_tol":3,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def sp2d():
    ps = dict(BAC=100,eta=1.0,D_plan=6,fi=6,si=1,alloc={4:100},
              ms_theta=[1.0],ms_e=[4],ms_phi=[1.0],
              CP=120,rho=0,alpha=0.30,A=30,delta_rec=0.25,psi=0.10,
              mu=0.30,Omega=1,tau_tol=4,
              eta_schedule={**{t:0.05 for t in range(1,4)},**{t:1.0 for t in range(4,7)}})
    return make_case(
        meta={"id":"SP-2d","group":"G3","Zstar":21.855,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":100,"CP":120,"alpha":0.30,"A":30,"tau_tol":4,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=30)

def spA():
    ps = dict(BAC=120,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:40,3:40,5:40},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=180,rho=0,alpha=0.20,A=36,delta_rec=0.30,psi=0.40,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-A","group":"G3","Zstar":59.43,"tol_Z":1.5,"outcome":"completed"},
        params={"BAC":120,"CP":180,"alpha":0.20,"psi":0.40,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=36)

def spB():
    ps = dict(BAC=60,eta=1.0,D_plan=3,fi=4,si=1,
              alloc={1:20,2:20,3:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,2,3],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0,alpha=0.20,A=18,delta_rec=0.30,psi=0.10,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-B","group":"G3","Zstar":28.975,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":60,"CP":90,"alpha":0.20,"H":4},
        projects_spec=[ps], B0=300, H=4, advance=18)

def spC():
    ps = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:20,3:20,5:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0.10,alpha=0,A=0,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-C","group":"G4","Zstar":26.35,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":60,"CP":90,"rho":0.10,"H":6},
        projects_spec=[ps], B0=300, H=6)

def spD():
    ps = dict(BAC=60,eta=1.0,D_plan=6,fi=6,si=1,
              alloc={1:20,3:20,5:20},
              ms_theta=[1/3,2/3,1.0],ms_e=[1,3,5],ms_phi=[1/3,1/3,1/3],
              CP=90,rho=0.10,alpha=0.20,A=18,delta_rec=0.30,psi=0.10,
              mu=0.50,Omega=2,tau_tol=2)
    return make_case(
        meta={"id":"SP-D","group":"G4","Zstar":27.227,"tol_Z":1.0,"outcome":"completed"},
        params={"BAC":60,"CP":90,"rho":0.10,"alpha":0.20,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=18)

def spE():
    ps = dict(BAC=100,eta=1.0,D_plan=8,fi=8,si=1,
              alloc={t:4 for t in range(1,9)},
              ms_theta=[0.5,1.0],ms_e=[1,5],ms_phi=[0.5,0.5],
              CP=100,mu=0.05,Omega=1,tau_tol=3)
    return make_case(
        meta={"id":"SP-E","group":"G5","Zstar":None,"tol_Z":None,"outcome":"active"},
        params={"BAC":100,"eta":1.0,"tau_tol":3,"H":8},
        projects_spec=[ps], B0=300, H=8)

def spF():
    ps = dict(BAC=100,eta=0.5,D_plan=20,fi=20,si=1,
              alloc={t:20 for t in range(1,21)},
              ms_theta=[0.5,1.0],ms_e=[1,20],ms_phi=[0.5,0.5],
              CP=100,mu=0.05,Omega=1,tau_tol=3)
    return make_case(
        meta={"id":"SP-F","group":"G5","Zstar":None,"tol_Z":None,"outcome":"active"},
        params={"BAC":100,"eta":0.5,"tau_tol":3,"H":20,"B0":500},
        projects_spec=[ps], B0=500, H=20)

def spG():
    ps = dict(BAC=100,eta=1.0,D_plan=12,fi=12,si=1,
              alloc={1:10,2:10,3:80,**{t:10 for t in range(4,13)}},
              ms_theta=[0.5,1.0],ms_e=[1,12],ms_phi=[0.5,0.5],
              CP=100,mu=0.50,Omega=1,tau_tol=3,
              eta_schedule={1:0.3,2:0.3,3:1.0,**{t:0.3 for t in range(4,13)}})
    return make_case(
        meta={"id":"SP-G","group":"G5","Zstar":None,"tol_Z":None,"outcome":"active"},
        params={"BAC":100,"tau_tol":3,"H":12},
        projects_spec=[ps], B0=300, H=12)

def sp5():
    ps = dict(BAC=100,eta=0.0,D_plan=6,fi=6,si=1,alloc={},
              ms_theta=[0.5,1.0],ms_e=[1,4],ms_phi=[0.5,0.5],
              CP=100,alpha=0.20,A=20,mu=0.10,Omega=1,tau_tol=3,term_at=4)
    return make_case(
        meta={"id":"SP-5","group":"G6","Zstar":2.85,"tol_Z":0.5,"outcome":"terminated"},
        params={"BAC":100,"eta":0.0,"A":20,"tau_tol":3,"H":6},
        projects_spec=[ps], B0=300, H=6, advance=20)

def spH():
    ps = dict(BAC=100,eta=0.5,D_plan=10,fi=10,si=1,alloc={1:60},
              ms_theta=[0.30,1.0],ms_e=[1,10],ms_phi=[0.40,0.60],
              CP=100,alpha=0.10,A=10,mu=0.05,Omega=0,tau_tol=2,term_at=3)
    return make_case(
        meta={"id":"SP-H","group":"G6","Zstar":7.075,"tol_Z":1.0,"outcome":"terminated"},
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
        meta={"id":"SP-I","group":"G6","Zstar":None,"tol_Z":None,"outcome":"terminated"},
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
        meta={"id":"MP-3","group":"G7","Zstar":41.925,"tol_Z":1.5,"outcome":"mixed"},
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
        meta={"id":"MP-A","group":"G7","Zstar":52.0,"tol_Z":5.0,"outcome":"both_completed"},
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
        meta={"id":"MP-B","group":"G7","Zstar":31.005,"tol_Z":1.5,"outcome":"mixed"},
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
        meta={"id":"MP-2","group":"G8","Zstar":37.79,"tol_Z":1.5,"outcome":"mixed"},
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
        meta={"id":"MP-4","group":"G8","Zstar":40.95,"tol_Z":1.5,"outcome":"low_terminated"},
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
        meta={"id":"MP-C","group":"G8","Zstar":28.54,"tol_Z":1.5,"outcome":"lo_terminated"},
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
        meta={"id":"MP-D","group":"G8","Zstar":85.999,"tol_Z":5.0,"outcome":"all_completed"},
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
        meta={"id":"MP-1","group":"G9","Zstar":34.153,"tol_Z":1.5,"outcome":"both_completed"},
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
        meta={"id":"MP-E","group":"G9","Zstar":71.937,"tol_Z":2.0,"outcome":"all_completed"},
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
        meta={"id":"SP-M","group":"G2","Zstar":39.298,"tol_Z":1.5,"outcome":"both_completed"},
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
        meta={"id":"SP-Mp","group":"G2","Zstar":30.0,"tol_Z":1.0,"outcome":"good_completed"},
        params={"n":2,"B0":90,"H":6},
        projects_spec=[pGood,pBad], B0=90, H=6)


ALL_CASES = {
    "SP-1":  sp1(),  "SP-3":  sp3(),  "SP-J":  spJ(),
    "SP-4":  sp4(),  "MP-K":  mpK(),  "MP-Kp": mpKp(),
    "SP-L":  spL(),  "SP-Lp": spLp(), "SP-M":  spM(),  "SP-Mp": spMp(),
    "SP-2":  sp2(),  "SP-2a": sp2a(), "SP-2b": sp2b(),
    "SP-2c": sp2c(), "SP-2d": sp2d(),
    "SP-A":  spA(),  "SP-B":  spB(),  "SP-C":  spC(),  "SP-D":  spD(),
    "SP-E":  spE(),  "SP-F":  spF(),  "SP-G":  spG(),
    "SP-5":  sp5(),  "SP-H":  spH(),  "SP-I":  spI(),
    "MP-3":  mp3(),  "MP-A":  mpA(),  "MP-B":  mpB(),
    "MP-2":  mp2(),  "MP-4":  mp4(),  "MP-C":  mpC(),
    "MP-D":  mpD(),  "MP-1":  mp1(),  "MP-E":  mpE(),
}

if __name__ == "__main__":
    HERE = Path(__file__).parent
    out_path = HERE / "cases.json"
    with open(out_path,"w") as f:
        json.dump(ALL_CASES, f, indent=2)
    print(f"Written {out_path}  ({len(ALL_CASES)} cases)")
    # quick sanity
    for k,c in ALL_CASES.items():
        n   = len(c["projects"])
        tls = [len(p["timeline"]) for p in c["projects"]]
        zs  = c["meta"]["Zstar"]
        print(f"  {k:8s}  n={n}  rows={tls}  Z*={zs}")