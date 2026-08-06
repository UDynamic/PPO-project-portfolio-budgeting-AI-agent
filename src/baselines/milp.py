"""
milp.py
=======
Level-1 Deterministic Full-Foresight MILP — PPM Budget Allocation
Implements Section 3 (Level 1) of the paper.

Mathematical Formulation
------------------------
All η_{i,t} are known constants at Level 1. Because η is deterministic,
every EVM signal (SPI, CPI, EAC, Δf) is a deterministic function of the
decision variables x_{i,t}, and both termination conditions reduce to
LINEAR inequalities in x.  No approximation or proxy is needed.

Condition 1 (Schedule, Eq.5):  Δf_{i,t} > Ω_i
  Equivalent (when rem_{i,t} > 0 and t < f_i + Ω_i):
      P_{i,t}  <  THRESH1_{i,t}
  where THRESH1_{i,t} = rem_{i,t} · P_plan_{i,t} / (f_i + Ω_i − t)
  is a pre-computable constant.

Condition 2 (Cost, Eq.6):  EAC_{i,t} > (1+μ_i)·BAC_i
  EAC = ACWP/P  →  ACWP > (1+μ)·BAC·P  →  LINEAR:
      Σ_{τ≤t} x_{i,τ}·[1 − (1+μ_i)·η_{i,τ}]  >  0
  Special case: ACWP=0 and t > s_i (zero spend since start) also fires Cond2.

Both conditions fire simultaneously for τ_i^{tol} consecutive periods → termination.

Termination settlement (Eq.8), linearized via McCormick:
  R^term_{i,t} = [P_{i,t}·CP_i·(1−ρ_i) − outstanding_advance_{i,t}] · δalive_{i,t}
  where δalive_{i,t} = alive_{i,t-1} − alive_{i,t} ∈ {0,1}
  P_{i,t}·δalive_{i,t}   → auxiliary w_{i,t}   (McCormick, exact for binary·[0,1])
  u_{i,j,t}·δalive_{i,t} → auxiliary v_{i,j,t} (binary AND)

Variables
---------
  x[i,t]       continuous ≥ 0        budget to project i at t
  u[i,j,t]     binary                milestone j of project i certified BY t
  P[i,t]       continuous [0,1]      cumulative actual progress
  B[t]         continuous ≥ 0        cash balance
  alive[i,t]   binary                1 if project i not yet terminated at t
  da[i,t]      binary = alive[t-1]-alive[t]  termination indicator
  c[i,t]       integer [0, τ_max]    cure-period counter
  b1[i,t]      binary                Cond1 active (schedule behind tolerance)
  b2[i,t]      binary                Cond2 active (cost trajectory unacceptable)
  b2s[i,t]     binary                Cond2 from inefficient spend (L_{i,t} > 0)
  zs[i,t]      binary                Cond2 from zero-spend stagnation (t>s, ACWP=0)
  q[i,t]       binary                both conditions simultaneously active
  w[i,t]       continuous [0,1]      = P[i,t] · da[i,t]  (McCormick)
  v[i,j,t]     binary                = u[i,j,t] · da[i,t] (binary AND)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Optional, Tuple

import pulp
from pulp import (
    PULP_CBC_CMD, LpBinary, LpMaximize, LpProblem,
    LpVariable, lpSum, value,
)
from scipy.special import betainc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: dict = dict(
    seed=42, n_projects=5, n_min=5, n_max=10, horizon=24,
    gamma=0.97, b1_ratio=(0.35, 0.55),
    s_max_frac=0.20, dur_min_frac=0.33, dur_max_frac=0.80,
    a_range=(1.5, 3.5), b_range=(2.0, 5.0),
    BAC_range=(50_000, 300_000), pi_range=(0.05, 0.20),
    rho_range=(0.05, 0.10), delta_rec=0.25, psi=0.10,
    n_ms_range=(3, 5), ms_min_gap=0.15, ms_lo=0.20, ms_hi=0.85,
    Omega_range=(2, 5), mu_range=(0.10, 0.30), tau_tol_range=(1, 3),
    eta_base_lo=0.65, eta_base_hi=1.00, eta_noise=0.03,
    eta_min=0.50, eta_max=1.10,
    solver_time_limit=300,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PURE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def scurve(tau: float, a: float, b: float) -> float:
    tc = max(0.0, min(1.0, tau))
    if tc <= 0.0: return 0.0
    if tc >= 1.0: return 1.0
    return float(betainc(a, b, tc))


def planned_progress(proj: dict, H: int) -> Dict[int, float]:
    """P_{i,t}^plan for t in 1..H using S-curve parameterisation (Eq.2)."""
    s, f, D = proj["s_i"], proj["f_i"], proj["D_plan"]
    a, b    = proj["a_i"], proj["b_i"]
    out: Dict[int, float] = {}
    for t in range(1, H + 1):
        if   t < s: out[t] = 0.0
        elif t > f: out[t] = 1.0
        else:       out[t] = scurve((t - s + 1) / D, a, b)
    return out


def derive_alpha(phi: List[float], drec: float, psi: float) -> float:
    """
    Advance ratio α so that Σ recovery deductions = α·CP exactly.
    Deduction applies at rate drec to milestones where cum_φ ≥ ψ.
    """
    cum = rec_frac = 0.0
    for phi_j in phi:
        cum += phi_j
        if cum >= psi:
            rec_frac += phi_j
    return drec * rec_frac


def compute_R_net(proj: dict) -> Tuple[List[float], float]:
    """
    Per-milestone net payment and retention release.

    R_{i,j}^net = φ_j·CP·(1−ρ) − δ_rec·φ_j·CP·1{cum_φ ≥ ψ}
    Payment identity: A + Σ R_net + R_ret = CP
    """
    CP, rho   = proj["CP_i"], proj["rho_i"]
    drec, psi = proj["delta_rec_i"], proj["psi_i"]
    A_i       = proj["alpha_i"] * CP
    cum       = adv_recov = 0.0
    R_net: List[float] = []
    for phi_j in proj["phi"]:
        cum += phi_j
        gross   = phi_j * CP
        ret_ded = rho * gross
        if cum >= psi:
            candidate = drec * gross
            remaining = max(0.0, A_i - adv_recov)
            rec_ded   = min(candidate, remaining)
            adv_recov += rec_ded
        else:
            rec_ded = 0.0
        R_net.append(gross - ret_ded - rec_ded)
    return R_net, rho * CP


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PORTFOLIO GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_project(idx: int, H: int, cfg: dict, rng: random.Random) -> dict:
    s_i    = rng.randint(1, max(1, int(H * cfg["s_max_frac"])))
    dur    = rng.randint(int(H * cfg["dur_min_frac"]), int(H * cfg["dur_max_frac"]))
    f_i    = min(s_i + dur - 1, H)
    D_plan = f_i - s_i + 1
    a_i    = rng.uniform(*cfg["a_range"])
    b_i    = rng.uniform(*cfg["b_range"])
    BAC_i  = rng.uniform(*cfg["BAC_range"])
    pi_i   = rng.uniform(*cfg["pi_range"])
    CP_i   = BAC_i * (1.0 + pi_i)
    rho_i  = rng.uniform(*cfg["rho_range"])
    drec_i = cfg["delta_rec"]
    psi_i  = cfg["psi"]
    n_ms   = rng.randint(*cfg["n_ms_range"])
    pts    = sorted(rng.uniform(cfg["ms_lo"], cfg["ms_hi"]) for _ in range(n_ms - 1))
    spaced: List[float] = []
    for v in pts:
        spaced.append(v if not spaced else max(v, spaced[-1] + cfg["ms_min_gap"]))
    theta  = [min(v, 0.95) for v in spaced] + [1.0]
    M_i    = len(theta)
    phi    = [1.0 / M_i] * M_i
    alpha_i = derive_alpha(phi, drec_i, psi_i)
    Omega_i   = rng.randint(*cfg["Omega_range"])
    mu_i      = rng.uniform(*cfg["mu_range"])
    tau_tol_i = rng.randint(*cfg["tau_tol_range"])
    e_i       = [s_i] * M_i
    lo, hi, noise = cfg["eta_base_lo"], cfg["eta_base_hi"], cfg["eta_noise"]
    eta: Dict[int, float] = {}
    for t in range(1, H + 1):
        if s_i <= t <= f_i:
            tau_n  = (t - s_i) / max(D_plan - 1, 1)
            base   = lo + (hi - lo) * math.sin(math.pi * tau_n)
            eta[t] = max(cfg["eta_min"], min(cfg["eta_max"],
                         base + rng.uniform(-noise, noise)))
        else:
            eta[t] = 0.0
    return dict(
        idx=idx, s_i=s_i, f_i=f_i, D_plan=D_plan,
        a_i=a_i, b_i=b_i,
        BAC_i=BAC_i, pi_i=pi_i, CP_i=CP_i,
        alpha_i=alpha_i, delta_rec_i=drec_i, psi_i=psi_i, rho_i=rho_i,
        M_i=M_i, theta=theta, phi=phi, e_i=e_i,
        Omega_i=Omega_i, mu_i=mu_i, tau_tol_i=tau_tol_i, eta=eta,
    )


def generate_portfolio(cfg: dict) -> dict:
    rng      = random.Random(cfg["seed"])
    n        = cfg.get("n_projects") or rng.randint(cfg["n_min"], cfg["n_max"])
    H        = cfg["horizon"]
    projects = [generate_project(i, H, cfg, rng) for i in range(n)]
    total_BAC = sum(p["BAC_i"] for p in projects)
    lo, hi   = cfg["b1_ratio"] if isinstance(cfg["b1_ratio"], tuple) \
               else (cfg["b1_ratio"], cfg["b1_ratio"])
    B1 = rng.uniform(lo, hi) * total_BAC
    return dict(n=n, H=H, projects=projects, B1=B1,
                gamma=cfg["gamma"], total_BAC=total_BAC, cfg=cfg)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PRE-COMPUTE TERMINATION THRESHOLD CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

def _precompute_term_constants(proj: dict, plan_prog: Dict[int, float], H: int):
    """
    For each period t in the activity window of project i, compute:

      thresh1[t]  :  P_{i,t} < thresh1[t]  ⟺  Cond1 fires
                     None if Cond1 is always-True or always-False at t (handled via flags)
      c1_always[t]:  True  → Cond1 fires regardless of x
      c1_never[t] :  True  → Cond1 cannot fire at t

      coeff2[t]   :  coefficient of x_{i,t} in L_{i,t} = Σ_{τ≤t} coeff2[τ]*x_{i,τ}
                     Cond2 fires when L_{i,t} > 0  (OR when ACWP=0, t>s)
    """
    s, f     = proj["s_i"], proj["f_i"]
    D_plan   = proj["D_plan"]
    Omega    = proj["Omega_i"]
    mu       = proj["mu_i"]
    eta      = proj["eta"]

    thresh1:   Dict[int, Optional[float]] = {}
    c1_always: Dict[int, bool]            = {}
    c1_never:  Dict[int, bool]            = {}
    coeff2:    Dict[int, float]           = {}

    for t in range(s, min(f, H) + 1):
        # ── Condition 1 threshold ──────────────────────────────────────────
        rem      = D_plan - (t - s)       # remaining planned periods
        P_plan_t = plan_prog[t]
        RHS1     = (f + Omega) - t        # time budget remaining inside tolerance

        if rem <= 0:
            # At or beyond planned finish: Δf = 0 ≤ Ω → never fires
            thresh1[t]   = None
            c1_always[t] = False
            c1_never[t]  = True
        elif RHS1 <= 0:
            # Even with SPI=∞, we are past tolerance → always fires
            thresh1[t]   = None
            c1_always[t] = True
            c1_never[t]  = False
        elif P_plan_t <= 0:
            # P_plan=0 (project just started, S-curve hasn't moved):
            # SPI = P/0 → undefined; treat as no schedule signal yet
            thresh1[t]   = None
            c1_always[t] = False
            c1_never[t]  = True
        else:
            # Normal case: Cond1 ⟺ P_{i,t} < rem * P_plan_t / RHS1
            th = rem * P_plan_t / RHS1
            thresh1[t]   = th
            c1_always[t] = False
            c1_never[t]  = False

        # ── Condition 2 coefficient ────────────────────────────────────────
        # Cond2: ACWP_{i,t} > (1+μ)·BAC·P_{i,t}
        #   ⟺ Σ_{τ≤t} x_{i,τ} > (1+μ) · Σ_{τ≤t} η_{i,τ}·x_{i,τ}
        #   ⟺ Σ_{τ≤t} [1 − (1+μ)·η_{i,τ}] · x_{i,τ}  >  0
        coeff2[t] = 1.0 - (1.0 + mu) * eta.get(t, 0.0)

    return thresh1, c1_always, c1_never, coeff2


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PRECOMPUTE ZERO-SPEND TERMINATION  (for logging)
# ═════════════════════════════════════════════════════════════════════════════

def _precompute_zero_alloc_termination(portfolio: dict) -> Dict[int, Optional[int]]:
    """
    Period at which project i terminates under zero spend.
    Under x=0: both conditions fire from t=s_i → counter decrements each period
    → termination at t = s_i + tau_tol_i.
    """
    result: Dict[int, Optional[int]] = {}
    for i, p in enumerate(portfolio["projects"]):
        t_term = p["s_i"] + p["tau_tol_i"]
        result[i] = t_term if t_term <= portfolio["H"] else None
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MILP  —  build_and_solve
# ═════════════════════════════════════════════════════════════════════════════

def build_and_solve(portfolio: dict) -> dict:
    """
    Build and solve the Level-1 MILP with exact endogenous termination.

    Both EVM termination conditions are linearized exactly using the
    deterministic η schedule.  No proxy variables needed.
    """
    n, H      = portfolio["n"], portfolio["H"]
    projects  = portfolio["projects"]
    B1        = portfolio["B1"]
    gamma     = portfolio["gamma"]
    T         = list(range(1, H + 1))
    cfg       = portfolio.get("cfg", {})
    time_limit = cfg.get("solver_time_limit", 300)

    # ── Pre-compute helpers ──────────────────────────────────────────────────
    plan_prog: Dict[int, Dict[int, float]] = {
        i: planned_progress(p, H) for i, p in enumerate(projects)
    }

    act: Dict[Tuple[int, int], int] = {
        (i, t): int(projects[i]["s_i"] <= t <= projects[i]["f_i"])
        for i in range(n) for t in T
    }

    A: Dict[int, float] = {
        i: projects[i]["alpha_i"] * projects[i]["CP_i"] for i in range(n)
    }

    R_net: Dict[Tuple[int, int], float] = {}
    R_ret: Dict[int, float]             = {}
    for i, p in enumerate(projects):
        rn, rr = compute_R_net(p)
        for j, v in enumerate(rn):
            R_net[i, j] = v
        R_ret[i] = rr

    # Payment identity check
    for i, p in enumerate(projects):
        total = A[i] + sum(R_net[i, j] for j in range(p["M_i"])) + R_ret[i]
        assert abs(total - p["CP_i"]) < 1.0, \
            f"Payment identity failed P{i}: {total:.4f} vs {p['CP_i']:.4f}"

    # Per-project recovery amount per milestone (for settlement linearization)
    # rec_j[i,j] = deduction applied to advance when milestone j is certified
    rec_j: Dict[Tuple[int, int], float] = {}
    for i, p in enumerate(projects):
        CP, rho   = p["CP_i"], p["rho_i"]
        drec, psi = p["delta_rec_i"], p["psi_i"]
        A_i       = A[i]
        cum = adv = 0.0
        for j, phi_j in enumerate(p["phi"]):
            cum += phi_j
            if cum >= psi:
                candidate = drec * phi_j * CP
                remaining = max(0.0, A_i - adv)
                ded       = min(candidate, remaining)
                adv      += ded
            else:
                ded = 0.0
            rec_j[i, j] = ded

    # Per-project termination threshold constants
    term_const: Dict[int, tuple] = {}
    for i, p in enumerate(projects):
        term_const[i] = _precompute_term_constants(p, plan_prog[i], H)

    zero_term = _precompute_zero_alloc_termination(portfolio)

    # ── Constants ────────────────────────────────────────────────────────────
    BIG_M   = max(p["BAC_i"] for p in projects) * 2.0
    tau_max = max(p["tau_tol_i"] for p in projects) + 1
    EPS     = 1e-5   # strict-inequality epsilon

    # ── PuLP variables ───────────────────────────────────────────────────────
    prob = LpProblem("PPM_L1_MILP", LpMaximize)

    x = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0.0)
         for i in range(n) for t in T}

    u = {(i, j, t): LpVariable(f"u_{i}_{j}_{t}", cat=LpBinary)
         for i in range(n)
         for j in range(projects[i]["M_i"])
         for t in T}

    P = {(i, t): LpVariable(f"P_{i}_{t}", lowBound=0.0, upBound=1.0)
         for i in range(n) for t in T}

    B = {t: LpVariable(f"B_{t}", lowBound=0.0) for t in T}

    # alive[i,t] = 1 if project i has NOT yet terminated at start of period t
    alive = {(i, t): LpVariable(f"alive_{i}_{t}", cat=LpBinary)
             for i in range(n) for t in T}

    # da[i,t] = alive[i,t-1] - alive[i,t] = 1 exactly in the termination period
    da = {(i, t): LpVariable(f"da_{i}_{t}", cat=LpBinary)
          for i in range(n) for t in T}

    # Cure-period counter
    c = {(i, t): LpVariable(f"c_{i}_{t}", lowBound=0, upBound=tau_max, cat="Integer")
         for i in range(n) for t in T}

    # b1[i,t] = 1 iff Condition 1 active (schedule behind tolerance)
    b1 = {(i, t): LpVariable(f"b1_{i}_{t}", cat=LpBinary)
          for i in range(n) for t in T}

    # b2s[i,t] = 1 iff L_{i,t} > 0  (inefficient-spend component of Cond2)
    b2s = {(i, t): LpVariable(f"b2s_{i}_{t}", cat=LpBinary)
           for i in range(n) for t in T}

    # zs[i,t] = 1 iff ACWP_{i,t}=0 AND t > s_i  (zero-spend stagnation)
    zs = {(i, t): LpVariable(f"zs_{i}_{t}", cat=LpBinary)
          for i in range(n) for t in T}

    # b2[i,t] = b2s OR zs  (either component triggers Cond2)
    b2 = {(i, t): LpVariable(f"b2_{i}_{t}", cat=LpBinary)
          for i in range(n) for t in T}

    # q[i,t] = b1 AND b2  (both conditions simultaneously active)
    q = {(i, t): LpVariable(f"q_{i}_{t}", cat=LpBinary)
         for i in range(n) for t in T}

    # w[i,t] = P[i,t] * da[i,t]  (McCormick linearization, P in [0,1])
    w = {(i, t): LpVariable(f"w_{i}_{t}", lowBound=0.0, upBound=1.0)
         for i in range(n) for t in T}

    # v[i,j,t] = u[i,j,t] * da[i,t]  (binary AND)
    v = {(i, j, t): LpVariable(f"v_{i}_{j}_{t}", cat=LpBinary)
         for i in range(n)
         for j in range(projects[i]["M_i"])
         for t in T}

    # ── Inflow expression ────────────────────────────────────────────────────
    def V_expr(t: int):
        """
        Period-t inflow V_t^{det}:
          + A_i at t = s_i
          + R_{i,j}^net + R_i^ret at milestone first certification
          + R^term_{i,t} = [P·CP·(1−ρ) − A_i + Σ_j rec_j·u_j] · da[i,t]
                         = CP·(1−ρ)·w[i,t] − A_i·da[i,t] + Σ_j rec_j·v[i,j,t]
        """
        terms: list = []
        for i, p in enumerate(projects):
            CP, rho = p["CP_i"], p["rho_i"]
            M       = p["M_i"]

            # Advance payment
            if t == p["s_i"]:
                terms.append(A[i])

            # Net milestone payments (+ retention at final milestone)
            for j in range(M):
                u_prev = u[i, j, t - 1] if t > 1 else 0
                pay    = R_net[i, j] + (R_ret[i] if j == M - 1 else 0.0)
                terms.append(pay * (u[i, j, t] - u_prev))

            # Termination settlement (linearized)
            if t > 1:
                # CP·(1−ρ)·w − A·da + Σ rec_j·v
                terms.append(CP * (1.0 - rho) * w[i, t])
                terms.append(-A[i] * da[i, t])
                for j in range(M):
                    terms.append(rec_j[i, j] * v[i, j, t])

        return lpSum(terms)

    # ── Objective ────────────────────────────────────────────────────────────
    prob += lpSum(
        gamma ** (t - 1) * (V_expr(t) - lpSum(act[i, t] * x[i, t] for i in range(n)))
        for t in T
    ), "Objective"

    # ── Cash balance ─────────────────────────────────────────────────────────
    prob += B[1] == B1, "B_init"
    for t in T[:-1]:
        prob += (
            B[t + 1] == B[t]
                      - lpSum(act[i, t] * x[i, t] for i in range(n))
                      + V_expr(t),
            f"Bal_{t}",
        )
    for t in T:
        prob += (
            lpSum(act[i, t] * x[i, t] for i in range(n)) <= B[t],
            f"Budget_{t}",
        )

    # ── Progress recursion ───────────────────────────────────────────────────
    for i, p in enumerate(projects):
        BAC, eta_p = p["BAC_i"], p["eta"]
        for t in T:
            P_prev = P[i, t - 1] if t > 1 else 0.0
            prob += (
                P[i, t] == P_prev + eta_p[t] * x[i, t] / BAC,
                f"Prog_{i}_{t}",
            )
            if act[i, t] == 0:
                prob += (x[i, t] == 0, f"Inact_{i}_{t}")

    # ── alive initialisation, monotonicity, da definition ───────────────────
    for i, p in enumerate(projects):
        # Before and at start: alive
        for t in T:
            if t <= p["s_i"]:
                prob += (alive[i, t] == 1, f"AliveStart_{i}_{t}")
                prob += (da[i, t] == 0,    f"DAStart_{i}_{t}")

        # Monotone: alive can only go 1→0 once
        for t in range(p["s_i"] + 1, H + 1):
            prob += (alive[i, t] <= alive[i, t - 1], f"AliveMono_{i}_{t}")

        # da[i,t] = alive[i,t-1] - alive[i,t]  (binary, in [0,1] by monotonicity)
        for t in range(p["s_i"] + 1, H + 1):
            prob += (da[i, t] == alive[i, t - 1] - alive[i, t], f"DA_{i}_{t}")

    # No allocation or certification when terminated
    for i in range(n):
        for t in T:
            prob += (x[i, t] <= BIG_M * alive[i, t],  f"XAlive_{i}_{t}")

    # ── Condition 1: b1[i,t] ────────────────────────────────────────────────
    for i, p in enumerate(projects):
        thresh1, c1_always, c1_never, _ = term_const[i]
        for t in T:
            if t < p["s_i"] or t > p["f_i"]:
                # Outside activity window: Cond1 never applies
                prob += (b1[i, t] == 0, f"B1Off_{i}_{t}")
                continue

            if c1_never.get(t, False):
                prob += (b1[i, t] == 0, f"B1Never_{i}_{t}")

            elif c1_always.get(t, False):
                # Cond1 fires unconditionally at this period
                # But only when project is still active (alive):
                prob += (b1[i, t] == alive[i, t], f"B1Always_{i}_{t}")

            else:
                th = thresh1[t]
                # b1=1 ⟺ P[i,t] < th
                # b1=0 → P[i,t] >= th:   P >= th - th*b1  (since P in [0,1], th>0)
                #   i.e.  P[i,t] + th*(b1[i,t]-1) >= 0
                #   i.e.  P[i,t] >= th - th*b1   →  P >= th*(1-b1)
                prob += (P[i, t] >= th * (1 - b1[i, t]) - (1 - alive[i, t]),
                         f"B1_lb_{i}_{t}")
                # b1=1 → P[i,t] < th:   P <= th - eps + (1-b1)*M_prog
                prob += (P[i, t] <= (th - EPS) + (1 - b1[i, t]) * 1.0
                                  + (1 - alive[i, t]),
                         f"B1_ub_{i}_{t}")
                # b1 can only be 1 when alive
                prob += (b1[i, t] <= alive[i, t], f"B1Alive_{i}_{t}")

    # ── Condition 2: b2s, zs, b2 ────────────────────────────────────────────
    for i, p in enumerate(projects):
        _, _, _, coeff2 = term_const[i]
        s_i = p["s_i"]

        for t in T:
            if t < s_i or t > p["f_i"]:
                prob += (b2s[i, t] == 0, f"B2SOff_{i}_{t}")
                prob += (zs[i, t]  == 0, f"ZSOff_{i}_{t}")
                prob += (b2[i, t]  == 0, f"B2Off_{i}_{t}")
                continue

            # L_{i,t} = sum_{τ=s_i}^{t} coeff2[τ] * x[i,τ]
            L_expr = lpSum(coeff2[tau] * x[i, tau]
                           for tau in range(s_i, t + 1)
                           if tau in coeff2)

            # b2s[i,t] = 1 iff L_{i,t} > 0
            # b2s=1 → L >= EPS:   L >= EPS - BIG_M*(1-b2s)
            prob += (L_expr >= EPS - BIG_M * (1 - b2s[i, t]),  f"B2S_lb_{i}_{t}")
            # b2s=0 → L <= 0:     L <= BIG_M * b2s
            prob += (L_expr <= BIG_M * b2s[i, t],              f"B2S_ub_{i}_{t}")
            prob += (b2s[i, t] <= alive[i, t],                  f"B2SAlive_{i}_{t}")

            # zs[i,t] = 1 iff ACWP_{i,t}=0 AND t > s_i
            # ACWP_{i,t} = sum_{τ=s_i}^{t} x[i,τ]
            if t == s_i:
                # At project start: no stagnation signal yet
                prob += (zs[i, t] == 0, f"ZSStart_{i}_{t}")
            else:
                ACWP_expr = lpSum(x[i, tau] for tau in range(s_i, t + 1))
                # zs=1 → ACWP=0:  ACWP <= BIG_M*(1-zs)
                prob += (ACWP_expr <= BIG_M * (1 - zs[i, t]),  f"ZS_ub_{i}_{t}")
                # zs=0 → ACWP>0:  ACWP >= EPS*(1-zs) ... but ACWP could be 0 with zs=0 at s_i
                # Better: zs=0 forces ACWP >= EPS only indirectly via objective
                # Just use upper bound; lower bound via b2s
                prob += (zs[i, t] <= alive[i, t], f"ZSAlive_{i}_{t}")

            # b2[i,t] = b2s[i,t] OR zs[i,t]
            prob += (b2[i, t] >= b2s[i, t],               f"B2_lb1_{i}_{t}")
            prob += (b2[i, t] >= zs[i, t],                f"B2_lb2_{i}_{t}")
            prob += (b2[i, t] <= b2s[i, t] + zs[i, t],   f"B2_ub_{i}_{t}")
            prob += (b2[i, t] <= alive[i, t],              f"B2Alive_{i}_{t}")

    # ── q[i,t] = b1 AND b2 ──────────────────────────────────────────────────
    for i in range(n):
        for t in T:
            prob += (q[i, t] <= b1[i, t],               f"Q_b1_{i}_{t}")
            prob += (q[i, t] <= b2[i, t],               f"Q_b2_{i}_{t}")
            prob += (q[i, t] >= b1[i, t] + b2[i, t] - 1, f"Q_and_{i}_{t}")

    # ── Cure-period counter c[i,t] ───────────────────────────────────────────
    # c[i,t] = c[i,t-1]+1 if q=1; = 0 if q=0
    for i, p in enumerate(projects):
        tau_tol = p["tau_tol_i"]
        for t in T:
            if t <= p["s_i"]:
                prob += (c[i, t] == 0, f"CStart_{i}_{t}")
            else:
                c_prev = c[i, t - 1]
                prob += (c[i, t] <= c_prev + 1,                             f"C_ub1_{i}_{t}")
                prob += (c[i, t] <= tau_max * q[i, t],                      f"C_ub2_{i}_{t}")
                prob += (c[i, t] >= c_prev + 1 - tau_max * (1 - q[i, t]),  f"C_lb_{i}_{t}")

            # Termination trigger: c[i,t] >= tau_tol → alive[i,t+1] = 0
            # Equivalently: alive[i,t+1] * tau_tol <= tau_tol - 1 when c >= tau_tol
            # Big-M form: c[i,t] <= (tau_tol-1) + tau_max * alive[i,t+1]
            if t < H:
                prob += (
                    c[i, t] <= (tau_tol - 1) + tau_max * alive[i, t + 1],
                    f"TermTrigger_{i}_{t}",
                )

    # ── Milestone certification ──────────────────────────────────────────────
    for i, p in enumerate(projects):
        M   = p["M_i"]
        e_i = p.get("e_i", [p["s_i"]] * M)
        for j in range(M):
            theta_j = p["theta"][j]
            e_ij    = e_i[j]
            for t in T:
                if t < e_ij:
                    prob += (u[i, j, t] == 0, f"EarlyWin_{i}_{j}_{t}")
                    continue
                # u[i,j,t]=1 ⟺ P[i,t] >= theta_j  AND  alive[i,t]
                prob += (
                    P[i, t] >= theta_j - (1 - u[i, j, t]),
                    f"CertLB_{i}_{j}_{t}",
                )
                prob += (
                    P[i, t] <= (theta_j - EPS) + (1 - u[i, j, t]) + (1 - alive[i, t]),
                    f"CertUB_{i}_{j}_{t}",
                )
                prob += (u[i, j, t] <= alive[i, t], f"UCertAlive_{i}_{j}_{t}")

        # Monotone
        for j in range(M):
            for t in range(2, H + 1):
                prob += (u[i, j, t] >= u[i, j, t - 1], f"UMono_{i}_{j}_{t}")

        # Ordered
        for j in range(M - 1):
            for t in T:
                prob += (u[i, j + 1, t] <= u[i, j, t], f"UOrd_{i}_{j}_{t}")

        # No certification before project start
        for j in range(M):
            for t in T:
                if t < p["s_i"]:
                    prob += (u[i, j, t] == 0, f"UPreStart_{i}_{j}_{t}")

    # ── McCormick linearization: w[i,t] = P[i,t] · da[i,t] ─────────────────
    # P in [0,1], da in {0,1}  → exact McCormick
    for i in range(n):
        for t in T:
            prob += (w[i, t] >= 0,                        f"W_lb0_{i}_{t}")
            prob += (w[i, t] >= P[i, t] + da[i, t] - 1,  f"W_lb1_{i}_{t}")
            prob += (w[i, t] <= P[i, t],                   f"W_ub1_{i}_{t}")
            prob += (w[i, t] <= da[i, t],                  f"W_ub2_{i}_{t}")

    # ── Binary AND: v[i,j,t] = u[i,j,t] · da[i,t] ──────────────────────────
    for i, p in enumerate(projects):
        for j in range(p["M_i"]):
            for t in T:
                prob += (v[i, j, t] <= u[i, j, t],                     f"V_ub1_{i}_{j}_{t}")
                prob += (v[i, j, t] <= da[i, t],                        f"V_ub2_{i}_{j}_{t}")
                prob += (v[i, j, t] >= u[i, j, t] + da[i, t] - 1,     f"V_lb_{i}_{j}_{t}")

    # ── Solve ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"SOLVING MILP  (n={n}, H={H}, time_limit={time_limit}s)")
    print("=" * 72)
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)
    obj_val = value(prob.objective) or 0.0
    print(f"Status: {pulp.LpStatus[prob.status]}   Obj: {obj_val:.4f}")

    return dict(
        status=pulp.LpStatus[prob.status],
        obj=obj_val,
        prob=prob,
        x=x, u=u, P=P, B=B,
        alive=alive, da=da, c=c,
        b1=b1, b2=b2, b2s=b2s, zs=zs, q=q,
        w=w, v=v,
        act=act, A=A, R_net=R_net, R_ret=R_ret,
        plan_prog=plan_prog,
        zero_term=zero_term,
        term_const=term_const,
        term_settlements={},   # API compatibility
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6.  STATE RECORD SYSTEM  —  build_records
# ═════════════════════════════════════════════════════════════════════════════

def build_records(portfolio: dict, sol: dict) -> dict:
    """
    Forward-simulate the MILP solution to produce exact per-period state records.
    Uses extracted x values and deterministic η to compute all EVM signals,
    termination counters, and cash-flow events.
    """
    n, H      = portfolio["n"], portfolio["H"]
    projects  = portfolio["projects"]
    B1        = portfolio["B1"]
    T         = list(range(1, H + 1))

    x_val: Dict[Tuple[int, int], float] = {
        k: max(0.0, value(v) or 0.0) for k, v in sol["x"].items()
    }
    u_val: Dict[Tuple[int, int, int], int] = {
        k: int(round(value(v) or 0.0)) for k, v in sol["u"].items()
    }
    P_lp:  Dict[Tuple[int, int], float] = {
        k: (value(v) or 0.0) for k, v in sol["P"].items()
    }
    B_lp:  Dict[int, float] = {
        t: (value(v) or 0.0) for t, v in sol["B"].items()
    }

    plan_prog = sol["plan_prog"]
    R_net     = sol["R_net"]
    R_ret     = sol["R_ret"]
    A         = sol["A"]
    act       = sol["act"]

    proj_records: dict = {}
    all_events:   list = []

    for i, p in enumerate(projects):
        s, f       = p["s_i"], p["f_i"]
        BAC, CP    = p["BAC_i"], p["CP_i"]
        alpha      = p["alpha_i"]
        drec, psi  = p["delta_rec_i"], p["psi_i"]
        rho        = p["rho_i"]
        theta, phi = p["theta"], p["phi"]
        M          = p["M_i"]
        e_i        = p.get("e_i", [s] * M)
        eta        = p["eta"]
        Omega      = p["Omega_i"]
        mu         = p["mu_i"]
        tau_tol    = p["tau_tol_i"]
        D_plan     = p["D_plan"]
        A_i        = alpha * CP

        P_sim    = 0.0
        ACWP_sim = 0.0
        certified: set = set()
        cum_phi_cert   = 0.0
        adv_recov      = 0.0
        status         = "pre_start"
        tau_rem        = tau_tol
        records: Dict[int, dict] = {}

        if s <= H:
            all_events.append(dict(t=s, proj=i, type="advance", amount=A_i))

        for t in T:
            x_t = x_val.get((i, t), 0.0)

            if   t < s: window = "pre_start"
            elif t > f: window = "post_plan"
            else:       window = "active"

            if window == "active" and status not in ("terminated", "completed"):
                status    = "active"
                P_sim     = min(1.0, P_sim + eta[t] * x_t / BAC)
                ACWP_sim += x_t

            EV_sim = P_sim * BAC

            # Milestone certification
            ms_new_sim: List[int] = []
            if window == "active" and status == "active":
                for j in range(M):
                    if (j not in certified
                            and P_sim >= theta[j] - 1e-6
                            and t >= e_i[j]
                            and all(k in certified for k in range(j))):
                        certified.add(j)
                        cum_phi_cert += phi[j]
                        gross   = phi[j] * CP
                        ret_ded = rho * gross
                        if cum_phi_cert >= psi:
                            candidate = drec * gross
                            remaining = max(0.0, A_i - adv_recov)
                            rec_ded   = min(candidate, remaining)
                            adv_recov += rec_ded
                        else:
                            rec_ded = 0.0
                        R_net_j = gross - ret_ded - rec_ded
                        ms_new_sim.append(j)
                        all_events.append(dict(t=t, proj=i, type=f"ms{j+1}_net", amount=R_net_j))
                        if j == M - 1:
                            all_events.append(dict(t=t, proj=i, type="retention", amount=rho * CP))

            # EVM signals
            SPI = CPI = EAC = f_hat = Delta_f = None
            cond1 = cond2 = False

            if window == "active" and status == "active":
                P_plan_t = plan_prog[i][t]

                if P_plan_t > 1e-9:
                    SPI = P_sim / P_plan_t
                else:
                    SPI = 1.0 if P_sim < 1e-9 else float("inf")

                if ACWP_sim > 1e-9:
                    CPI = EV_sim / ACWP_sim
                    EAC = ACWP_sim / P_sim if P_sim > 1e-9 else float("inf")
                else:
                    if t > s:
                        # Zero spend since start: stagnation → worst case
                        CPI = float("inf")
                        EAC = float("inf")
                    else:
                        CPI = 1.0
                        EAC = BAC

                elapsed  = t - s
                rem_per  = D_plan - elapsed
                if SPI is not None and SPI > 1e-9 and not math.isinf(SPI):
                    f_hat   = t + rem_per / SPI
                    Delta_f = f_hat - f
                else:
                    f_hat   = float("inf")
                    Delta_f = float("inf")

                cond1 = math.isinf(Delta_f) or (Delta_f > Omega)
                cond2 = (EAC is not None and
                         (math.isinf(EAC) or EAC > (1.0 + mu) * BAC))

            # Cure-period counter (Eq.13)
            if window == "active" and status == "active":
                if cond1 and cond2:
                    tau_rem = tau_rem - 1
                else:
                    tau_rem = tau_tol

            # LP-derived milestone events (for cross-reference)
            u_now  = {j: u_val.get((i, j, t), 0)     for j in range(M)}
            u_prev = ({j: u_val.get((i, j, t - 1), 0) for j in range(M)}
                      if t > 1 else {j: 0 for j in range(M)})
            ms_new_lp = [j for j in range(M) if u_now[j] - u_prev.get(j, 0) > 0]

            records[t] = dict(
                proj=i, t=t, window=window, status=status,
                x_lp=x_t, P_lp=P_lp.get((i, t)), u_lp=u_now, u_prev_lp=u_prev,
                ms_new_lp=ms_new_lp,
                P_sim=P_sim, ACWP_sim=ACWP_sim, EV_sim=EV_sim,
                P_plan=plan_prog[i][t], eta=eta.get(t, 0.0),
                SPI=SPI, CPI=CPI, EAC=EAC, f_hat=f_hat, Delta_f=Delta_f,
                cond1=cond1, cond2=cond2, tau_rem=tau_rem,
                ms_new_sim=ms_new_sim,
                theta=list(theta), Omega=Omega, mu=mu, BAC=BAC, CP=CP,
            )

            # Termination execution (fires when counter reaches 0)
            if window == "active" and status == "active" and tau_rem <= 0:
                status = "terminated"
                records[t]["status"] = "terminated"
                outstanding_advance  = max(0.0, A_i - adv_recov)
                R_term = P_sim * CP * (1.0 - rho) - outstanding_advance
                all_events.append(dict(
                    t=t + 1, proj=i,
                    type="termination_settlement", amount=R_term,
                ))
            elif window == "active" and status == "active" and P_sim >= 1.0 - 1e-6:
                status = "completed"
                records[t]["status"] = "completed"

        proj_records[i] = dict(
            records=records, status=status,
            P_final=P_sim, ACWP_final=ACWP_sim, certified=certified,
        )

    # Portfolio records
    events_by_t: Dict[int, list] = {}
    for ev in all_events:
        events_by_t.setdefault(ev["t"], []).append(ev)

    port_records: Dict[int, dict] = {}
    running_B = B1
    for t in T:
        outflow = sum(x_val.get((i, t), 0.0) for i in range(n))
        inflow  = sum(ev["amount"] for ev in events_by_t.get(t, []))
        active_projs = [
            i for i in range(n)
            if proj_records[i]["records"][t]["window"] == "active"
            and proj_records[i]["records"][t]["status"] == "active"
        ]
        port_records[t] = dict(
            t=t, B_lp=B_lp.get(t, 0.0), B_sim=running_B,
            outflow=outflow, inflow=inflow, net=inflow - outflow,
            n_active=len(active_projs), active_proj=active_projs,
            events=events_by_t.get(t, []),
        )
        running_B = running_B - outflow + inflow

    running = 0.0
    for ev in sorted(all_events, key=lambda e: (e["t"], e["proj"])):
        running += ev["amount"]
        ev["running_total"] = running

    return dict(
        proj=proj_records, portfolio=port_records, events=all_events,
        x_val=x_val, u_val=u_val, P_lp=P_lp, B_lp=B_lp,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7.  VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate(portfolio: dict, sol: dict, rec: dict, tol: float = 1e-3) -> dict:
    n, H      = portfolio["n"], portfolio["H"]
    projects  = portfolio["projects"]
    T         = list(range(1, H + 1))
    x_val, u_val = rec["x_val"], rec["u_val"]
    P_lp, B_lp  = rec["P_lp"], rec["B_lp"]
    R_net, R_ret, A, act = sol["R_net"], sol["R_ret"], sol["A"], sol["act"]
    errors: dict = {}

    errors["C1_B_nonneg"] = [
        (t, round(B_lp.get(t, 0.0), 4))
        for t in T if B_lp.get(t, 0.0) < -tol
    ]

    errors["C2_budget"] = [
        dict(t=t,
             spend=round(sum(x_val.get((i, t), 0.0) for i in range(n)), 4),
             bal=round(B_lp.get(t, 0.0), 4))
        for t in T
        if sum(x_val.get((i, t), 0.0) for i in range(n)) > B_lp.get(t, 0.0) + tol
    ]

    prog_errs = []
    for i, p in enumerate(projects):
        Pc = 0.0
        for t in T:
            Pc  = min(1.0, Pc + p["eta"][t] * x_val.get((i, t), 0.0) / p["BAC_i"])
            lp  = P_lp.get((i, t), 0.0)
            if abs(lp - Pc) > tol * 10:
                prog_errs.append(dict(proj=i, t=t, lp=round(lp, 6), sim=round(Pc, 6)))
    errors["C3_progress"] = prog_errs[:10]

    bal_errs = []
    Bc = portfolio["B1"]
    for t in T[:-1]:
        out   = sum(act.get((i, t), 0) * x_val.get((i, t), 0.0) for i in range(n))
        inf_t = sum(A[i] for i in range(n) if t == projects[i]["s_i"])
        for i, p in enumerate(projects):
            for j in range(p["M_i"]):
                u_now  = u_val.get((i, j, t), 0)
                u_prev = u_val.get((i, j, t - 1), 0) if t > 1 else 0
                if u_now - u_prev > 0:
                    pay    = R_net.get((i, j), 0.0)
                    if j == p["M_i"] - 1:
                        pay += R_ret.get(i, 0.0)
                    inf_t += pay
        Bc_next = Bc - out + inf_t
        lp_next = B_lp.get(t + 1, 0.0)
        if abs(lp_next - Bc_next) > tol * 100:
            bal_errs.append(dict(t=t, lp=round(lp_next, 2), comp=round(Bc_next, 2)))
        Bc = Bc_next
    errors["C4_balance"] = bal_errs[:5]

    cert_errs = []
    for i, p in enumerate(projects):
        Pc  = 0.0
        e_i = p.get("e_i", [p["s_i"]] * p["M_i"])
        for t in T:
            Pc = min(1.0, Pc + p["eta"][t] * x_val.get((i, t), 0.0) / p["BAC_i"])
            for j in range(p["M_i"]):
                u_ij    = u_val.get((i, j, t), 0)
                reached = Pc >= p["theta"][j] - tol and t >= e_i[j]
                if reached and u_ij == 0:
                    cert_errs.append(dict(issue="missed",    proj=i, ms=j, t=t, P=round(Pc, 5)))
                if not reached and u_ij == 1:
                    cert_errs.append(dict(issue="premature", proj=i, ms=j, t=t, P=round(Pc, 5)))
    errors["C5_cert"] = cert_errs[:10]

    errors["C6_mono"] = [
        (i, j, t)
        for i, p in enumerate(projects)
        for j in range(p["M_i"])
        for t in range(2, H + 1)
        if u_val.get((i, j, t), 0) < u_val.get((i, j, t - 1), 0)
    ][:10]

    errors["C7_order"] = [
        (i, j, t)
        for i, p in enumerate(projects)
        for j in range(p["M_i"] - 1)
        for t in T
        if u_val.get((i, j + 1, t), 0) > u_val.get((i, j, t), 0)
    ][:10]

    errors["C8_inactive"] = [
        (i, t, round(x_val.get((i, t), 0.0), 4))
        for i in range(n) for t in T
        if act.get((i, t), 0) == 0 and x_val.get((i, t), 0.0) > tol
    ][:10]

    pay_errs = []
    for i, p in enumerate(projects):
        if rec["proj"][i]["status"] == "completed":
            total = (A[i]
                     + sum(R_net.get((i, j), 0.0) for j in range(p["M_i"]))
                     + R_ret.get(i, 0.0))
            if abs(total - p["CP_i"]) > tol * p["CP_i"]:
                pay_errs.append(dict(proj=i, total=round(total, 4), CP=round(p["CP_i"], 4)))
    errors["C9_payment_id"] = pay_errs

    errors["C10_nonneg"] = [
        (i, t, round(x_val.get((i, t), 0.0), 6))
        for (i, t) in x_val if x_val.get((i, t), 0.0) < -tol
    ]

    errors["_terminations"] = {
        i: next(
            (t for t in T if rec["proj"][i]["records"][t]["status"] == "terminated"),
            None,
        )
        for i in range(n) if rec["proj"][i]["status"] == "terminated"
    }

    all_pass = all(len(v) == 0 for k, v in errors.items() if not k.startswith("_"))
    return dict(errors=errors, all_pass=all_pass)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  PRINT / CSV / MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".4f", width=8, none="--"):
    if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
        return f"{none:>{width}}"
    return f"{v:{width}{fmt}}"


def print_portfolio(pf: dict) -> None:
    print("\n" + "=" * 72)
    print("PORTFOLIO PARAMETERS")
    print("=" * 72)
    print(f"  n={pf['n']}  H={pf['H']}  B1={pf['B1']:,.0f}  "
          f"totalBAC={pf['total_BAC']:,.0f}  gamma={pf['gamma']}")
    print(f"\n  {'i':>2}  {'s':>3}  {'f':>3}  {'D':>3}  "
          f"{'BAC':>9}  {'CP':>9}  {'π%':>5}  {'α%':>5}  "
          f"{'ρ%':>5}  {'M':>2}  {'Ω':>3}  {'μ%':>5}  {'τtol':>4}")
    print("  " + "─" * 75)
    for p in pf["projects"]:
        print(f"  {p['idx']:>2}  {p['s_i']:>3}  {p['f_i']:>3}  {p['D_plan']:>3}  "
              f"{p['BAC_i']:>9,.0f}  {p['CP_i']:>9,.0f}  "
              f"{p['pi_i']*100:>5.1f}  {p['alpha_i']*100:>5.1f}  "
              f"{p['rho_i']*100:>5.1f}  {p['M_i']:>2}  "
              f"{p['Omega_i']:>3}  {p['mu_i']*100:>5.1f}  {p['tau_tol_i']:>4}")


def print_solution(pf: dict, sol: dict, rec: dict) -> None:
    n, H  = pf["n"], pf["H"]
    T     = list(range(1, H + 1))
    x_val = rec["x_val"]
    B_lp  = rec["B_lp"]
    u_val = rec["u_val"]
    print("\n" + "=" * 72)
    print(f"MILP SOLUTION  —  Z*_L1 = {sol['obj']:,.4f}  [{sol['status']}]")
    print("=" * 72)
    print("\n── PER-PROJECT OUTCOME ──")
    for i, p in enumerate(pf["projects"]):
        pr  = rec["proj"][i]
        tx  = sum(x_val.get((i, t), 0.0) for t in T)
        ms_str = []
        for j in range(p["M_i"]):
            ct = next(
                (t for t in T if u_val.get((i, j, t), 0) == 1
                 and u_val.get((i, j, t - 1), 0) == 0), None)
            ms_str.append(f"m{j+1}@{ct}" if ct else f"m{j+1}:--")
        print(f"  P{i}  total_x={tx:,.0f}  P_final={pr['P_final']:.4f}  "
              f"status={pr['status']}  {' '.join(ms_str)}")


def print_validation(val: dict) -> None:
    labels = {
        "C1_B_nonneg":  "B_t >= 0",
        "C2_budget":    "Σ x <= B_t",
        "C3_progress":  "Progress recursion",
        "C4_balance":   "Balance recursion",
        "C5_cert":      "u=1 ⟺ P>=θ, t>=e",
        "C6_mono":      "u monotone",
        "C7_order":     "u ordered",
        "C8_inactive":  "x=0 outside window",
        "C9_payment_id":"Payment identity",
        "C10_nonneg":   "x >= 0",
    }
    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    for k, desc in labels.items():
        errs = val["errors"].get(k, [])
        flag = "PASS ✓" if not errs else "FAIL ✗"
        print(f"  {flag}  {desc}")
        for e in errs[:3]:
            print(f"         → {e}")
    terms = val["errors"].get("_terminations", {})
    print(f"\n  Terminations: {terms}" if terms else "\n  No terminations.")
    print(f"  Overall: {'ALL PASS ✓' if val['all_pass'] else 'SOME FAILED ✗'}")


def export_csv(pf: dict, rec: dict, prefix: str = "milp_out") -> None:
    H, n = pf["H"], pf["n"]
    T    = list(range(1, H + 1))
    rows = []
    for i in range(n):
        for t in T:
            r = rec["proj"][i]["records"][t]
            row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            row["u_lp"]       = "".join(str(r["u_lp"].get(j, 0)) for j in range(pf["projects"][i]["M_i"]))
            row["ms_new_lp"]  = str(r["ms_new_lp"])
            row["ms_new_sim"] = str(r["ms_new_sim"])
            row["theta"]      = str(r["theta"])
            for k in list(row.keys()):
                if isinstance(row[k], float) and (math.isinf(row[k]) or math.isnan(row[k])):
                    row[k] = "inf"
            rows.append(row)
    fn = f"{prefix}_project_states.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [CSV] {fn}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--n_projects", type=int,   default=None)
    parser.add_argument("--horizon",    type=int,   default=None)
    parser.add_argument("--b1_ratio",   type=float, default=None)
    parser.add_argument("--no_csv",     action="store_true")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.seed       is not None: cfg["seed"]       = args.seed
    if args.n_projects is not None: cfg["n_projects"] = args.n_projects
    if args.horizon    is not None: cfg["horizon"]    = args.horizon
    if args.b1_ratio   is not None: cfg["b1_ratio"]   = (args.b1_ratio, args.b1_ratio)

    pf  = generate_portfolio(cfg)
    print_portfolio(pf)
    sol = build_and_solve(pf)
    rec = build_records(pf, sol)
    val = validate(pf, sol, rec)
    print_solution(pf, sol, rec)
    print_validation(val)
    if not args.no_csv:
        export_csv(pf, rec)
    print(f"\n  Z*_L1 = {sol['obj']:,.4f}\n")


if __name__ == "__main__":
    main()