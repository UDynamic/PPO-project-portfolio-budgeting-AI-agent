# src/env/helper.py
#
# Pure, stateless helpers for PortfolioBudgetingEnv.
# No I/O, no global state, no randomness except through the rng argument.
#
# Sections
# --------
#   DISTRIBUTIONS        — sample / sample_int
#   S-CURVE              — beta_cdf / planned_progress / earned_schedule
#   SAMPLERS             — sample_project / sample_milestones / sample_efficiency
#   EVM                  — compute_* / update_evm
#   BREACHES             — evaluate_breaches / update_tolerance / check_termination
#   PAYMENTS             — check_certifications / deliver_payments / compute_termination_settlement
#   RENDER               — reset_history / render
#   PLAY INFRASTRUCTURE  — db init, config seeding, export, config loading

from __future__ import annotations
import math
import os
import sqlite3
import sys
from datetime import datetime
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sample_ordinal(seq, idx: int, proj_idx: int = 0) -> float:
    if seq and isinstance(seq[0], (list, tuple)):
        seq = seq[proj_idx % len(seq)]
    return float(seq[idx % len(seq)])


def sample(rng, dist: str, p1, p2=None, p3=None, p4=None) -> float:
    if dist == "fixed":             return float(p1)
    if dist == "uniform":           return float(rng.uniform(p1, p2))
    if dist == "normal":            return float(rng.normal(p1, p2))
    if dist == "lognormal":         return float(rng.lognormal(p1, p2))
    if dist == "triangular":        return float(rng.triangular(p1, p3, p2))
    if dist == "beta":              return float(rng.beta(p1, p2))
    if dist == "categorical":
        r = float(rng.random())
        if r < p1:        return int(p3)
        if r < p1 + p2:   return int(p4)
        return int(p3)
    if dist == "truncated_normal":
        while True:
            v = float(rng.normal(p1, p2))
            if p3 <= v <= p4:
                return v
    if dist == "custom_ordinal":
        raise ValueError(
            "custom_ordinal cannot be sampled via sample(); "
            "call sample_ordinal(seq, idx, proj_idx) directly."
        )
    raise ValueError(f"Unknown distribution: {dist!r}")


def sample_int(rng, dist: str, p1, p2=None, p3=None, p4=None) -> int:
    return max(1, round(sample(rng, dist, p1, p2, p3, p4)))


# ═══════════════════════════════════════════════════════════════════════════════
# S-CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    dx    = x / 200
    total = 0.0
    for k in range(200):
        t      = (k + 0.5) * dx
        total += (t ** (a - 1.0)) * ((1.0 - t) ** (b - 1.0)) * dx
    B = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return min(1.0, total / B)


def planned_progress(t_project: int, duration: int, a: float, b: float) -> float:
    return beta_cdf(t_project / duration, a, b)


def earned_schedule(progress_actual: float, duration: int, a: float, b: float) -> float:
    if progress_actual <= 0.0: return 0.0
    if progress_actual >= 1.0: return float(duration)
    lo, hi = 0.0, float(duration)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid / duration, a, b) < progress_actual:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _s(rng, cfg, key, proj_idx: int = 0):
    dist = cfg[f"{key}_dist"]
    if dist == "custom_ordinal":
        return sample_ordinal(cfg[f"{key}_p1"], idx=proj_idx, proj_idx=0)
    return sample(rng, dist,
                  cfg[f"{key}_p1"], cfg[f"{key}_p2"],
                  cfg[f"{key}_p3"], cfg[f"{key}_p4"])

def _si(rng, cfg, key, proj_idx: int = 0):
    dist = cfg[f"{key}_dist"]
    if dist == "custom_ordinal":
        return max(1, round(sample_ordinal(cfg[f"{key}_p1"], idx=proj_idx, proj_idx=0)))
    return sample_int(rng, dist,
                      cfg[f"{key}_p1"], cfg[f"{key}_p2"],
                      cfg[f"{key}_p3"], cfg[f"{key}_p4"])


def sample_project(rng, cfg: dict, i: int) -> dict:
    bac              = max(1.0, _s(rng, cfg, "bac", proj_idx=i))
    profit_percent   = max(0.0, _s(rng, cfg, "profit_percent", proj_idx=i))
    price            = bac * (1.0 + profit_percent)
    planned_start    = max(0, int(round(_s(rng, cfg, "planned_start", proj_idx=i))))
    planned_duration = max(1, _si(rng, cfg, "planned_duration", proj_idx=i))
    planned_finish   = planned_start + planned_duration
    scurve_a         = max(0.5, _s(rng, cfg, "scurve_a", proj_idx=i))
    scurve_b         = max(0.5, _s(rng, cfg, "scurve_b", proj_idx=i))
    advance_percent  = max(0.0, min(0.3,  _s(rng, cfg, "advance_percent",  proj_idx=i)))
    advance_trigger  = max(0.0, min(1.0,  _s(rng, cfg, "advance_trigger",  proj_idx=i)))
    advance_recovery = max(0.0, min(1.0,  _s(rng, cfg, "advance_recovery", proj_idx=i)))
    retention_rate   = max(0.0, min(0.1,  _s(rng, cfg, "retention_rate",   proj_idx=i)))
    progress_delay_cap    = max(0.0, min(1.0, _s(rng, cfg,  "progress_delay_cap",    proj_idx=i)))
    finish_delay_cap      = max(1,           _si(rng, cfg,  "finish_delay_cap",       proj_idx=i))
    termination_tolerance = max(1,           _si(rng, cfg,  "termination_tolerance",  proj_idx=i))
    return {
        "i": i,
        "planned_start": planned_start, "planned_finish": planned_finish,
        "planned_duration": planned_duration,
        "bac": bac, "profit_percent": profit_percent, "price": price,
        "scurve_a": scurve_a, "scurve_b": scurve_b,
        "advance_percent": advance_percent, "advance_trigger": advance_trigger,
        "advance_recovery": advance_recovery, "retention_rate": retention_rate,
        "progress_delay_cap": progress_delay_cap,
        "finish_delay_cap": finish_delay_cap,
        "termination_tolerance": termination_tolerance,
    }


def _ordinal_list(rng, cfg: dict, key: str, n: int, proj_idx: int = 0) -> list:
    dist = cfg.get(f"{key}_dist", "fixed")
    p1   = cfg.get(f"{key}_p1")
    if dist == "custom_ordinal":
        if isinstance(p1, (list, tuple)):
            seq = p1
            if seq and isinstance(seq[0], (list, tuple)):
                seq = seq[proj_idx % len(seq)]
            base = list(seq)
            while len(base) < n:
                base.append(base[-1])
            return base[:n]
        return [float(p1)] * n
    return [_s(rng, cfg, key, proj_idx=proj_idx) for _ in range(n)]


def sample_milestones(rng, cfg: dict, proj: dict) -> list[dict]:
    i    = proj.get("i", 0)
    n_ms = max(1, _si(rng, cfg, "n_milestones", proj_idx=i))

    thresholds_cfg = _ordinal_list(rng, cfg, "progress_threshold", n_ms, proj_idx=i)
    weights_cfg    = _ordinal_list(rng, cfg, "payment_weight",     n_ms, proj_idx=i)
    ts_raw_cfg     = _ordinal_list(rng, cfg, "timestep_threshold", n_ms, proj_idx=i)

    thresholds = [max(0.0, min(1.0, float(v))) for v in thresholds_cfg]
    thresholds[-1] = 1.0

    raw_w   = [max(0.0, float(v)) for v in weights_cfg]
    total_w = sum(raw_w)
    if total_w > 1e-9:
        weights = [w / total_w for w in raw_w]
    else:
        weights = [1.0 / n_ms] * n_ms
    weights[-1] = max(0.0, 1.0 - sum(weights[:-1]))

    ts_is_ordinal = cfg.get("timestep_threshold_dist") == "custom_ordinal"

    price          = proj["price"]
    advance_amount = proj["advance_percent"] * price
    adv_rec_rate   = proj["advance_recovery"]
    ret_rate       = proj["retention_rate"]

    milestones = [{
        "j": 0,
        "progress_threshold": 0.0,
        "timestep_threshold": proj["planned_start"],
        "payment_weight": 0.0,
        "gross_payment": advance_amount,
        "advance_recovery": 0.0,
        "advance_recovery_remain": advance_amount,
        "retention_withheld": 0.0,
        "retention_released": 0.0,
        "net_payment": advance_amount,
    }]

    remaining_advance    = advance_amount
    cumulative_retention = 0.0

    for idx in range(n_ms):
        j        = idx + 1
        is_final = (idx == n_ms - 1)

        if is_final and not ts_is_ordinal:
            ts = proj["planned_duration"] - 1   # project-time: last step of project
        elif ts_is_ordinal:
            ts = int(round(float(ts_raw_cfg[idx])))
        else:
            ts_frac = max(0.0, min(1.0, float(ts_raw_cfg[idx])))
            ts = max(1, round(
                thresholds[idx] * proj["planned_duration"] * ts_frac))

        gross    = weights[idx] * price
        recovery = remaining_advance if is_final else min(gross * adv_rec_rate, remaining_advance)
        remaining_advance -= recovery

        ret_held = gross * ret_rate
        cumulative_retention += ret_held
        ret_released = cumulative_retention if is_final else 0.0

        milestones.append({
            "j": j,
            "progress_threshold":      thresholds[idx],
            "timestep_threshold":      ts,
            "payment_weight":          weights[idx],
            "gross_payment":           gross,
            "advance_recovery":        recovery,
            "advance_recovery_remain": remaining_advance,
            "retention_withheld":      ret_held,
            "retention_released":      ret_released,
            "net_payment":             gross - recovery - ret_held + ret_released,
        })

    return milestones


def sample_efficiency(rng, cfg: dict, proj_idx: int = 0, t_project: int = 0) -> float:
    dist = cfg.get("efficiency_dist", "fixed")
    if dist == "custom_ordinal":
        seq = cfg.get("efficiency_p1", [1.0])
        return max(0.01, sample_ordinal(seq, t_project, proj_idx))
    return max(0.01, float(_s(rng, cfg, "efficiency")))


def efficiency_lower_bound(cfg: dict, proj_idx: int = 0) -> float:
    """
    Return the minimum possible efficiency value for a project, given the
    configured distribution.  Used to compute a dynamic per-project allocation
    cap: alloc_cap = bac / eta_min, guaranteeing that a single full allocation
    is always sufficient to complete the project regardless of the draw.

    Supported distributions
    -----------------------
    fixed          → the single value is both min and max
    uniform        → min(p1, p2)
    custom_ordinal → min of the sequence (per-project slice honoured)

    All other distributions fall back to 0.01 (the global efficiency floor),
    which is conservative but safe — the cap becomes very large and Rule 3
    (budget cap) takes over.
    """
    dist = cfg.get("efficiency_dist", "fixed")
    p1   = cfg.get("efficiency_p1")
    p2   = cfg.get("efficiency_p2")

    if dist == "fixed":
        return max(0.01, float(p1))

    if dist == "uniform":
        return max(0.01, float(min(p1, p2)))

    if dist == "custom_ordinal":
        seq = p1
        if seq and isinstance(seq[0], (list, tuple)):
            seq = seq[proj_idx % len(seq)]
        return max(0.01, float(min(seq)))

    # triangular, normal, lognormal, beta, etc. — no closed-form lower bound;
    # fall back to the global efficiency floor so the cap is safely large.
    return 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# EVM
# ═══════════════════════════════════════════════════════════════════════════════

def _spi(progress: float, t_project: int, duration: int, a: float, b: float) -> float:
    es = earned_schedule(progress, duration, a, b)
    at = float(t_project)
    return es / at if at > 1e-9 else 1.0

def _cpi(bcwp: float, acwp: float) -> float:
    return bcwp / acwp if acwp > 1e-9 else 1.0

def _eac(acwp: float, bac: float, bcwp: float, cpi: float, spi: float) -> float:
    composite = cpi * spi
    return (acwp + (bac - bcwp) / composite) if composite > 1e-9 else bac

def _tcpi(bac: float, bcwp: float, acwp: float) -> float:
    work_rem   = bac - bcwp
    budget_rem = bac - acwp
    if budget_rem > 1e-9: return work_rem / budget_rem
    return 0.0 if work_rem <= 0.0 else float("inf")

def _forecast_finish(planned_start: int, planned_duration: int, spi: float,
                     planned_finish: int, finish_delay_cap: int) -> float:
    if spi > 1e-9:
        return planned_start + planned_duration / spi
    return float(planned_finish + finish_delay_cap + 1)


def update_evm(ps: dict, proj: dict) -> None:
    bac  = proj["bac"]
    bcwp = ps["progress_actual"] * bac
    acwp = ps["outflow"]

    spi  = _spi(ps["progress_actual"], ps["t_project"],
                proj["planned_duration"], proj["scurve_a"], proj["scurve_b"])
    cpi  = _cpi(bcwp, acwp)
    eac  = _eac(acwp, bac, bcwp, cpi, spi)
    tcpi = _tcpi(bac, bcwp, acwp)

    ps["spi"]              = spi
    ps["cpi"]              = cpi
    ps["eac"]              = eac
    ps["tcpi"]             = tcpi
    ps["progress_delay_t"] = ps["progress_plan_t"] - ps["progress_actual"]


# ═══════════════════════════════════════════════════════════════════════════════
# BREACHES
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_breaches(ps: dict, proj: dict) -> dict:
    op = ps["progress_delay_t"] > proj["progress_delay_cap"]
    return {
        "over_progress_delay":  op,
        "over_any":             op,
        "over_duration_window": False,
    }


def update_tolerance(ps: dict, proj: dict, flags: dict) -> None:
    if flags["over_any"]:
        ps["tolerance_remain"] = max(0, ps["tolerance_remain"] - 1)
    else:
        ps["tolerance_remain"] = proj["termination_tolerance"]


def check_termination(t: int, ps: dict, proj: dict,
                      flags: dict) -> tuple[bool, bool]:
    odw = t >= proj["planned_finish"] + proj["finish_delay_cap"]
    flags["over_duration_window"] = odw
    if odw:
        return True, True
    if ps["tolerance_remain"] <= 0:
        return True, False
    return False, False


# ═══════════════════════════════════════════════════════════════════════════════
# PAYMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def check_certifications(t: int, ps: dict,
                         milestones: list[dict],
                         milestone_state: list[dict],
                         planned_start: int = 0) -> list[int]:
    result = []
    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] == 0 or ms_state["certified"]:
            continue
        # timestep_threshold is in project-time; convert to episode-time
        episode_threshold = planned_start + ms["timestep_threshold"]
        if (ps["progress_actual"] >= ms["progress_threshold"]
                and t >= episode_threshold):
            result.append(ms["j"])
    return result


def deliver_payments(certified_js: list[int], t: int, ps: dict,
                     milestones: list[dict],
                     milestone_state: list[dict]) -> float:
    total = 0.0
    for ms, ms_state in zip(milestones, milestone_state):
        if ms["j"] not in certified_js:
            continue
        net = ms["net_payment"]
        ms_state.update(certified=True, certified_t=t, payment_released=net)
        ps["inflow"] += net
        total        += net
    return total


def compute_termination_settlement(proj: dict, ps: dict) -> float:
    return ps["progress_actual"] * proj["price"] - ps["inflow"]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

_W, _LW, _VW, _GAP = 80, 17, 12, 3

def _sep(c="═"):  return c * _W

def _row(ll, lv, rl="", rv=""):
    left = f"  {ll:<{_LW}} : {lv:>{_VW}}"
    return left + (f"{' '*_GAP}{rl:<{_LW}} : {rv:>{_VW}}" if rl or rv else "")

def _fmt(v, d=2): return f"{v:,.{d}f}"

def _tag(s):
    return {"active":"ACTIVE","pending":"PENDING",
            "completed":"COMPLETED","terminated":"TERMINATED"}.get(
            (s or "").lower(), (s or "UNKNOWN").upper())


_episode_id = ""
_n_projects = 0


def reset_history(episode_id: str, n_projects: int) -> None:
    global _episode_id, _n_projects
    _episode_id, _n_projects = episode_id, n_projects


def render(t: int, budget: float, horizon: int, initial_budget: float,
           projects: list[dict], proj_state: list[dict],
           milestones: list[list[dict]], milestone_state: list[list[dict]],
           episode_id: str = "", net_cashflow: float = 0.0,
           cum_reward: float = 0.0, allocations=None, **_) -> None:

    print()
    print(_sep())
    print(_row(f"PERIOD {t} / {horizon}", "", "net_cashflow", _fmt(net_cashflow)))
    print(_row("reward", f"{cum_reward:+.4f}", "budget_available", _fmt(budget)))
    print(_sep())

    for idx, (proj, ps, ms_list, ms_sl) in enumerate(
            zip(projects, proj_state, milestones, milestone_state)):
        tol     = ps.get("tolerance_remain", proj["termination_tolerance"])
        tol_max = proj["termination_tolerance"]
        alloc_amount = (allocations[idx] if allocations is not None
                        else ps.get("allocation_action"))
        print(f"  [{_tag(ps.get('status','pending'))}]  Project {proj['i']}")
        print(_row("BAC",    _fmt(proj["bac"]),   "Price",             _fmt(proj["price"])))
        print(_row("t_proj", str(ps.get("t_project", 0)),
                   "tolerance_remain", f"{tol}/{tol_max}"))
        print(_row("inflow", _fmt(ps.get("inflow", 0.0)),
                   "outflow", _fmt(ps.get("outflow", 0.0))))
        if alloc_amount is not None:
            print(_row("→ allocated", _fmt(alloc_amount), "", ""))
        print(_sep("─"))
        print(_row("catchup_t",      _fmt(ps.get("catchup_alloc_t",      0.0)),
                   "catchup_next_t", _fmt(ps.get("catchup_alloc_next_t", 0.0))))
        print(_row("reach_plan_t",      _fmt(ps.get("reach_plan_t",      0.0)),
                   "reach_plan_next_t", _fmt(ps.get("reach_plan_next_t", 0.0))))
        print(_sep("─"))
        j = ps.get("target_milestone_j")
        print(_row("target",         f"j={j}" if j is not None else "none",
                   "net_payment",    _fmt(ps.get("target_net_payment",    0.0))))
        print(_row("progress_gap",   _fmt(ps.get("target_progress_gap",   0.0), 4),
                   "timestep_gap",   str(ps.get("target_timestep_gap",   0))))
        print(_row("required_alloc", _fmt(ps.get("target_required_alloc", 0.0)),
                   "target_npv",     _fmt(ps.get("target_npv",            0.0), 4)))
        print(_sep())


# ═══════════════════════════════════════════════════════════════════════════════
# PLAY INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Everything below is used exclusively by play.py.
# It is NOT part of the gymnasium environment API.
#
# Public surface
# --------------
#   play_startup(no_db, runs_dir, schema_path)  → run_dir | None
#   play_init_db(run_dir, schema_path)           → conn | None
#   play_seed_configs(conn, cfgs_dir)            → None
#   play_select_config(conn, cfgs_dir, name)     → cfg dict | None
#   play_export(db_path, fmt, out_dir, ...)      → None
#   parse_action(raw, n, budget)                 → np.ndarray | None
#   default_action(n, active_mask)               → np.ndarray

import numpy as np   # already a project dependency; safe to import here


# ── console helpers ───────────────────────────────────────────────────────────

def _banner(text: str, width: int = 60) -> None:
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def _ok(label: str, detail: str = "") -> None:
    print(f"  ✓  {label}" + (f"  ({detail})" if detail else ""))

def _warn(label: str, detail: str = "") -> None:
    print(f"  ⚠  {label}" + (f"  ({detail})" if detail else ""))

def _fail(label: str, detail: str = "") -> None:
    print(f"  ✗  {label}" + (f"  — {detail}" if detail else ""))


# ── run folder ────────────────────────────────────────────────────────────────

def play_startup(no_db: bool, runs_dir: str, schema_path: str) -> "str | None":
    """
    Create the timestamped run folder.
    Returns run_dir on success, None if --no-db or schema missing.
    """
    _banner("STARTUP")

    if no_db:
        _warn("DB disabled", "--no-db flag set; skipping run folder")
        return None

    if not os.path.exists(schema_path):
        _fail("schema.sql not found", schema_path)
        return None

    stamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    _ok("Run folder created", run_dir)
    return run_dir


# ── DB init ───────────────────────────────────────────────────────────────────

_ALL_TABLES = [
    "environment_config",
    "projects_profile",
    "milestones_profile",
    "portfolios",
    "projects_status",
    "projects_observation",
    "portfolio_observation",
    "milestones_status",
    "training_log",
]


def play_init_db(run_dir: str,
                 schema_path: str) -> "sqlite3.Connection | None":
    """
    Create database.db inside run_dir, apply schema, verify tables.
    Returns open connection or None on failure.
    """
    _banner("DATABASE INIT")

    db_path = os.path.join(run_dir, "database.db")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        with open(schema_path) as f:
            conn.executescript(f.read())
        conn.commit()
        _ok("Database created & schema applied", db_path)
    except Exception as exc:
        _fail("Database init failed", str(exc))
        return None

    # verify
    rows  = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    found = {row["name"] for row in rows}
    all_ok = True
    for table in _ALL_TABLES:
        if table in found:
            _ok(f"table: {table}")
        else:
            _fail(f"table: {table}", "MISSING")
            all_ok = False
    if not all_ok:
        print("\n  [!] Schema verification failed. Check schema.sql.")

    return conn


# ── config seeding ────────────────────────────────────────────────────────────

def play_seed_configs(conn: sqlite3.Connection, cfgs_dir: str) -> None:
    """
    Discover every .py CONFIG dict in cfgs_dir and INSERT OR IGNORE into
    environment_config.  Safe to call multiple times.
    """
    import importlib.util
    import json
    from datetime import timezone

    py_files = sorted(
        f for f in os.listdir(cfgs_dir)
        if f.endswith(".py") and not f.startswith("__")
    )
    if not py_files:
        _warn("No config files found", cfgs_dir)
        return

    seeded = 0
    for fname in py_files:
        fpath = os.path.join(cfgs_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg = getattr(mod, "CONFIG", None)
            if cfg is None:
                _warn(f"No CONFIG dict in {fname}")
                continue
            _seed_one_config(conn, cfg, json, timezone)
            seeded += 1
        except Exception as exc:
            _warn(f"Seed failed: {fname}", str(exc))

    _ok("Configs seeded", f"{seeded} / {len(py_files)} files")


def _seed_one_config(conn: sqlite3.Connection, cfg: dict,
                     json_mod, timezone_mod) -> None:
    """Insert a single CONFIG dict; silently skips duplicates (INSERT OR IGNORE)."""

    def _to_sqlite(v):
        if isinstance(v, (list, dict)):
            return json_mod.dumps(v)
        return v

    created_at = datetime.now(timezone_mod.utc).isoformat()

    cols = ["config_id", "config_name", "created_at"] + [
        k for k in cfg if k not in ("config_id", "config_name")
    ]
    values = (
        [cfg.get("config_id"), cfg.get("config_name"), created_at]
        + [_to_sqlite(cfg.get(k)) for k in cols[3:]]
    )
    placeholders = ", ".join("?" * len(cols))
    col_names    = ", ".join(cols)

    try:
        conn.execute(
            f"INSERT OR IGNORE INTO environment_config ({col_names}) "
            f"VALUES ({placeholders})",
            values,
        )
        conn.commit()
    except Exception as exc:
        print(f"  [!] Insert failed for {cfg.get('config_id')}: {exc}")


# ── config selection / loading ────────────────────────────────────────────────

def play_list_configs(conn: sqlite3.Connection) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT config_id, config_name, created_at "
            "FROM environment_config ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def play_prompt_config(conn: sqlite3.Connection) -> "str | None":
    """
    Interactive config selector.
    Returns a config_id string, or None to quit.
    """
    configs = play_list_configs(conn)
    if not configs:
        print("  [!] No configs found in database.")
        return None

    print()
    print("  Available configurations:")
    print(f"  {'#':>3}  {'Config ID':<40}  {'Name'}")
    print("  " + "─" * 70)
    for idx, c in enumerate(configs):
        print(f"  {idx:>3}  {c['config_id']:<40}  {c['config_name']}")
    print()

    while True:
        raw = input("  Select config number (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            sel = int(raw)
            if 0 <= sel < len(configs):
                return configs[sel]["config_id"]
            print(f"  [!] Enter a number between 0 and {len(configs)-1}.")
        except ValueError:
            print("  [!] Enter a valid number.")


def play_load_config_by_id(config_id: str, cfgs_dir: str) -> "dict | None":
    """
    Load the CONFIG dict for a given config_id from cfgs_dir.
    Tries filename convention first, then scans all files.
    """
    import importlib.util

    candidate = os.path.join(cfgs_dir, config_id.replace("-", "_") + ".py")
    fpath     = candidate if os.path.exists(candidate) else None

    if fpath is None:
        for fname in os.listdir(cfgs_dir):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue
            full = os.path.join(cfgs_dir, fname)
            spec = importlib.util.spec_from_file_location(fname[:-3], full)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if getattr(mod, "CONFIG", {}).get("config_id") == config_id:
                fpath = full
                break

    if fpath is None:
        print(f"  [!] Cannot resolve config_id {config_id!r}.")
        return None

    module_name = os.path.basename(fpath)[:-3]
    spec        = importlib.util.spec_from_file_location(module_name, fpath)
    mod         = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "CONFIG", None)


def play_load_config_by_name(name: str, cfgs_dir: str) -> "dict | None":
    """Match by config_id or config_name substring."""
    import importlib.util

    name_upper = name.strip().upper()
    for fname in sorted(os.listdir(cfgs_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = os.path.join(cfgs_dir, fname)
        spec  = importlib.util.spec_from_file_location(fname[:-3], fpath)
        mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg   = getattr(mod, "CONFIG", {})
        if (name_upper in cfg.get("config_id",   "").upper() or
                name_upper in cfg.get("config_name", "").upper()):
            return cfg

    print(f"  [!] No config matching {name!r} found in {cfgs_dir}.")
    return None


# ── export ────────────────────────────────────────────────────────────────────

def play_export(db_path: str,
                fmt: str = "csv",
                out_dir: "str | None" = None,
                tables_subset: "list[str] | None" = None,
                episode_id: "str | None" = None) -> None:
    """Export tables from db_path into out_dir."""
    _banner("EXPORT")

    if not os.path.exists(db_path):
        _fail("Database not found", db_path)
        return

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(db_path), "exports")

    try:
        import pandas as pd
    except ImportError:
        _fail("pandas missing", "pip install pandas")
        return

    os.makedirs(out_dir, exist_ok=True)
    tables_to_export = tables_subset or _ALL_TABLES

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    tables_to_export = [t for t in tables_to_export if t in existing]

    if not tables_to_export:
        _fail("No tables to export")
        conn.close()
        return

    print(f"  Format : {fmt}")
    print(f"  Output : {out_dir}")
    if episode_id:
        print(f"  Filter : episode_id = {episode_id}")

    data: dict = {}
    for table in tables_to_export:
        try:
            cols = [r[1] for r in conn.execute(
                f"PRAGMA table_info({table})").fetchall()]
            if episode_id and "episode_id" in cols:
                data[table] = pd.read_sql_query(
                    f"SELECT * FROM {table} WHERE episode_id = ?",
                    conn, params=(episode_id,))
            else:
                data[table] = pd.read_sql_query(
                    f"SELECT * FROM {table}", conn)
        except Exception as exc:
            _warn(f"Failed to load {table}", str(exc))

    conn.close()

    if fmt == "csv":
        for name, df in data.items():
            path = os.path.join(out_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            _ok(f"{name:<30} → {os.path.basename(path)}", f"{len(df)} rows")
    elif fmt == "excel":
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            _fail("openpyxl missing", "pip install openpyxl")
            return
        path = os.path.join(out_dir, "database_export.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
                _ok(f"{name:<30} → sheet '{name[:31]}'", f"{len(df)} rows")
        print(f"\n  Saved: {path}")
    elif fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            _fail("pyarrow missing", "pip install pyarrow")
            return
        for name, df in data.items():
            path = os.path.join(out_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
            _ok(f"{name:<30} → {os.path.basename(path)}", f"{len(df)} rows")

    print(f"\n  Done. Files written to: {out_dir}")


# ── action helpers ────────────────────────────────────────────────────────────

def parse_action(raw: str, n: int, budget: float) -> "np.ndarray | None":
    """
    Parse a comma-separated string of monetary amounts into a fraction vector.
    Returns None if input is malformed.
    """
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) != n:
        print(f"  [!] Expected {n} value(s), got {len(parts)}.")
        return None
    try:
        amounts = [float(p) for p in parts]
    except ValueError:
        print("  [!] Non-numeric input.")
        return None
    if any(a < 0 for a in amounts):
        print("  [!] Allocations must be non-negative.")
        return None

    total = sum(amounts)
    if total <= 0.0:
        return np.zeros(n, dtype=np.float32)

    fractions = [a / budget for a in amounts] if budget > 0 else [0.0] * n
    frac_sum  = sum(fractions)
    if frac_sum > 1.0 + 1e-9:
        fractions = [f / frac_sum for f in fractions]
    return np.array(fractions, dtype=np.float32)


def default_action(n: int, active_mask: "list[bool]") -> np.ndarray:
    """Equal split across active projects."""
    n_active = sum(active_mask)
    if n_active == 0:
        return np.zeros(n, dtype=np.float32)
    share = 1.0 / n_active
    return np.array(
        [share if active_mask[i] else 0.0 for i in range(n)],
        dtype=np.float32,
    )