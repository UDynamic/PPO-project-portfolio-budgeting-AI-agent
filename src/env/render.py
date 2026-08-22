# src/env/render.py
#
# ANSI render functions for the Portfolio Budgeting environment.
#
# All formatting and print logic lives here.  Two consumers:
#   - env.render()          calls render_episode() / render_step()
#   - src/play/play.py      imports from here; contains no formatting itself
#
# Four panels per project, printed sequentially:
#   1. Project Identity       (static params)
#   2. Payment Profile        (milestone table with advance/recovery/retention/net/status)
#   3. Boundary & Termination (breach flags + cure counter)
#   4. Period Timeseries      (Group A: periodic CF, Group B: cumulative CF, Group C: EVM)
#
# Preceded by a Portfolio Summary block.
# Plain f-strings only.  No external dependencies.
#
# History store
# -------------
# The timeseries panel needs rows from all prior periods.  render.py owns
# a module-level _history dict keyed by (episode_id, proj_index).
# Call reset_history(episode_id) at the start of each episode.

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

W = 110   # total print width


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY STORE  (timeseries accumulation)
# ─────────────────────────────────────────────────────────────────────────────

_history: dict[tuple[str, int], list[dict]] = {}
_cum:     dict[tuple[str, int], dict]       = {}


def reset_history(episode_id: str, n_projects: int) -> None:
    """
    Clear history for a new episode and initialise per-project cumulative
    accumulators.  Call once from env.reset() before the first render.
    """
    for i in range(n_projects):
        key = (episode_id, i)
        _history[key] = []
        _cum[key]     = {
            "inflow": 0.0, "outflow": 0.0, "net_cf": 0.0,
            "deficit": 0.0, "alloc": 0.0, "interest": 0.0,
            "budget_draw": 0.0,
        }


def record_row(episode_id: str, proj_index: int,
               t: int, ps: dict, proj: dict, cfg: dict,
               cashflow: dict) -> dict:
    """
    Build one timeseries row for *proj_index* at period *t*, update the
    cumulative accumulators, append to _history, and return the row.

    *cashflow* is the proj_cashflow[i] dict from env.step().
    """
    key = (episode_id, proj_index)
    cum = _cum.setdefault(key, {
        "inflow": 0.0, "outflow": 0.0, "net_cf": 0.0,
        "deficit": 0.0, "alloc": 0.0, "interest": 0.0, "budget_draw": 0.0,
    })

    eac     = ps.get("eac", proj["budget"])
    eac_bac = eac / proj["budget"] if proj["budget"] > 0 else 1.0
    alloc   = cashflow.get("allocation",    0.0)
    interest= cashflow.get("interest_cost", 0.0)
    draw    = cashflow.get("treasury_draw", 0.0)

    adv_in  = cashflow.get("advance",          0.0)
    ms_in   = cashflow.get("milestone_net",    0.0)
    ret_in  = cashflow.get("retention_release",0.0)
    sett_in = cashflow.get("settlement",       0.0)

    inflow_p  = adv_in + ms_in + ret_in + max(0.0, sett_in)
    outflow_p = alloc + interest
    net_cf_p  = inflow_p - outflow_p
    deficit_p = max(0.0, -net_cf_p)

    cum["inflow"]      += inflow_p
    cum["outflow"]     += outflow_p
    cum["net_cf"]      += net_cf_p
    cum["deficit"]     += deficit_p
    cum["alloc"]       += alloc
    cum["interest"]    += interest
    cum["budget_draw"] += draw

    pdt     = cfg.get("plan_deviation_threshold", 0.10)
    b_idle  = alloc < 1e-9 and ps.get("status") == "active"
    b_dev   = ps.get("plan_deviation", 0.0) > pdt
    b_sched = ps.get("schedule_slip",  0.0) > proj["schedule_cap"]
    b_cost  = eac_bac > proj["cost_cap"]
    b_dl    = (
        t >= proj["finish"] + proj["schedule_cap"]
        and ps.get("status") == "active"
    )
    b_any   = b_idle or b_dev or (b_sched and b_cost) or b_dl

    row = {
        "t":                   t,
        "status":              ps.get("status") or "not_started",
        "inflow_period":       inflow_p,
        "outflow_period":      outflow_p,
        "net_cf_period":       net_cf_p,
        "cash_deficit_period": deficit_p,
        "allocation":          alloc,
        "interest_cost":       interest,
        "budget_draw":         draw,
        "inflow_cum":          cum["inflow"],
        "outflow_cum":         cum["outflow"],
        "net_cf_cum":          cum["net_cf"],
        "cash_deficit_cum":    cum["deficit"],
        "alloc_cum":           cum["alloc"],
        "interest_cum":        cum["interest"],
        "budget_draw_cum":     cum["budget_draw"],
        "progress":            ps.get("progress",       0.0),
        "progress_plan":       ps.get("progress_plan",  0.0),
        "plan_deviation":      ps.get("plan_deviation", 0.0),
        "spi":                 ps.get("spi",            1.0),
        "cpi":                 ps.get("cpi",            1.0),
        "tcpi":                ps.get("tcpi",           1.0),
        "eac":                 eac,
        "eac_bac":             eac_bac,
        "schedule_slip":       ps.get("schedule_slip",  0.0),
        "forecast_finish":     ps.get("forecast_finish", float(proj["finish"])),
        "cure_remaining":      ps.get("cure_remaining", proj.get("cure_length", 0)),
        "breach_idle":         b_idle,
        "breach_deviation":    b_dev,
        "breach_schedule":     b_sched,
        "breach_cost":         b_cost,
        "breach_deadline":     b_dl,
        "any_breach":          b_any,
    }

    _history.setdefault(key, []).append(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = W) -> str:
    return char * width

def _header(title: str, char: str = "═", width: int = W) -> str:
    pad   = max(0, width - len(title) - 4)
    left  = pad // 2
    right = pad - left
    return f"{char * left}  {title}  {char * right}"

def _fmt_mu(v) -> str:
    return f"{v:>12,.2f}" if v is not None else f"{'—':>12}"

def _fmt_f4(v) -> str:
    return f"{v:>10.4f}" if v is not None else f"{'—':>10}"

def _fmt_f2(v) -> str:
    return f"{v:>8.2f}" if v is not None else f"{'—':>8}"

def _fmt_pct(v) -> str:
    return f"{v*100:>7.1f}%" if v is not None else f"{'—':>8}"

def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"

def _breach(v: bool) -> str:
    return "X BREACH" if v else "OK      "

def _status_str(v) -> str:
    return (v or "NOT_STARTED").upper()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — PROJECT IDENTITY
# ─────────────────────────────────────────────────────────────────────────────

def print_project_identity(proj: dict) -> None:
    """Print static project parameters."""
    print(_header(f"Project {proj['i']}  |  Identity & Parameters"))
    rows = [
        ("Budget (BAC)",           f"{proj['budget']:,.2f}"),
        ("Contract Price",         f"{proj['price']:,.2f}"),
        ("Margin",                 f"{proj['margin']*100:.1f}%"),
        ("Start",                  str(proj["start"])),
        ("Finish (planned)",       str(proj["finish"])),
        ("Duration",               str(proj["duration"])),
        ("S-curve a",              f"{proj['scurve_a']:.3f}"),
        ("S-curve b",              f"{proj['scurve_b']:.3f}"),
        ("Advance %",              f"{proj['advance_percent']*100:.1f}%"),
        ("Advance Trigger",        f"{proj['advance_trigger']*100:.1f}%"),
        ("Advance Recovery Rate",  f"{proj['advance_recovery']*100:.1f}%"),
        ("Retention Rate",         f"{proj['retention_rate']*100:.1f}%"),
        ("Schedule Cap (periods)", str(proj["schedule_cap"])),
        ("Cost Cap (xBAC)",        f"{proj['cost_cap']:.2f}x"),
        ("Cure Length",            str(proj["cure_length"])),
    ]
    col_w = 28
    for label, val in rows:
        print(f"  {label:<{col_w}} {val}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — PAYMENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def print_payment_profile(proj: dict, milestones: list, ps: dict) -> None:
    """
    Print the milestone payment table including advance, each milestone,
    and the deferred retention release row.
    """
    print(_header(f"Project {proj['i']}  |  Payment Profile"))

    price            = proj["price"]
    advance_pct      = proj["advance_percent"]
    adv_recovery_rt  = proj["advance_recovery"]
    retention_rt     = proj["retention_rate"]
    advance_received = ps.get("advance_received", 0.0)

    # column widths: label, threshold, weight, gross, adv_recov,
    #                cum_recov, retention, net_pay, earliest, status
    c = [22, 10, 8, 12, 12, 14, 12, 12, 10, 16]
    hdr = (
        f"  {'Milestone':<{c[0]}} {'Threshold':>{c[1]}} {'Weight':>{c[2]}}"
        f" {'Gross':>{c[3]}} {'Adv.Recov':>{c[4]}} {'Cum.Recov':>{c[5]}}"
        f" {'Retention':>{c[6]}} {'Net Pay':>{c[7]}} {'Earliest':>{c[8]}} {'Status':<{c[9]}}"
    )
    print(hdr)
    print("  " + _sep("─", W - 2))

    # Advance row
    adv_gross  = advance_pct * price
    adv_status = "PAID" if advance_received > 0 else "PENDING"
    print(
        f"  {'Advance':<{c[0]}} {'0%':>{c[1]}} {advance_pct*100:>{c[2]-1}.1f}%"
        f" {adv_gross:>{c[3]},.2f} {'---':>{c[4]}} {'---':>{c[5]}}"
        f" {'---':>{c[6]}} {adv_gross:>{c[7]},.2f}"
        f" {('t='+str(proj['start'])):>{c[8]}} {adv_status:<{c[9]}}"
    )

    # Milestone rows — recompute recovery caps using cumulative logic
    cum_recovered = 0.0
    for ms in milestones:
        is_final         = ms["threshold"] >= 1.0
        gross            = ms["payment_weight"] * price
        desired_recovery = gross * adv_recovery_rt
        remaining_cap    = max(0.0, advance_received - cum_recovered)
        recovery         = min(desired_recovery, remaining_cap)
        cum_recovered   += recovery
        retention        = 0.0 if is_final else gross * retention_rt
        net              = gross - recovery - retention

        if ms["certified"]:
            ms_status = f"CERTIFIED t={ms.get('certified_t')}"
        elif ps.get("progress", 0.0) >= ms["threshold"]:
            ms_status = "ELIGIBLE"
        else:
            ms_status = "PENDING"

        label = "Final MS / Completion" if is_final else f"MS {ms['j']+1}"
        print(
            f"  {label:<{c[0]}} {ms['threshold']*100:>{c[1]-1}.0f}%"
            f" {ms['payment_weight']*100:>{c[2]-1}.1f}%"
            f" {gross:>{c[3]},.2f} {recovery:>{c[4]},.2f} {cum_recovered:>{c[5]},.2f}"
            f" {retention:>{c[6]},.2f} {net:>{c[7]},.2f}"
            f" {('t='+str(ms['earliest_t'])):>{c[8]}} {ms_status:<{c[9]}}"
        )

    # Retention release row
    expected_retention = sum(
        ms["payment_weight"] * price * retention_rt
        for ms in milestones if ms["threshold"] < 1.0
    )
    ret_held     = ps.get("retention_held", 0.0)
    ret_released = ps.get("retention_released", False)
    display_ret  = ret_held if ret_held > 0 else expected_retention
    ret_status   = (
        "RELEASED" if ret_released
        else (f"HELD {ret_held:,.2f}" if ret_held > 0 else "PENDING")
    )
    print(
        f"  {'Retention Release':<{c[0]}} {'100%':>{c[1]}} {'---':>{c[2]}}"
        f" {display_ret:>{c[3]},.2f} {'---':>{c[4]}} {'---':>{c[5]}}"
        f" {'---':>{c[6]}} {display_ret:>{c[7]},.2f}"
        f" {('t='+str(proj['finish'])):>{c[8]}} {ret_status:<{c[9]}}"
    )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — BOUNDARY & TERMINATION STATUS
# ─────────────────────────────────────────────────────────────────────────────

def print_termination_status(proj_index: int, term: dict) -> None:
    """Print breach flags and cure counter for one project."""
    print(_header(f"Project {proj_index}  |  Boundary & Termination Status"))
    fields = [
        ("Idle Breach",      term.get("breach_idle",      False)),
        ("Deviation Breach", term.get("breach_deviation", False)),
        ("Schedule Breach",  term.get("breach_schedule",  False)),
        ("Cost Breach",      term.get("breach_cost",      False)),
        ("Deadline Breach",  term.get("breach_deadline",  False)),
        ("Any Breach",       term.get("any_breach",       False)),
    ]
    col_w = 22
    for label, val in fields:
        print(f"  {label:<{col_w}}  {_breach(val)}")
    cure     = term.get("cure_remaining")
    cure_str = _fmt_int(cure)
    indicator = "  << LOW" if cure is not None and cure <= 2 else ""
    print(f"  {'Cure Periods Remaining':<{col_w}}  {cure_str}{indicator}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — PERIOD TIMESERIES
# ─────────────────────────────────────────────────────────────────────────────

def print_timeseries(episode_id: str, proj_index: int) -> None:
    """Print all three sub-tables (periodic CF, cumulative CF, EVM) for one project."""
    rows = _history.get((episode_id, proj_index), [])
    if not rows:
        return

    print(_header(f"Project {proj_index}  |  Period Timeseries"))

    # Group A: periodic cash flow
    print(
        f"\n  {'t':>3}  {'Status':<12}"
        f"  {'Inflow':>11}  {'Outflow':>11}  {'Net CF':>11}  {'Deficit':>10}"
        f"  {'Alloc':>11}  {'Interest':>11}  {'BudgetDraw':>11}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}  {_status_str(r['status']):<12}"
            f"  {r['inflow_period']:>11,.2f}  {r['outflow_period']:>11,.2f}"
            f"  {r['net_cf_period']:>11,.2f}  {r['cash_deficit_period']:>10,.2f}"
            f"  {r['allocation']:>11,.2f}  {r['interest_cost']:>11,.2f}"
            f"  {r['budget_draw']:>11,.2f}"
        )

    # Group B: cumulative cash flow
    print(
        f"\n  {'t':>3}"
        f"  {'Sum Inflow':>11}  {'Sum Outflow':>11}  {'Sum NetCF':>11}  {'Sum Deficit':>11}"
        f"  {'Sum Alloc':>11}  {'Sum Interest':>12}  {'Sum BudgDraw':>12}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}"
            f"  {r['inflow_cum']:>11,.2f}  {r['outflow_cum']:>11,.2f}"
            f"  {r['net_cf_cum']:>11,.2f}  {r['cash_deficit_cum']:>11,.2f}"
            f"  {r['alloc_cum']:>11,.2f}  {r['interest_cum']:>12,.2f}"
            f"  {r['budget_draw_cum']:>12,.2f}"
        )

    # Group C: EVM metrics
    print(
        f"\n  {'t':>3}"
        f"  {'Progress':>9}  {'Plan':>9}  {'Deviation':>9}"
        f"  {'SPI(t)':>7}  {'CPI':>7}  {'TCPI':>7}"
        f"  {'EAC':>12}  {'EAC/BAC':>8}  {'SchedSlip':>10}  {'FcstFinish':>11}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}"
            f"  {r['progress']:>9.4f}  {r['progress_plan']:>9.4f}  {r['plan_deviation']:>9.4f}"
            f"  {r['spi']:>7.4f}  {r['cpi']:>7.4f}  {r['tcpi']:>7.4f}"
            f"  {r['eac']:>12,.2f}  {r['eac_bac']:>8.4f}"
            f"  {r['schedule_slip']:>10.2f}  {r['forecast_finish']:>11.2f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_portfolio_summary(
    t: int,
    budget: float,
    horizon: int,
    total_reward: float,
    info: Optional[dict] = None,
) -> None:
    """
    Print the portfolio-level header block, plus per-project cashflow table
    if *info* is provided (i.e. after a step has executed).
    """
    print()
    print(_sep("="))
    print(
        f"  PORTFOLIO"
        f"   Period {t}"
        f"   Budget {budget:,.2f}"
        f"   Horizon {horizon}"
        f"   Cumulative Reward {total_reward:+.4f}"
    )
    print(_sep("="))

    if info:
        print(
            f"\n  {'P':>2}  {'Alloc':>11}  {'Interest':>11}  {'Advance':>11}"
            f"  {'Milestone':>11}  {'Retention':>11}  {'Settlement':>11}  {'Net CF':>11}"
        )
        print("  " + _sep("─", W - 2))
        for cf in info.get("cashflow", []):
            inflow  = (
                cf["advance"] + cf["milestone_net"]
                + cf["retention_release"] + max(0.0, cf["settlement"])
            )
            outflow = cf["allocation"] + cf["interest_cost"]
            net     = inflow - outflow
            print(
                f"  {cf['i']:>2}  {cf['allocation']:>11,.2f}  {cf['interest_cost']:>11,.2f}"
                f"  {cf['advance']:>11,.2f}  {cf['milestone_net']:>11,.2f}"
                f"  {cf['retention_release']:>11,.2f}  {cf['settlement']:>11,.2f}"
                f"  {net:>11,.2f}"
            )
        reward = info.get("_reward", 0.0)
        print(f"\n  Step reward: {reward:+.4f}   Budget after: {budget:,.2f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE RENDER — all panels for every project in one call
# ─────────────────────────────────────────────────────────────────────────────

def render_all_projects(
    episode_id: str,
    projects: list[dict],
    proj_states: list[dict],
    milestones_list: list[list[dict]],
    cfg: dict,
    cashflow_by_proj: dict[int, dict],
    t: int,
) -> None:
    """
    Record a timeseries row then print all four panels for every project.

    Parameters
    ----------
    episode_id       : current episode UUID (for history keying)
    projects         : list of project param dicts
    proj_states      : list of project state dicts (same order)
    milestones_list  : list of milestone lists (same order)
    cfg              : environment config dict (for plan_deviation_threshold)
    cashflow_by_proj : {proj_index: cashflow_dict} from env.step info
    t                : episode timestep just executed (used for row label)
    """
    for i, (proj, ps) in enumerate(zip(projects, proj_states)):
        cashflow = cashflow_by_proj.get(i, {
            "allocation": 0.0, "advance": 0.0,
            "milestone_net": 0.0, "retention_release": 0.0,
            "settlement": 0.0, "interest_cost": 0.0, "treasury_draw": 0.0,
        })

        row = record_row(episode_id, i, t, dict(ps), proj, cfg, cashflow)

        term = {
            "cure_remaining":   row["cure_remaining"],
            "breach_idle":      row["breach_idle"],
            "breach_deviation": row["breach_deviation"],
            "breach_schedule":  row["breach_schedule"],
            "breach_cost":      row["breach_cost"],
            "breach_deadline":  row["breach_deadline"],
            "any_breach":       row["any_breach"],
        }

        print_project_identity(proj)
        print_payment_profile(proj, milestones_list[i], ps)
        print_termination_status(i, term)
        print_timeseries(episode_id, i)