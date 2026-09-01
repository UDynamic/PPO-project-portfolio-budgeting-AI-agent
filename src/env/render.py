# src/env/render.py
#
# ANSI render functions for the Portfolio Budgeting environment.
#
# All formatting and print logic lives here.  Two consumers:
#   - env.render()       calls render()
#   - src/play/play.py   imports render() directly; contains no formatting
#
# Four panels per project, printed sequentially:
#   1. Project Identity       static params
#   2. Payment Profile        milestone table (reads precomputed profile values)
#   3. Boundary & Termination breach flags + tolerance counter
#   4. Period Timeseries      Group A: periodic CF, Group B: cumulative CF,
#                             Group C: EVM
#
# Preceded by a Portfolio Summary block.
#
# Key name conventions match schema and module contracts exactly:
#   proj["bac"], proj["planned_start/finish/duration"]
#   proj["finish_delay_cap"], proj["cost_overrun_cap"]
#   proj["termination_tolerance"], proj["profit_percent"]
#   ps["progress_actual"], ps["progress_delay_t"]
#   ps["projected_finish_delay"], ps["projected_finish"]
#   ps["tolerance_remain"], ps["breach_flags"]
#   ms["progress_threshold"], ms["timestep_threshold"]
#   ms["gross_payment"], ms["net_payment"], etc.
#
# History store
# -------------
# The timeseries panel needs rows from all prior periods.  render.py owns
# a module-level _history dict keyed by (episode_id, proj_index).
# Call reset_history(episode_id, n_projects) at the start of each episode.

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

W = 110   # total print width


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY STORE
# ─────────────────────────────────────────────────────────────────────────────

_history: dict[tuple[str, int], list[dict]] = {}
_cum:     dict[tuple[str, int], dict]       = {}


def reset_history(episode_id: str, n_projects: int) -> None:
    """
    Clear history for a new episode and initialise cumulative accumulators.
    Call once from env.reset() before the first render.
    """
    for i in range(n_projects):
        key           = (episode_id, i)
        _history[key] = []
        _cum[key]     = {
            "inflow": 0.0, "outflow": 0.0, "net_cf": 0.0,
            "deficit": 0.0, "alloc": 0.0, "interest": 0.0,
            "treasury_draw": 0.0,
        }


def _record_row(episode_id: str, proj_index: int,
                t: int, ps: dict, proj: dict,
                cf: dict) -> dict:
    """
    Build one timeseries row, update cumulative accumulators, append to
    history, and return the row.

    Breach flags are read from ps["breach_flags"] — not recomputed here.
    cf is proj_cf[i] from env.step().
    """
    key = (episode_id, proj_index)
    cum = _cum.setdefault(key, {
        "inflow": 0.0, "outflow": 0.0, "net_cf": 0.0,
        "deficit": 0.0, "alloc": 0.0, "interest": 0.0, "treasury_draw": 0.0,
    })

    bac      = proj["bac"]
    eac      = ps.get("eac", bac)
    eac_bac  = eac / bac if bac > 0 else 1.0

    alloc    = cf.get("allocation",   0.0)
    interest = cf.get("interest",     0.0)
    draw     = cf.get("treasury_draw",0.0)
    adv_in   = cf.get("advance",      0.0)
    ms_in    = cf.get("milestone_net",0.0)
    sett_in  = cf.get("settlement",   0.0)

    inflow_p  = adv_in + ms_in + max(0.0, sett_in)
    outflow_p = alloc + interest
    net_cf_p  = inflow_p - outflow_p
    deficit_p = max(0.0, -net_cf_p)

    cum["inflow"]        += inflow_p
    cum["outflow"]       += outflow_p
    cum["net_cf"]        += net_cf_p
    cum["deficit"]       += deficit_p
    cum["alloc"]         += alloc
    cum["interest"]      += interest
    cum["treasury_draw"] += draw

    flags = ps.get("breach_flags", {})

    row = {
        "t":                    t,
        "status":               ps.get("status") or "pending",
        "inflow_period":        inflow_p,
        "outflow_period":       outflow_p,
        "net_cf_period":        net_cf_p,
        "cash_deficit_period":  deficit_p,
        "allocation":           alloc,
        "interest":             interest,
        "treasury_draw":        draw,
        "inflow_cum":           cum["inflow"],
        "outflow_cum":          cum["outflow"],
        "net_cf_cum":           cum["net_cf"],
        "cash_deficit_cum":     cum["deficit"],
        "alloc_cum":            cum["alloc"],
        "interest_cum":         cum["interest"],
        "treasury_draw_cum":    cum["treasury_draw"],
        "progress_actual":      ps.get("progress_actual",   0.0),
        "progress_plan_t":      ps.get("progress_plan_t",   0.0),
        "progress_delay_t":     ps.get("progress_delay_t",  0.0),
        "spi":                  ps.get("spi",               1.0),
        "cpi":                  ps.get("cpi",               1.0),
        "tcpi":                 ps.get("tcpi",              1.0),
        "eac":                  eac,
        "eac_bac":              eac_bac,
        "projected_finish_delay": ps.get("projected_finish_delay", 0.0),
        "projected_finish":     ps.get("projected_finish", float(proj["planned_finish"])),
        "tolerance_remain":     ps.get("tolerance_remain", proj["termination_tolerance"]),
        "abandoned":            flags.get("abandoned",           False),
        "over_progress_delay":  flags.get("over_progress_delay", False),
        "over_finish_delay":    flags.get("over_finish_delay",   False),
        "over_cost_overrun":    flags.get("over_cost_overrun",   False),
        "over_duration_window": flags.get("over_duration_window",False),
        "over_any":             flags.get("over_any",            False),
    }

    _history.setdefault(key, []).append(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
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

def _fmt_pct(v) -> str:
    return f"{v*100:>7.1f}%" if v is not None else f"{'—':>8}"

def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"

def _breach(v: bool) -> str:
    return "X BREACH" if v else "OK      "

def _status_str(v) -> str:
    return (v or "PENDING").upper()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — PROJECT IDENTITY
# ─────────────────────────────────────────────────────────────────────────────

def _print_project_identity(proj: dict) -> None:
    print(_header(f"Project {proj['i']}  |  Identity & Parameters"))
    col_w = 28
    rows = [
        ("BAC",                    f"{proj['bac']:,.2f}"),
        ("Contract Price",         f"{proj['price']:,.2f}"),
        ("Profit %",               f"{proj['profit_percent']*100:.1f}%"),
        ("Planned Start",          str(proj["planned_start"])),
        ("Planned Finish",         str(proj["planned_finish"])),
        ("Planned Duration",       str(proj["planned_duration"])),
        ("S-curve a",              f"{proj['scurve_a']:.3f}"),
        ("S-curve b",              f"{proj['scurve_b']:.3f}"),
        ("Advance %",              f"{proj['advance_percent']*100:.1f}%"),
        ("Advance Recovery Rate",  f"{proj['advance_recovery']*100:.1f}%"),
        ("Retention Rate",         f"{proj['retention_rate']*100:.1f}%"),
        ("Finish Delay Cap",       str(proj["finish_delay_cap"])),
        ("Cost Overrun Cap",       f"{proj['cost_overrun_cap']:.2f}x"),
        ("Termination Tolerance",  str(proj["termination_tolerance"])),
    ]
    for label, val in rows:
        print(f"  {label:<{col_w}} {val}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — PAYMENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def _print_payment_profile(proj: dict, milestones: list[dict],
                            ps: dict) -> None:
    """
    Print the milestone payment table.

    Reads all payment values from the precomputed milestone profile
    (gross_payment, advance_recovery, retention_withheld,
    retention_released, net_payment).  No financial recomputation here.

    j=0  advance
    j=1..n-1  interim milestones
    j=n  final milestone (includes retention_released)
    """
    print(_header(f"Project {proj['i']}  |  Payment Profile"))

    # column widths
    c = [22, 10, 8, 12, 12, 12, 12, 12, 10, 16]
    hdr = (
        f"  {'Milestone':<{c[0]}} {'Threshold':>{c[1]}} {'Weight':>{c[2]}}"
        f" {'Gross':>{c[3]}} {'Adv.Rec':>{c[4]}} {'Ret.With':>{c[5]}}"
        f" {'Ret.Rel':>{c[6]}} {'Net Pay':>{c[7]}} {'EarliestT':>{c[8]}}"
        f" {'Status':<{c[9]}}"
    )
    print(hdr)
    print("  " + _sep("─", W - 2))

    for ms in milestones:
        j        = ms["j"]
        is_final = ms["progress_threshold"] >= 1.0

        if j == 0:
            label = "Advance"
        elif is_final:
            label = "Final Milestone"
        else:
            label = f"Milestone {j}"

        if ms["certified"]:
            status = f"PAID  t={ms['certified_t']}"
        elif ps.get("progress_actual", 0.0) >= ms["progress_threshold"]:
            status = "ELIGIBLE"
        else:
            status = "PENDING"

        threshold_str = (
            "—" if j == 0
            else f"{ms['progress_threshold']*100:.0f}%"
        )
        weight_str = (
            "—" if j == 0
            else f"{ms['payment_weight']*100:.1f}%"
        )

        print(
            f"  {label:<{c[0]}} {threshold_str:>{c[1]}} {weight_str:>{c[2]}}"
            f" {ms['gross_payment']:>{c[3]},.2f}"
            f" {ms['advance_recovery']:>{c[4]},.2f}"
            f" {ms['retention_withheld']:>{c[5]},.2f}"
            f" {ms['retention_released']:>{c[6]},.2f}"
            f" {ms['net_payment']:>{c[7]},.2f}"
            f" {('t='+str(ms['timestep_threshold'])):>{c[8]}}"
            f" {status:<{c[9]}}"
        )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — BOUNDARY & TERMINATION STATUS
# ─────────────────────────────────────────────────────────────────────────────

def _print_termination_status(proj_index: int, ps: dict,
                               proj: dict) -> None:
    print(_header(f"Project {proj_index}  |  Boundary & Termination Status"))
    flags = ps.get("breach_flags", {})
    col_w = 26
    fields = [
        ("Abandoned (idle)",       flags.get("abandoned",           False)),
        ("Over Progress Delay",    flags.get("over_progress_delay", False)),
        ("Over Finish Delay",      flags.get("over_finish_delay",   False)),
        ("Over Cost Overrun",      flags.get("over_cost_overrun",   False)),
        ("Over Duration Window",   flags.get("over_duration_window",False)),
        ("Any Breach",             flags.get("over_any",            False)),
    ]
    for label, val in fields:
        print(f"  {label:<{col_w}}  {_breach(val)}")

    tol     = ps.get("tolerance_remain", proj["termination_tolerance"])
    low_str = "  << LOW" if tol <= 2 else ""
    print(f"  {'Tolerance Remaining':<{col_w}}  {_fmt_int(tol)}{low_str}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — PERIOD TIMESERIES
# ─────────────────────────────────────────────────────────────────────────────

def _print_timeseries(episode_id: str, proj_index: int) -> None:
    rows = _history.get((episode_id, proj_index), [])
    if not rows:
        return

    print(_header(f"Project {proj_index}  |  Period Timeseries"))

    # Group A: periodic cash flow
    print(
        f"\n  {'t':>3}  {'Status':<12}"
        f"  {'Inflow':>11}  {'Outflow':>11}  {'Net CF':>11}  {'Deficit':>10}"
        f"  {'Alloc':>11}  {'Interest':>11}  {'Treasury':>11}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}  {_status_str(r['status']):<12}"
            f"  {r['inflow_period']:>11,.2f}  {r['outflow_period']:>11,.2f}"
            f"  {r['net_cf_period']:>11,.2f}  {r['cash_deficit_period']:>10,.2f}"
            f"  {r['allocation']:>11,.2f}  {r['interest']:>11,.2f}"
            f"  {r['treasury_draw']:>11,.2f}"
        )

    # Group B: cumulative cash flow
    print(
        f"\n  {'t':>3}"
        f"  {'Sum Inflow':>11}  {'Sum Outflow':>11}  {'Sum NetCF':>11}"
        f"  {'Sum Deficit':>11}  {'Sum Alloc':>11}  {'Sum Int':>11}"
        f"  {'Sum Treasury':>12}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}"
            f"  {r['inflow_cum']:>11,.2f}  {r['outflow_cum']:>11,.2f}"
            f"  {r['net_cf_cum']:>11,.2f}  {r['cash_deficit_cum']:>11,.2f}"
            f"  {r['alloc_cum']:>11,.2f}  {r['interest_cum']:>11,.2f}"
            f"  {r['treasury_draw_cum']:>12,.2f}"
        )

    # Group C: EVM
    print(
        f"\n  {'t':>3}"
        f"  {'Progress':>9}  {'Plan':>9}  {'Delay':>9}"
        f"  {'SPI(t)':>7}  {'CPI':>7}  {'TCPI':>7}"
        f"  {'EAC':>12}  {'EAC/BAC':>8}"
        f"  {'FinDelay':>9}  {'FcstFinish':>11}"
    )
    print("  " + _sep("─", W - 2))
    for r in rows:
        print(
            f"  {r['t']:>3}"
            f"  {r['progress_actual']:>9.4f}  {r['progress_plan_t']:>9.4f}"
            f"  {r['progress_delay_t']:>9.4f}"
            f"  {r['spi']:>7.4f}  {r['cpi']:>7.4f}  {r['tcpi']:>7.4f}"
            f"  {r['eac']:>12,.2f}  {r['eac_bac']:>8.4f}"
            f"  {r['projected_finish_delay']:>9.2f}"
            f"  {r['projected_finish']:>11.2f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_portfolio_summary(t: int, budget: float, initial_budget: float,
                              horizon: int, proj_cf: list[dict],
                              period_inflow: float,
                              period_outflow: float) -> None:
    print()
    print(_sep("="))
    budget_norm = budget / initial_budget if initial_budget > 0 else 1.0
    print(
        f"  PORTFOLIO"
        f"   Period {t}"
        f"   Budget {budget:,.2f}  ({budget_norm:.1%} of initial)"
        f"   Horizon {horizon}"
    )
    print(_sep("="))

    if proj_cf:
        print(
            f"\n  {'P':>2}  {'Alloc':>11}  {'Interest':>11}"
            f"  {'Advance':>11}  {'Milestone':>11}  {'Settlement':>11}"
            f"  {'Net CF':>11}"
        )
        print("  " + _sep("─", W - 2))
        for idx, cf in enumerate(proj_cf):
            inflow  = (
                cf.get("advance", 0.0)
                + cf.get("milestone_net", 0.0)
                + max(0.0, cf.get("settlement", 0.0))
            )
            outflow = cf.get("allocation", 0.0) + cf.get("interest", 0.0)
            net     = inflow - outflow
            print(
                f"  {idx:>2}"
                f"  {cf.get('allocation',    0.0):>11,.2f}"
                f"  {cf.get('interest',      0.0):>11,.2f}"
                f"  {cf.get('advance',       0.0):>11,.2f}"
                f"  {cf.get('milestone_net', 0.0):>11,.2f}"
                f"  {cf.get('settlement',    0.0):>11,.2f}"
                f"  {net:>11,.2f}"
            )
        print(
            f"\n  Period inflow: {period_inflow:,.2f}"
            f"   Period outflow: {period_outflow:,.2f}"
            f"   Net: {period_inflow - period_outflow:+,.2f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC RENDER ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render(t: int,
           budget: float,
           horizon: int,
           initial_budget: float,
           projects: list[dict],
           proj_state: list[dict],
           milestones: list[list[dict]],
           proj_cf: Optional[list[dict]] = None,
           period_inflow: float = 0.0,
           period_outflow: float = 0.0,
           episode_id: str = "") -> None:
    """
    Full render for one timestep.  Called from env.render() and play.py.

    Records a timeseries row for each project then prints all panels.
    """
    if proj_cf is None:
        proj_cf = [{} for _ in projects]

    _print_portfolio_summary(
        t, budget, initial_budget, horizon,
        proj_cf, period_inflow, period_outflow,
    )

    for i, (proj, ps, ms_list, cf) in enumerate(
        zip(projects, proj_state, milestones, proj_cf)
    ):
        _record_row(episode_id, i, t, ps, proj, cf)
        _print_project_identity(proj)
        _print_payment_profile(proj, ms_list, ps)
        _print_termination_status(i, ps, proj)
        _print_timeseries(episode_id, i)