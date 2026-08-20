# run_env.py

import re
import sys
import importlib
import sqlite3
from db_init import init_db

# Pre-compiled ANSI escape stripper — used for box padding calculations
_ANSI_RE = re.compile(r'\033\[[0-9;]*m')
from env import PortfolioEnv


# ═════════════════════════════════════════════════════════════
# CONFIG REGISTRY
# Maps a short label → (module_name, seed_function_name, description)
# Add a new entry here whenever a new config_seed_*.py is created.
# ═════════════════════════════════════════════════════════════

CONFIG_REGISTRY = {
    "1": ("config_seed_sp",  "seed_single_project", "Single project  — CFG-SINGLE-001"),
    "2": ("config_seed_dp",  "seed_dual_project",   "Dual project    — CFG-DUAL-001"),
}


# ═════════════════════════════════════════════════════════════
# ANSI COLOR PALETTE  — neon theme
# ═════════════════════════════════════════════════════════════

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"
    BRED     = "\033[91m"
    BGREEN   = "\033[92m"
    BYELLOW  = "\033[93m"
    BBLUE    = "\033[94m"
    BMAGENTA = "\033[95m"
    BCYAN    = "\033[96m"
    BWHITE   = "\033[97m"
    # Neon pink — 256-colour (hot pink / neon rose)
    PINK     = "\033[38;5;206m"
    ORANGE   = "\033[38;5;214m"


def c(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def status_color(status):
    if status is None or status == "NOT_STARTED":
        return c("NOT_STARTED", C.DIM, C.WHITE)
    if status == "active":
        return c("ACTIVE",      C.BGREEN,   C.BOLD)
    if status == "completed":
        return c("COMPLETED",   C.BCYAN,    C.BOLD)
    if status == "terminated":
        return c("TERMINATED",  C.BRED,     C.BOLD)
    return c(status.upper(), C.BYELLOW)


# ═════════════════════════════════════════════════════════════
# LAYOUT PRIMITIVES
# ═════════════════════════════════════════════════════════════

W = 70   # ruler width

def ruler(ch="─", color=C.DIM):
    print(c("  " + ch * W, color))

def pink_ruler(ch="─"):
    """Neon pink divider — used for period borders."""
    print(c("  " + ch * W, C.PINK, C.BOLD))

def section(title, color=C.BYELLOW):
    print()
    print(c("  " + title, C.BOLD, color))
    ruler()

def kv(label, value, unit="", value_color=C.BWHITE, indent=4):
    pad = " " * indent
    lbl = c(f"{pad}{label:<36}", C.CYAN)
    val = c(str(value), value_color, C.BOLD)
    u   = c(f"  {unit}", C.DIM) if unit else ""
    print(lbl + val + u)

def blank():
    print()

def header(title, color=C.BBLUE):
    blank()
    ruler("═", color)
    print(c(f"  {title:^{W}}", C.BOLD, color))
    ruler("═", color)

def sub_header(title, color=C.DIM):
    print(c(f"    ── {title}", color))


# ═════════════════════════════════════════════════════════════
# NEXT-PERIOD TRAJECTORY BOX
# ═════════════════════════════════════════════════════════════

def print_trajectory_box(next_period: int, prog_actual: float,
                         prog_plan: float, delta: float,
                         next_ms_threshold_gap: float,
                         next_ms_net_payment: float,
                         next_ms_earliest_t,
                         next_ms_is_final: bool,
                         retention_held: float,
                         t_episode: int):
    """
    Renders the 'Next Period Trajectory' box in orange/neon style.

    next_period  — the episode period the agent is about to execute
                   (always t_episode + 1, passed in from print_period_state)
    t_episode    — the current episode period shown in the period header
    """
    BOX_W   = 52          # inner content width (chars between │ and │)
    oc      = C.ORANGE
    dlt_col = C.BGREEN if delta >= 0 else C.BRED

    # ── payment proximity coloring ───────────────────────────
    if next_ms_threshold_gap <= 0.0:
        gap_col = C.BGREEN    # already past threshold, awaiting earliest_t
    elif next_ms_threshold_gap <= 0.10:
        gap_col = C.BGREEN    # very close
    elif next_ms_threshold_gap <= 0.25:
        gap_col = C.BYELLOW
    else:
        gap_col = C.BRED

    # Earliest-t lock status — compare against next_period (the period being decided)
    if next_ms_earliest_t is None:
        lock_str = "—"
        lock_col = C.DIM
    elif next_period >= next_ms_earliest_t:
        lock_str = f"t={next_ms_earliest_t}  (unlocked ✔)"
        lock_col = C.BGREEN
    else:
        periods_locked = next_ms_earliest_t - next_period
        lock_str = f"t={next_ms_earliest_t}  ({periods_locked} period{'s' if periods_locked != 1 else ''} away)"
        lock_col = C.BYELLOW

    final_tag = c("  ★ FINAL", C.BYELLOW) if next_ms_is_final else ""
    pay_col   = C.BGREEN if next_ms_net_payment > 0 else C.DIM

    top_label = " Next Period Trajectory "
    top_fill  = "─" * (BOX_W - len(top_label) - 1)
    top_line  = "┌" + "─" + top_label + top_fill + "┐"
    div_line  = "├" + "─" * BOX_W + "┤"
    bot_line  = "└" + "─" * BOX_W + "┘"

    def box_row(label, value, val_col=C.ORANGE, suffix=""):
        colored_inner = (
            c(f"  {label:<28}", oc) +
            c(str(value), val_col, C.BOLD) +
            (c(suffix, C.DIM) if suffix else "")
        )
        raw_len = len(_ANSI_RE.sub("", colored_inner))
        pad = max(0, BOX_W - raw_len)
        left  = c("│", oc)
        right = c("│", oc)
        print(f"    {left}{colored_inner}{' ' * pad}{right}")

    def box_section(title):
        inner_plain = f"  {title}"
        pad = BOX_W - len(inner_plain)
        left  = c("│", oc)
        right = c("│", oc)
        print(f"    {left}{c(inner_plain, C.DIM)}{' ' * max(0, pad)}{right}")

    print()
    print("    " + c(top_line, oc))

    # ── Trajectory section ────────────────────────────────────
    box_row("Next period",               next_period)
    box_row("Progress so far",           f"{prog_actual:.4f}")
    box_row("Plan target (end of next)", f"{prog_plan:.4f}")
    box_row("Δ vs plan",                 f"{delta:+.4f}", dlt_col)

    # ── Divider ───────────────────────────────────────────────
    print("    " + c(div_line, oc))

    # ── Next milestone section ────────────────────────────────
    box_section(f"Next Milestone{final_tag}")
    box_row("Progress gap to trigger",
            f"{next_ms_threshold_gap:.4f}" if next_ms_threshold_gap > 0 else "threshold reached",
            gap_col)
    box_row("Net payment on trigger",
            f"{next_ms_net_payment:,.2f}" if next_ms_net_payment > 0 else "—",
            pay_col,
            "  monetary units")
    box_row("Earliest eligible period",  lock_str, lock_col)
    if retention_held > 0:
        box_row("Retention held (to release)",
                f"{retention_held:,.2f}", C.BCYAN, "  monetary units")

    print("    " + c(bot_line, oc))


# ═════════════════════════════════════════════════════════════
# STATIC PROFILE  (printed once after reset)
# ═════════════════════════════════════════════════════════════

def print_portfolio_profile(env: PortfolioEnv, state: dict, episode_number: int = 1):

    header("PORTFOLIO ENVIRONMENT — STATIC PROFILE")

    # ── EPISODE ──────────────────────────────────────────────
    section("EPISODE")
    kv("Episode",         f"#{episode_number}",            value_color=C.BWHITE)
    kv("Episode ID",      env.episode_id,                  value_color=C.DIM)
    kv("Config ID",       env.config_id,                   value_color=C.BMAGENTA)
    kv("Method",          env.method,                      value_color=C.BYELLOW)
    kv("Horizon",         state["horizon"],  "periods",    C.BYELLOW)
    kv("Discount factor", f"{env.discount:.4f}", "per period", C.BWHITE)

    # ── PORTFOLIO FINANCIALS ──────────────────────────────────
    section("PORTFOLIO FINANCIALS")
    total_bac   = sum(p["budget"] for p in env.projects)
    total_price = sum(p["price"]  for p in env.projects)
    total_adv   = sum(p["advance_percent"] * p["price"]
                      for p in env.projects if p["start"] == 0)
    initial_budget = env.cfg["initial_budget_p1"]
    kappa_raw = initial_budget / total_bac if total_bac > 0 else 0.0
    kappa_col = C.BGREEN if kappa_raw > 1.2 else C.BYELLOW if kappa_raw > 0.9 else C.BRED

    kv("Initial budget (B₀)",          f"{initial_budget:,.2f}", "monetary units", C.BGREEN)
    kv("Portfolio BAC (ΣBAC)",         f"{total_bac:,.2f}",      "monetary units", C.BWHITE)
    kv("Portfolio price (ΣP)",         f"{total_price:,.2f}",    "monetary units", C.BWHITE)
    kv("Budget tightness (κ=B₀/ΣBAC)", f"{kappa_raw:.3f}",      "B₀/ΣBAC",        kappa_col)
    kv("Advance credited (t=0)",       f"{total_adv:,.2f}",      "monetary units", C.BCYAN)
    kv("Opening budget",               f"{state['budget']:,.2f}", "monetary units", C.BGREEN)

    # ── PROJECTS  (nested: params → milestones → stochastic) ─
    section(f"PROJECTS  ({len(env.projects)} total)")

    for proj, ms_list in zip(env.projects, env.milestones):
        # ── Project header ────────────────────────────────────
        blank()
        print(c(f"  ┌─ Project {proj['i']} ", C.BCYAN, C.BOLD) +
              c("─" * (W - 12 - len(str(proj['i']))), C.BCYAN))

        # ── Contract parameters ───────────────────────────────
        print(c(f"  │", C.BCYAN))
        print(c(f"  │  ", C.BCYAN) + c("CONTRACT PARAMETERS", C.DIM))
        kv("BAC (budget at completion)",  f"{proj['budget']:,.2f}",     "monetary units", C.BWHITE,  indent=6)
        kv("Price (contract value)",      f"{proj['price']:,.2f}",      "monetary units", C.BGREEN,  indent=6)
        kv("Margin",                      f"{proj['margin']*100:.1f}%", "",               C.BYELLOW, indent=6)
        kv("Start period",                proj["start"],                 "",               C.WHITE,   indent=6)
        kv("Planned finish",              proj["finish"],                "",               C.WHITE,   indent=6)
        kv("Duration",                    proj["duration"],              "periods",        C.WHITE,   indent=6)
        kv("Schedule cap (max slip)",     proj["schedule_cap"],          "periods",        C.BYELLOW, indent=6)
        kv("Cost cap (max EAC/BAC)",      f"{proj['cost_cap']:.2f}×",   "",               C.BYELLOW, indent=6)
        kv("Cure period length",          proj["cure_length"],           "periods",        C.BYELLOW, indent=6)
        kv("Advance payment",             f"{proj['advance_percent']*100:.0f}%",
                                          f"= {proj['advance_percent']*proj['price']:.2f}",
                                          C.BCYAN, indent=6)
        kv("Advance recovery rate",       f"{proj['advance_recovery']*100:.0f}%",
                                          "deducted per milestone gross", C.DIM, indent=6)
        kv("Retention rate",              f"{proj['retention_rate']*100:.0f}%",
                                          "held per milestone, released at completion",
                                          C.DIM, indent=6)

        # ── S-curve ───────────────────────────────────────────
        print(c(f"  │", C.BCYAN))
        print(c(f"  │  ", C.BCYAN) + c("S-CURVE", C.DIM))
        a, b = proj["scurve_a"], proj["scurve_b"]
        shape = "front-loaded" if a < b else "back-loaded" if a > b else "symmetric"
        kv("α",          f"{a:.3f}", "", C.BWHITE,   indent=6)
        kv("β",          f"{b:.3f}", "", C.BWHITE,   indent=6)
        kv("Shape",      shape,      "", C.BMAGENTA, indent=6)

        # ── Stochastic parameters ─────────────────────────────
        print(c(f"  │", C.BCYAN))
        print(c(f"  │  ", C.BCYAN) + c("STOCHASTIC PARAMETERS", C.DIM))
        kv("Efficiency η distribution", env.cfg["efficiency_dist"], "", C.BMAGENTA, indent=6)
        kv("η lower bound",             env.cfg["efficiency_p1"],   "", C.BWHITE,   indent=6)
        kv("η upper bound",             env.cfg["efficiency_p2"],   "", C.BWHITE,   indent=6)

        # ── Milestones ────────────────────────────────────────
        print(c(f"  │", C.BCYAN))
        print(c(f"  │  ", C.BCYAN) +
              c(f"MILESTONES  ({len(ms_list)} total)", C.DIM))

        for ms in ms_list:
            gross     = ms["payment_weight"] * proj["price"]
            after_rec = gross * (1 - proj["advance_recovery"])
            after_ret = after_rec * (1 - proj["retention_rate"])
            is_final  = ms["threshold"] == 1.0

            print(c(f"  │", C.BCYAN))
            ms_tag = c(" ★ FINAL", C.BYELLOW) if is_final else ""
            print(c(f"  │    MS {ms['j']}  ", C.BCYAN) +
                  c(f"threshold {ms['threshold']*100:.1f}%", C.BYELLOW if is_final else C.WHITE) +
                  ms_tag)
            kv("Earliest certification", ms["earliest_t"],                  "period",         C.DIM,    indent=10)
            kv("Payment weight",         f"{ms['payment_weight']*100:.2f}%", "",               C.WHITE,  indent=10)
            kv("Gross payment",          f"{gross:.2f}",                    "monetary units", C.BGREEN, indent=10)
            kv("After advance recovery", f"{after_rec:.2f}",               "monetary units", C.BYELLOW,indent=10)
            kv("After retention",        f"{after_ret:.2f}",               "monetary units", C.BCYAN,  indent=10)

        # ── Project footer ────────────────────────────────────
        print(c(f"  └" + "─" * W, C.BCYAN))

    # ── HOW TO PLAY ────────────────────────────────────────────
    blank()
    ruler("═", C.PINK)
    print(c(f"  {'HOW TO PLAY':^{W}}", C.BOLD, C.PINK))
    ruler("─", C.PINK)
    blank()
    lines = [
        "At each period you will see the current status of every active project.",
        "You then decide how much budget to allocate to each active project.",
        "",
        "  • Enter one number per active project, separated by spaces.",
        "    Example — 2 active projects:   40 60",
        "    Example — 1 active project:    80",
        "",
        "  • Allocations must be ≥ 0 and their sum must not exceed the available",
        "    budget shown at the prompt.",
        "",
        "  • Allocation drives project progress: more funding → faster execution",
        "    → earlier milestones → earlier cash inflows.",
        "",
        "  • The environment ends when all projects are completed or terminated,",
        "    the budget is exhausted, or the episode horizon is reached.",
        "",
        "  • Your score is the discounted sum of all cash inflows (NPV).",
        "    Advance payments, milestone payments, and retention releases all count.",
        "    Termination penalties subtract from your score.",
    ]
    for line in lines:
        print(c(f"  {line}", C.BWHITE if line.startswith("  •") else
                              C.BCYAN  if line.startswith("At") or line.startswith("You") else
                              C.DIM    if line == "" else C.WHITE))
    blank()
    ruler("═", C.PINK)
    blank()

def print_milestone_history_box(ms_history: list):
    BOX_W = 52
    oc    = C.BCYAN

    top_label = " Milestone History "
    top_fill  = "─" * (BOX_W - len(top_label) - 1)
    top_line  = "┌" + "─" + top_label + top_fill + "┐"
    bot_line  = "└" + "─" * BOX_W + "┘"

    def ms_row(label, value, val_col=C.BWHITE, suffix=""):
        colored_inner = (
            c(f"  {label:<28}", oc) +
            c(str(value), val_col, C.BOLD) +
            (c(suffix, C.DIM) if suffix else "")
        )
        raw_len = len(_ANSI_RE.sub("", colored_inner))
        pad = max(0, BOX_W - raw_len)
        left  = c("│", oc)
        right = c("│", oc)
        print(f"    {left}{colored_inner}{' ' * pad}{right}")

    print()
    print("    " + c(top_line, oc))

    for ms in ms_history:
        certified  = ms["certified"]
        t_str      = f"t={ms['certified_t']}" if certified else "—"
        status_str = "CERTIFIED" if certified else "PENDING"
        status_col = C.BGREEN if certified else C.BYELLOW
        final_tag  = c("  ★ FINAL", C.BYELLOW) if ms["is_final"] else ""
        label_str  = (
            f"  MS {ms['j']}  "
            f"{ms['threshold']*100:.0f}%  "
            f"{t_str}"
        )
        net_label  = "net received" if certified else "net expected"
        net_col    = C.BGREEN if certified else C.DIM

        # header row
        colored_inner = (
            c(label_str, oc) +
            c(status_str, status_col, C.BOLD) +
            final_tag
        )
        raw_len = len(_ANSI_RE.sub("", colored_inner))
        pad = max(0, BOX_W - raw_len)
        print(f"    {c('│', oc)}{colored_inner}{' ' * pad}{c('│', oc)}")

        # amounts row
        ms_row(
            f"    gross {ms['gross']:.2f}",
            f"{ms['net']:.2f}",
            net_col,
            f"  {net_label}"
        )

    print("    " + c(bot_line, oc))


# ═════════════════════════════════════════════════════════════
# PERIOD STATE  (printed every step before allocation)
# ═════════════════════════════════════════════════════════════

def print_period_state(state: dict):
    t = state["t_episode"]

    blank()
    pink_ruler("─")
    print(
        c(f"  PERIOD {t}", C.BOLD, C.PINK) +
        c("  │  Budget: ", C.DIM) +
        c(f"{state['budget']:,.2f}", C.BGREEN, C.BOLD) +
        c(f"  │  Horizon: {state['horizon']}", C.DIM)
    )
    pink_ruler("─")

    for p in state["projects"]:
        status  = p["status"] or "NOT_STARTED"
        prog    = p["progress"]
        plan    = p["progress_plan"]
        delta   = prog - plan
        spi     = p["spi"]
        cpi     = p["cpi"]
        eac     = p["eac"]
        eac_bac = eac / p["budget"] if p["budget"] > 0 else 1.0
        slip    = p["schedule_slip"]
        cure    = p["cure_remaining"]
        tp      = p["t_project"]

        spi_col = C.BGREEN if spi >= 0.95 else C.BYELLOW if spi >= 0.80 else C.BRED
        cpi_col = C.BGREEN if cpi >= 0.95 else C.BYELLOW if cpi >= 0.80 else C.BRED
        slp_col = C.BGREEN if slip <= 0 else C.BYELLOW if slip <= 2 else C.BRED
        cur_col = C.BGREEN if cure >= 2 else C.BYELLOW if cure == 1 else C.BRED
        eac_col = C.BGREEN if eac_bac <= 1.05 else C.BYELLOW if eac_bac <= 1.20 else C.BRED

        blank()
        sub_header(f"Project {p['i']}  [{status_color(status)}]", C.BCYAN)
        kv("Status",         status_color(status), "", C.BWHITE)
        tcpi     = p.get("tcpi", 1.0)
        tcpi_col = C.BGREEN if tcpi <= 1.05 else C.BYELLOW if tcpi <= 1.10 else C.BRED

        kv("SPI(t)",         f"{spi:.4f}",         "", spi_col)
        kv("CPI",            f"{cpi:.4f}",         "", cpi_col)
        kv("TCPI",           f"{tcpi:.4f}",         "", tcpi_col)
        kv("EAC",            f"{eac:,.2f}",         "", eac_col)
        kv("EAC / BAC",      f"{eac_bac:.4f}×",    "", eac_col)
        kv("Schedule slip",  f"{slip:+.2f}",        "periods", slp_col)
        kv("Cure remaining", str(cure),             "periods", cur_col)

        # ── Next-period trajectory box ────────────────────────
        # FIX 1: next_period is always the upcoming episode period (t + 1),
        #         not the project-local counter tp.
        # FIX 2: the lock check inside print_trajectory_box now compares
        #         next_ms_earliest_t against next_period (t+1), so it
        #         correctly reflects whether the milestone is reachable in
        #         the period the agent is about to execute.
        if status == "active" and tp is not None:
            print_milestone_history_box(p["ms_history"])
            print_trajectory_box(
                next_period=t + 1,          # ← FIX 1: was `tp`
                prog_actual=prog,
                prog_plan=plan,
                delta=delta,
                next_ms_threshold_gap=p["next_ms_threshold_gap"],
                next_ms_net_payment=p["next_ms_net_payment"],
                next_ms_earliest_t=p["next_ms_earliest_t"],
                next_ms_is_final=p["next_ms_is_final"],
                retention_held=p["retention_held"],
                t_episode=t,
            )

    blank()


# ═════════════════════════════════════════════════════════════
# STEP RESULT
# ═════════════════════════════════════════════════════════════

def print_step_result(t_executed: int, reward: float, total_reward: float,
                      allocations: list, state_after: dict, info: dict):
    blank()
    pink_ruler("─")
    print(c(f"  ✔  Period {t_executed} complete", C.BGREEN, C.BOLD))
    pink_ruler("─")

    cashflow = info.get("cashflow", [])
    portfolio_inflow  = info.get("portfolio_inflow",  0.0)
    portfolio_outflow = info.get("portfolio_outflow", 0.0)

    for cf in cashflow:
        i      = cf["i"]
        alloc  = cf["allocation"]
        adv    = cf["advance"]
        ms_net = cf["milestone_net"]
        ms_grs = cf["milestone_gross"]
        ret    = cf["retention_release"]
        sett   = cf["settlement"]

        had_activity = (alloc > 0 or adv > 0 or ms_net != 0
                        or ret > 0 or sett != 0)
        if not had_activity:
            continue

        blank()
        sub_header(f"Project {i} — Cash Flow", C.BCYAN)

        if alloc > 0:
            kv("  ↳ Allocation (outflow)",
               f"-{alloc:,.2f}", "monetary units", C.BRED)

        if adv > 0:
            kv("  ↳ Advance payment (inflow)",
               f"+{adv:,.2f}", "monetary units", C.BGREEN)
        if ms_net > 0:
            kv("  ↳ Milestone payment net (inflow)",
               f"+{ms_net:,.2f}", "monetary units", C.BGREEN)
            if ms_grs > 0 and abs(ms_grs - ms_net) > 0.005:
                kv("     (gross before deductions)",
                   f"{ms_grs:,.2f}", "monetary units", C.DIM)
        if ret > 0:
            kv("  ↳ Retention released (inflow)",
               f"+{ret:,.2f}", "monetary units", C.BCYAN)
        if sett < 0:
            kv("  ↳ Termination penalty (outflow)",
               f"{sett:,.2f}", "monetary units", C.BRED)

        proj_net = adv + ms_net + ret + (sett or 0) - alloc
        net_col  = C.BGREEN if proj_net >= 0 else C.BRED
        kv("  Net this period",
           f"{proj_net:+,.2f}", "monetary units", net_col)

    blank()
    ruler("─", C.DIM)
    kv("Portfolio inflow  (total)",
       f"+{portfolio_inflow:,.2f}", "monetary units", C.BGREEN)
    kv("Portfolio outflow (total)",
       f"-{portfolio_outflow:,.2f}", "monetary units", C.BRED)

    rwd_col = C.BGREEN if reward > 0 else C.DIM if reward == 0 else C.BRED
    blank()
    kv("Reward this period (NPV)",  f"{reward:.4f}",       "", rwd_col)
    kv("Cumulative reward",         f"{total_reward:.4f}", "", C.BWHITE)
    kv("Budget after step",
       f"{state_after['budget']:,.2f}", "monetary units", C.BGREEN)


# ═════════════════════════════════════════════════════════════
# INPUT
# ═════════════════════════════════════════════════════════════

def get_allocations(state: dict) -> list:
    n = len(state["projects"])
    allocatable = [p for p in state["projects"] if p["status"] == "active"]

    if not allocatable:
        print(c("  ○  No active projects this period — zero allocation auto-sent.", C.DIM))
        return [0.0] * n

    budget  = state["budget"]
    indices = [p["i"] for p in allocatable]
    allocations = [0.0] * n

    blank()
    ruler("─")
    kv("Available budget",
       f"{budget:,.2f}", "monetary units", C.BGREEN)
    kv("Active projects",
       "  ".join(c(f"P{i}", C.BCYAN, C.BOLD) for i in indices), "", C.BWHITE)
    blank()

    proj_hints = "  ".join(
        c(f"P{p['i']}  BAC={p['budget']:,.0f}", C.DIM)
        for p in allocatable
    )
    print(c(f"  Enter one allocation per active project, space-separated.", C.WHITE))
    print(c(f"  Projects: {proj_hints}", C.DIM))
    print(c(f"  Example ({len(allocatable)} project{'s' if len(allocatable)>1 else ''}): "
            f"{'  '.join(str(int(budget/len(allocatable))) for _ in allocatable)}", C.DIM))
    blank()

    while True:
        try:
            prompt = (
                c("  ❯ ", C.PINK, C.BOLD) +
                c(f"alloc [{', '.join(f'P{i}' for i in indices)}]"
                  f"  (total ≤ {budget:,.2f}): ", C.WHITE)
            )
            raw = input(prompt)
            values = [float(v) for v in raw.strip().split()]

            if len(values) != len(allocatable):
                print(c(f"  ✗  Need {len(allocatable)} value(s), got {len(values)}. Try again.", C.BRED))
                continue
            if any(v < 0 for v in values):
                print(c("  ✗  All values must be ≥ 0. Try again.", C.BRED))
                continue
            if sum(values) > budget + 1e-6:
                print(c(f"  ✗  Total {sum(values):,.2f} exceeds budget {budget:,.2f}. Try again.", C.BRED))
                continue

            for idx, p in enumerate(allocatable):
                allocations[p["i"]] = values[idx]
            return allocations

        except ValueError:
            print(c("  ✗  Invalid input — numbers only, space-separated.", C.BRED))


# ═════════════════════════════════════════════════════════════
# EPISODE SUMMARY
# ═════════════════════════════════════════════════════════════

def print_episode_summary(env: PortfolioEnv, total_reward: float, periods_done: int):
    header("EPISODE COMPLETE", C.BGREEN)

    blank()
    section("EPISODE RESULT", C.BGREEN)
    kv("Episode ID",       env.episode_id,               value_color=C.DIM)
    kv("Periods executed", periods_done,                  value_color=C.BWHITE)
    kv("Total reward",     f"{total_reward:.4f}",
       value_color=C.BGREEN if total_reward > 0 else C.BRED)

    section("PROJECT OUTCOMES", C.BGREEN)
    for proj, ps in zip(env.projects, env.proj_state):
        status = ps["status"] or "not_started"
        blank()
        sub_header(f"Project {proj['i']}", C.BCYAN)
        kv("Final status",    status_color(status),           "", C.BWHITE)
        kv("Progress",        f"{ps['progress']:.4f}",
           "", C.BGREEN if ps["progress"] >= 1.0 else C.BYELLOW)
        kv("ACWP",            f"{ps['acwp']:,.2f}",           "monetary units", C.BWHITE)
        kv("BAC",             f"{proj['budget']:,.2f}",       "monetary units", C.DIM)
        kv("EAC",             f"{ps['eac']:,.2f}",            "monetary units", C.BWHITE)
        kv("Cost overrun",    f"{ps['cost_overrun']:+,.2f}",  "monetary units",
           C.BGREEN if ps["cost_overrun"] <= 0 else C.BRED)
        kv("CPI",             f"{ps['cpi']:.4f}",
           "", C.BGREEN if ps["cpi"] >= 1.0 else C.BRED)
        kv("SPI",             f"{ps['spi']:.4f}",
           "", C.BGREEN if ps["spi"] >= 1.0 else C.BRED)
        kv("Schedule slip",   f"{ps['schedule_slip']:+.2f}", "periods",
           C.BGREEN if ps["schedule_slip"] <= 0 else C.BRED)
        kv("Retention released",
           c("Yes", C.BGREEN) if ps["retention_released"] else c("No", C.BRED),
           "", C.BWHITE)

    blank()
    ruler("═", C.BGREEN)
    blank()


# ═════════════════════════════════════════════════════════════
# CONFIG SELECTOR
# ═════════════════════════════════════════════════════════════

METHOD_REGISTRY = {
    "1": ("manual",     "Manual",      True,  "Interactive manual allocation"),
    "2": ("rl",         "RL Agent",    False, "Trained PPO model — not yet available"),
    "3": ("milp",       "MILP",        False, "Deterministic upper bound solver — not yet available"),
    "4": ("baseline_1", "Baseline 1",  False, "Placeholder — not yet defined"),
    "5": ("baseline_2", "Baseline 2",  False, "Placeholder — not yet defined"),
    "6": ("baseline_3", "Baseline 3",  False, "Placeholder — not yet defined"),
}


def select_config() -> str:
    blank()
    ruler("═", C.BMAGENTA)
    print(c(f"  {'SELECT CONFIGURATION':^{W}}", C.BOLD, C.BMAGENTA))
    ruler("═", C.BMAGENTA)
    blank()

    for key, (_, _, description) in CONFIG_REGISTRY.items():
        print(c(f"    [{key}]  {description}", C.BWHITE))

    blank()

    while True:
        prompt = (
            c("  ❯ ", C.PINK, C.BOLD) +
            c(f"Enter choice [{'/'.join(CONFIG_REGISTRY.keys())}]: ", C.WHITE)
        )
        choice = input(prompt).strip()
        if choice in CONFIG_REGISTRY:
            break
        print(c(f"  ✗  Invalid choice. Enter one of: {', '.join(CONFIG_REGISTRY.keys())}", C.BRED))

    module_name, func_name, description = CONFIG_REGISTRY[choice]
    blank()
    print(c(f"  ✔  Selected: {description}", C.BGREEN, C.BOLD))
    blank()

    module = importlib.import_module(module_name)
    seed_fn = getattr(module, func_name)
    return seed_fn, description


def select_method() -> str:
    blank()
    ruler("═", C.BMAGENTA)
    print(c(f"  {'SELECT METHOD':^{W}}", C.BOLD, C.BMAGENTA))
    ruler("═", C.BMAGENTA)
    blank()

    selectable = []
    for key, (method_id, label, available, description) in METHOD_REGISTRY.items():
        if available:
            print(c(f"    [{key}]  {label:<14}", C.BWHITE) +
                  c(f"  {description}", C.DIM))
            selectable.append(key)
        else:
            print(c(f"    [ ]  {label:<14}", C.DIM) +
                  c(f"  {description}", C.DIM))

    blank()

    while True:
        prompt = (
            c("  ❯ ", C.PINK, C.BOLD) +
            c(f"Enter choice [{'/'.join(selectable)}]: ", C.WHITE)
        )
        choice = input(prompt).strip()
        if choice in selectable:
            break
        print(c(f"  ✗  Invalid choice. Enter one of: {', '.join(selectable)}", C.BRED))

    method_id, label, _, _ = METHOD_REGISTRY[choice]
    blank()
    print(c(f"  ✔  Method: {label}", C.BGREEN, C.BOLD))
    blank()
    return method_id


def get_episode_number(conn: sqlite3.Connection, config_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT episode_id) FROM portfolios WHERE config_id = ?",
        (config_id,)
    ).fetchone()
    return (row[0] or 0) + 1


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run():
    conn = init_db()

    seed_fn, _ = select_config()
    config_id = seed_fn(conn)

    method = select_method()
    episode_number = get_episode_number(conn, config_id)

    env = PortfolioEnv(conn, config_id, method=method)

    state = env.reset()
    print_portfolio_profile(env, state, episode_number)

    total_reward = 0.0
    periods_done = 0

    while True:
        print_period_state(state)
        allocations = get_allocations(state)
        t_executed  = state["t_episode"]

        state, reward, done, info = env.step(allocations)
        periods_done += 1
        total_reward += reward

        print_step_result(t_executed, reward, total_reward, allocations, state, info)

        if done:
            print_period_state(state)
            print_episode_summary(env, total_reward, periods_done)
            break

    conn.close()


if __name__ == "__main__":
    run()