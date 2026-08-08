import sqlite3
from db_init import init_db
from config_seed_sp import seed_single_project
from env import PortfolioEnv


# ═════════════════════════════════════════════════════════════
# ANSI COLOR PALETTE
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
# LAYOUT PRIMITIVES  — kv-list style, zero box-drawing bugs
# ═════════════════════════════════════════════════════════════

W = 70   # ruler width

def ruler(ch="─", color=C.DIM):
    print(c("  " + ch * W, color))

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
    """Indented label for a sub-group within a section."""
    print(c(f"    ── {title}", color))


# ═════════════════════════════════════════════════════════════
# STATIC PROFILE  (printed once after reset)
# ═════════════════════════════════════════════════════════════

def print_portfolio_profile(env: PortfolioEnv, state: dict):

    header("PORTFOLIO ENVIRONMENT — STATIC PROFILE")

    # ── EPISODE ──────────────────────────────────────────────
    section("EPISODE")
    kv("Episode ID",      env.episode_id,            value_color=C.DIM)
    kv("Config ID",       env.config_id,             value_color=C.BMAGENTA)
    kv("Method",          env.method,                value_color=C.BYELLOW)
    kv("Horizon",         state["horizon"],  "periods",      C.BYELLOW)
    kv("Discount factor", f"{env.discount:.4f}", "per period", C.BWHITE)

    # ── PORTFOLIO FINANCIALS ──────────────────────────────────
    section("PORTFOLIO FINANCIALS")
    total_bac   = sum(p["budget"] for p in env.projects)
    total_price = sum(p["price"]  for p in env.projects)
    total_adv   = sum(p["advance_percent"] * p["price"]
                      for p in env.projects if p["start"] == 0)
    kappa = state["budget"] / total_bac   # use opening budget before advance
    kappa_raw = 200 / total_bac
    kappa_col = C.BGREEN if kappa_raw > 1.2 else C.BYELLOW if kappa_raw > 0.9 else C.BRED

    kv("Initial budget (B₀)",      f"{200:,.2f}",          "monetary units", C.BGREEN)
    kv("Portfolio BAC (ΣBAC)",     f"{total_bac:,.2f}",    "monetary units", C.BWHITE)
    kv("Portfolio price (ΣP)",     f"{total_price:,.2f}",  "monetary units", C.BWHITE)
    kv("Budget tightness (κ=B₀/ΣBAC)", f"{kappa_raw:.3f}", "B₀/ΣBAC",       kappa_col)
    kv("Advance credited (t=0)",   f"{total_adv:,.2f}",    "monetary units", C.BCYAN)
    kv("Opening budget",           f"{state['budget']:,.2f}", "monetary units", C.BGREEN)

    # ── PROJECTS ─────────────────────────────────────────────
    section(f"PROJECTS  ({len(env.projects)} total)")

    for proj in env.projects:
        blank()
        sub_header(f"Project {proj['i']}", C.BCYAN)
        kv("BAC (budget at completion)",  f"{proj['budget']:,.2f}",      "monetary units", C.BWHITE)
        kv("Price (contract value)",      f"{proj['price']:,.2f}",       "monetary units", C.BGREEN)
        kv("Margin",                      f"{proj['margin']*100:.1f}%",  "",               C.BYELLOW)
        kv("Start period",                proj["start"],                  "",               C.WHITE)
        kv("Planned finish",              proj["finish"],                 "",               C.WHITE)
        kv("Duration",                    proj["duration"],               "periods",        C.WHITE)
        kv("Schedule cap (max slip)",     proj["schedule_cap"],           "periods",        C.BYELLOW)
        kv("Cost cap (max EAC/BAC)",      f"{proj['cost_cap']:.2f}×",    "",               C.BYELLOW)
        kv("Cure period length",          proj["cure_length"],            "periods",        C.BYELLOW)
        kv("Advance payment",             f"{proj['advance_percent']*100:.0f}%",
                                          f"= {proj['advance_percent']*proj['price']:.2f}", C.BCYAN)
        kv("Advance recovery rate",       f"{proj['advance_recovery']*100:.0f}%",
                                          "deducted per milestone gross", C.DIM)
        kv("Retention rate",              f"{proj['retention_rate']*100:.0f}%",
                                          "held per milestone, released at completion", C.DIM)
        kv("S-curve α",                   f"{proj['scurve_a']:.3f}",     "",               C.BWHITE)
        kv("S-curve β",                   f"{proj['scurve_b']:.3f}",     "",               C.BWHITE)
        a, b = proj["scurve_a"], proj["scurve_b"]
        shape = "front-loaded" if a < b else "back-loaded" if a > b else "symmetric"
        kv("S-curve shape",               shape,                          "",               C.BMAGENTA)

    # ── MILESTONES ────────────────────────────────────────────
    section("MILESTONES")

    for i, (proj, ms_list) in enumerate(zip(env.projects, env.milestones)):
        blank()
        sub_header(f"Project {i}  —  {len(ms_list)} milestone(s)", C.BCYAN)
        for ms in ms_list:
            gross     = ms["payment_weight"] * proj["price"]
            after_rec = gross * (1 - proj["advance_recovery"])
            after_ret = after_rec * (1 - proj["retention_rate"])
            is_final  = ms["threshold"] == 1.0
            tag       = c("  ★ FINAL", C.BYELLOW) if is_final else ""

            label = f"  MS {ms['j']}  threshold {ms['threshold']*100:.1f}%{'' if not is_final else ' ★'}"
            print(c(f"    {label}", C.BYELLOW if is_final else C.WHITE))
            kv("Earliest certification",  ms["earliest_t"],               "period",         C.DIM,    indent=8)
            kv("Payment weight",          f"{ms['payment_weight']*100:.2f}%", "",            C.WHITE,  indent=8)
            kv("Gross payment",           f"{gross:.2f}",                 "monetary units", C.BGREEN, indent=8)
            kv("After advance recovery",  f"{after_rec:.2f}",             "monetary units", C.BYELLOW,indent=8)
            kv("After retention",         f"{after_ret:.2f}",             "monetary units", C.BCYAN,  indent=8)

    # ── STOCHASTIC ────────────────────────────────────────────
    section("STOCHASTIC PARAMETERS")
    kv("Efficiency η distribution",  env.cfg["efficiency_dist"],  "", C.BMAGENTA)
    kv("η lower bound",              env.cfg["efficiency_p1"],    "", C.BWHITE)
    kv("η upper bound",              env.cfg["efficiency_p2"],    "", C.BWHITE)

    blank()
    ruler("═", C.BBLUE)
    print(c(f"  {'ENTERING EPISODE — allocate at each period prompt':^{W}}", C.BOLD, C.BBLUE))
    ruler("═", C.BBLUE)
    blank()


# ═════════════════════════════════════════════════════════════
# PERIOD STATE  (printed every step before allocation)
# ═════════════════════════════════════════════════════════════

def print_period_state(state: dict):
    t = state["t_episode"]

    blank()
    ruler("─", C.BBLUE)
    print(
        c(f"  PERIOD {t}", C.BOLD, C.BBLUE) +
        c("  │  Budget: ", C.DIM) +
        c(f"{state['budget']:,.2f}", C.BGREEN, C.BOLD) +
        c(f"  │  Horizon: {state['horizon']}", C.DIM)
    )
    ruler("─", C.BBLUE)

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
        dlt_col = C.BGREEN if delta >= 0 else C.BRED
        slp_col = C.BGREEN if slip <= 0 else C.BYELLOW if slip <= 2 else C.BRED
        cur_col = C.BGREEN if cure >= 2 else C.BYELLOW if cure == 1 else C.BRED
        eac_col = C.BGREEN if eac_bac <= 1.05 else C.BYELLOW if eac_bac <= 1.20 else C.BRED

        blank()
        sub_header(f"Project {p['i']}  [{status_color(status)}]", C.BCYAN)
        kv("Status",            status_color(status),          "", C.BWHITE)
        kv("Project period",    str(tp) if tp else "—",        "", C.WHITE)
        kv("Progress (actual)", f"{prog:.4f}",                 "", C.BWHITE)
        kv("Progress (plan)",   f"{plan:.4f}",                 "", C.DIM)
        kv("Δ vs plan",         f"{delta:+.4f}",               "", dlt_col)
        kv("SPI",               f"{spi:.4f}",                  "", spi_col)
        kv("CPI",               f"{cpi:.4f}",                  "", cpi_col)
        kv("EAC",               f"{eac:,.2f}",                 "", eac_col)
        kv("EAC / BAC",         f"{eac_bac:.4f}×",             "", eac_col)
        kv("Schedule slip",     f"{slip:+.2f}",                "periods", slp_col)
        kv("Cure remaining",    str(cure),                     "periods", cur_col)

    blank()


# ═════════════════════════════════════════════════════════════
# STEP RESULT
# ═════════════════════════════════════════════════════════════

def print_step_result(t_executed: int, reward: float, total_reward: float,
                      allocations: list, state_after: dict):
    blank()
    ruler("─", C.BGREEN)
    print(c(f"  ✔  Period {t_executed} complete", C.BGREEN, C.BOLD))
    ruler("─", C.BGREEN)

    for p in state_after["projects"]:
        a = allocations[p["i"]]
        if a > 0 or p["status"] == "active":
            kv(f"Allocation → Project {p['i']}",
               f"{a:,.2f}", "monetary units",
               C.BWHITE if a > 0 else C.DIM)

    rwd_col = C.BGREEN if reward > 0 else C.DIM if reward == 0 else C.BRED
    blank()
    kv("Reward this period (NPV)", f"{reward:.4f}",       "", rwd_col)
    kv("Cumulative reward",        f"{total_reward:.4f}", "", C.BWHITE)
    kv("Budget after step",        f"{state_after['budget']:,.2f}", "monetary units", C.BGREEN)


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

    while True:
        try:
            prompt = (
                c("  ❯ ", C.BBLUE, C.BOLD) +
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
        kv("Retention released", c("Yes", C.BGREEN) if ps["retention_released"] else c("No", C.BRED),
           "", C.BWHITE)

    blank()
    ruler("═", C.BGREEN)
    blank()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run():
    conn = init_db()
    config_id = seed_single_project(conn)
    env = PortfolioEnv(conn, config_id, method="rl")

    state = env.reset()
    print_portfolio_profile(env, state)

    total_reward = 0.0
    periods_done = 0

    while True:
        print_period_state(state)
        allocations = get_allocations(state)
        t_executed  = state["t_episode"]

        state, reward, done, _ = env.step(allocations)
        periods_done += 1
        total_reward += reward

        print_step_result(t_executed, reward, total_reward, allocations, state)

        if done:
            print_period_state(state)
            print_episode_summary(env, total_reward, periods_done)
            break

    conn.close()


if __name__ == "__main__":
    run()