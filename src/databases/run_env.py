import sqlite3
from db_init import init_db
from config_seed_sp import seed_single_project
from env import PortfolioEnv


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────

def print_state(state: dict):
    t = state["t_episode"]
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║  PERIOD {t}  —  deciding allocation for period {t}       ".ljust(59) + "║")
    print(f"║  Budget available : {state['budget']:>10,.2f}".ljust(59) + "║")
    print(f"║  Horizon          : {state['horizon']:>10}".ljust(59) + "║")
    print("╚" + "═" * 58 + "╝")

    for p in state["projects"]:
        status = p["status"] or "NOT_STARTED"
        print()
        print(f"  ┌─ Project {p['i']}  [{status}]")
        print(f"  │  budget         {p['budget']:>10,.2f}   "
              f"start/finish  {p['start']} / {p['finish']}")
        print(f"  │  t_project      {str(p['t_project'] or '—'):>10}   "
              f"cure left     {p['cure_remaining']}")
        print(f"  │  progress       {p['progress']:>10.4f}   "
              f"plan target   {p['progress_plan']:.4f}")
        print(f"  │  SPI / CPI      {p['spi']:>6.3f} / {p['cpi']:<6.3f}   "
              f"EAC           {p['eac']:,.2f}")
        print(f"  └  forecast fin   {p['forecast_finish']:>10.2f}   "
              f"slip          {p['schedule_slip']:+.2f}")
    print()


def print_step_result(t_executed: int, reward: float, total_reward: float):
    print()
    print(f"  ✔  Period {t_executed} complete")
    print(f"     reward this period  : {reward:>10.4f}")
    print(f"     cumulative reward   : {total_reward:>10.4f}")


# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────

def get_allocations(state: dict) -> list:
    n = len(state["projects"])
    allocatable = [
        p for p in state["projects"]
        if p["status"] == "active"
    ]

    if not allocatable:
        print("  — No active projects this period. Zero allocation sent.")
        return [0.0] * n

    budget = state["budget"]
    indices = [p["i"] for p in allocatable]

    print(f"  Available budget : {budget:,.2f}")
    print(f"  Active projects  : {indices}")
    print()

    allocations = [0.0] * n

    while True:
        try:
            raw = input(
                f"  Allocate to projects {indices} "
                f"(space-separated, total ≤ {budget:,.2f}): "
            )
            values = [float(v) for v in raw.strip().split()]

            if len(values) != len(allocatable):
                print(f"  ✗  Expected {len(allocatable)} value(s). Got {len(values)}. Try again.")
                continue
            if any(v < 0 for v in values):
                print("  ✗  All allocations must be ≥ 0. Try again.")
                continue
            if sum(values) > budget + 1e-6:
                print(f"  ✗  Total {sum(values):,.2f} exceeds budget {budget:,.2f}. Try again.")
                continue

            for idx, p in enumerate(allocatable):
                allocations[p["i"]] = values[idx]
            return allocations

        except ValueError:
            print("  ✗  Invalid input. Enter numbers separated by spaces.")


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def run():
    conn = init_db()
    config_id = seed_single_project(conn)
    env = PortfolioEnv(conn, config_id, method="rl")

    print()
    print("  Portfolio Environment — Manual Mode")
    print("  ─────────────────────────────────────────────────────")
    print("  Each period: observe state → enter allocations → execute.")
    print("  Press Ctrl+C to quit.")

    # reset() generates the portfolio, credits t=0 advances, returns
    # pre-action observation with progress_plan already set.
    state = env.reset()

    total_reward = 0.0
    periods_done = 0

    while True:
        # Show current state — this is the observation BEFORE action.
        print_state(state)

        # Get allocations for this period.
        allocations = get_allocations(state)

        # Execute the period.
        t_executed = state["t_episode"]   # capture before step increments it
        state, reward, done, _ = env.step(allocations)

        periods_done += 1
        total_reward += reward
        print_step_result(t_executed, reward, total_reward)

        if done:
            # Show the final state after the last period.
            print_state(state)
            print("╔" + "═" * 58 + "╗")
            print("║  EPISODE COMPLETE".ljust(59) + "║")
            print(f"║  Total reward  : {total_reward:>10.4f}".ljust(59) + "║")
            print(f"║  Periods run   : {periods_done:>10}".ljust(59) + "║")
            print(f"║  Episode ID    : {env.episode_id}  ║")
            print("╚" + "═" * 58 + "╝")
            print()
            break

    conn.close()


if __name__ == "__main__":
    run()