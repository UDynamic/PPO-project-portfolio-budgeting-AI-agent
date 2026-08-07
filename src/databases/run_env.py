import sqlite3
from db_init import init_db
from config_seed_sp import seed_single_project
from env import PortfolioEnv


def print_state(state: dict):
    print()
    print("=" * 60)
    print(f"  EPISODE TIMESTEP : {state['t_episode']}")
    print(f"  BUDGET           : {state['budget']:,.2f}")
    print(f"  HORIZON          : {state['horizon']}")
    print("=" * 60)

    for p in state["projects"]:
        status = p["status"] or "not_started"
        print(f"\n  Project {p['i']}  [{status.upper()}]")
        print(f"    budget         : {p['budget']:,.2f}")
        print(f"    start / finish : {p['start']} / {p['finish']}")
        print(f"    t_project      : {p['t_project']}")
        print(f"    progress       : {p['progress']:.4f}  (plan: {p['progress_plan']:.4f})")
        print(f"    SPI / CPI      : {p['spi']:.3f} / {p['cpi']:.3f}")
        print(f"    EAC            : {p['eac']:,.2f}")
        print(f"    forecast_finish: {p['forecast_finish']:.2f}  (slip: {p['schedule_slip']:.2f})")
        print(f"    cure_remaining : {p['cure_remaining']}")
    print()


def get_allocations(state: dict) -> list:
    n = len(state["projects"])
    active = [
        p for p in state["projects"]
        if p["status"] == "active"
    ]
    not_started = [
        p for p in state["projects"]
        if p["status"] is None and state["t_episode"] >= p["start"]
    ]
    allocatable = active + not_started

    if not allocatable:
        print("  No active projects to allocate to.")
        return [0.0] * n

    print(f"  Available budget : {state['budget']:,.2f}")
    print(f"  Active projects  : {[p['i'] for p in allocatable]}")
    print()

    allocations = [0.0] * n

    while True:
        raw = input(f"  Enter allocations for projects {[p['i'] for p in allocatable]}"
                    f" (space-separated, total <= {state['budget']:,.2f}): ")
        try:
            values = [float(v) for v in raw.strip().split()]
            if len(values) != len(allocatable):
                print(f"  Expected {len(allocatable)} values. Try again.")
                continue
            if any(v < 0 for v in values):
                print("  Allocations must be >= 0. Try again.")
                continue
            if sum(values) > state["budget"] + 1e-6:
                print(f"  Total {sum(values):,.2f} exceeds budget {state['budget']:,.2f}. Try again.")
                continue
            for idx, p in enumerate(allocatable):
                allocations[p["i"]] = values[idx]
            return allocations
        except ValueError:
            print("  Invalid input. Enter numbers separated by spaces.")


def run():
    conn = init_db()
    config_id = seed_single_project(conn)
    env = PortfolioEnv(conn, config_id, method="rl")

    print()
    print("  Portfolio Environment — Manual Mode")
    print("  Type allocations at each step.")
    print("  Press Ctrl+C to quit.")
    print()

    state = env.reset()

    total_reward = 0.0
    step = 0

    while True:
        print_state(state)
        allocations = get_allocations(state)
        state, reward, done, _ = env.step(allocations)

        step += 1
        total_reward += reward

        print()
        print(f"  Reward this step : {reward:.4f}")
        print(f"  Cumulative reward: {total_reward:.4f}")

        if done:
            print_state(state)
            print("=" * 60)
            print("  EPISODE COMPLETE")
            print(f"  Total reward : {total_reward:.4f}")
            print(f"  Total steps  : {step}")
            print(f"  Episode ID   : {env.episode_id}")
            print("=" * 60)
            break

    conn.close()


if __name__ == "__main__":
    run()