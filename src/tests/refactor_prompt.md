# Portfolio Budgeting RL Environment — MVP Refactor
# Stateless session. No memory. Work from uploaded files only.

---

## What this project is

A Gymnasium-compatible RL environment for project portfolio budget allocation
under cash flow uncertainty. Construction/engineering domain. The portfolio
manager is the contractor. The agent's only action is a continuous budget
allocation vector across active projects.

Academic contribution of a Master's thesis targeting IJPM, EJOR, or
Automation in Construction. The environment is the primary deliverable.
A PPO agent trained on it is the demonstration.

---

## Uploaded files

| File | What it is |
|------|------------|
| `env.py` | Environment logic. Not yet Gymnasium-compatible. Custom step/reset API. Contains known bugs (listed below). |
| `play.py` | Plain-terminal manual play loop. Connects to current env.py. Replace Textual UI. Keep as-is — it becomes the render reference. |
| `db_init.py` | SQLite schema init. |
| `config_seed_sp.py` | Seeds single-project config into DB. |
| `config_seed_dp.py` | Seeds dual-project config into DB. |

`ui.py` (Textual TUI) is discarded. Do not reference it.

---

## Target folder structure

```
root/
  src/
    env/                        <- Gymnasium package lives here
      __init__.py               <- registers gym.make("PortfolioBudgeting-v0")
      env.py                    <- PortfolioBudgetingEnv(gym.Env)
      sampler.py                <- distribution sampling via self.np_random
      payments.py               <- advance / milestone / retention / settlement
      evm.py                    <- SPI(t), CPI, TCPI, EAC, earned_schedule()
      render.py                 <- ANSI render (ported from play.py)
      db_logger.py              <- SQLite audit ledger, side effect only
      configs/
        single_project.py
        dual_project.py
    db/                         <- database files and schema
      db_init.py
      database.db
    play/                       <- manual play scripts
      play.py
    tests/
      test_check_env.py
      test_step_sequence.py
      test_payments.py
```

Keep the Gymnasium env package isolated in `src/env/`.
Everything else goes in its own folder under `src/`.
Do not mix concerns across folders.

---

## Domain model (do not change any of this)

### Portfolio
- `n` projects, global budget `B_t`, horizon `T`
- Each period: agent allocates `x_i(t) >= 0`, subject to `sum(x_i) <= B_t`
- Budget updated: `B += total_inflow - total_outflow`

### Project parameters
`budget` (BAC), `price` (= BAC × (1+margin)), `start`, `finish`, `duration`,
`scurve_a`, `scurve_b`, `advance_percent`, `advance_recovery`, `retention_rate`,
`schedule_cap`, `cost_cap`, `cure_length`

### Progress
`P_i(t) = P_i(t-1) + (x_i(t) / BAC) × η`, where `η ~ cfg distribution`
Planned: `P̄_i(t) = BetaCDF(t_project / duration, a, b)`
Deviation: `δ_i(t) = P̄_i(t) - P_i(t)` (positive = behind plan)

### EVM signals
- `SPI(t)` = Earned Schedule / Actual Time (ES method, not classical)
- `CPI` = BCWP / ACWP
- `TCPI` = (BAC - BCWP) / (BAC - ACWP)
- `EAC` = ACWP + (BAC - BCWP) / (CPI × SPI(t))
- `schedule_slip` = forecast_finish - planned_finish
- `forecast_finish` = start + duration / SPI(t)

### Payment mechanics (FIDIC 14.2 style)
- **Advance**: `advance_percent × price` at project start
- **Milestone**: triggered when `progress >= threshold AND t >= earliest_t`
  - `gross = weight × price`
  - `recovery = min(gross × advance_recovery, remaining_advance_unrecovered)`
  - `retention = gross × retention_rate` (interim only; zero for final milestone)
  - `net = gross - recovery - retention`
- **Retention release**: fires the period AFTER final milestone (one-period lag)
- **Settlement on termination**: `progress × price - (advance_received + milestone_net_inflows)`

### Termination
Breach conditions (any triggers cure counter decrement):
- `breach_idle`: alloc < ε while active
- `breach_deviation`: δ > plan_deviation_threshold
- `breach_schedule AND breach_cost`: both slip > schedule_cap AND EAC > cost_cap × BAC
- `breach_deadline`: t >= finish + schedule_cap while active

Cure counter τ: starts at cure_length, decrements on breach, resets on clean period.
Termination fires when τ <= 0 OR breach_deadline is true (use elif — not both).

### Step sequence — FIXED ORDER, do not reorder

```
Phase 1  INFLOWS
  1a. Advance payment if project starts this period
  1b. Milestone certification → net payment
  1c. Retention release if _release_retention_next_period flag set

Phase 2  INTEREST
  treasury_draw = max(0, cumulative_cost_BEFORE_this_alloc - cumulative_inflows)
  interest = (annual_rate / 12) × treasury_draw
  Debited as outflow. Does NOT drive progress.

Phase 3  ALLOCATION & PROGRESS
  cumulative_cost += allocation
  progress += (allocation / BAC) × η

Phase 4  EVM UPDATE
  Recompute SPI(t), CPI, TCPI, EAC, forecast_finish, schedule_slip, plan_deviation

Phase 5  TERMINATION CHECK
  Evaluate breach flags.
  Update cure counter.
  If progress >= 1.0: set _release_retention_next_period = True, status = completed
  If terminated: compute settlement, add to inflows.

Phase 6  RECORD
  DB write (side effect, wrapped in try/except — never load-bearing)
  conn.commit() every step unconditionally
  budget += total_inflow - total_outflow
  reward = discount^t × (total_inflow - total_outflow)
```

### Reward
`reward(t) = discount^t × (total_inflow_t - total_outflow_t)`

### State space: 8n + 1
Per project (8): progress, progress_plan, plan_deviation, spi, cpi, eac/BAC,
schedule_slip, cure_remaining
Portfolio (1): budget (normalised by initial budget)

### Action space
`Box(low=0, high=1, shape=(n,), dtype=float32)` — allocation fractions.
Scale to actual budget inside env. Clip and rescale if sum > 1.

---

## Known bugs in env.py — fix these during refactor

1. **Early-continue crash**: `_write_project_row` called with undefined
   `interest_cost`, `treasury_draw` when `t < proj["start"]`. Pass `0.0, 0.0`.

2. **DB commit only on done**: commit every step unconditionally.

3. **Double termination**: cure and deadline can both fire same period,
   doubling settlement. Second branch must be `elif`.

4. **Retention release timing**: currently releases same period as completion.
   Must set `_release_retention_next_period = True` and release next period.

5. **Interest base**: computed on `cumulative_cost + alloc`. Must use
   `cumulative_cost` before alloc lands (Phase 2 before Phase 3).

6. **`advance_trigger` dead parameter**: sampled but never used. Remove from
   sampling and schema, or wire it up. Do not leave it silently unused.

---

## Do not change

- Beta CDF numerical integration and `earned_schedule` binary search — correct,
  no scipy, keep dependency-light.
- DB schema — correct. Add tables if needed, remove nothing.
- Config seeding approach — keep as separate files per scenario.
- Domain logic, payment mechanics, EVM formulas — correct and match the paper.

---

## Gymnasium wrapper requirements

```python
class PortfolioBudgetingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict, render_mode=None):
        # observation_space: Box(shape=(8n+1,), dtype=float32)
        # action_space: Box(low=0, high=1, shape=(n,), dtype=float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # sets self.np_random — use instead of random module
        return obs, info           # obs: np.float32 array

    def step(self, action):
        return obs, reward, terminated, truncated, info
        # terminated = all projects done
        # truncated  = t >= horizon

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()   # port from play.py

    def close(self):
        pass
```

Replace all `random.xxx` calls with `self.np_random.xxx` equivalents.
Config is a plain dict passed to `__init__` — not read from DB at reset time.

---

## ANSI render

All formatting and print functions live in `src/env/render.py`. Written once,
used in two places:
- `env.render()` calls into render.py
- `play.py` imports from render.py for the manual loop

Four blocks per project, printed sequentially:
1. Project Identity (static params)
2. Payment Profile (milestone table with advance/recovery/retention/net/status)
3. Boundary & Termination Status (breach flags + cure counter)
4. Period Timeseries (three sub-tables: periodic CF, cumulative CF, EVM metrics)

Preceded by a Portfolio Summary block.
Plain f-strings only. No external dependencies.
play.py is a thin loop — input collection and calls to render.py only.
No formatting logic in play.py itself.

---

## Validation — must pass before anything else is done

```python
from gymnasium.utils.env_checker import check_env
env = PortfolioBudgetingEnv(config)
check_env(env)   # zero warnings, zero errors
```

---

## Immediate task order

1. Create `src/env/` package with `__init__.py` registering `PortfolioBudgeting-v0`
2. Port `env.py` into `src/env/env.py` as `gymnasium.Env` subclass with all bugs fixed
3. Extract `sampler.py`, `payments.py`, `evm.py` as separate modules
4. Define `observation_space`, `action_space`, `_get_obs()`, `_get_info()`
5. Run `check_env()` — fix everything it reports
6. Extract all formatting/print functions from `play.py` into `src/env/render.py`
7. Wire `env.render()` to call render.py functions
8. Rewrite `play.py` as a thin loop that imports from render.py — no formatting logic in play.py
9. Move `db_init.py` and DB files to `src/db/`
10. Move `play.py` to `src/play/` and update its import paths