# Portfolio Budgeting RL Environment

A Gymnasium-compatible environment for project portfolio budget allocation under
cash-flow uncertainty. Construction/engineering domain. The portfolio manager is
the contractor. The agent's only action is a continuous budget allocation vector
across active projects.

Academic contribution: Master's thesis targeting IJPM, EJOR, or Automation in
Construction. The environment is the primary deliverable; a PPO agent trained on
it is the demonstration.

---

## Folder structure

```
root/
  src/
    env/
      __init__.py               ← registers gym.make("PortfolioBudgeting-v0")
      env.py                    ← PortfolioBudgetingEnv(gym.Env)
      sampler.py                ← distribution sampling, Beta CDF, earned_schedule
      payments.py               ← advance / milestone / retention / settlement
      evm.py                    ← SPI(t), CPI, TCPI, EAC, update_evm()
      render.py                 ← ANSI render (history store + all print functions)
      db_logger.py              ← SQLite audit ledger, side-effect only
      configs/
        single_project.py       ← CFG-SINGLE-001 (CONFIG dict + seed())
        dual_project.py         ← CFG-DUAL-001   (CONFIG dict + seed())
    db/
      db_init.py                ← init_db(), verify()
      schema.sql                ← SQLite schema
      database.db               ← created on first run (not committed)
    play/
      play.py                   ← manual play loop
    tests/
      test_check_env.py
      test_step_sequence.py
      test_payments.py
```

---

## Installation

```bash
pip install gymnasium numpy
```

No other dependencies. Beta CDF and earned schedule use only the standard library.

---

## Quick start

### Gymnasium API

```python
import sys
sys.path.insert(0, "src")
import env  # triggers gym.register

import gymnasium as gym
from env.configs.single_project import CONFIG

e = gym.make("PortfolioBudgeting-v0", config=CONFIG)
obs, info = e.reset(seed=42)

done = False
while not done:
    action = e.action_space.sample()
    obs, reward, terminated, truncated, info = e.step(action)
    done = terminated or truncated

e.close()
```

### Manual play

```bash
# single-project config, no DB
python src/play/play.py --config single --no-db

# dual-project config, with DB logging, fixed seed
python src/play/play.py --config dual --seed 7
```

Controls each period:

| Input | Effect |
|-------|--------|
| Enter | Equal split across active projects |
| `30,20` | Explicit amounts (comma-separated, one per project) |
| `q` / `quit` | Exit immediately |

### Seed configs into DB

```bash
python src/env/configs/single_project.py
python src/env/configs/dual_project.py
```

### Validate Gymnasium compliance

```python
from gymnasium.utils.env_checker import check_env
from env.configs.single_project import CONFIG
from env.env import PortfolioBudgetingEnv

check_env(PortfolioBudgetingEnv(CONFIG))
```

---

## Domain model

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
- `TCPI` = (BAC − BCWP) / (BAC − ACWP)
- `EAC` = ACWP + (BAC − BCWP) / (CPI × SPI(t))
- `schedule_slip` = forecast_finish − planned_finish
- `forecast_finish` = start + duration / SPI(t)

### Payment mechanics (FIDIC 14.2 style)

- **Advance**: `advance_percent × price` at project start
- **Milestone**: triggered when `progress >= threshold AND t >= earliest_t`
  - `gross = weight × price`
  - `recovery = min(gross × advance_recovery, remaining_advance_unrecovered)`
  - `retention = gross × retention_rate` (interim only; zero for final milestone)
  - `net = gross − recovery − retention`
- **Retention release**: fires the period after final milestone (one-period lag)
- **Settlement on termination**: `progress × price − (advance_received + milestone_net_inflows)`

### Step phase order

```
Phase 1  INFLOWS
  1a. Advance payment if project starts this period
  1b. Milestone certification → net payment
  1c. Retention release if _release_retention_next_period flag set

Phase 2  INTEREST
  treasury_draw = max(0, cumulative_cost_BEFORE_alloc − cumulative_inflows)
  interest = (annual_rate / 12) × treasury_draw

Phase 3  ALLOCATION & PROGRESS
  cumulative_cost += allocation
  progress += (allocation / BAC) × η

Phase 4  EVM UPDATE
  Recompute SPI(t), CPI, TCPI, EAC, forecast_finish, schedule_slip, plan_deviation

Phase 5  TERMINATION CHECK
  Evaluate breach flags. Update cure counter.
  If progress >= 1.0: set _release_retention_next_period = True, status = completed
  If terminated: compute settlement.

Phase 6  RECORD
  DB write (side-effect, try/except — never load-bearing)
  budget += total_inflow − total_outflow
  reward = discount^t × (total_inflow − total_outflow)
```

### Termination

Breach conditions (any triggers cure counter decrement):
- `breach_idle`: alloc < ε while active
- `breach_deviation`: δ > plan_deviation_threshold
- `breach_schedule AND breach_cost`: slip > schedule_cap AND EAC > cost_cap × BAC
- `breach_deadline`: t >= finish + schedule_cap while active

Cure counter τ: starts at cure_length, decrements on breach, resets on clean period.
Termination fires when τ <= 0 (elif) or breach_deadline is true.

### Spaces

- **Observation**: `Box(shape=(8n+1,), dtype=float32)` — 8 signals per project + normalised budget
- **Action**: `Box(low=0, high=1, shape=(n,), dtype=float32)` — allocation fractions

Per-project observation (8): `progress, progress_plan, plan_deviation, spi, cpi, eac/BAC, schedule_slip, cure_remaining`  
Portfolio (1): `budget / initial_budget`

### Reward

`reward(t) = discount^t × (total_inflow_t − total_outflow_t)`

---

## Configs

Two baseline configs are provided as Python dicts. Both have two interfaces:

- **`CONFIG`** dict — pass directly to `PortfolioBudgetingEnv(config=CONFIG)`. No DB required.
- **`seed(conn)`** — insert the config into `environment_config` for DB audit logging.

| Config | ID | Projects | Initial Budget | BAC |
|--------|----|----------|----------------|-----|
| `single_project.py` | CFG-SINGLE-001 | 1 | 200 | 100 (fixed) |
| `dual_project.py` | CFG-DUAL-001 | 2 | 400 | uniform(80, 150) |

---

## DB logging

Pass an open `sqlite3.Connection` to `PortfolioBudgetingEnv(conn=conn)`.
All writes are wrapped in `try/except` — a DB failure never raises inside the env.
`conn.commit()` is called unconditionally every step.

Schema tables:

| Table | Written |
|-------|---------|
| `environment_config` | by `seed()` in config files |
| `projects_profile` | once per episode at `reset()` |
| `milestones_profile` | once per episode at `reset()` |
| `portfolios` | every step |
| `projects_status` | every step, per project |
| `milestones_status` | on certification |
| `training_log` | by PPO trainer (not env) |

---

## Known limitations / planned extensions

- `advance_trigger` is sampled and stored but not yet wired into certification
  logic. Wiring it is a planned extension (Bug #6 from the original MVP).
- Per-project config tables (for structurally different projects in a single
  config) are a planned extension. The dual-project config currently uses wide
  distributions to achieve project variety within a single shared config.
- `training_log` is written by the PPO trainer, not by the environment.





---------------
You need to run in this order:

1. **Init the DB and seed the config** (one-time setup):
```bash
python src/db/db_init.py
python src/env/configs/single_project.py
```

2. **Run the manual play loop**:
```bash
python src/play/play.py --config single --seed 42
```

Or skip DB entirely for a quick smoke test:
```bash
python src/play/play.py --config single --seed 42 --no-db
```

---

Before running, confirm your working directory is the repo root, and that `src/` exists with the structure from the readme. The play script inserts its own `sys.path` entries so you don't need to `pip install` anything beyond `gymnasium` and `numpy`.

If `check_env` validation matters before manual play:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "src/env")
from gymnasium.utils.env_checker import check_env
from configs.single_project import CONFIG
from env import PortfolioBudgetingEnv
check_env(PortfolioBudgetingEnv(CONFIG))
print("check_env passed")
EOF
```