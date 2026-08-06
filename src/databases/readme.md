# Database Documentation
## Contractor Portfolio Budgeting — RL Training Database

---

## Overview

Single SQLite file (`training.db`) stores everything — environment configuration, episode data, and post-hoc solver results. All components read and write to this one file.

Seven tables. Two concerns:

- **Left side** — problem definition (static, written once)
- **Right side** — solution traces (temporal, written per step)

```
environment_config ──→ portfolios
                   ──→ projects_profile ──→ projects_status
                   ──→ milestones_profile ──→ milestones_status
                        training_log (standalone)
```

---

## The System in Brief

An RL agent (PPO) manages cash allocation across a contractor's active project portfolio. At each timestep the agent decides how much budget to allocate to each active project. A stochastic efficiency parameter η resolves after each decision, driving project progress, milestone certification, and cash inflows.

After training, three naive baselines and one MILP upper bound replay every recorded episode using the same realized η sequence. All four solvers write their traces into the same tables as the RL agent, distinguished by the `method` column.

**Methods:** `rl` · `milp` · `equal` · `spi` · `cpi`

---

## Table Registry

| Table | Type | Rows |
|---|---|---|
| `environment_config` | Static | One per experiment configuration |
| `portfolios` | Temporal | One per episode × timestep × method |
| `projects_profile` | Static | One per project per episode |
| `projects_status` | Temporal | One per project × episode timestep × method |
| `milestones_profile` | Static | One per milestone per project per episode |
| `milestones_status` | Event | One per milestone × method |
| `training_log` | Training | One per PPO update |

---

## Table Details

---

### environment_config

Defines the parameter distributions the portfolio generator samples from. Each row is one experiment configuration. The generator reads one row at environment reset and produces a portfolio conforming to those distributions.

Different rows enable different experimental regimes — abundant budget, scarce budget, stress tests, calibrated contractor data.

**Identifier columns:** `config_id`, `config_name`, `created_at`

**Parameter columns:** one set of five columns per sampled parameter:

```
{parameter}_dist    TEXT    distribution family
{parameter}_p1      REAL    first parameter
{parameter}_p2      REAL    second parameter
{parameter}_p3      REAL    third parameter (null if unused)
{parameter}_p4      REAL    fourth parameter (null if unused)
```

**Supported distribution families:**

| Family | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| `fixed` | value | — | — | — |
| `uniform` | min | max | — | — |
| `normal` | mean | std | — | — |
| `lognormal` | mean | std | — | — |
| `triangular` | min | mode | max | — |
| `truncated_normal` | mean | std | min | max |
| `categorical` | prob_1 | prob_2 | anchor_1 | anchor_2 |
| `beta` | alpha | beta | — | — |

**Sampled parameters:**

Portfolio level: `n_projects` · `initial_budget` · `budget_tightness` · `discount`

Project level: `budget` · `margin` · `start` · `duration` · `scurve_a` · `scurve_b` · `advance_percent` · `advance_trigger` · `advance_recovery` · `retention_rate` · `schedule_cap` · `cost_cap` · `cure_length` · `efficiency`

Milestone level: `n_milestones` · `threshold` · `payment_weight`

---

### projects_profile

Static parameters of each project instance. Written once at episode start by the generator. Never updated.

One row per project per episode. The `config_id` link records which environment configuration produced this project.

**Key columns:** `episode_id` · `config_id` · `i` (project index) · `budget` · `price` · `margin` · `start` · `finish` · `duration` · `scurve_a` · `scurve_b` · `advance_percent` · `advance_trigger` · `advance_recovery` · `retention_rate` · `schedule_cap` · `cost_cap` · `cure_length`

---

### projects_status

Temporal trace of each project at every episode timestep for every method. Written at every timestep during episode execution. Post-hoc solvers append their own rows after the RL episode completes.

Every project gets a row at every episode timestep regardless of whether it is active. The `status` column explains the row.

**Status values:**

| status | Meaning |
|---|---|
| `null` | Project not yet started or episode not reached this period |
| `active` | Project executing normally |
| `completed` | Project finished successfully |
| `terminated` | Project stopped due to breach |

Temporal columns are `null` when status is null. Payment columns follow their own null logic — see below.

**Key columns:** `episode_id` · `i` · `t_episode` · `t_project` · `method` · `status` · `allocation` · `efficiency` · `progress` · `progress_plan` · `progress_increment` · `spi` · `cpi` · `eac` · `schedule_slip` · `cost_overrun` · `forecast_finish` · `cure_remaining`

**Payment columns null logic:**

| Column | Zero means | Null means |
|---|---|---|
| `advance_amount` | not the start period | — (always zero or nonzero) |
| `payment_net` | milestone window passed uncertified | before first milestone window |
| `retention_release` | — | project not yet completed |
| `settlement` | — | no termination event yet |

---

### milestones_profile

Static definition of each milestone. Written once at episode start. Never updated.

One row per milestone per project per episode.

**Key columns:** `episode_id` · `i` · `j` (milestone index) · `threshold` · `earliest_t` · `payment_weight`

`threshold` — progress fraction required for certification (θᵢⱼ)
`earliest_t` — earliest episode timestep at which certification is contractually valid (θᵢⱼᵉ)
`payment_weight` — fraction of contract price Pᵢ released at certification (φᵢⱼ)

Weights across all milestones of one project sum to 1.0.

---

### milestones_status

Certification trace per milestone per method. One row per milestone per method. Written at certification or episode end.

Not a timestep table — milestones are events, not continuous observations.

**Key columns:** `episode_id` · `i` · `j` · `method` · `certified` · `certified_t` · `payment_released`

`certified_t` is null if the milestone was never reached. `payment_released` is null if never certified.

---

### portfolios

Portfolio-level temporal trace. One row per episode timestep per method. Records the aggregate cash state of the portfolio at each period.

`outflow` is the sum of all project allocations at this timestep. Stored here for reporting convenience — it is derivable from `projects_status` but pre-computed avoids repeated aggregation queries.

**Key columns:** `episode_id` · `config_id` · `t_episode` · `method` · `balance` · `inflow` · `outflow` · `reward` · `done`

`done = 1` marks the final timestep of the episode.

---

### training_log

PPO learning metrics recorded per training update. Used for convergence charts and training diagnostics. Not linked to a specific episode — linked to the global training step.

**Key columns:** `update` · `episode_id` · `timestep_global` · `reward_mean` · `reward_std` · `policy_loss` · `value_loss` · `entropy` · `kl_divergence` · `clip_fraction` · `learning_rate`

---

## How an Episode Flows Through the Database

```
RESET
  generator reads environment_config row
  writes: projects_profile (N rows)
          milestones_profile (N × M rows)
          portfolios row at t=0 (method=rl)

STEP LOOP (t = 1 .. T)
  agent decides allocation vector
  environment resolves η, updates state
  recorder writes:
      portfolios row at t (method=rl)
      projects_status rows at t (one per project, method=rl)
  if milestone certified:
      milestones_status row updated (method=rl)

EPISODE END
  portfolios done=1 at final t

POST-HOC (Phase 2)
  for each solver in [milp, equal, spi, cpi]:
      reads projects_status (method=rl) for η sequence
      replays episode deterministically
      writes portfolios rows (method=solver)
      writes projects_status rows (method=solver)
      writes milestones_status rows (method=solver)

REPORTING (Phase 3)
  reads all methods from portfolios, projects_status, milestones_status
  computes normalized improvement per episode
  generates figures and tables
```

---

## Connections Between Tables

```
environment_config.config_id
    → portfolios.config_id
    → projects_profile.config_id

projects_profile (episode_id, i)
    → projects_status (episode_id, i)
    → milestones_profile (episode_id, i)

milestones_profile (episode_id, i, j)
    → milestones_status (episode_id, i, j)

portfolios (episode_id, t_episode)
    → projects_status (episode_id, t_episode)
```

---

## Post-hoc Solvers

All three solvers run after the RL episode is fully recorded.

**MILP** — full foresight deterministic upper bound. Reads the complete realized η sequence from `projects_status` (method=rl). Solves the deterministic allocation problem with perfect knowledge of all future η values. Writes optimal allocations as method=milp rows. This is Z* — the performance ceiling no online policy can exceed.

**Equal split** — divides available budget equally across all active projects at each timestep regardless of their state.

**SPI baseline** — allocates budget proportional to each project's Schedule Performance Index. Projects falling behind schedule receive more budget.

**CPI baseline** — allocates budget proportional to each project's Cost Performance Index. Projects with better cost efficiency receive more budget.

All baselines use the same realized η from the RL trace. They are deterministic replays, not new environment runs.

---

## Evaluation Metrics

**Normalized improvement** — where the agent sits between random and optimal:

```
normalized_improvement = (agent_return - random_return) / (milp_return - random_return)
```

Target: ≥ 65% across in-distribution test episodes.

**Budget regime** — reported separately for three κ regimes:

```
κ ≫ 1   abundant   budget far exceeds total project cost
κ ≈ 1   tight      budget roughly matches total project cost
κ ≪ 1   scarce     budget insufficient to fund all projects simultaneously
```

κ is computed from `environment_config` and stored in `portfolios` at t=0.