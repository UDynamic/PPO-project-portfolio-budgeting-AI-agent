# Baseline Solvers & Test Configs

`src/tests/solvers/`

---

## Files

| File | Purpose |
|---|---|
| `baseline_solvers.py` | Solver classes + `run_baselines()` runner called from `play.py` |
| `add_baseline_fixtures.py` | Batch script that runs every solver × config and writes regression fixtures |

---

## Solvers

Four solvers are available. Three are deterministic baselines; the fourth is the trained optimal policy.

### `catchup_t` — CatchupTSolver

Allocates `catchup_alloc_t` to each active project. This is the minimum
amount needed to keep the project within its progress-delay tolerance this
period — it brings actual progress up to `plan_nt − delay_cap` without
overshooting the plan.

**Late-phase switch (finishing intent):**
`catchup_alloc_t` is derived from the S-curve planned progress, which
saturates at `1.0` once `t_project >= planned_duration`. At that point
`plan_nt = 1.0` and the gap to the tolerance floor collapses to zero,
so `catchup_alloc_t = 0`. Without correction, the solver allocates nothing
while the project is still active and the project gets terminated by
`over_duration_window` before completing.

The solver detects when a project has reached or passed its planned duration
and switches to `reach_plan_t` instead:

```python
if ps["status"] == "active" and ps["t_project"] >= proj["planned_duration"]:
    amounts.append(ps.get("reach_plan_t", 0.0))    # finishing intent
else:
    amounts.append(ps.get("catchup_alloc_t", 0.0)) # normal catchup
```

`reach_plan_t` is the amount needed to close the remaining progress gap to
`1.0` in the current period, keeping spend positive for as long as work remains.

**Duration convention:** a project with contractual duration `X+1` periods has
`planned_duration = X`. This means `t_project = 0, 1, …, X` are the planned
active periods (X+1 actions total), and the late-phase switch fires from
`t_project = X` onward. Example: `planned_duration=3` covers
`t_project = 0,1,2,3` (4 periods), plus up to `finish_delay_cap=2` late
periods, giving 6 efficiency values in total.

Verified on config `TC-1LMAIF-R` (`planned_duration=3`, `planned_start=2`):

| Episode t | t_project | catchup_alloc_t | reach_plan_t | Allocated | Phase |
|-----------|-----------|-----------------|--------------|-----------|-------|
| 2 | 0 | 49.88 | 64.88 | 49.88 | normal catchup |
| 3 | 1 | 37.82 | 52.82 | 37.82 | normal catchup |
| 4 | 2 | 0.00 | 10.35 | 0.00 | normal catchup |
| 5 | 3 | 0.00 | 10.35 | **10.35** | **late-phase switch** |
| 6 | 4 | 0.00 | 0.93 | **0.93** | **late-phase switch** → completed |

---

### `reach_t` — ReachTSolver

Allocates `reach_plan_t` to each active project every period. Fully closes
the gap to the current period's plan target — no breach, no carry-over.
More aggressive than `catchup_t`, higher cost, but tracks the S-curve exactly
when budget permits.

---

### `catchup_next_t` — CatchupNextTSolver

One-period lookahead: allocates enough now so that after this period's
progress the project will be within tolerance of *next* period's plan target.
Sits between `catchup_t` (reactive minimum) and `reach_t` (full catch-up)
in terms of spend.

---

### `optimal` — trained PPO policy

The trained agent. Used as the upper-bound reference in regression and
evaluation runs.

---

## Shared action conversion

All three baseline solvers use `_amounts_to_action(env, amounts)` from `_BaseSolver`:

1. Non-active projects are zeroed.
2. Each amount is divided by the current budget to get a fraction.
3. If the sum of fractions exceeds `1.0` the vector is rescaled proportionally.

---

## Test configs

### Shared parameters (fixed across all configs)

| Parameter | Value |
|---|---|
| `retention_rate` | 0.05 |
| `progress_delay_cap` | 0.15 |
| `finish_delay_cap` | 2 |
| `annual_interest_rate` | 0.12 |
| `budget_tightness` | 3.0 |
| `profit_percent` | 0.20 |
| `discount` | 0.95 |
| `scurve_a` | 1.5 |
| `scurve_b` | 3.0 |

The S-curve parameters `a=1.5, b=3` produce a front-loaded progress profile.
Plan values for `planned_duration=3` (evaluated at `t/n`, `n=duration+1`):

| t_project | plan progress |
|---|---|
| 0 | 0.3975 |
| 1 | 0.7845 |
| 2 | 0.9693 |
| 3 | 1.0000 |

---

### Naming convention

```
TC-{N}{S}{D}{P}{E}
```

| Segment | Values | Meaning |
|---|---|---|
| `N` | `1` `2` `3` | number of projects |
| `S` | `0` `L` `X` | start time: t=0 / Late / miXed |
| `D` | `S` `M` | duration: Single period / Multi period |
| `P` | `F` `AF` `AIF` | payments: Final only / Advance+Final / Advance+Interim+Final |
| `E` | `P` `R` | efficiency: Perfect η=1 / Random seeded |

---

### Active configs (4 of 48)

#### `TC-10SF-R` — single atom

One project, starts at t=0, single period, final payment only, random efficiency.
The simplest possible episode. Used to validate the core loop: one action,
one milestone, one payment, retention deducted from final. Regression baseline
for all other configs.

| Parameter | Value |
|---|---|
| `planned_start` | 0 |
| `planned_duration` | 0 |
| `n_milestones` | 1 |
| `progress_threshold` | [1.0] |
| `payment_weight` | [1.0] |
| `timestep_threshold` | [0] |
| `advance_percent` | 0 |
| `efficiency` | `[0.91, 1.17, 0.83]` |

---

#### `TC-1LMAIF-R` — delayed start + full milestone payment mechanics

One project, late start (`planned_start=2`), multi-period (`planned_duration=3`),
advance + interim + final payments, random efficiency.

**Purpose:** verify delayed start handling, multi-milestone payment sequencing,
advance trigger, advance recovery across milestones, and retention deduction.

**Key design decisions:**

- `advance_trigger=0.8` — the advance fires when progress ≥ 80%, which falls
  between `plan[t=1]=0.7845` and `plan[t=2]=0.9693`. This means the advance
  is paid *after* m_1 fires, so **m_1 carries no advance recovery deduction**.

- `advance_recovery=0.5` — 50% of each milestone payment is deducted as
  recovery *after* the advance has triggered. With `payment_weight=[0.3, 0.7, 1.0]`
  (incremental: 0.30, 0.40, 0.30 of BAC):
  - m_1 (t=1): advance not yet triggered → **deduction = 0%**
  - m_2 (t=2): advance triggered → deduction = 0.5 × 30% = **15% of BAC**... wait,
    incremental weight at m_2 is 0.40 → deduction = 0.5 × 40% = **20% of BAC = full advance recovered ✓**
  - m_3 (t=3): advance fully recovered → **deduction = 0%**

- Retention (5% of BAC) is deducted from the final milestone payment (m_3),
  applied after advance recovery is complete.

- Efficiency is uniform `[0.91, 1.17, 0.83, 0.91, 1.17, 0.83]` — no
  termination incentive; this config is for payment mechanic verification only.

| Parameter | Value |
|---|---|
| `planned_start` | 2 |
| `planned_duration` | 3 |
| `advance_percent` | 0.20 |
| `advance_trigger` | 0.80 |
| `advance_recovery` | 0.50 |
| `n_milestones` | 3 |
| `progress_threshold` | [0.40, 0.97, 1.00] |
| `payment_weight` (cumul.) | [0.30, 0.70, 1.00] |
| `timestep_threshold` | [1, 2, 3] |
| `efficiency` | `[0.91, 1.17, 0.83, 0.91, 1.17, 0.83]` |

---

#### `TC-10MAIF-R` — diminishing efficiency / early termination

One project, starts at t=0, multi-period (`planned_duration=3`),
advance + interim + final, diminishing efficiency.

**Purpose:** verify that early termination is the optimal action. The project
is economical in its first two active periods (advance paid, m_1 collected)
but becomes increasingly wasteful from m_2 onward.

**Key design decisions:**

- `advance_trigger=0.0` — advance paid immediately at episode start.

- `advance_recovery=0.2` — recovery spread evenly across all milestones.
  With incremental weights `[0.30, 0.40, 0.30]` and recovery rate 0.20,
  each milestone deducts 20% of its payment as recovery:
  - m_1: 0.20 × 30% = 6% of BAC recovered
  - m_2: 0.20 × 40% = 8% of BAC recovered
  - m_3: 0.20 × 30% = 6% of BAC recovered → total = 20% = full advance ✓

- Efficiency series `[0.95, 0.91, 0.55, 0.40, 0.30, 0.20]`:
  - t_project=0,1 (GOOD): near-full efficiency, advance and m_1 collected cheaply
  - t_project=2,3,4,5 (BAD): steep cost cliff, m_2 onward is uneconomical
  - Late periods cost on average **2.57×** more per unit of progress than early periods

| t_project | efficiency | cost/unit | phase |
|---|---|---|---|
| 0 | 0.95 | 1.05× | GOOD — advance period |
| 1 | 0.91 | 1.10× | GOOD — m_1 fires |
| 2 | 0.55 | 1.82× | BAD — m_2 territory |
| 3 | 0.40 | 2.50× | BAD |
| 4 | 0.30 | 3.33× | BAD — late period 1 |
| 5 | 0.20 | 5.00× | BAD — late period 2 |

| Parameter | Value |
|---|---|
| `planned_start` | 0 |
| `planned_duration` | 3 |
| `advance_percent` | 0.20 |
| `advance_trigger` | 0.00 |
| `advance_recovery` | 0.20 |
| `n_milestones` | 3 |
| `progress_threshold` | [0.40, 0.97, 1.00] |
| `payment_weight` (cumul.) | [0.30, 0.70, 1.00] |
| `timestep_threshold` | [1, 2, 3] |
| `efficiency` | `[0.95, 0.91, 0.55, 0.40, 0.30, 0.20]` |

---

#### `TC-3XMAIF-R` — multi-project gap and overlap

Three projects, mixed start times (`planned_start=[0, 6, 7]`), each
`planned_duration=3`, advance + interim + final, varied efficiency per project.

**Purpose:** verify that the portfolio handles projects in three distinct
temporal relationships: one running alone early, a gap period, then two
overlapping. Tests budget sharing, per-project milestone sequencing, and
that pending/active/completed state transitions work correctly across a
heterogeneous portfolio.

**Project timeline:**

| Project | planned_start | active periods | relationship |
|---|---|---|---|
| P0 | t=0 | t=0–3 | runs alone |
| P1 | t=6 | t=6–9 | gap between P0 and P1/P2 |
| P2 | t=7 | t=7–10 | overlaps with P1 |

- BAC: `[100, 200, 300]` — different sizes to test budget allocation under constraint
- Each project has its own efficiency series (no shared pattern — varied to stress
  the per-project state tracking)
- Milestone structure is identical across all three projects

| Parameter | Value |
|---|---|
| `planned_start` | [0, 6, 7] |
| `planned_duration` | 3 (all projects) |
| `bac` | [100, 200, 300] |
| `advance_percent` | 0.20 |
| `advance_trigger` | 0.00 |
| `advance_recovery` | 0.20 |
| `n_milestones` | 3 (all projects) |
| `progress_threshold` | [0.40, 0.97, 1.00] each |
| `payment_weight` (cumul.) | [0.30, 0.70, 1.00] each |
| `timestep_threshold` | [1, 2, 3] each |
| `efficiency` (P0) | `[0.91, 1.17, 0.83, 0.91, 1.17, 0.83]` |
| `efficiency` (P1) | `[1.09, 0.84, 1.18, 1.09, 0.84, 1.18]` |
| `efficiency` (P2) | `[0.97, 1.13, 0.86, 0.97, 1.13, 0.86]` |

---

## Generating regression fixtures

`add_baseline_fixtures.py` runs every solver over every config and writes
the exported DB tables as CSV ground-truth fixtures under:

```
tests/regression/ground_truth/<config_id>/<method>/<table>.csv
```

**Basic usage** (from the project root):

```bash
python src/tests/solvers/add_baseline_fixtures.py
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--configs TC-1LMAIF-R ...` | all | Only run these config IDs |
| `--methods catchup_t ...` | all | Only run these solver methods |
| `--overwrite` | off | Overwrite existing fixtures without prompting |
| `--seed N` | `0` | RNG seed passed to `env.reset()` |
| `--no-db` | off | Skip DB logging and fixture export |

**Example — regenerate one config, one method:**

```bash
python src/tests/solvers/add_baseline_fixtures.py \
    --configs TC-1LMAIF-R \
    --methods catchup_t \
    --overwrite
```

Episode IDs take the form `<config_id>__<method>__` (empty `run_id`) so
results are fully deterministic and reproducible across runs with the same seed.