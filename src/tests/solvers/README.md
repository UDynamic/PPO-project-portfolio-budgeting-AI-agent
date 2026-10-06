# Baseline Solvers

`src/tests/solvers/`

---

## Overview

This directory contains three deterministic baseline solvers for
`PortfolioBudgetingEnv`, plus the script that uses them to generate
regression fixtures.

Each solver is a rule-based policy that reads fields already computed
by the environment (`catchup_alloc_t`, `reach_plan_t`, etc.) and
converts them into a budget-fraction action vector via the shared
`_amounts_to_action` helper. No learning, no randomness.

---

## Files

| File | Purpose |
|---|---|
| `baseline_solvers.py` | Solver classes + `run_baselines()` runner called from `play.py` |
| `add_baseline_fixtures.py` | Batch script that runs every solver × config and writes regression fixtures |

---

## Solvers

### `catchup_t` — CatchupTSolver

Allocates `catchup_alloc_t` to each active project. This is the minimum
amount needed to keep the project within its progress-delay tolerance this
period — it brings actual progress up to `plan_nt − delay_cap`, but does
not overshoot the plan.

**Late-phase switch (finishing intent):**
`catchup_alloc_t` is derived from the S-curve planned progress, which
saturates at `1.0` once `t_project >= planned_duration`. At that point
`plan_nt = 1.0` and the gap to the tolerance floor collapses to zero,
so `catchup_alloc_t = 0`. If left uncorrected, the solver would allocate
nothing while the project is still active, and the project would be
terminated by `over_duration_window` before completing.

To prevent this, the solver detects when a project has reached or passed
its planned duration and switches to `reach_plan_t` instead:

```python
if ps["status"] == "active" and ps["t_project"] >= proj["planned_duration"]:
    amounts.append(ps.get("reach_plan_t", 0.0))   # finishing intent
else:
    amounts.append(ps.get("catchup_alloc_t", 0.0)) # normal catchup
```

`reach_plan_t` is the amount needed to close the remaining progress gap
to `1.0` in the current period, so it expresses an explicit intent to
finish the project and keeps spending positive for as long as work remains.

Example (config `TC-1LMAIF-R`, `planned_duration=3`, `planned_start=2`):

| Episode t | t_project | catchup_alloc_t | reach_plan_t | Allocated | Phase |
|-----------|-----------|-----------------|--------------|-----------|-------|
| 2 | 0 | 49.88 | 64.88 | 49.88 | normal catchup |
| 3 | 1 | 37.82 | 52.82 | 37.82 | normal catchup |
| 4 | 2 | 0.00 | 10.35 | 0.00 | normal catchup |
| 5 | 3 | 0.00 | 10.35 | **10.35** | **late-phase switch** |
| 6 | 4 | 0.00 | 0.93 | **0.93** | **late-phase switch** → completed |

---

### `reach_t` — ReachTSolver

Allocates `reach_plan_t` to each active project every period. This fully
closes the gap to the current period's plan target — no breach, no
carry-over. More aggressive than `catchup_t` and higher cost, but
guarantees the project tracks the S-curve exactly when budget permits.

---

### `catchup_next_t` — CatchupNextTSolver

Allocates `catchup_alloc_next_t` to each active project. This is a
one-period lookahead: it allocates enough now so that after this period's
progress the project will be within tolerance of *next* period's plan
target. Sits between `catchup_t` (reactive minimum) and `reach_t`
(full catch-up) in terms of spend.

---

## Shared action conversion

All three solvers use `_amounts_to_action(env, amounts)` from `_BaseSolver`:

1. Non-active projects are zeroed.
2. Each amount is divided by the current budget to get a fraction.
3. If the sum of fractions exceeds `1.0` the vector is rescaled
   proportionally — the environment enforces the same constraint
   internally, so this just avoids unnecessary clipping.

---

## Generating regression fixtures

`add_baseline_fixtures.py` runs every solver over every config in
`db/configs/` and writes the exported DB tables as CSV ground-truth
fixtures under:

```
tests/regression/ground_truth/<config_id>/<method>/<table>.csv
```

The fixtures are the canonical reference for regression tests — no
bespoke harvesting, the same export path used by `play.py`.

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

The script uses a single run directory and SQLite database for the whole
batch, identified by episode IDs of the form
`<config_id>__<method>__` (empty `run_id` so the ID is fully
deterministic and reproducible across runs with the same seed).