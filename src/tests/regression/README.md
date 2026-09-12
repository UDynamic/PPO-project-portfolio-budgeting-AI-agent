# Regression Test Suite

## Structure

```
tests/regression/
    harness.py              # parametrized pytest — auto-discovers all fixtures
    conftest.py             # pytest config
    README.md               # this file
    add_fixture.py          # helper: copy a play.py export into ground_truth/
    ground_truth/
        <config_id>/
            optimal/
                actions.json
                portfolios.csv
                projects_status.csv
                projects_observation.csv
                milestones_status.csv
            greedy_catchup/
                ...
            greedy_reach_plan/
                ...
            uniform/
                ...
```

## Running tests

```bash
# All fixtures
pytest tests/regression/ -v

# One config
pytest tests/regression/ -k "TC-10SF-R" -v

# One method across all configs
pytest tests/regression/ -k "optimal" -v

# Short tracebacks
pytest tests/regression/ --tb=short
```

## Adding a new fixture

**Option A — automated (recommended)**

After a verified play.py run, note the exports path printed in the terminal, then:

```bash
python tests/regression/add_fixture.py \
    --exports "path/to/runs/2026-09-12_15-20-03/exports" \
    --config TC-10SF-R \
    --method optimal \
    --actions "[110.0]"
```

`--actions` is a JSON list of per-period allocations. Single-project configs
use flat scalars; multi-project configs use lists-of-lists:
```
--actions "[[50.0, 30.0], [20.0, 10.0]]"
```

**Option B — manual**

1. Copy the exports folder into `ground_truth/<config_id>/<method>/`.
2. Create `actions.json` with the allocation sequence you used.
3. Run `pytest tests/regression/ -k "<config_id>__<method>"` to verify.

## Workflow when changing env.py / helper.py

1. `pytest tests/regression/` — confirm all current fixtures pass (baseline).
2. Apply code change.
3. `pytest tests/regression/` — failures identify which field/period broke.
4. If the change is an intentional fix (e.g. pending catchup bug):
   - Re-run play.py with the same actions on the fixed env.
   - Use `add_fixture.py` to overwrite the affected fixture's CSVs.
   - Add a comment in actions.json explaining what changed and why.
5. Re-run to confirm green.

## Tolerance

Numeric fields are compared within `TOLERANCE = 1e-3` (absolute).
String fields (e.g. `status`) use exact equality.
Adjust `TOLERANCE` in `harness.py` if needed.

## Fields asserted per table

Defined in `harness.py` as `PORTFOLIO_FIELDS`, `PROJECT_STATUS_FIELDS`,
`PROJECT_OBS_FIELDS`, `MILESTONE_STATUS_FIELDS`.
Extend these lists as confidence in new fields grows.