# Schema Update — Parameter Rename Log

Use this file to update all other files that reference the old parameter names:
`env.py`, `sampler.py`, `payments.py`, `evm.py`, `db_logger.py`, `render.py`,
`single_project.py`, `dual_project.py`, `play.py`.

---

## Renames — environment_config table

| Old name | New name |
|----------|----------|
| `initial_budget_dist / _p1–p4` | `budget_available_dist / _p1–p4` |
| `budget_dist / _p1–p4` | `bac_dist / _p1–p4` |
| `margin_dist / _p1–p4` | `profit_percent_dist / _p1–p4` |
| `plan_deviation_threshold` (single scalar) | `progress_delay_cap_dist / _p1–p4` (full dist + 4 params, no default) |
| `schedule_cap_dist / _p1–p4` | `finish_delay_cap_dist / _p1–p4` |
| `cost_cap_dist / _p1–p4` | `cost_overrun_cap_dist / _p1–p4` |
| `cure_length_dist / _p1–p4` | `termination_tolerance_dist / _p1–p4` |
| `threshold_dist / _p1–p4` | `progress_threshold_dist / _p1–p4` |
| `earliest_t_fraction_dist / _p1–p4` | `timestep_threshold_dist / _p1–p4` |

---

## Renames — projects_profile table

| Old name | New name |
|----------|----------|
| `budget` | `bac` |
| `margin` | `profit_percent` |
| `plan_deviation_threshold` | `progress_delay_cap` |
| `schedule_cap` | `finish_delay_cap` |
| `cost_cap` | `cost_overrun_cap` |
| `cure_length` | `termination_tolerance` |

---

## Renames — milestones_profile table

| Old name | New name |
|----------|----------|
| `threshold` | `progress_threshold` |
| `earliest_t` | `timestep_threshold` |

---

## Renames — portfolios table

| Old name | New name |
|----------|----------|
| `budget` | `budget_available` |

---

## Renames — projects_status table

| Old name | New name |
|----------|----------|
| `progress` | `progress_actual` |
| `progress_increment` | `progress_actual_periodic` |
| `schedule_slip` | `projected_finish_delay` |
| `plan_deviation` | `progress_delay` |
| `cost_overrun` | `projected_cost_overrun` |
| `forecast_finish` | `projected_finish` |
| `cure_remaining` | `tolerance_remain` |
| `advance_amount` | `advance_received` |

---

## New columns — projects_status table

| Column | Type | Description |
|--------|------|-------------|
| `progress_plan_periodic` | REAL | planned progress increment this period (was absent) |

---

## Structural changes — milestones_status table

The old table only recorded `certified`, `certified_t`, `payment_released`.
The new table records the full payment breakdown to support the payment profile display.

| Column | Type | Description |
|--------|------|-------------|
| `progress_threshold` | REAL | denormalised from milestones_profile for query convenience |
| `payment_weight` | REAL | denormalised from milestones_profile |
| `timestep_threshold` | INTEGER | denormalised from milestones_profile (renamed from earliest_t) |
| `gross` | REAL | weight × price |
| `advance_recovery` | REAL | advance recovered this milestone |
| `cumulative_recovery` | REAL | running total advance recovered at this milestone |
| `retention_withheld` | REAL | retention held this milestone (0 for final) |
| `payment_net` | REAL | net paid (replaces old payment_released) |
| `certified` | INTEGER | 0 / 1 (unchanged) |
| `certified_t` | INTEGER | period when certified (unchanged) |

---

## Structural changes — environment_config table

`plan_deviation_threshold` was a single scalar column with `DEFAULT 0.10`.
It is now `progress_delay_cap_dist / _p1 / _p2 / _p3 / _p4` — a full
distribution with 4 parameter slots, consistent with all other caps.
No default value. Must be explicitly set in every config.

---

## Files to update

### `single_project.py` and `dual_project.py`

- Rename all CONFIG dict keys to match new names above.
- Add `progress_delay_cap_dist / _p1–p4` (replace the single `plan_deviation_threshold` scalar).
- Rename INSERT column list and VALUES tuple to match.
- The INSERT now has 119 columns (was 115): +4 for progress_delay_cap dist/params, -1 for the old scalar = net +3. Recount placeholders.

### `env.py`

- All `proj["budget"]` → `proj["bac"]`
- All `proj["margin"]` → `proj["profit_percent"]`
- All `proj["plan_deviation_threshold"]` → `proj["progress_delay_cap"]`
- All `proj["schedule_cap"]` → `proj["finish_delay_cap"]`
- All `proj["cost_cap"]` → `proj["cost_overrun_cap"]`
- All `proj["cure_length"]` → `proj["termination_tolerance"]`
- All `ps["progress"]` → `ps["progress_actual"]`
- All `ps["progress_increment"]` → `ps["progress_actual_periodic"]`
- All `ps["schedule_slip"]` → `ps["projected_finish_delay"]`
- All `ps["plan_deviation"]` → `ps["progress_delay"]`
- All `ps["cost_overrun"]` → `ps["projected_cost_overrun"]`
- All `ps["forecast_finish"]` → `ps["projected_finish"]`
- All `ps["cure_remaining"]` → `ps["tolerance_remain"]`
- All `self.budget` (portfolio level) stays as `budget` internally; DB write column renamed to `budget_available`.
- Add `ps["progress_plan_periodic"]` computation alongside `ps["progress_actual_periodic"]`.

### `sampler.py`

- `sample_project`: rename all dict keys returned to match new names.
- `sample_milestones`: rename `threshold` → `progress_threshold`, `earliest_t` → `timestep_threshold`.
- Config key lookups: `cfg["budget_dist"]` → `cfg["bac_dist"]`, etc. Full list matches environment_config rename table above.

### `evm.py`

- `ps["plan_deviation"]` → `ps["progress_delay"]`
- `ps["cost_overrun"]` → `ps["projected_cost_overrun"]`
- `ps["forecast_finish"]` → `ps["projected_finish"]`
- `ps["schedule_slip"]` → `ps["projected_finish_delay"]`
- `proj["cost_cap"]` → `proj["cost_overrun_cap"]`
- `proj["schedule_cap"]` → `proj["finish_delay_cap"]`

### `payments.py`

- `ms["threshold"]` → `ms["progress_threshold"]`
- `ms["earliest_t"]` → `ms["timestep_threshold"]`
- `proj["budget"]` → `proj["bac"]`

### `db_logger.py`

- `write_profiles`: column names in INSERT match new projects_profile and milestones_profile names.
- `write_portfolio_row`: `budget` column → `budget_available`.
- `write_project_row`: all column names updated per projects_status rename table.
- `write_milestone_status`: expand INSERT to include all new milestones_status columns.

### `render.py`

- All display labels and dict key lookups updated to new names.
- Payment profile table already matches the target display — column data keys update only.

---

## Column count change — environment_config

Old: 115 columns
New: 119 columns

Difference: `plan_deviation_threshold` (1 scalar) replaced by
`progress_delay_cap_dist, _p1, _p2, _p3, _p4` (5 columns) = net +4.

Update placeholder count in `single_project.py` and `dual_project.py` from 115 to 119.