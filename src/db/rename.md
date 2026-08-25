# Parameter Rename Map
# Old name → New name, with the file/context each appears in.
# Use this to refactor: env.py, sampler.py, payments.py, evm.py,
# db_logger.py, render.py, single_project.py, dual_project.py

---

## environment_config table

| Old | New |
|-----|-----|
| `initial_budget_dist / _p1–p4` | `budget_available_dist / _p1–p4` |
| `budget_dist / _p1–p4` | `bac_dist / _p1–p4` |
| `margin_dist / _p1–p4` | `profit_percent_dist / _p1–p4` |
| `start_dist / _p1–p4` | `planned_start_dist / _p1–p4` |
| `duration_dist / _p1–p4` | `planned_duration_dist / _p1–p4` |
| `plan_deviation_threshold` (single scalar) | `progress_delay_cap_dist / _p1–p4` (full dist) |
| `schedule_cap_dist / _p1–p4` | `finish_delay_cap_dist / _p1–p4` |
| `cost_cap_dist / _p1–p4` | `cost_overrun_cap_dist / _p1–p4` |
| `cure_length_dist / _p1–p4` | `termination_tolerance_dist / _p1–p4` |
| `threshold_dist / _p1–p4` | `progress_threshold_dist / _p1–p4` |
| `earliest_t_fraction_dist / _p1–p4` | `timestep_threshold_dist / _p1–p4` |

Column count change: 115 → 119
Reason: `plan_deviation_threshold` was 1 scalar, now 5 columns (dist + p1–p4).

---

## projects_profile table

| Old | New |
|-----|-----|
| `budget` | `bac` |
| `margin` | `profit_percent` |
| `start` | `planned_start` |
| `finish` | `planned_finish` |
| `duration` | `planned_duration` |
| `plan_deviation_threshold` | `progress_delay_cap` |
| `schedule_cap` | `finish_delay_cap` |
| `cost_cap` | `cost_overrun_cap` |
| `cure_length` | `termination_tolerance` |

---

## milestones_profile table

| Old | New |
|-----|-----|
| `threshold` | `progress_threshold` |
| `earliest_t` | `timestep_threshold` |
| *(absent)* | `gross_payment` (new — computed at reset) |
| *(absent)* | `advance_recovery` (new — computed at reset) |
| *(absent)* | `advance_recovery_remain` (new — computed at reset) |
| *(absent)* | `retention_withheld` (new — computed at reset) |
| *(absent)* | `retention_released` (new — computed at reset) |
| *(absent)* | `net_payment` (new — computed at reset) |

---

## portfolios table

| Old | New |
|-----|-----|
| `budget` | `budget_available` |
| `inflow` | *(removed — redundant with net_cashflow)* |
| `outflow` | *(removed — redundant with net_cashflow)* |
| *(absent)* | `net_cashflow` (new) |

---

## projects_status table

| Old | New |
|-----|-----|
| `progress` | `progress_actual` |
| `progress_increment` | `progress_actual_periodic` |
| *(absent)* | `progress_plan_periodic` (new) |
| `schedule_slip` | `projected_finish_delay` |
| `plan_deviation` | `progress_delay` |
| `cost_overrun` | `projected_cost_overrun` |
| `forecast_finish` | `projected_finish` |
| `cure_remaining` | `tolerance_remain` |
| `advance_amount` | `advance_received` → renamed to `inflow` (cumulative) |
| `payment_net` | `payment_net` → stays, but now under cumulative `inflow` |
| `treasury_draw` | `deficit` |
| `eac_bac_ratio` | *(removed — derivable as EAC/BAC; use projected_cost_overrun)* |
| *(absent)* | `allocation_action` (new — raw agent input before interest) |
| *(absent)* | `inflow` (new — cumulative) |
| *(absent)* | `outflow` (new — cumulative) |
| *(absent)* | `termination_settlement` (new — standalone) |
| *(absent)* | `abandoned` (new breach flag) |
| *(absent)* | `over_duration_window` (new breach flag) |
| *(absent)* | `over_progress_delay` (new breach flag) |
| *(absent)* | `over_finish_delay` (new breach flag) |
| *(absent)* | `over_cost_overrun` (new breach flag) |
| *(absent)* | `over_any` (new breach flag) |
| *(absent)* | `target_milestone` (new — next uncertified milestone index) |
| *(absent)* | `target_progress` (new — progress threshold to unlock it) |
| *(absent)* | `target_timestep` (new — earliest period it can be certified) |
| *(absent)* | `target_net_payment` (new — net payment when certified) |

---

## milestones_status table

| Old | New |
|-----|-----|
| `certified` | *(removed — certified_t NOT NULL is sufficient signal)* |
| `certified_t` | `certified_t` (unchanged) |
| `payment_released` | `net_payment` |
| *(absent)* | `certification_delay` (new — certified_t minus timestep_threshold) |

---

## Python dict keys to update (env.py, sampler.py, payments.py, evm.py)

### proj[] dict (sampled in sampler.py, read everywhere)

| Old key | New key |
|---------|---------|
| `proj["budget"]` | `proj["bac"]` |
| `proj["margin"]` | `proj["profit_percent"]` |
| `proj["start"]` | `proj["planned_start"]` |
| `proj["finish"]` | `proj["planned_finish"]` |
| `proj["duration"]` | `proj["planned_duration"]` |
| `proj["plan_deviation_threshold"]` | `proj["progress_delay_cap"]` |
| `proj["schedule_cap"]` | `proj["finish_delay_cap"]` |
| `proj["cost_cap"]` | `proj["cost_overrun_cap"]` |
| `proj["cure_length"]` | `proj["termination_tolerance"]` |

### ps[] dict (project state, lives in env.py)

| Old key | New key |
|---------|---------|
| `ps["progress"]` | `ps["progress_actual"]` |
| `ps["progress_increment"]` | `ps["progress_actual_periodic"]` |
| `ps["schedule_slip"]` | `ps["projected_finish_delay"]` |
| `ps["plan_deviation"]` | `ps["progress_delay"]` |
| `ps["cost_overrun"]` | `ps["projected_cost_overrun"]` |
| `ps["forecast_finish"]` | `ps["projected_finish"]` |
| `ps["cure_remaining"]` | `ps["tolerance_remain"]` |
| `ps["treasury_draw"]` | `ps["deficit"]` |

### ms[] dict (milestone, lives in sampler.py + payments.py)

| Old key | New key |
|---------|---------|
| `ms["threshold"]` | `ms["progress_threshold"]` |
| `ms["earliest_t"]` | `ms["timestep_threshold"]` |
| `ms["payment_released"]` | `ms["net_payment"]` |

### cfg[] dict (config, lives in single_project.py + dual_project.py)

| Old key | New key |
|---------|---------|
| `cfg["initial_budget_dist"]` | `cfg["budget_available_dist"]` |
| `cfg["budget_dist"]` | `cfg["bac_dist"]` |
| `cfg["margin_dist"]` | `cfg["profit_percent_dist"]` |
| `cfg["start_dist"]` | `cfg["planned_start_dist"]` |
| `cfg["duration_dist"]` | `cfg["planned_duration_dist"]` |
| `cfg["plan_deviation_threshold"]` | `cfg["progress_delay_cap_dist"]` + `_p1–p4` |
| `cfg["schedule_cap_dist"]` | `cfg["finish_delay_cap_dist"]` |
| `cfg["cost_cap_dist"]` | `cfg["cost_overrun_cap_dist"]` |
| `cfg["cure_length_dist"]` | `cfg["termination_tolerance_dist"]` |
| `cfg["threshold_dist"]` | `cfg["progress_threshold_dist"]` |
| `cfg["earliest_t_fraction_dist"]` | `cfg["timestep_threshold_dist"]` |

---

## Files and what to change in each

### sampler.py
- All `cfg["..."]` key lookups — use cfg rename table above
- `sample_project()` return dict — use proj rename table above
- `sample_milestones()` — rename `threshold` → `progress_threshold`, `earliest_t` → `timestep_threshold`
- Add computation of `gross_payment`, `advance_recovery`, `advance_recovery_remain`, `retention_withheld`, `retention_released`, `net_payment` per milestone (currently absent, needed for milestones_profile insert)

### payments.py
- `ms["threshold"]` → `ms["progress_threshold"]`
- `ms["earliest_t"]` → `ms["timestep_threshold"]`
- `proj["budget"]` → `proj["bac"]`
- `proj["cure_length"]` → `proj["termination_tolerance"]`

### evm.py
- `ps["plan_deviation"]` → `ps["progress_delay"]`
- `ps["cost_overrun"]` → `ps["projected_cost_overrun"]`
- `ps["forecast_finish"]` → `ps["projected_finish"]`
- `ps["schedule_slip"]` → `ps["projected_finish_delay"]`
- `proj["cost_cap"]` → `proj["cost_overrun_cap"]`
- `proj["schedule_cap"]` → `proj["finish_delay_cap"]`
- `proj["duration"]` → `proj["planned_duration"]`
- `proj["start"]` → `proj["planned_start"]`
- `proj["finish"]` → `proj["planned_finish"]`



### env.py
- All proj[], ps[], ms[], cfg[] keys — use all tables above
- `self.budget` internal name stays; DB write column is `budget_available`
- Add `ps["allocation_action"]` separate from `ps["allocation"]`
- Add `ps["deficit"]` replacing `ps["treasury_draw"]`
- Add all six breach flag keys to ps[]
- Add `target_milestone`, `target_progress`, `target_timestep`, `target_net_payment` to ps[]
- Phase 5: rename breach flag variables to match new names

### db_logger.py
- `write_profiles()`: column names → new projects_profile and milestones_profile names; add new milestones_profile columns
- `write_portfolio_row()`: `budget` → `budget_available`; remove `inflow`/`outflow`; add `net_cashflow`
- `write_project_row()`: all column names → new projects_status names; add all new columns
- `write_milestone_status()`: `payment_released` → `net_payment`; add `certification_delay`; remove `certified`

### render.py
- All display label strings and dict key lookups → new names
- Payment profile table column data keys → new milestones_profile names
- Breach flag display → new breach flag names (abandoned, over_*)

### single_project.py and dual_project.py
- All CONFIG dict keys → new names
- INSERT column list → new environment_config column names
- VALUES tuple → same data, reordered to match; placeholder count 115 → 119