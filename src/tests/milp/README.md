# PPM Environment Verification Suite — MILP Baseline(MVP)

Test suite and report generator for the MILP baseline.



All scripts live in `tests/milp/` and are self-contained.

---

## Layout

```
root/
├── baselines/
│   └── milp.py                ← the baseline being tested
└── tests/
    └── milp/
        ├── case_data.py       ← single source of truth for all cases (edit here)
        ├── cases.json         ← generated — do not edit by hand
        ├── test_milp.py       ← pytest suite (drives milp.py; generates report)
        ├── test_report.py     ← report generator (tables + TikZ figures → report.tex)
        └── report.tex         ← generated after every test run
```

---



## How it works



### Testing

`test_milp.py` drives the real solver for every case:

1. Translates each `cases.json` spec into a `milp.py`-compatible portfolio dict.
2. Calls `baselines/milp.py` → `build_and_solve()` then `build_records()`.
3. Asserts the **live MILP solver output** against the hand-solved ground-truth
  cells (each stored with an absolute tolerance in `cases.json`).

Results are cached within a session — all 11 test categories for a case share
one solver call.

### Report generation

After all tests finish, `test_milp.py` calls `test_report.generate_report()`
via a `pytest_sessionfinish` hook.  The report contains:

1. **Verification table** — pass / fail / skip per case per test category,
  populated from the live pytest run.
2. **Figure legends** and **conventions**.
3. **Per-case blocks** (one per case, grouped by G1–G9):
  - Parameter summary table
  - Per-project timeline data table
  - Portfolio strip data table
  - **Project figure** — three panels per project row:
    - EVM signals (BCWS / BCWP / ACWP, milestone thresholds, termination line)
    - Milestone profile (spend bars down, inflow bars up, advance bar)
    - SPI / CPI index plot
  - **Portfolio figure** — three panels:
    - Cash balance step-plot (with B₁ reference)
    - Period cash flows (inflow / outflow bars + net line)
    - Cumulative discounted NCF

All axes are **dynamic**: x-axis scales to each case's horizon H;
middle project panel uses symmetric ±BAC range;
portfolio panels use data-driven ranges with auto-computed ticks.

---



## Quick-start

```bash
# 1. Regenerate cases.json from case definitions
python tests/milp/case_data.py

# 2. Run tests AND generate report.tex in one step
pytest tests/milp/test_milp.py -v

# 3. Generate report.tex standalone (without running tests)
python tests/milp/test_report.py

# 4. Generate report for a subset of cases
python tests/milp/test_report.py --cases SP-1 SP-3 MP-K
```

---



## Running tests

```bash
# All cases, all tests
pytest tests/milp/test_milp.py -v

# Single case
pytest tests/milp/test_milp.py -v -k SP-1

# All cases in a group
pytest tests/milp/test_milp.py -v -k G3

# One test category across all cases
pytest tests/milp/test_milp.py -v -k objective_value
pytest tests/milp/test_milp.py -v -k milestone_certification
pytest tests/milp/test_milp.py -v -k cash_balance_nonneg

# Short output, stop on first failure
pytest tests/milp/test_milp.py -q --tb=short -x
```

> **Note:** `report.tex` is always written at session end, even on partial runs
> (e.g. `-k SP-1` or `-x`).  Cells for untested cases show `---` in the table.



### The 11 test categories


| Key        | Test method                    | What it checks                                         |
| ---------- | ------------------------------ | ------------------------------------------------------ |
| `budget`   | `test_budget_allocation`       | MILP `x_{i,t}` non-negative, within tol of hand-solved |
| `progress` | `test_progress_accumulation`   | `P` from MILP `x` in `[0,1]`, matches stored cells     |
| `evm`      | `test_evm_signals`             | SPI, CPI ≥ 0 and match stored cells                    |
| `termctr`  | `test_termination_counter`     | `τ_rem` decrements only when both conditions fire      |
| `mscert`   | `test_milestone_certification` | MILP certifies only when `P ≥ θ_j` AND `t ≥ e_j`       |
| `payment`  | `test_payment_identity`        | `A + ΣR_net + R_ret = CP` for completed projects       |
| `cashbal`  | `test_cash_balance_nonneg`     | MILP `B_t ≥ 0` at all `t`, matches stored cells        |
| `zstar`    | `test_objective_value`         | MILP `Z*` within `tol_Z` of hand-solved value          |
| `outcome`  | `test_outcome_status`          | completed / terminated status matches `meta.outcome`   |
| `monotone` | `test_progress_monotone`       | `P_{i,t}` non-decreasing across MILP solution          |
| `advance`  | `test_advance_timing`          | advance reflected in MILP `B_t` from `t = s_i`         |


Cases SP-E, SP-F, SP-G, SP-I have no `Z*` target — `test_objective_value` is
skipped for them.

---



## Report output

`report.tex` is designed to be `\input{}` into a parent document.
Required packages in the parent:

```latex
\usepackage{booktabs, longtable, xcolor, pifont}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\definecolor{msgreen}{RGB}{0,140,60}
```

Then include with:

```latex
\input{tests/milp/report}
```

---



## Adding a new test case

1. Open `tests/milp/case_data.py` and write a builder function:

```python
def sp_new():
    ps = dict(
        BAC=100, eta=1.0, D_plan=4, fi=4, si=1,
        alloc={1: 50, 3: 50},
        ms_theta=[0.5, 1.0], ms_e=[1, 3], ms_phi=[0.5, 0.5],
        CP=120, rho=0, alpha=0, A=0,
        mu=0.30, Omega=1, tau_tol=2,
    )
    return make_case(
        meta={"id": "SP-New", "group": "G1",
              "Zstar": 19.025, "tol_Z": 1.0, "outcome": "completed"},
        params={"BAC": 100, "CP": 120, "H": 4},
        projects_spec=[ps], B0=300, H=4,
    )
```

1. Add it to `ALL_CASES` in `case_data.py` and to `CASE_ORDER` in `test_report.py`.
2. Regenerate and run:

```bash
python tests/milp/case_data.py
pytest tests/milp/test_milp.py -v -k SP-New
```

---



## When `baselines/milp.py` changes

Run the full suite. Failing tests identify exactly which periods and cases
diverge from the hand-solved ground truth. The generated `report.tex` marks
those cells in red (✗) in the verification table.

```bash
pytest tests/milp/test_milp.py -q --tb=short
```

---



## Updating tolerances

Every cell in `cases.json` is `{"v": <value>, "tol": <tolerance>}`.
Default tolerances are set by `V()` in `case_data.py` (2% of magnitude,
minimum 0.5). To tighten or relax the `Z*` assertion, change `"tol_Z"` in
that case's `meta` dict, then regenerate.

---



## File dependency

```
case_data.py ──► cases.json ──► test_milp.py  ──► build_and_solve()  [milp.py]
                                    │                build_records()   [milp.py]
                                    │
                                    └──► test_report.py ──► report.tex
                                              ▲
                                         (also standalone:
                                          python test_report.py)
```

---



## Conventions

- `γ = 0.95` throughout.
- Advance `A_i` arrives at `t = s_i − 1` (period 0 when `s_i = 1`).
- Discounting: `γ^{t−1}` for flows at period `t ≥ 1`; advance at `t = 0`
contributes at face value.
- Payment identity `A + ΣR_net + R_ret = CP` checked only for completed projects.
- Retention released as a lump at the period the final milestone certifies.
- Advance recovery deductions use `δ_rec × φ_j × CP` (pre-retention base,
FIDIC convention).

