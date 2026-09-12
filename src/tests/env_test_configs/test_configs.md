# Test configs

<!-- ⚠️ removed two set's of columns:
```
budget_available_dist       TEXT NOT NULL,
    budget_available_p1         REAL NOT NULL,
    budget_available_p2         REAL,
    budget_available_p3         REAL,
    budget_available_p4         REAL,
```

the other one is :
```
    -- cost_overrun_cap : max tolerated EAC / BAC ratio; e.g. 1.30 = 30% overrun
    cost_overrun_cap_dist       TEXT NOT NULL,
    cost_overrun_cap_p1         REAL NOT NULL,
    cost_overrun_cap_p2         REAL,
    cost_overrun_cap_p3         REAL,
    cost_overrun_cap_p4         REAL,
``` -->

---

## Naming convention

```
TC-{N}{S}{D}{P}{E}
```

| Segment | Position | Values | Meaning |
|---------|----------|--------|---------|
| `TC` | prefix | fixed | Test Config |
| `N` | projects | `1` `2` `3` | number of projects |
| `S` | start | `0` `L` `X` | all at t=0 / all Late / miXed |
| `D` | duration | `S` `M` | Single period / Multi period |
| `P` | payments | `F` `AF` `AIF` | Final only / Advance+Final / Advance+Interim+Final |
| `E` | efficiency | `P` `R` | Perfect η=1 / Random seeded |

---

## Full config list (**Total: 48 configs**)

### 1-project configs (N = 12)

```
|Name||||||
TC-10SF-P     1 project | t=0    | Single | Final only        | Perfect η
TC-10SF-R     1 project | t=0    | Single | Final only        | Random η
TC-10MAF-P    1 project | t=0    | Multi  | Advance+Final     | Perfect η
TC-10MAF-R    1 project | t=0    | Multi  | Advance+Final     | Random η
TC-10MAIF-P   1 project | t=0    | Multi  | Adv+Interim+Final | Perfect η
TC-10MAIF-R   1 project | t=0    | Multi  | Adv+Interim+Final | Random η
TC-1LSF-P     1 project | Late   | Single | Final only        | Perfect η
TC-1LSF-R     1 project | Late   | Single | Final only        | Random η
TC-1LMAF-P    1 project | Late   | Multi  | Advance+Final     | Perfect η
TC-1LMAF-R    1 project | Late   | Multi  | Advance+Final     | Random η
TC-1LMAIF-P   1 project | Late   | Multi  | Adv+Interim+Final | Perfect η
TC-1LMAIF-R   1 project | Late   | Multi  | Adv+Interim+Final | Random η
```

### 2-project configs (N = 18)

```
TC-20SF-P     2 projects | all t=0  | Single | Final only        | Perfect η
TC-20SF-R     2 projects | all t=0  | Single | Final only        | Random η
TC-20MAF-P    2 projects | all t=0  | Multi  | Advance+Final     | Perfect η
TC-20MAF-R    2 projects | all t=0  | Multi  | Advance+Final     | Random η
TC-20MAIF-P   2 projects | all t=0  | Multi  | Adv+Interim+Final | Perfect η
TC-20MAIF-R   2 projects | all t=0  | Multi  | Adv+Interim+Final | Random η
TC-2LSF-P     2 projects | all Late | Single | Final only        | Perfect η
TC-2LSF-R     2 projects | all Late | Single | Final only        | Random η
TC-2LMAF-P    2 projects | all Late | Multi  | Advance+Final     | Perfect η
TC-2LMAF-R    2 projects | all Late | Multi  | Advance+Final     | Random η
TC-2LMAIF-P   2 projects | all Late | Multi  | Adv+Interim+Final | Perfect η
TC-2LMAIF-R   2 projects | all Late | Multi  | Adv+Interim+Final | Random η
TC-2XSF-P     2 projects | miXed   | Single | Final only        | Perfect η
TC-2XSF-R     2 projects | miXed   | Single | Final only        | Random η
TC-2XMAF-P    2 projects | miXed   | Multi  | Advance+Final     | Perfect η
TC-2XMAF-R    2 projects | miXed   | Multi  | Advance+Final     | Random η
TC-2XMAIF-P   2 projects | miXed   | Multi  | Adv+Interim+Final | Perfect η
TC-2XMAIF-R   2 projects | miXed   | Multi  | Adv+Interim+Final | Random η
```

### 3-project configs (N = 18)

```
TC-30SF-P     3 projects | all t=0  | Single | Final only        | Perfect η
TC-30SF-R     3 projects | all t=0  | Single | Final only        | Random η
TC-30MAF-P    3 projects | all t=0  | Multi  | Advance+Final     | Perfect η
TC-30MAF-R    3 projects | all t=0  | Multi  | Advance+Final     | Random η
TC-30MAIF-P   3 projects | all t=0  | Multi  | Adv+Interim+Final | Perfect η
TC-30MAIF-R   3 projects | all t=0  | Multi  | Adv+Interim+Final | Random η
TC-3LSF-P     3 projects | all Late | Single | Final only        | Perfect η
TC-3LSF-R     3 projects | all Late | Single | Final only        | Random η
TC-3LMAF-P    3 projects | all Late | Multi  | Advance+Final     | Perfect η
TC-3LMAF-R    3 projects | all Late | Multi  | Advance+Final     | Random η
TC-3LMAIF-P   3 projects | all Late | Multi  | Adv+Interim+Final | Perfect η
TC-3LMAIF-R   3 projects | all Late | Multi  | Adv+Interim+Final | Random η
TC-3XSF-P     3 projects | miXed   | Single | Final only        | Perfect η
TC-3XSF-R     3 projects | miXed   | Single | Final only        | Random η
TC-3XMAF-P    3 projects | miXed   | Multi  | Advance+Final     | Perfect η
TC-3XMAF-R    3 projects | miXed   | Multi  | Advance+Final     | Random η
TC-3XMAIF-P   3 projects | miXed   | Multi  | Adv+Interim+Final | Perfect η
TC-3XMAIF-R   3 projects | miXed   | Multi  | Adv+Interim+Final | Random η
```



---

## Notes on degenerate cases

**Single period + mixed start** (`2XSF`, `3XSF`): one project starts at t=0 and completes in one period, the other starts later. The portfolio horizon spans at least 2 periods. These are valid but the "mixed" label means the late-starting project has a `planned_start > 0`. Keep them — they test that pending-project handling works correctly.

**Single period + all late** (`2LSF`, `3LSF`): all projects start at the same late period and each completes in one period. Valid — tests that the portfolio idles correctly during the pre-start periods.

**Final only + single period**: no advance, no interim — the entire payment is the single final milestone. This is the simplest possible case and is the correct starting point for TC-10SF.

---

## File naming

Each config lives in its own file:

```
src/env/configs/
  tc_1_0sf_p.py      TC-10SF-P
  tc_1_0sf_r.py      TC-10SF-R
  tc_1_0maf_p.py     TC-10MAF-P
  ...
  tc_3_xmaif_r.py    TC-3XMAIF-R
```

File name pattern: `tc_{N}_{S}{D}{P}_{E}.py` all lowercase with underscores. The `CONFIG_ID` string inside each file is the canonical `TC-...` form.