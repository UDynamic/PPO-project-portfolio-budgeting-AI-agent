# Period t — Step Structure

---

## ① Init
> Runs before the agent acts. Establishes the state the agent observes.

| # | Process | Detail |
|---|---------|--------|
| 1a | ... | ... |
| 1b | ... | ... |
| 1c | ... | ... |

---

## ② Act
> Agent or human submits allocation vector x(t).

```
x(t) = [x_0, x_1, ..., x_n]    subject to: sum(x_i) <= B_t
```

---

## ③ On Allocation

| # | Process | Detail |
|---|---------|--------|
| 3a | ... | ... |
| 3b | ... | ... |

> **Termination Check A** — evaluated after 3b

```
if breach:
    ├── compute settlement
    ├── skip remaining sub-processes for this project
    └── go to ④ Lock
```

| # | Process | Detail |
|---|---------|--------|
| 3c | ... | ... |
| 3d | ... | ... |

> **Termination Check B** — evaluated after 3d

```
if breach:
    ├── compute settlement
    ├── skip remaining sub-processes for this project
    └── go to ④ Lock
```

| # | Process | Detail |
|---|---------|--------|
| 3e | ... | ... |

---

## ④ Lock
> Episode ledger updated. State closed for period t.

| # | Process | Detail |
|---|---------|--------|
| 4a | Budget update | B_{t+1} = B_t + total_inflow - total_outflow |
| 4b | Reward | r_t = discount^t x (total_inflow - total_outflow) |
| 4c | DB write | Side-effect only. try/except. conn.commit(). |
| 4d | Advance t | t = t + 1 |