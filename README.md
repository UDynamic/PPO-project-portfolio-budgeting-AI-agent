# Contractor Project Portfolio Budgeting under Performance-Driven Cash Flow Uncertainty

### A Constrained MDP Framework with Reinforcement Learning

> **Primary output:** English journal article (target: IJPM / EJOR / Automation in Construction)
> **Secondary output:** Persian Master's thesis (Industrial Engineering, derived from article)
> **Stack:** Python · Gymnasium · LaTeX · SQLite

---

## What this is

A reinforcement learning framework for **dynamic budget allocation** across a contractor's active project portfolio. The portfolio manager — a contractor receiving milestone-based payments under FIDIC contract mechanics — must decide how to distribute the available cash balance across projects at each discrete time period, under uncertainty about project execution outcomes.

The core contribution is a **Constrained MDP (CMDP) environment** (Gymnasium-compatible) that embeds:

- FIDIC Sub-Clause 14.2 advance payment recovery mechanics
- Endogenous project termination via a cure period counter (τ_{i,t}^{rem})
- Performance-driven cash flow uncertainty (stochastic productivity η induces uncertain milestone timing and therefore uncertain payment arrival — no separate payment delay model needed)
- EVM-based state signals (BCWS, BCWP, ACWP, SPI, CPI, EAC)

A **PPO agent** is trained on this environment and benchmarked against classical OR baselines and a deterministic MILP upper bound.

---

## Three-level relaxation hierarchy

| Level  | Model                             | Purpose                                         |
| ------ | --------------------------------- | ----------------------------------------------- |
| **L1** | Deterministic full-foresight MILP | Performance upper bound (oracle)                |
| **L2** | Multi-stage stochastic program    | Establishes intractability of exact solution    |
| **L3** | Constrained MDP + PPO agent       | Primary contribution — tractable learned policy |

---

## Key design decisions

**Scope:** Portfolio selection is exogenous. The agent governs committed, contracted projects only — it allocates budget, it does not choose which projects to take on.

**Perspective:** The contractor (not the client). The agent receives milestone payments and must manage cash flow to sustain execution.

**Uncertainty source:** Stochastic project productivity η. Uncertain execution pace → uncertain milestone timing → uncertain payment arrival. η is the single source of uncertainty in the system. Its realized value is recorded at every step, so all post-hoc solvers operate on a fully deterministic record.

**Termination:** Endogenous. Projects are terminated by the environment when the cure period counter hits zero, not by the agent directly.

**Budget regime:** Parameterized by κ = B₀ / ΣBAC_i. Three regimes: abundant (κ ≫ 1), tight (κ ≈ 1), scarce (κ ≪ 1). Experiments cover all three.

---

## Pipeline

The system runs in three sequential phases.

**Phase 1 — Training**
The PPO agent interacts with the generative CMDP environment. Every episode — states, actions, rewards, and the full realized η sequence — is recorded to the database.

**Phase 2 — Post-hoc analysis**
Run on completed episodes. All solvers read from the database and write their results back to the same episode records.

- **MILP upper bound:** Solves the deterministic full-foresight problem over each episode's realized η sequence. This is a retrospective calculation, not a separate simulation.
- **Naive baselines (×3):** Each baseline replays the same realized η sequence under its fixed allocation rule.

**Phase 3 — Reporting**
Reads from the database only. Computes normalized improvement, regime breakdowns, and generates all article figures and tables.

```
Phase 1: env + agent → DB (episodes, steps, realized η)
Phase 2: DB → MILP solver → DB (upper bounds)
          DB → baselines   → DB (baseline returns)
Phase 3: DB → metrics + plots → article/figures/, article/tables/
```

Entry points: `src/pipeline/train.py` · `src/pipeline/postprocess.py` · `src/pipeline/report.py`

---

## Filing system

```
PPO-project-portfolio-budgeting-AI-agent/
│
├── src/
│   ├── notation.yaml
│   │
│   ├── database/
│   │   ├── schema.sql
│   │   ├── db.py
│   │   ├── models.py
│   │   └── queries/
│   │       ├── episodes.py
│   │       ├── steps.py
│   │       └── results.py
│   │
│   ├── schema/
│   │   ├── portfolio.py
│   │   ├── project.py
│   │   └── episode.py
│   │
│   ├── environment/
│   │   ├── env.py
│   │   ├── mechanics/
│   │   │   ├── fidic.py
│   │   │   ├── evm.py
│   │   │   ├── cure_period.py
│   │   │   └── productivity.py
│   │   ├── recorder.py
│   │   └── generator.py
│   │
│   ├── agent/
│   │   ├── ppo.py
│   │   ├── network.py
│   │   ├── trainer.py
│   │   └── evaluate.py
│   │
│   ├── solvers/
│   │   ├── base.py
│   │   ├── milp/
│   │   │   ├── solver.py
│   │   │   ├── formulation.py
│   │   │   └── warmstart.py
│   │   └── baselines/
│   │       ├── equal_split.py
│   │       ├── proportional_spi.py
│   │       └── greedy_cpi.py
│   │
│   ├── pipeline/
│   │   ├── train.py
│   │   ├── postprocess.py
│   │   └── report.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── regime.py
│   │
│   └── plots/
│       ├── portfolio_plot.py
│       ├── convergence.py
│       └── benchmark.py
│
├── tests/
│   ├── cases/
│   │   ├── case_data.py
│   │   └── loader.py
│   │
│   ├── environment/
│   │   ├── test_fidic.py
│   │   ├── test_evm.py
│   │   ├── test_cure_period.py
│   │   └── test_productivity.py
│   │
│   ├── solvers/
│   │   ├── test_milp.py
│   │   ├── test_equal_split.py
│   │   ├── test_proportional_spi.py
│   │   └── test_greedy_cpi.py
│   │
│   └── integration/
│       ├── test_recorder.py
│       ├── test_postprocess.py
│       └── test_pipeline.py
│
├── experiments/
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── abundant.yaml
│   │   ├── tight.yaml
│   │   └── scarce.yaml
│   ├── results/
│   │   ├── id_evaluation/
│   │   ├── ood_evaluation/
│   │   ├── regime_analysis/
│   │   └── sensitivity/
│   └── runs/
│       ├── checkpoints/
│       └── logs/
│
├── data/
│   ├── raw/
│   ├── calibrated/
│   └── synthetic/
│
├── article/
│   ├── main.tex
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_literature.tex
│   │   ├── 03_problem_formulation.tex
│   │   ├── 04_solution_method.tex
│   │   ├── 05_experiments.tex
│   │   ├── 06_conclusion.tex
│   │   ├── appendix_A.tex
│   │   ├── appendix_B.tex
│   │   └── appendix_C.tex
│   ├── figures/
│   │   ├── tikz/
│   │   └── plots/
│   ├── tables/
│   ├── submission/
│   └── build/
│
├── thesis/
│   ├── main.tex
│   ├── chapters/
│   └── build/
│
├── docs/
│   ├── notation.md
│   ├── architecture.md
│   ├── pipeline.md
│   ├── testing.md
│   ├── lit_review/
│   └── deprecated/
│
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

## Notation map

The file `src/notation.yaml` is the **single source of truth** for all parameter names. It maps LaTeX symbols to Python variable names to database column names. Any new parameter must be registered here first.

Example entry:

```yaml
cost_overrun_cap:
  latex: "\\mu_i"
  python: "cost_overrun_cap"
  db_column: "mu"
  description: "Maximum allowable EAC as a fraction of BAC before termination"
  units: "fraction"
  scope: "project"
```

The human-readable version is auto-generated at `docs/notation.md`.

---

## Test architecture

Testing is separated into three layers with distinct purposes.

**Layer 1 — Environment mechanics** (`tests/environment/`)
Verifies that the CMDP environment correctly implements the problem mechanics. Given a fixed portfolio, a fixed η sequence, and a fixed action sequence, the environment must produce exact expected state transitions, cash flows, termination events, and EVM signals. Fully deterministic. No agent involved.

**Layer 2 — Solver correctness** (`tests/solvers/`)
Verifies that each solver — MILP and all three baselines — correctly implements its algorithm. Given a hand-built case with a known correct answer, each solver must return the pre-verified result exactly. These are regression tests against `tests/cases/case_data.py`, which is the single source of truth for all hand-built cases.

**Layer 3 — Integration** (`tests/integration/`)
Verifies that the pipeline phases connect correctly: episodes are recorded completely to the database, the post-hoc pipeline reads and writes correctly, and a full train → postprocess → report smoke test completes without error.

The hand-built cases in `tests/cases/case_data.py` are shared across Layers 1 and 2. `tests/cases/loader.py` provides format conversion for each consumer.

---

## Test case taxonomy

Hand-built cases in `tests/cases/case_data.py` follow a four-category structure:

| Category        | Purpose                                  | Example cases                                   |
| --------------- | ---------------------------------------- | ----------------------------------------------- |
| **Sanity**      | Verify fundamental mechanics work at all | SP-1 (single project, abundant budget)          |
| **Stress**      | Push one parameter to its extreme        | T-X-01 (max cost overrun), S-X-02 (min budget)  |
| **Interaction** | Two failure modes simultaneously         | MP-K (payment + performance pressure together)  |
| **Boundary**    | Sit exactly at a threshold               | T-B-01 (EAC exactly at 1.1·BAC cap)             |

---

## Evaluation framework

**In-distribution (ID) test:** 500 fresh portfolios sampled from calibrated contractor distributions. Metric: normalized improvement = (agent score − random baseline) / (MILP − random baseline). Target: ≥ 65%.

**Out-of-distribution (OOD) test:** The hand-built case library. Metric: qualitative case analysis — does the agent behave correctly for each named failure mode?

**Budget regime analysis:** Performance reported separately for κ ≫ 1, κ ≈ 1, κ ≪ 1.

**Convergence criterion:** Mean normalized improvement ≥ 65% with variance ≤ 5% over the last 100 evaluation episodes.

---

## Training strategy

A **generative environment** with a mixture training distribution:

- **75%** of episodes: portfolios sampled from calibrated contractor distributions (realistic)
- **25%** of episodes: portfolios sampled uniformly from the full parameter space (coverage)

This prevents policy collapse on underrepresented configurations while maintaining fidelity to real-world conditions.

---

## Setup

```bash
git clone <repo>
cd PPO-project-portfolio-budgeting-AI-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Verify environment mechanics
pytest tests/environment/

# Verify solver correctness
pytest tests/solvers/

# Run integration smoke test
pytest tests/integration/

# Generate case figures
python src/plots/portfolio_plot.py tests/cases/case_data.py

# Run full pipeline
python src/pipeline/train.py --config experiments/configs/base.yaml
python src/pipeline/postprocess.py
python src/pipeline/report.py
```

---

## Article structure

| Section             | File                                          | Status              |
| ------------------- | --------------------------------------------- | ------------------- |
| Introduction        | `article/sections/01_introduction.tex`        | Draft               |
| Literature Review   | `article/sections/02_literature.tex`          | Draft (PRISMA 2020) |
| Problem Formulation | `article/sections/03_problem_formulation.tex` | Draft               |
| Solution Method     | `article/sections/04_solution_method.tex`     | Pending             |
| Experiments         | `article/sections/05_experiments.tex`         | Pending             |
| Conclusion          | `article/sections/06_conclusion.tex`          | Pending             |
| Appendix A          | `article/sections/appendix_A.tex`             | Draft               |
| Appendix B          | `article/sections/appendix_B.tex`             | Draft               |
| Appendix C          | `article/sections/appendix_C.tex`             | Draft               |

Build:

```bash
cd article
latexmk -pdf main.tex
```

---

## Target venues

IJPM · EJOR · Automation in Construction

Positioning: OR/applied engineering venues. The primary contribution is the CMDP environment formulation and the three-level relaxation hierarchy, not the RL algorithm itself.

---

## Acknowledgements

*Contractor project data provided under anonymization agreement. Acknowledgement language to be confirmed following data verification sign-off.*