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

**Uncertainty source:** Stochastic project productivity η. Uncertain execution pace → uncertain milestone timing → uncertain payment arrival. This is sufficient to justify the "cash flow uncertainty" framing without requiring an explicit payment delay model.

**Termination:** Endogenous. Projects are terminated by the environment when the cure period counter hits zero, not by the agent directly. This is a state-space mechanic, not an action.

**Budget regime:** Parameterized by κ = B₀ / ΣBAC_i. Three regimes: abundant (κ ≫ 1), tight (κ ≈ 1), scarce (κ ≪ 1). Experiments cover all three.

---



## Filing system

```
├── article/              English journal article (LaTeX)
│   ├── sections/         Section .tex files
│   ├── figures/
│   │   ├── tikz/         TikZ figure source
│   │   └── plots/        Python-generated figures
│   ├── tables/
│   ├── submission/       Journal submission packages
│   └── build/            LaTeX build artifacts (gitignored)
│
├── thesis/               Persian Master's thesis (derived from article)
│   ├── chapters/
│   └── build/
│
├── src/                  All Python source code
│   ├── notation.yaml     Symbol ↔ Python ↔ DB mapping (single source of truth)
│   ├── schema/           Portfolio data schema (Pydantic + JSON Schema)
│   ├── environment/      Gymnasium CMDP environment
│   ├── milp/             Deterministic MILP solver (Level 1)
│   ├── baselines/        Naive allocation policies (equal split, proportional, greedy SPI)
│   ├── agent/            PPO actor-critic
│   ├── database/         SQLite integration layer
│   ├── evaluation/       Benchmarking and metrics
│   ├── plots/            Plotting scripts (portfolio_plot.py etc.)
│   └── utils/
│
├── tests/                Test case library
│   ├── cases/            Hand-built structured test cases (cases.json)
│   ├── milp/             MILP solver tests + figures
│   ├── environment/      Environment correctness tests
│   └── sanity/           Sanity check cases (isolation, stress, boundary)
│
├── data/                 Data (raw contractor data gitignored)
│   ├── raw/              Anonymized contractor project data
│   ├── calibrated/       Fitted parameter distributions
│   └── synthetic/        Generated episodes
│
├── experiments/          Experiment configs, results, model checkpoints
│   ├── configs/
│   ├── results/
│   │   ├── baseline_comparison/
│   │   ├── sensitivity/
│   │   ├── scalability/
│   │   └── ood_evaluation/
│   └── runs/
│       ├── pretrained/
│       └── finetuned/
│
└── docs/                 Documentation
    ├── notation.md       Human-readable notation map
    ├── architecture.md   Design decisions
    ├── lit_review/       Literature review pipeline + archive
    └── deprecated/       Old versions kept for reference
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



## Test case taxonomy

Hand-built cases in `tests/cases/cases.json` follow a four-category structure:


| Category        | Purpose                                  | Example cases                                  |
| --------------- | ---------------------------------------- | ---------------------------------------------- |
| **Sanity**      | Verify fundamental mechanics work at all | SP-1 (single project, abundant budget)         |
| **Stress**      | Push one parameter to its extreme        | T-X-01 (max cost overrun), S-X-02 (min budget) |
| **Interaction** | Two failure modes simultaneously         | MP-K (payment + performance pressure together) |
| **Boundary**    | Sit exactly at a threshold               | T-B-01 (EAC exactly at 1.1·BAC cap)            |


MILP upper bounds are pre-computed for all cases and stored in the database.

---



## Evaluation framework

**In-distribution (ID) test:** 500 fresh portfolios sampled from calibrated contractor distributions. Metric: normalized improvement = (agent score − random baseline) / (MILP − random baseline). Target: ≥ 65%.

**Out-of-distribution (OOD) test:** The hand-built case library above. Metric: qualitative case analysis — does the agent behave correctly for each named failure mode?

**Budget regime analysis:** Performance reported separately for κ ≫ 1, κ ≈ 1, κ ≪ 1.

**Convergence criterion:** Mean normalized improvement ≥ 65% with variance ≤ 5% over the last 100 evaluation episodes.

---



## Training strategy

A **generative environment** with a mixture training distribution:

- **75%** of episodes: portfolios sampled from calibrated contractor distributions (realistic)
- **25%** of episodes: portfolios sampled uniformly from the full parameter space (coverage)

This prevents policy collapse on underrepresented configurations (e.g., single-project portfolios, extreme budget stress) while maintaining fidelity to real-world conditions.

---



## Setup

```bash
# Clone and create virtual environment
git clone <repo>
cd PPO-project-portfolio-budgeting-AI-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run MILP tests
pytest tests/milp/

# Run sanity checks
pytest tests/sanity/

# Generate MILP case figures
python src/plots/portfolio_plot.py tests/cases/cases.json

# Train PPO agent
python src/agent/ppo_trainer.py --config experiments/configs/base.yaml
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


Build the article:

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