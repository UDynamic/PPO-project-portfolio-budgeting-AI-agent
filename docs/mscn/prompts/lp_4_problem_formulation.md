Here is your carry-forward prompt. Copy it exactly into the new session.

---

**PROMPT FOR NEW SESSION:**

---

You are assisting a Master's student in Industrial Engineering with his thesis. Before we begin the actual work, read this entire brief carefully so you are fully context-loaded. When you are ready, ask me to paste the draft LaTeX file.

---

**WHO I AM AND WHAT I AM BUILDING**

I am a Master's student in Industrial Engineering with a strong ML/AI and software development background. My thesis introduces a reinforcement learning framework for project portfolio management (PPM) budgeting optimization under cash flow uncertainty. It is positioned as a management science contribution targeting journals such as IJPM, EJOR, or Automation in Construction.

The thesis has four sections. Section 1 is the introduction. Section 2 is the literature review following PRISMA 2020. Section 3 is the problem formulation — this is what we are working on today. Section 4 is the experiments and agent evaluation.

---

**THE CORE RESEARCH FRAMING**

The central argument of the thesis is this: portfolio selection has been studied extensively in the literature. This work assumes that problem is already solved upstream. A portfolio of committed projects exists, contracts have been signed, and the decision-maker is a high-level board or portfolio manager whose only recurring decision is how to allocate budget across active projects at each discrete time step.

That single budget allocation decision is the lever through which all strategic and operational outcomes flow — contractor performance recovery, milestone achievement, cash flow balance, and ultimately project success or termination.

The research hypothesis is: there is no established framework that models this portfolio-level budget allocation decision as a sequential stochastic optimization problem. This work fills that gap.

---

**THE ENVIRONMENT — MECHANISMS ALREADY DESIGNED**

The core contribution is a configurable synthetic Gymnasium-compatible environment modeled as a Constrained MDP (CMDP). The RL agent trained on this environment is a demonstration of its utility, not the primary contribution. The environment itself is the primary novel contribution.

The environment encodes the following mechanisms, all of which must appear as constraints or variables in the LP formulation:

- Discrete time steps representing budget allocation periods (e.g. monthly)
- Multiple concurrent active projects, each with its own state
- Budget feasibility constraint: total allocation across projects cannot exceed available portfolio budget at each time step
- Cash inflow from owner to contractor triggered by milestone achievement
- Cash outflow representing contractor expenditure and cost of work performed
- Advance payment mechanism: an upfront payment made at contract start, recovered progressively from subsequent milestone payments as a fixed recovery fraction
- Retention mechanism: a percentage withheld from each milestone payment, released upon project completion or practical completion
- Contractor performance modeled via CPI (cost performance index) and SPI (schedule performance index), both stochastic in the real environment
- S-curve budget consumption profiles: budget consumption follows a sigmoid trajectory over project life
- Internal management intervention action: the decision-maker can trigger a performance recovery intervention on an underperforming project at a cost, which stochastically improves CPI/SPI
- Project termination action: the decision-maker can declare a project failed and terminate it, triggering a termination cost and removing the project from the active portfolio
- All contract mechanics are calibrated to FIDIC contract forms (Red, Yellow, Silver Books)
- Environment parameters are calibrated using anonymized data from two major EPC contractors, supplemented by World Bank ICR data and Flyvbjerg datasets

---

**THE THREE-LEVEL MODELING ARC — THIS IS THE STRUCTURE OF SECTION 3**

Section 3 must be written as a three-part movement. This is the agreed strategy:

Level 1 — Deterministic Full-Foresight LP: Assume all contractor performance parameters are known with certainty. CPI trajectories, payment timing, cash flow profiles are all fixed. Write the complete formal LP (technically a MILP because termination is a binary decision variable). This is the problem formulation we are refactoring today. This LP serves as the theoretical upper bound on performance — no real policy can outperform it because no real decision-maker has perfect foresight.

Level 2 — Stochastic Extension and Intractability Argument: Replace deterministic parameters with stochastic processes. Show that the resulting stochastic program faces scenario-tree explosion as the number of projects and time steps grows. This motivates the need for a learning-based approach.

Level 3 — CMDP Reformulation: Introduce the Constrained MDP as the tractable reformulation. Show the explicit mapping between LP elements and MDP elements: decision variables map to actions, constraint parameters map to state variables, the objective maps to the reward function, and feasibility constraints map to the constrained action space. This is where the Gymnasium environment and PPO agent are introduced.

Today we are working only on Level 1 — the deterministic LP. The file is named 03_problem_formulation.tex.

---

**NOTATION AND DOMAIN VOCABULARY — USE THESE CONSISTENTLY**

The following terms and symbols are used consistently across the thesis. Do not deviate from them:

- $\mathcal{P}$ — set of active projects, indexed by $i$
- $\mathcal{T}$ — set of discrete time steps, indexed by $t$
- $B_t$ — total portfolio budget available at time step $t$
- $x_{i,t}$ — budget allocated to project $i$ at time step $t$ (decision variable)
- $\text{CPI}_{i,t}$ — cost performance index of project $i$ at time $t$
- $\text{SPI}_{i,t}$ — schedule performance index of project $i$ at time $t$
- $\text{BAC}_i$ — budget at completion for project $i$
- $M_{i,k}$ — value of milestone $k$ for project $i$
- $\alpha_i$ — advance payment fraction for project $i$
- $\rho_i$ — advance payment recovery rate (fraction of each milestone payment)
- $r_i$ — retention rate for project $i$
- $z_{i,t}$ — binary termination decision variable: 1 if project $i$ is terminated at time $t$
- $\delta_{i,t}$ — binary intervention decision variable: 1 if recovery intervention is triggered on project $i$ at time $t$
- EVM, EPC, FIDIC, CPI, SPI, BAC, S-curve — standard domain terminology, always used as defined above

---

**BASELINE COMPARISON STRATEGY — CONTEXT FOR WHY THE LP MATTERS**

The LP formulation serves multiple roles. It is the rigorous OR-tradition entry point that signals methodological credibility to EJOR and management science reviewers. It is the upper bound benchmark against which the RL agent and other baselines are evaluated in Section 4. It also sets up the intractability argument that motivates the CMDP approach.

The planned baselines for Section 4 are: the deterministic LP optimal (upper bound, perfect foresight), a two-stage stochastic program (classical OR baseline), a rule-based heuristic approximating experienced human decision-making, and the PPO RL agent. The LP we write today anchors the top of that comparison stack.

---

**ACADEMIC POSITIONING**

The thesis is positioned within the Industrial Engineering and Operations Research tradition. The MDP formalism is framed as a direct descendant of Bellman's dynamic programming and stochastic control theory — not as an import from ML. The RL agent is presented as a computationally tractable solver for a problem whose exact solution (the MINLP stochastic program) is intractable at realistic scale. This framing makes the work credible to both OR/IE and ML audiences.

The work also positions the CMDP environment as the analytical engine for a potential portfolio-level decision-support module in incumbent project management information systems such as Oracle Primavera P6 — which currently lacks portfolio-level optimization capability. This practitioner framing appears in the conclusion as a deployment context and future work direction.

---

**WRITING STYLE AND QUALITY STANDARDS**

- LaTeX throughout, compiled with pdflatex and natbib
- File to be delivered as clean .tex, no compile errors
- Notation must be consistent with the symbols defined above
- All mechanisms must appear formally as constraints, not just described in prose
- The LP must be numbered as equations and presented in standard OR format: objective function first, then constraints grouped logically, then variable domain declarations
- Prose around the LP must explain the economic interpretation of each constraint, not just its mathematical form
- No overclaiming: the LP is introduced explicitly as a deterministic idealization and theoretical upper bound, not as a proposed solution

---

**NOW:**

Please confirm you have read and understood this full brief. Then ask me to paste the contents of 03_problem_formulation.tex so we can begin the refactor.

---