# RL-Based Portfolio Budgeting: Paper Strategy & Architecture

---

## 1. Core Problem & Motivation

### 1.1 The Challenge
Portfolio budgeting under uncertainty faces three critical barriers:

- **Computational Intractability:** Classical OR (MIP, Stochastic Programming) requires
  minutes to hours per instance, making real-time decisions and large-scale scenario
  analysis infeasible.
- **Scalability Failure:** MIP solve time grows exponentially with portfolio size;
  re-optimization after budget changes requires full re-solve.
- **Data Limitations:** Real-world project data is confidential; regional economic
  distortions (inflation, sanctions) invalidate direct use for international publication.

### 1.2 The Opportunity
A trained RL agent, once deployed, produces allocation plans in milliseconds — enabling
use cases that classical OR simply cannot support at scale.

---

## 2. Core Contributions (Ordered by Priority)

### Contribution 1 — Computational Speedup (Primary, Must-Have)
> "Our RL agent achieves a 10,000× speedup over classical MIP solvers (100ms vs.
> 15–45 min/instance), enabling real-time decision support and large-scale scenario
> analysis previously infeasible with traditional OR methods."

**Why this matters:**
- Portfolio manager evaluating 1,000 scenarios → MIP: ~250 hours, RL: ~2 minutes
- Urgent budget reallocation → MIP: wait 30 min, RL: instant response
- Monte Carlo simulation over uncertainty → MIP: infeasible, RL: trivial

**How you prove it:**
- Table: solve time vs. portfolio size (N = 10, 20, 50, 100, 200 projects)
- Show MIP time grows exponentially; RL inference stays flat (~100ms)
- Highlight practical scenario: "1,000 budget scenarios evaluated in under 2 minutes"

---

### Contribution 2 — Solution Quality Under Uncertainty (Primary, Must-Have)
> "The RL agent outperforms deterministic MIP by ~18% in expected portfolio return
> and matches stochastic programming performance — without requiring explicit scenario
> enumeration — achieving ~92% of the perfect-information upper bound."

**Why this matters:**
- Deterministic MIP ignores uncertainty → poor real-world performance
- Stochastic programming requires manual scenario definition → expert bottleneck
- RL learns robust policies directly from stochastic simulation

**How you prove it:**
- Comparison table: RL vs. Greedy vs. MIP vs. Stochastic Programming
- Metric: expected return, variance, budget utilization rate
- Perfect-information MIP as upper bound → show RL gap is small (~8%)
- Sensitivity analysis: vary uncertainty level (low/medium/high) and show RL stays robust

---

### Contribution 3 — Interpretability Analysis (Secondary, Add If Time Permits)
> "The RL policy provides interpretable allocation signals through attention-based
> analysis, revealing which project features drive budget decisions."

**Why this matters for OR journals:**
- Practitioners don't trust black boxes
- Reviewers at EJOR/Omega want auditability
- Adds a "managerial insights" section with real substance

**Implementation options (pick one):**

| Option | Effort | Output |
|--------|--------|--------|
| Attention mechanism in policy network | 1 week | Heatmap of project feature focus |
| SHAP value analysis post-training | 3 days | Feature importance ranking |
| Comparative RL vs. MIP decision analysis | 1 week | "RL agrees with MIP on 85% of top projects" |
| Policy distillation to decision tree | 2 weeks | Human-readable allocation rules |

**Recommendation:** Go with SHAP or comparative analysis — lowest effort, publishable output.

---

## 3. What You Build

### 3.1 Synthetic Environment (Your Data Foundation)
- Generate 10,000 portfolio instances
- Parameters calibrated from international OR/PPM literature
- Each instance: N projects, budget constraint B, stochastic returns/costs/risks
- Validate distributions against literature statistics
- Publish full parameter spec as JSON (reproducibility)

**Why synthetic data is acceptable:**
- Real data is confidential and regionally distorted
- Literature-calibrated synthetic data is standard practice in OR computation papers
- You explicitly acknowledge and justify this in Section 4.4.1

### 3.2 RL Agent
- Algorithm: PPO (stable, well-understood, easy to tune)
- State: current budget remaining, project feature vectors, portfolio state
- Action: budget allocation vector across N projects
- Reward: expected portfolio return under uncertainty (penalize constraint violations)
- Architecture: MLP or attention-based policy network

### 3.3 Baselines (All Required)

| Baseline | Implementation | Role in Paper |
|----------|---------------|---------------|
| Greedy Heuristic | 1 day | Shows you beat naive approaches |
| Deterministic MIP | 2–3 days (Gurobi API) | Classical OR benchmark |
| Stochastic Programming | 3–4 days (scenario-based MIP) | Advanced OR benchmark |
| Perfect-Information MIP | 1 day (variant of MIP) | Theoretical upper bound |

**Note:** You don't implement OR algorithms from scratch. You use Gurobi's Python API
as a solver. That's standard and fully acceptable.
```python
from gurobipy import Model, GRB

model = Model("portfolio_mip")
# define variables, constraints, objective
model.optimize()
# record: solution quality + solve time
```

---

## 4. Paper Architecture

### 4.1 Introduction
- Hook: "A portfolio manager needs to evaluate 500 budget scenarios. Classical MIP
  would take 5 days. Our RL agent does it in 50 seconds."
- Problem statement: computational intractability + uncertainty
- Why RL: speed, implicit uncertainty handling, no re-solve needed
- Contributions: 2–3 bullet points (from Section 2 above)
- Paper roadmap

### 4.2 Literature Review
- Project Portfolio Management (PPM) and classical OR methods
- Stochastic optimization: scenario trees, robust optimization
- RL in combinatorial optimization and OR (recent 2020–2025 papers)
- Gap: no work addresses real-time portfolio budgeting with RL at scale
- Why PSPLIB and standard benchmarks are unsuitable for this problem

### 4.3 Problem Formulation
- Define the MDP formally:
  - State $s_t$: budget remaining, project features, selection state
  - Action $a_t$: budget allocation decision
  - Reward $r_t$: realized portfolio value under uncertainty
  - Transition: stochastic cost/return realization
- Stochastic elements: project cost $\tilde{c}_i \sim \mathcal{N}(\mu_c, \sigma_c^2)$,
  return $\tilde{r}_i \sim \mathcal{N}(\mu_r, \sigma_r^2)$, risk events
- Justify all distributional choices with literature citations

### 4.4 Methodology
- **4.4.1 Synthetic Environment:** instance generation, calibration, validation stats
- **4.4.2 RL Agent:** network architecture, PPO hyperparameters, training details
- **4.4.3 Baselines:** MIP formulation, stochastic programming setup, greedy rule
- **4.4.4 Perfect-Information Upper Bound:** how it's computed, role in analysis

### 4.5 Computational Experiments
- **4.5.1 Setup:** instance sizes, hardware, metrics (solve time, expected return,
  budget utilization, variance)
- **4.5.2 Main Comparison:** RL vs. all baselines — speed and quality
- **4.5.3 Scalability Analysis:** performance as N (portfolio size) grows
- **4.5.4 Robustness Analysis:** vary uncertainty level, budget tightness, N
- **4.5.5 Interpretability:** (if included) attention maps or SHAP analysis

### 4.6 Managerial Insights
- "What does 10,000× speedup mean for a real portfolio manager?"
- Scenario analysis use case: evaluate hundreds of budget plans in seconds
- Dynamic reoptimization: respond to mid-year budget changes instantly
- Democratization: no OR expert needed for deployment

### 4.7 Conclusion
- Restate contributions clearly
- Acknowledge synthetic data limitation honestly
- Future work: real data validation, multi-period portfolios, hybrid RL+OR

---

## 5. Key Metrics to Report

| Metric | What It Shows |
|--------|--------------|
| Solve time (ms / min) | Contribution 1: speed |
| Expected portfolio return | Contribution 2: solution quality |
| Gap to upper bound (%) | How close to optimal |
| Budget utilization rate | Constraint satisfaction |
| Performance under uncertainty levels | Robustness |
| Variance of returns | Risk management |

---

## 6. Reproducibility Checklist

- [ ] GitHub repo: environment, RL agent, all baselines
- [ ] JSON parameter spec: all calibrated distributions with literature sources
- [ ] Raw results: all tables/figures reproducible from published data
- [ ] Limitations section: synthetic data acknowledged, calibration justified

---

## 7. Journal Target

| Journal | Fit | Notes |
|---------|-----|-------|
| EJOR | Primary | Strong on computation + practical impact, publishes RL |
| Computers & OR | Secondary | Highly receptive to RL/computational methods |
| Omega | Alternative | Good if managerial insights section is strong |
| IEEE TEM | Alternative | Good if interpretability contribution is included |

---

## 8. Implementation Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1. Foundation | Synthetic environment + greedy baseline + basic RL | 3 weeks |
| 2. Baselines | MIP + stochastic programming + upper bound | 4 weeks |
| 3. RL Optimization | Hyperparameter tuning + architecture refinement | 4 weeks |
| 4. Analysis | All comparisons + sensitivity + visualizations | 3 weeks |
| 5. Interpretability | SHAP or attention analysis (optional) | 1–2 weeks |
| 6. Writing | Full paper draft + revision | 3 weeks |

**Total: ~18 weeks** for a complete, submittable paper.

---

## 9. The Framing That Works for OR Reviewers

Do NOT say: "RL is better than OR."

DO say: "RL enables use cases that OR cannot support due to computational constraints.
For problems requiring real-time response or large-scale scenario analysis, our approach
is the only practical option. For offline, single-instance optimization, classical OR
remains the gold standard."

This is a complementary framing — OR reviewers will appreciate it, and it's honest.
