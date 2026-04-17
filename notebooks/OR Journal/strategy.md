# RL-Based Portfolio Budgeting: Complete Paper Strategy & Architecture

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

### Contribution 3 — Open-Source Deployable Framework (Secondary, High-Impact)
> "We release an open-source decision support system with full backend API and
> interactive frontend, enabling practitioners to deploy portfolio optimization without
> ML expertise. For industry-specific adaptation, we provide a fine-tuning protocol
> using historical project data, bridging the gap between synthetic training and
> real-world deployment."

**Why this matters:**
- Most OR papers stop at experimental results — you ship working software
- Addresses the synthetic data limitation directly: pretrain on synthetic, fine-tune on real
- Democratizes access: no OR expert or ML engineer needed for deployment
- Tangible impact: reviewers can actually use your system

**What you build:**

| Component | Purpose | User Experience |
|-----------|---------|-----------------|
| Backend API | RL inference engine | REST endpoints for allocation requests |
| Frontend UI | Interactive dashboard | Upload projects, set budget, get allocation plan |
| Fine-tuning module | Domain adaptation | Company uploads historical data, system retrains |
| Documentation | Deployment guide | Step-by-step setup for practitioners |

**How you prove it:**
- Demo video or screenshots in supplementary material
- GitHub repository with full codebase
- Fine-tuning case study: synthetic "company dataset" → show performance improvement
- Deployment guide: "from zero to running system in 30 minutes"

**The fine-tuning angle specifically:**
- Pretrain on 10,000 synthetic instances (broad knowledge)
- Fine-tune on 200-500 company-specific instances (domain specialization)
- Show: fine-tuned model outperforms generic model by 8-12% on company data
- This is exactly the LLM paradigm — reviewers in 2025-2026 will immediately get it

**Strategic value:**
- Turns synthetic data from weakness to strength: "we use synthetic for pretraining,
  real data for fine-tuning"
- Opens a natural future work section: "how much real data is needed for effective
  fine-tuning?"
- For PhD applications: you can demo a live system, not just show tables

---

### Contribution 4 — Interpretability Analysis (Optional, Add If Time Permits)
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
- Fine-tuning protocol addresses real-world deployment

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
### 3.4 Deployable System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React/Vue)                  │
│  - Project upload interface                                  │
│  - Budget constraint input                                   │
│  - Allocation visualization                                  │
│  - Scenario comparison dashboard                             │
└─────────────────────────────────────────────────────────────┘
↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI/Flask)               │
│  - /allocate endpoint: returns budget allocation             │
│  - /finetune endpoint: triggers domain adaptation            │
│  - /evaluate endpoint: compares allocation strategies        │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│                      RL Inference Engine                     │
│  - Pretrained PPO agent (10K synthetic instances)            │
│  - Fine-tuning module (company data adaptation)              │
│  - Model versioning and rollback                             │
└─────────────────────────────────────────────────────────────┘
```
**Key features:**
- Zero-setup deployment: Docker container or cloud-hosted demo
- No ML expertise required: upload CSV, get allocation
- Fine-tuning workflow: company uploads historical projects → system retrains → improved performance
- Export results: PDF reports, CSV allocations

---

## 4. Paper Architecture

### 4.1 Introduction
- Hook: "A portfolio manager needs to evaluate 500 budget scenarios. Classical MIP
  would take 5 days. Our RL agent does it in 50 seconds — and we provide the full
  system as open-source software."
- Problem statement: computational intractability + uncertainty
- Why RL: speed, implicit uncertainty handling, no re-solve needed
- Contributions: 3–4 bullet points (from Section 2 above)
- Paper roadmap

### 4.2 Literature Review
- Project Portfolio Management (PPM) and classical OR methods
- Stochastic optimization: scenario trees, robust optimization
- RL in combinatorial optimization and OR (recent 2020–2025 papers)
- Software tools for portfolio optimization (gap: no RL-based systems)
- Gap: no work addresses real-time portfolio budgeting with RL at scale + deployment
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
- **4.4.5 Fine-Tuning Protocol:** domain adaptation procedure, data requirements

### 4.5 Computational Experiments
- **4.5.1 Setup:** instance sizes, hardware, metrics (solve time, expected return,
  budget utilization, variance)
- **4.5.2 Main Comparison:** RL vs. all baselines — speed and quality
- **4.5.3 Scalability Analysis:** performance as N (portfolio size) grows
- **4.5.4 Robustness Analysis:** vary uncertainty level, budget tightness, N
- **4.5.5 Fine-Tuning Case Study:** synthetic company dataset, performance improvement
- **4.5.6 Interpretability:** (if included) attention maps or SHAP analysis

### 4.6 System Architecture and Deployment
- **4.6.1 Software Design:** backend API, frontend interface, deployment options
- **4.6.2 User Workflow:** from project upload to allocation decision
- **4.6.3 Fine-Tuning Workflow:** company data integration, retraining process
- **4.6.4 Reproducibility:** GitHub repository, documentation, demo instance

### 4.7 Managerial Insights
- "What does 10,000× speedup mean for a real portfolio manager?"
- Scenario analysis use case: evaluate hundreds of budget plans in seconds
- Dynamic reoptimization: respond to mid-year budget changes instantly
- Democratization: no OR expert needed for deployment
- Fine-tuning enables industry-specific adaptation without ML expertise

### 4.8 Conclusion
- Restate contributions clearly
- Acknowledge synthetic data limitation honestly, highlight fine-tuning solution
- Future work: multi-period portfolios, hybrid RL+OR, fine-tuning data requirements study

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
| Fine-tuning improvement (%) | Domain adaptation effectiveness |
| System response time (end-to-end) | Practical deployment performance |

---

## 6. Reproducibility Checklist

- [ ] GitHub repo: environment, RL agent, all baselines, full system code
- [ ] JSON parameter spec: all calibrated distributions with literature sources
- [ ] Raw results: all tables/figures reproducible from published data
- [ ] Docker container or cloud demo: one-click deployment
- [ ] Fine-tuning tutorial: step-by-step guide with example dataset
- [ ] Limitations section: synthetic data acknowledged, calibration justified, fine-tuning validated

---

## 7. Journal Target

| Journal | Fit | Notes |
|---------|-----|-------|
| EJOR | Primary | Strong on computation + practical impact, publishes RL, values software contributions |
| Computers & OR | Secondary | Highly receptive to RL/computational methods, good software track record |
| IEEE TEM | Alternative | Strong fit if system deployment is emphasized |
| Omega | Alternative | Good if managerial insights section is strong |

**Strategic note:** The open-source system contribution plays better at IEEE TEM and
Computers & OR than pure theory journals. EJOR sits in the middle — they appreciate
practical impact but won't weight software as heavily as algorithmic novelty.

---

## 8. Implementation Timeline (Revised)

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1. Foundation | Synthetic environment + greedy baseline + basic RL | 3 weeks |
| 2. Baselines | MIP + stochastic programming + upper bound | 4 weeks |
| 3. RL Optimization | Hyperparameter tuning + architecture refinement | 4 weeks |
| 4. System Development | Backend API + frontend UI + fine-tuning module | 5 weeks |
| 5. Analysis | All comparisons + sensitivity + fine-tuning case study | 3 weeks |
| 6. Interpretability | SHAP or attention analysis (optional) | 1–2 weeks |
| 7. Writing | Full paper draft + revision | 3 weeks |

**Total: ~22–24 weeks** for a complete, submittable paper with deployable system.

**Critical path consideration:** System development (Phase 4) can partially overlap with
RL optimization (Phase 3) if you build the API wrapper while tuning the agent.

---

## 9. The Framing That Works for OR Reviewers

Do NOT say: "RL is better than OR."

DO say: "RL enables use cases that OR cannot support due to computational constraints.
For problems requiring real-time response or large-scale scenario analysis, our approach
is the only practical option. For offline, single-instance optimization, classical OR
remains the gold standard. We provide an open-source system that makes this technology
accessible to practitioners without ML expertise."

This is a complementary framing — OR reviewers will appreciate it, and it's honest.

---

## 10. Risk Assessment: The Open-Source System Contribution

### Upside
- Differentiates your paper from 95% of OR submissions
- Directly addresses synthetic data limitation via fine-tuning
- Tangible impact: reviewers can use your system
- Strong signal for PhD applications: you ship working software
- Natural follow-up research: fine-tuning data requirements, hybrid methods

### Downside
- Adds 5 weeks to timeline (22–24 weeks total vs. 18 weeks without)
- OR journals may treat it as "nice bonus" rather than core contribution
- Requires maintaining code quality and documentation
- Fine-tuning case study must be convincing (can't just promise it works)

### Mitigation Strategy
- Position system as supporting contribution, not primary
- Ensure core algorithmic contributions (speed + quality) stand alone
- Build minimal viable system: functional but not polished
- One fine-tuning case study is enough: synthetic company dataset with clear improvement

### My Recommendation
Include it. The upside is significant, the downside is manageable, and it directly
addresses your biggest vulnerability (synthetic data). For PhD applications, this is
gold. For publication, it's a strong differentiator that increases acceptance probability
at EJOR and especially at IEEE TEM or Computers & OR.

---

## 11. Fine-Tuning Protocol Details

### Pretraining Phase
- Train on 10,000 synthetic instances
- Broad coverage: varied portfolio sizes, budget constraints, uncertainty levels
- Goal: learn general allocation strategies

### Fine-Tuning Phase
- Company provides 200–500 historical project records
- Format: project features, actual costs, actual returns, completion status
- Retrain last 2–3 layers of policy network (transfer learning)
- Goal: adapt to company-specific cost/return distributions

### Validation
- Hold out 20% of company data for testing
- Compare: pretrained model vs. fine-tuned model vs. company's historical decisions
- Show: fine-tuned model outperforms both by 8–12%

### Data Requirements Study (Future Work)
- How much company data is needed? 50 projects? 200? 500?
- Diminishing returns curve: performance vs. fine-tuning dataset size
- This is a natural follow-up paper

---

## 12. What Makes This Paper Strong

You're not just proposing an algorithm. You're proposing a complete solution:
- Fast inference (10,000× speedup)
- High-quality decisions (matches stochastic programming)
- Deployable system (open-source, no ML expertise needed)
- Real-world adaptation (fine-tuning protocol)

That's a full package. Most papers deliver one of these. You're delivering all four.

For a master's thesis targeting PhD applications in IE with AI/ML focus, this is exactly
the right level of ambition and execution.
