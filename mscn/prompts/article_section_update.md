# Section 3 — Problem Formulation: Update Instructions

## Preamble

### Interaction Protocol

The update proceeds in a controlled, iterative loop:

1. Researcher provides the current document.
2. Researcher states a modification request.
3. Assistant evaluates the request and its downstream effects on the section.
4. Assistant surfaces unresolved ambiguities and requests researcher approval on open details.
5. Researcher provides clarifications and approvals.
6. Assistant delivers the updated file as a clean, compilable `.tex` file.

---

### Authoring Constraints (Apply to All Updates)

- Problem formulation must adhere to classical OR structure: sets, parameters, decision variables, binary variables, state variables, constraints, transition equations, and objective-function implications.
- Citations amplify and provide evidence for the model — both for conceptual mechanisms (project management literature) and for mathematical novelty. They do not replace reasoning.
- The section is 4 pages of formulation + 2 supplementary pages: one parameter table, one pseudocode algorithm.
- The model remains **purely parametric and distribution-free**. Distribution fitting and calibration belong to the Methodology section.

---

## Part I — Structural Redesign

### 1.1 Hierarchical Organisation: Project → Portfolio

Restructure the section using an object-oriented analogy:

- A **project** is the atomic unit — the class blueprint.
- A **portfolio** is a collection of projects — the instantiated set of objects.

**Required ordering:**

1. Define a single project: its attributes, state variables, cash-flow mechanics, and budget allocation sub-problem.
2. Aggregate across the project set to derive portfolio-level dynamics, constraints, and performance metrics.

The project definition must be proven consistent with the contractual dynamics and mechanisms established in the current design. The operational decision is **budget allocation only**.

### 1.2 Scope Pivot: General Resource Allocation with EPC Instantiation

**Core fix:** The formulation pivots from an EPC/oil-and-gas-specific framing to a **general resource allocation problem formulation**, applicable across industries and sectors. The EPC oil-and-gas context enters as the empirical instantiation in the Methodology section via anonymised contractor data.

**General formulation (Problem Formulation section):**
S-curves, Earned Value Management (EVM), progress-to-budget conversion, schedule/cost performance indices, and abandonment thresholds are documented across construction, software, defence, and IT project management. Cite this literature correctly at the general level:
- S-curves: Kenley & Wilson (1986), construction-general
- EVM: Fleming & Koppelman, PMI — sector-agnostic
- CPI stability: Christensen & Heise (1993) — defence contracts (not EPC; this is now correct usage)
- Abandonment option theory: Dixit & Pindyck — finance-generic
- Constrained MDPs: Borkar & Jain — domain-agnostic

---

## Part II — Mechanism-Level Updates

### 2.1 Advance Payment and Recovery — Redesign

**Design correction:** The previous design conflated advance payment with the pool of contractual milestone payments. This is incorrect. The corrected design is:

- All milestone payments (interim + final) sum to **100% of the Contract Price**.
- Advance payment is a **separate, supportive liquidity instrument**. It is disbursed at project initiation, does not count toward milestone revenue, and is **recovered through deductions from subsequent milestone payments**.
- Advance payment and retention are **independent mechanisms** and must not be conflated.

**Formal properties:**

- Advance payment amount: $\alpha_i \cdot CP_i$, where $\alpha_i$ is the contractually fixed advance ratio and $CP_i$ is the Contract Price of project $i$.
- Recovery: cumulative deductions from certified interim payments, constrained to equal $\alpha_i \cdot CP_i$ over the project life.
- Advance payment is **not debt**. It carries no interest and no balance-sheet liability. It is a timing shift in contractual cash inflows.
- Recovery schedule is **parameterised and not optimised** — it is contractually determined ex ante.
- This prevents double-counting of financing costs and preserves separation between contractual cash-flow mechanics and any endogenous financing decisions.

The subsection text below is the approved version and must be preserved verbatim in the updated file:

```
\subsubsection{Advance Payment and Recovery Mechanism}

In EPC and EPC/Turnkey projects, advance payment constitutes a contractual mechanism
through which a portion of the Contract Price is disbursed to the contractor at an early
stage of the project, prior to the certification of corresponding physical progress.
Industry-standard contract forms and empirical studies consistently characterise advance
payment as an early liquidity support instrument, secured by an Advance Payment Guarantee
and contractually recovered through deductions from subsequent interim payment certificates
\cite{fidic1999silver, fidic2017silver, quollnetadvancepayment}. These characteristics
motivate the explicit modeling of advance payment as a distinct cash-flow mechanism that
affects the temporal distribution of project cash inflows without altering total contractual
revenue.

[... remainder of approved subsection text ...]
```

### 2.2 Uncertainty Parameter: Budget-to-Progress Efficiency

**Role in the model:** This is the key stochastic parameter representing contractor performance — the ratio of physical progress achieved per unit of budget allocated.

**Modeling stance:**

- This parameter is the primary source of uncertainty in the model and the key attribute engineered for agent training.
- It implicitly aggregates all inner and outer factors affecting conversion efficiency (internal performance, local legislation, inflation, labour productivity, etc.). Explicit modelling of these sub-factors is **out of scope** and designated as future work.
- Distribution calibration using literature and real-world contractor data belongs to the **Methodology section**.
- In the Problem Formulation, this parameter is treated as a **stochastic scalar** drawn from a to-be-calibrated distribution, without specifying that distribution here.

**Agent learning target:** The agent observes this parameter over time and learns allocation policies that are robust to its variability.

### 2.3 Intervention Mechanism

**Conceptual scope:** Intervention represents management intervention to recover operational performance. It is triggered by the environment — not chosen by the agent — when the schedule performance gap exceeds a threshold.

**Formal properties:**

- Trigger condition: $\text{SPI gap} > \theta^{\text{int}}$ (intervention threshold), enforced by the environment.
- Effect: adds cost to the project for a defined number of future periods, and recovers performance to a ratio of the stochastic efficiency parameter. This recovery ratio decays over time.
- No other mechanism for increasing project performance is modelled. Evidence must be provided in the section that this single intervention mechanism is sufficient to represent management's performance-recovery options within the model's scope.
- Intervention is mandatory when triggered; the agent does not control it.

### 2.4 Credit and Treasury Systems — Dropped

**Decision:** The credit system and treasury/cash-pooling mechanism are **excluded from the model scope**.

**Rationale:** Insufficient literature to rigorously model and calibrate these mechanisms.

**Replacement:** The soft budget constraint is replaced by a **hard budget constraint**:

$$b_t \geq 0 \quad \forall t$$

Expenditure cannot exceed available funds in any period. There is no borrowing, no credit line, and no portfolio-level cash pool.

---

## Part III — Project Termination and Abandonment

### 3.1 Termination Conditions

A project is terminated when **three thresholds are simultaneously breached**:

1. Schedule performance falls below a minimum acceptable level.
2. Cost performance falls below a minimum acceptable level.
3. expected finish date is over the threshold. (due to the delay penalty cap and client losing leverage)

obviously the Intervention has failed to recover performance (or the added cost of intervention further reinforced the non-allocation policy).

These thresholds are modelled as binary trigger conditions. When all three are satisfied jointly, the environment forces project termination and initiates the settlement mechanism.



### 3.2 Termination Settlement Mechanism

expected delay is the key indicator for the termination.
according to the literature provide evidence on that for settlement, the client payment delay is deducted from the total delay, and the remained delay is owned by client up to 50% conventionally.

so the remaining delay must not cross the threshold of the delay penalty cap by the timestep count.
termination has no completion therfore no delay penalty.
but the delay penalty horizon specific to each project is the the max delay tolerable by the client.

At termination, a financial settlement is computed to balance the project accounts between contractor and client:

- If the contractor is **behind** (client has overpaid relative to certified progress): the contractor receives a payment and work stops immediately. The client's outstanding payment is delivered under the delayed payment model. The contractor completes work up to the settlement point — this triggers a **mandatory allocation** for that timestep.
- If the contractor is **ahead** (contractor has delivered more than billed): the contractor completes remaining billable work and receives payment.

This settlement-driven mandatory allocation is **distinct** from any general planning-based mandatory allocation (see §3.3).

### 3.3 Mandatory Allocation — Revised Design

**Previous design (dropped):** Mandatory allocation as a floor equal to the numerically planned budget on the S-curve. This is dropped. Rationale: expecting a contractor with sub-unit performance to catch up by allocating exactly the planned amount is internally inconsistent.

**Retained mechanism (settlement use only):** Mandatory allocation is kept exclusively as a settlement enforcement utility:

- When a project is terminated and settlement requires the contractor to complete work, the required budget allocation in the next timestep is **mandatory**.
- If the available budget is insufficient to cover this mandatory allocation, it is **deferred to the next period** with allocation priority above all other projects.
- The model must track a **per-project mandatory allocation register**: a state variable recording any outstanding mandatory allocation obligation arising from termination settlement.
- Budget allocation logic: the agent must clear all mandatory settlement allocations before allocating to any other project in the portfolio.

### 3.4 Reputation Cost of Termination — Excluded

Although reputation effects of project abandonment are qualitatively documented in the literature, rigorous quantitative modelling does not exist, and reputation effects fall outside the fixed-portfolio structure assumed in this model (no project entries after initialisation). This parameter is **out of scope** and may be noted as a direction for future work.

---

## part IV - project completion

at the finish all the payments and retention are released to the contractor as the final payment.

this final payment includes delay penalty.

according to the literature provide evidence on that for settlement, the client payment delay is deducted from the total delay, and the remained delay is owned by client up to 50% conventionally.

so the remaining delay will have penalty as a portion of the contract value for each timestep until a cap.


## Summary of Design Decisions

| # | Topic | Decision | Status |
|---|-------|----------|--------|
| 1 | Section structure | Project-first, portfolio-aggregated | **Approved** |
| 2 | Scope framing | General formulation + EPC instantiation | **Approved** |
| 3 | Advance payment | Separate from milestone sum; recovered via deductions | **Approved** |
| 4 | Advance payment ↔ retention | Treated as independent mechanisms | **Approved** |
| 5 | Budget-to-progress efficiency | Key stochastic parameter; calibrated in Methodology | **Approved** |
| 6 | Sub-factors of efficiency | Out of scope; future work | **Approved** |
| 7 | Intervention | Environment-triggered; single performance-recovery mechanism | **Approved** |
| 8 | Credit system | Dropped; hard budget constraint adopted | **Approved** |
| 9 | Treasury / cash pooling | Dropped; out of scope | **Approved** |
| 10 | Termination conditions | Three simultaneous thresholds | **Approved** |
| 11 | Settlement mechanism | Financial balancing; triggers mandatory allocation | **Approved** |
| 12 | Planning-based mandatory allocation | Dropped | **Approved** |
| 13 | Settlement mandatory allocation | Retained with priority queue logic | **Approved** |
| 14 | Reputation cost | Out of scope; future work | **Approved** |
