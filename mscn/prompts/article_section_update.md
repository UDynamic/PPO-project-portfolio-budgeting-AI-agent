# Update prompt

## focusing on a specific design : 

``` text
You are assisting me as a rigorous academic research and OR/MS modeling advisor for my master's thesis. 
My research concerns project portfolio budgeting, EPC/oil-and-gas contract cash-flow modeling, and optimization/RL-based decision support.

Respond in a highly structured, professional, analytical style. 
The answer must read like a research design document, not a casual explanation.

Core style requirements:
1. Always start with a clear document structure or roadmap at the top.
2. Use precise section headings and numbered subsections.
3. Explain what has already been designed, what assumptions are being made, and what decisions remain open.
4. Distinguish clearly between:
   - conceptual interpretation,
   - mathematical modeling choice,
   - real-world EPC/contract-finance justification,
   - thesis contribution,
   - unresolved ambiguity.
5. Use direct, rigorous language, but keep the tone readable and slightly friendly.
6. Do not overuse citations unless explicitly requested; if citations are needed, use them to justify modeling choices, not to replace reasoning.
7. When discussing mathematical model design, formulate it as close as possible to a classic OR problem formulation:
   - sets,
   - parameters,
   - decision variables,
   - binary variables,
   - state variables,
   - constraints,
   - transition equations,
   - objective-function implications.
8. When there are multiple modeling alternatives, compare them strategically instead of choosing blindly.
9. Put elaborative strategic tables at the end for classification, comparison, and approval/rejection by the researcher.
10. Explicitly mark what is recommended, what is optional, and what needs the researcher’s approval.
11. Treat me as the scientist/researcher who will approve, reject, or modify the proposed design.
12. Avoid vague advice. Every recommendation should have a modeling implication.
13. Keep the model purely parametric and distribution-free unless I explicitly ask for uncertainty modeling or calibration.
14. If the topic concerns EPC contract cash flows, distinguish carefully between:
   - advance payment,
   - advance payment guarantee,
   - retention,
   - milestone payments,
   - bank credit,
   - project-specific borrowing,
   - portfolio-level cash pooling.
15. Use LaTeX notation for formulas where useful.
16. End with a concise “current scientific recommendation” or “current design state.”

When I ask you to summarize or review a design we developed, do not just repeat it. 
Organize it into:
- Current State of the Design,
- Mechanism Description,
- Mathematical Representation,
- Decided Elements,
- Open Design Ambiguities,
- Strategic Tables,
- Final Recommendation.

When I ask for a problem formulation subsection, write in thesis-ready academic language and include:
- mechanism interpretation,
- variables and parameters,
- constraints,
- state transitions,
- modeling assumptions,
- relationship to other model components,
- why the formulation is novel or useful.

When I ask for a literature review paragraph, write in academic prose without heavy mathematical notation unless requested.

Default tone:
Focused, precise, analytical, structured, and professional, with light friendliness only where appropriate.
```
---

## section 3 : problem formulation

**GUIDELINES:**
- the problem formulation must be defined as close as possible to the classic OR problem definition. with mechanisms and events coded as mathematical constraints and binary variables etc.
- the problem formulation is a major contribution. so the Citations must amplify and provide evidence on the presented model. both at the conceptual model design from project management literature and both novelty of the mathematical model.
- the section must be 4 pages with 2 added pages. one for the parameter table, another for the pseudo code algorithm of the problem
- model stays purely parametric and distribution free. we calibrate the model and fit distributions in the Methodology section.

---

**CONTEXT:**

the problem is like this analogy:
the projects of portfolios are fixed pre arranged set of glasses on the table to filled with our jar. 
the water in the jar is the available budget. 
this available budget is expected to be provided from interested credit for the portfolio. milestone payments are the only other positive cash inflow and they compensate and help with increasing the project total payments minus capital cost.
if budget allocated strategically, earning the milestone payments help increase this gap of total project income minus credit cost.
the challenge of making a project profitable is to increase remained gross profit for the project. after deduction of the credit cost from total earned money.

`Note: it's a sequential budget allocation problem. agent learns to allocate budget for each project at each timestep until termination state.`

this is my problem formulation section.
remember it as the last state of the local file.

here it is:

---

**GOAL:**
I want to modify some designs.

**PROCESS:**
I need to be aware of the effects on the whole section and places needing according updates.

1. first I'll give you the document.
2. then i mention my idea and the modification I have in mind as **REQUEST**.
3. you think and evaluate update.
4. you ask me for resolving your confusions and clarified approval of me on that detail
5. I give you clarification and approvals.
6. you give me the updated file as a proper no bugged tex file for the section.

---
**REQUESTS:**

1. the organization:
just like project,
portfolio have it's attributes and aggregated variables and performance metrics.

I'd like to first define what a project is.
and then by scaling the collection of the projects introduce the portfolio level dynamics.

so first we should elaborate on the project budget allocation problem,
then aggregate it over a portfolio of them.

as in object oriented programming, the class blueprint is a project,
this project definition and attribute setting and model formulation must be according to the current design and proved to be aligned with the majority of the contractual dynamics and mechanisms,
the action is focused on the operational budget allocation.


- Pivoting the direction of paper: making a general resource allocation problem formulation in the general difinition of the project accross different industries and sectors,
then solving it in the methodology section leveraging the letter of recieving anonymized data from a major EPC oil and gas contractor
```
**The core fix:** you've been trying to find oil & gas-EPC-specific literature for mechanisms that are actually *generic project-control phenomena* — S-curves, EVM, progress-to-budget conversion, schedule/cost performance indices, abandonment thresholds. That literature search kept failing not because you were searching badly, but because **the generic version of every one of these mechanisms is genuinely well-documented across construction, software, defense, and IT project management**, while the oil & gas-EPC-specific *numbers* are the part that's thin. Your pivot separates these two things instead of conflating them, which is exactly the separation a Q1 reviewer wants to see:

- **Layer 1 — the general resource-allocation-under-uncertainty formulation.** S-curves (Kenley & Wilson 1986, construction generally), EVM (Fleming & Koppelman, PMI, all sector-agnostic), CPI stability (Christensen & Heise 1993 — *defense* contracts, not EPC, and that's fine now because you're not claiming it's EPC-specific), abandonment-option theory (Dixit & Pindyck — finance-generic), constrained MDPs (Borkar & Jain — domain-agnostic). All of this literature is real, well-established, and was never actually about oil & gas — you were the one stretching it there. Presented as the general case, every one of these citations is now used *correctly*, not stretched.
- **Layer 2 — the oil & gas EPC instantiation.** Here, exactly where the literature legitimately runs out, you cite **Merrow/IPA** (genuinely oil & gas EPC-specific, the strongest thing you have) for what it actually measured, and for everything else, you have a real, citable, methodologically standard fallback: **anonymized data from a contractor, via a data-sharing or NDA-style acknowledgment.** This is not a workaround — it is exactly how applied OR/RL papers handle proprietary industry calibration. It's more credible than Tier 4 "expert judgment" alone, because it's actual data, not just your recollection of it.
```

2. **The advance payment redesign**
the one updated design is the advanced payment and it's recovery through milestones to come. 
the previous design may be considering the accumulation of the advanced and all other payments to be all the project payments.
that's not correct,
all the other milestone payments including interim and final payments must add up to the 100% of the project payment.
advanced payment is a supportive payment and receiving it must be recovered through certain amount in certain milestones to come.
here's the probolem formulation section on it:

``` tex
\subsubsection{Advance Payment and Recovery Mechanism}

In EPC and EPC/Turnkey projects, advance payment constitutes a contractual mechanism through which a portion of the Contract Price is disbursed to the contractor at an early stage of the project, prior to the certification of corresponding physical progress. Industry-standard contract forms and empirical studies consistently characterize advance payment as an early liquidity support instrument, secured by an Advance Payment Guarantee and contractually recovered through deductions from subsequent interim payment certificates \cite{fidic1999silver, fidic2017silver, quollnetadvancepayment}. These characteristics motivate the explicit modeling of advance payment as a distinct cash-flow mechanism that affects the temporal distribution of project cash inflows without altering total contractual revenue.

In the proposed portfolio optimization framework, advance payment is modeled as a project-specific, upfront cash inflow event that occurs at project initiation and is followed by a structured recovery process over the project execution horizon. Let each project $i \in \mathcal{P}$ be characterized by a fixed Contract Price $CP_i$ and an exogenously specified advance payment ratio $\alpha_i$, consistent with contractual practice in EPC projects \cite{fidic1999silver}. The advance payment amount is assumed to be fully determined at contract effectiveness and is treated as a deterministic parameter of the project. This modeling choice reflects the fact that advance payment terms are contractually agreed ex ante and are not subject to operational decision-making during project execution.

To capture the recovery of the advance payment, the model introduces a recovery state variable that tracks the remaining unrecovered portion of the advance over time. Recovery is enforced through deductions from interim payment certificates and is constrained such that the cumulative recovery over the project life exactly equals the initial advance payment amount. This constraint encodes the contractual requirement that the Employer does not pay more than the agreed Contract Price, while allowing flexibility in the timing of recovery, consistent with industry practice where recovery may begin after a progress threshold or from the first certified payment \cite{fidic1999silver, constructionknowledgehubclause14}. In the problem formulation, the recovery schedule itself is parameterized and not optimized, reflecting its contractual nature.

Importantly, advance payment is not modeled as debt or borrowing. Unlike credit facilities or external financing instruments, advance payment does not generate interest, repayment obligations beyond contractual recovery, or balance-sheet liabilities for the contractor. Instead, it is treated as a timing shift in contractual cash inflows, aligned with its interpretation in both contract standards and construction finance literature \cite{quollnetadvancepayment}. This distinction is critical for avoiding double counting of financing costs and for preserving the separation between contractual cash-flow mechanisms and endogenous credit decisions in the portfolio model.

From a portfolio-level perspective, the advance payment mechanism creates an early positive cash inflow that augments the available budget in initial periods, followed by systematically reduced net inflows during recovery periods. By explicitly representing advance payment and its recovery, the model captures an essential liquidity trade-off faced by EPC contractors managing multiple concurrent projects. This formulation allows the optimization model to account for the interaction between advance payments, mandatory budget allocations, and project-specific credit usage, without introducing distributional assumptions or stochastic processes at this stage. Consequently, advance payment and recovery are incorporated as deterministic, contract-driven constraints that shape feasible cash-flow trajectories across the project portfolio, while leaving calibration and uncertainty modeling to subsequent methodological sections.
```

Note: advance payment and advance recovery is independant from the retention mechanism

3. the mandatory allocation is at least numerical planned budget of the s-curve. (how do you expect with performance less than 1 to catch up with less than the plan.)

4. uncertainty modeling:
the key parameter of **budget to progress efficiency** is the key parameter modeled as the contractor's performance. 
this parameter is the key attribute to be engineered for the agents training.
this stochastic number is behind all the inner and outer factors effecting on the budget to progress efficiency. 
specific modeling of the factors effecting on this parameter including contractor's internal performance parameters, local legislations, inflation etc.
this is out of scope.
we will calibrate the distribution of this parameter using literature and real-world data but that's it.
it's a natural future work to study on the key factors effecting this parameter.
we want for the agent to learn to allocate by monitoring this parameter.

5. Note on the intervention mechanism:
purely local representing on-site project MC contractor's part as mediator. intervention is mentioned because management won't stay and look at the performance to plumet. they're obligated for high performance on time operational delivery. currently the only mechanism for increasing project performance is this. we must provide evidence that this is enough, and no other effort for increasing project performance is outside this intervention mechanism. for the performance to go up you need to bear the cost.

6. on the credit system: 
the whole system is to be exclude and out of scope.
there is no enough literature to model and calibrate it.
instead of soft budget constraint we use hard budget constraint.
you can't spend the money you don't have.


7. termination abandonment modeling of the project:
**under a hard budget constraint, starvation is the normal operating mode whenever the optimizer must choose which projects to feed**
because we don't assume the credit mechanism or contractor's treasury in the project financing. we will model another design backed by literture in concept.


8. the reputation cost of terminating a project
because the portfolio is fixed beforehand and no project entries assumed in the model, this parameter and it's effect however qualitatively studied in the literature, for the lack of rigorous quantitative study and being out of the portfolio structure is assumed out of scope.







---
<!-- 
Ok this all seams so pinishing as an environment. 

I've decided:
Let's drop the credit system, 
Let's drop the treasury system, 
let's drop the mandatory allocation (there is no academic or legal contract mentioning explicit mandatory allocation. they get the contractor's agreement on the plan, the finish date is the main goal)

Let's work on Project termination:
according to the literature there is norm in project delay, spi and cpi. 
this norm is backed by the literature. 
passing that norm would trigger termination settlement. 

If a project finish is expected to be delayed more than the norm.
norm for each project categorie is existing in tthe literature.

so we could add another to each project as delayed finish threshold. 

there is two other conditions to termination. 

SPI and CPI below the norm and their threshold. 

so these three threshold's if passed simultaneusly, the termination settlement is the resault. 

in this case the Management intervention for performance recovery didn't work or the increased cost of it even made the not allocation policy more stronger.

Note on intervention mechanism: mandated by the environment. if the spi gap is more than intervention threshold the intervention is triggered. 
adding to the cost for specific periods in future, revocering the performance to a ratio times the uncertainty parameter of the performance. 
boosting performance. 
these ratio has got a decay over time.)

in the settlement we studied previously to balance the project finance. 
if contractor is behind, will get the payment and vice versa. 
a clean financial balancing things out. 
better settlement is for the contractor to complete the work until balanced with client's payments. 
if the client has to pay the work is immediately stoped, and the payment will be delivered according to delayed payment model. 
the contractor will finish the work at that timestep. (adding to the mandatory allocation of the portfolio)

this is different with the general mandatory allocation interpreted from the contract plan. 
that mandatory allocation is droped. 
but the idea of having that mechanism is good for forcing allocation on project termination after settlement.
-->