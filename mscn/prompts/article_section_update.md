# Update prompt

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

4. ⚠️ the credit remains project specific. it's maxed to a certain ratio relative the bac of the project. (you can only borrow so much) but the received credit is added to the avialable budget and could be spended on another project.

5. uncertainty modeling:
the key parameter of **budget to progress efficiency** is the key parameter modeled as the contractor's performance. 
this parameter is the key attribute to be engineered for the agents training.
this stochastic number is behind all the inner and outer factors effecting on the budget to progress efficiency. 
specific modeling of the factors effecting on this parameter including contractor's internal performance parameters, local legislations, inflation etc.
this is out of scope.
we will calibrate the distribution of this parameter using literature and real-world data but that's it.
it's a natural future work to study on the key factors effecting this parameter.
we want for the agent to learn to allocate by monitoring this parameter.

6. Note on the intervention mechanism:
purely local representing on-site project MC contractor's part as mediator. intervention is mentioned because management won't stay and look at the performance to plumet. they're obligated for high performance on time operational delivery. currently the only mechanism for increasing project performance is this. we must provide evidence that this is enough, and no other effort for increasing project performance is outside this intervention mechanism. for the performance to go up you need to bear the cost.