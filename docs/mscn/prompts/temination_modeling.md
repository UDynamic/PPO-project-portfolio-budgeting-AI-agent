ok I'm gonna answer your questions in the updated form of the content you created.
let's build upon that.

Let me write the clean construction first. Confirm or correct it, then we update the sections.

TerminationTolerance_i is another project parameter that is the margin for failure for any specific project.
some clients maybe more forgiving and some maybe too punishing.
each time the two conditions simultaneously stay active one will be deducted from the Termination tolerance. 
when this get's to zero the project will be terminated.

each time at least one of the conditions go off this tolerance will reset to the original value


---

## Termination Mechanism — Clean Construction

---

**delay penalty per periods of delay $r_i$ and Cap delay penalty is given as a project parameter:** $$r_i^{\max}$$
it's a percent ratio to be multiplied by the contract price:
$$ \text{completion delay penalty} = r_i^{\max} \times CP_i$$

### Stage 1 — Continuous Penalty Assessment (every period $t$, every active project $i$)

this is for termination condition monitoring.

**Forecast finish deviation.**
At each period $t$, the expected finish date is a simple linear estimated remaining work from current SPI:

$$\hat{f}_{i,t} = s_i + \frac{D_i^{\mathrm{plan}} - (t - s_i  ) }{\mathrm{SPI}_{i,t}}$$

The forecast finish deviation in timesteps (positive = late, negative = early):

$$\Delta f_{i,t} = \hat{f}_{i,t} - f_i$$


**Per-period penalty in objective :**
we maximize the income and minimize the penalties.
this is the only penalty in the objective function.
and the maximum net present value of the cashflow is the maximization objective.


$$c_{i,t}^{\mathrm{pen}} = CP_i \cdot \min{\left( r_i^{\max}, \Delta f_{i,j} \times r_i \right)}$$

This is not cumulative, it’s a state penalty signal.
and the delay penalty enters cashflow only at completion to be deducted from the last payment.
this is for objective. no cash transferred at this amount mid project.
if the project completes before the delay cap, then the delay penalty is enforced and deducted from the last payment with retentions. 

where $r_i$ is the per-period penalty rate and $r_i^{\max}$ is the maximum total penalty ratio, both project-specific calibrated parameters. The penalty is symmetric: early delivery and late delivery are penalized equally, discouraging both budget-pushing and schedule lag.

---

### Stage 2 — Termination flag (two conditions, both must hold at period $t$)

**Condition 1 — Schedule tolerance exhausted:**

$$\Delta f_{i,t} > \frac{r_i^{\max}}{r_i}$$

The forecast finish deviation exceeds the maximum tolerable delay — the penalty cap has been reached and the project is expected to deliver beyond the contractual limit. The penalty is saturated; further delay carries no additional contractual deterrent.

**Condition 2 — Cost performance below threshold:**

$$\mathrm{CPI}_{i,t} < \kappa_i$$

The cost performance index has fallen below a project-specific threshold $\kappa_i$, indicating that the contractor is spending significantly more than the budgeted value of work performed. The project is both temporally and monetarily unrecoverable.

**Termination get's executed** at the period $t_i^{\mathrm{term}}$ at which both conditions remain active simultaneously for the termination tolerance amount of times, this makes the termination tolerance zero and at the timestep that it's zero the projects get's terminated:

$$\Delta f_{i,t_i^{\mathrm{term}}} > \frac{r_i^{\max}}{r_i} \quad \text{and} \quad \mathrm{CPI}_{i,t_i^{\mathrm{term}}} < \kappa_i$$



so the immediate termination flag won't terminate project at the step.
both parties when this flag get's active intervene in real world with the goal of performance recovery.
it's know as the grace or cure period in the literature and we just modeled it.

This is an environment transition, not an agent action. The agent influences both conditions indirectly through its allocation decisions — allocation drives progress which affects $\mathrm{SPI}_{i,t}$, $\mathrm{CPI}_{i,t}$, and therefore $\Delta f_{i,t}$ — but does not decide termination.
but seas that it's flagged for termination and entered to the grace period.


---

### Stage 3 — Termination Settlement (computed at $t_i^{\mathrm{term}}$ effected at the next step)
the $t_i^{\mathrm{term}}$ is the period that the termination tolerance becomes zero. so the next step the project get's terminated and the settlement is as below:

**Value of work delivered:**

$$V_i^{\mathrm{earned}} = P_{i,t_i^{\mathrm{term}}}^{\mathrm{actual}} \cdot CP_i$$

**Total payments already received by contractor** (cumulative cash transferred from client to contractor up to termination, includes advanced payment and milestone payments until termination period which the retention and the recovery have already ben deducted from them. so the total received payment is):

$$V_i^{\mathrm{received}} = A_i + \sum_{j \in \mathcal{M}_i^{\mathrm{cert}}(t_i^{\mathrm{term}})} R_{i,j}^{\mathrm{gross}}$$

where $R_{i,j}^{\mathrm{gross}}$ is the gross milestone payments received that client deducted advanced recovery and the retention from.

**Net settlement obligation:**

$$\Omega_i = V_i^{\mathrm{earned}} - V_i^{\mathrm{received}}$$

- If $\Omega_i > 0$: the client owes the contractor — client is expected to pays $\Omega_i$ to contractor at the termination timestep but since payments are stochastic this payment too follows the same stochastic timing.
- If $\Omega_i < 0$: the contractor owes the client — contractor repays $|\Omega_i|$ to client.

Because termination is the contractor's fault under FIDIC, and the contractor has provided an advance payment guarantee, the client holds priority in recovery.
contractor will pay when he has the money.

---

### Stage 4 — Settlement Recovery Mechanic (post-termination periods)

**Settlement queue.**
Upon termination of project $i$, a settlement obligation $\Omega_i$ is entered into a priority queue $\mathcal{Q}_t$ maintained by the environment. If multiple projects terminate, obligations are ordered by ascending $|\Omega_i|$ — smallest first.

all projects have equal priority until one is terminated.
the priority for each project could be from zero to the number of projects in the portfolio.
each project is terminated, it's priority get's increased.
the terminated projects have higher priorities and the termination with lower termination cost has the highest priority for budget allocation at the amount of termination cost.

Payment is all-or-nothing: no partial settlement.
any partial budget freezes until settlement resolves.

this priority is because the unresolved termination follows with legal limitations that may disrupt payment system for the portfolio manager's company. 

**Priority allocation.**
At each subsequent period $t > t_i^{\mathrm{term}}$, before any budget is allocated to active projects, If available cash $B_t$ covers the full obligation, it is discharged immediately and $B_t$ is reduced accordingly. If not, the obligation carries forward to period $t+1$ with the same priority and step passes with no allocation.

The settlement queue is resolved before the agent's allocation vector $\mathbf{x}_t$ is applied, making it a hard priority constraint that the agent cannot override.

---
Now I'm not sure how to do all these in linear programming especially with all the integration and update to the work that already we have.

and about the stochastic payment system I must say:
the payment related to interim payments only is stochastic in timing.
certified milestone will get payment is a near future modeling the stochastic payments from the client.
the amount is the amount on the payment plan and equal to the milestone payment.
of course the retention and the advanced recovery is deducted from it.
