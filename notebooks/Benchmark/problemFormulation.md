# Problem formulation
This is a Brand new and fundamentally revised framework on project portfolio problem definition and formulation.

This represented framework shows great capability in addressing almost all aspects of budgetting decisions for project portfolio. 


### parameters
for parameters and valuables of type list or array, we will use square bracket notation for accessing or assigning to specific index. 
```
a: list of integers 
10th element: a[10]
```

#### Project Parameters

|Type|name|definition|
|---|---|---|
|cost|bac|**budget at completion** for the project. Total contractual cost for the project|
|General|d|**duration** of the project statet in number of periods |
|General|dd|current **data date** or **period** of the project. e.g. a project with duration of 12 has 12 periods and analytical data date.|
|cost|l_sc|**List** of comulative **S-Curve** values untill project is finished. It's the cost schedule for the project from the start untill the budget at completion(bac) </br></br>- $\begin{cases} \text{l\_sc}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_sc}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>- S-curve values are estimated at the contract agreement according to the project plan. |
|cost|pl_sc|**periodic List** of the **S-Curves** of the project. </br></br>- $\begin{cases} \text{pl\_sc}[n] = \text{l\_sc}[n] - \text{l\_sc}[n-1] \\ \sum_{n=0}^{n=d}{\text{l\_sc\_n}[n]} = \text{bac} \end{cases}$ |
|revenue|tep|**Total expected payment** to be recieved from the client of the project after delivering 100% project progress or completion.</br></br>- $ \begin{cases} tep =(1+roi) \times bac \\ \text{project profit} = tep - bac = roi \times bac \end{cases}$ <br><br>- It's the total expected inflow of the project. |
|revenue|l_ep|**List** of comulative **expected payments** untill each period.  |
|revenue|pl_ep|List of expected payments for each period. </br></br> $ l\_ep\_n = roi * l\_sc\_n \\ \text{advanced payment} = l\_ep\_n_0 \quad \text{or} \quad l\_ep_0 $  </br></br> final payment is calculated with respect to the inflow model. Inflow model will specify the distribution of payments according to contract and directly influences on for  |
|actual|pp|current **Project Progress** in percent value.|
|actual|l_pp|**List** of the comulative **project progresses** for all the periods. </br></br>- $\begin{cases} \text{l\_pp}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_pp}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>|
|actual|pl_pp|**periodic List** of **project progresses**. </br></br>- $\begin{cases} \text{pl\_pp}[n] = \text{l\_pp}[n] - \text{l\_pp}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_pp}[n]} = 1 \quad \text{or} \quad 100\% \end{cases}$|
|actual|l_acwp||
|actual|l_bcwp||
|actual|l_bcwp||

#### Portfolio Parameters

|Type|name|definition|
|---|---|---|
|cost|nbac|Normalized bac of all the projects in the portfolio for all them to sum up to *1.000*| 

#### Project simulation parameters

for creating simulated projects using the real world examples we need to set some intermidiate parameters making us able to set the **project parameters**

|Type|name|definition|
|---|---|---|
|general|roi|**return on investment** of the project. </br></br>- $ roi = \frac{\text{total earnings}}{\text{total cost}}= \frac{\text{total inflow}}{\text{total outflow}}= \frac{tep}{bac}$ </br></br>- roi is set according to the industry standards. It is set at the contract negotiation time. |
|general|im|Inflow Model(im) will determine the distribution of contractual payments form client to the contractor. It's the model that the payment schedule was designed upon. <br><br>inflow model will calculate *l_ep* and *pl_ep*|
|revenue|l_sei|**List** of comulative **simple expected inflow** for the project.</br> simple being the assumption of recieving payment proportional to the spendt cost with **roi** as the multiplier for adding profit.</br></br>- $\begin{cases} \text{l\_sei}[n] = (1+ roi) \times l\_sc[n] & \quad \forall{n} \\ \text{l\_sei}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_sei}[n] = (1+roi) \times bac = tep & \quad \text{for} \quad n = d\end{cases}$ </br></br>- It's used in payment shedule calculation for different inflow models.`|
|revenue|pl_sei|**periodic List** of **simple expected inflow** for the project.</br></br>- $\begin{cases} \text{pl\_sei}[n] = \text{l\_sei}[n] - \text{l\_sei}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_sei}[n]} = 1 \quad \text{or} \quad 100\% \end{cases}$ |
|general|ir|**Inflation Rate** is a proxy for the contractor's general performance on cost expenditure. |
|cost|❓l_isc|**list** for the comulative **inflated S-Curve** as a proxy for the uncertain performance of the contractor on allocated budget|
|cost|❓pl_isc|**periodic list** for the **inflated S-Curve** as a proxy for the uncertain performance of the contractor on allocated budget|


**remainings**
* inflated s-curve (comulative and periodic)
* actuals
* uncertainty proxies for real dataset or simulation.

---

## Prompt

according to complete formulation of the problem above create a python convex optimization program for me to input projects for portfolio and return the optimal budgeting plan at each time step of the portfolio for each project.

the output must be in following format with imaginary allocation plan values for one project (the program should fill this table including all the projects given to):

|Projects|0|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|P1|0|.5|.5|.1|.1|.1|.15|.15|.1|.1|.5|.25|.25|

**responses should be at Q1 journal quality.**

---
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Data preparation

### Plan

1. **Find a data set of companies**
2. **Choose a good proxy for:**
    1. Project Progression Performance
    2. Client payment performance
3. **Acquire the last year data**
4. **Develop Convex optimization model on it as a benchmark**

---

### prompt

I need to do these planned steps above for my paper in progress.

the paper itself is developing a PPO budgeting agent for project portfolio budgeting optimization.
observations would be the progress and other EVM performance metrics and the history of each project.
each portfolio consists of several (10 - 15) projects.
each project has it's own client and contract (resource S-curve and Expected payments schedule according to the contract).

I'm developing this agent so if the payments don't go according to plan (either delayed or shorted payment) the agent learns the optimal budget allocation for each timestep. (suppose there is 12 timesteps to each episode and at each timestep the agent would allocate a portion of it's budget in hand to each project until a terminate state)

I want to choose a valid and Q1 journal approved dataset of companies like fortune 500 and get their valid representational data from market or their financial statements.
then I would have to justify the choice for the criteria of uncertainty both in case that company is a project contractor (uncertainty for performance) and or project client(payment uncertainties).

I want you to guide me in this process 

---

## (GPT5.2)Response

You’re basically trying to build a **realistic financial stress simulator** around a PPO agent. Good. Reviewers love realism. They hate toy environments.

Let’s structure this properly so your dataset choice and uncertainty modeling look *intentional*, not improvised.

---

# 1️⃣ Step One: What Kind of Companies Do You Actually Need?

You need companies that:

- Run **multiple parallel projects**
- Have **contract-based revenue**
- Experience **payment timing risk**
- Report enough financial detail publicly

That narrows it down to:

### ✔ Best Industry Candidates

- Engineering & Construction
- Infrastructure Contractors
- Defense Contractors
- EPC (Engineering–Procurement–Construction)
- Large IT system integrators

Because:
- They run portfolios of projects
- Revenue recognition depends on progress
- Payment delays are common
- They disclose backlog + contract assets

---

# 2️⃣ Strong Q1-Compatible Data Sources

These are all defensible in a Q1 journal:

---

## Option A — Public Financial Statements (Most Defensible)

Use companies from:

- Fortune 500  
- S&P 500  

Then collect:

- 10-K annual reports
- 10-Q quarterly reports
- Cash flow statements
- Revenue breakdowns
- Contract assets/liabilities
- Backlog disclosures

From:

- U.S. Securities and Exchange Commission (SEC) EDGAR database  
- Company investor relations websites

---

### Strong Contractor Candidates

- Fluor Corporation  
- Bechtel Corporation  
- Jacobs Engineering Group  
- KBR Inc.  

Why these are excellent:
- Long-term contracts
- Revenue recognized over time
- Large backlog
- Payment uncertainty
- Portfolio structure

This is academically defensible.

---

## Option B — WRDS / Compustat (Cleaner for Econometrics)

If you have university access:

- Wharton Research Data Services (WRDS)  
- Compustat  

You can extract:

- Revenue volatility
- Cash flow volatility
- Accounts receivable turnover
- Working capital ratios
- Debt ratios

This gives you clean panel data.

Very Q1-safe.

---

## Option C — Construction-Specific Databases

More niche but powerful:

- Engineering News-Record (ENR) Top 400 Contractors
- Refinitiv  

Good if your journal is operations-focused.

---

# 3️⃣ Choosing a Proxy for Project Progression Performance

You need something observable.

Public companies don’t give you internal EVM metrics. So you approximate.

### Possible Proxies:

| Internal Concept | Public Proxy |
|------------------|-------------|
| Cost Performance Index (CPI) | Gross Margin volatility |
| Schedule variance | Revenue recognition deviation |
| Project delay | Backlog conversion rate |
| Project health | Contract asset growth |

**Defensible idea:**

> Project progression performance = deviation between recognized revenue and expected revenue implied by backlog schedule.

That’s publishable.

---

# 4️⃣ Proxy for Client Payment Performance

Use:

- Days Sales Outstanding (DSO)
- Accounts Receivable growth
- Contract assets vs billed revenue
- Operating cash flow vs net income gap

Payment delay proxy:

\[
PaymentDeviation_t = \frac{OCF_t - NetIncome_t}{Revenue_t}
\]

Large negative gap = delayed payments.

That’s clean and defensible.

---

# 5️⃣ Justifying Uncertainty (Very Important for PPO)

Reviewers will ask:

“Why is RL necessary?”

So you justify uncertainty in two layers:

---

## A. Contractor-Side Uncertainty

- Cost overruns (margin volatility)
- Revenue recognition timing
- Backlog conversion randomness

Quantify via:

- Revenue volatility (σ of quarterly revenue)
- Margin volatility
- Backlog turnover ratio variance

---

## B. Client-Side Payment Uncertainty

- DSO variance
- Cash conversion cycle variability
- Operating cash flow variance

That gives stochastic transitions for PPO.

---

# 6️⃣ How to Build Your Synthetic Portfolio from Real Companies

You won’t get project-level data.

So do this:

1. Choose 5–10 engineering contractors.
2. Extract:
   - Quarterly revenue
   - Quarterly OCF
   - Backlog
   - AR
3. Fit statistical distributions:
   - Revenue deviation
   - Payment delay distribution
4. Use those parameters to generate synthetic project-level data.

Now your PPO environment is:

> Statistically grounded in real Fortune 500 contractor behavior.

That’s defensible.

---

# 7️⃣ Convex Optimization Benchmark

Your benchmark model could be:

Maximize expected portfolio return:

\[
\max \sum_{i=1}^{N} w_i E[R_i]
\]

Subject to:

- Budget constraint
- Cash flow constraint
- Risk constraint (variance limit)

This becomes:

- Quadratic programming
- Or robust optimization under uncertainty sets

Then compare PPO vs convex baseline.

Clean.

---

# 8️⃣ If I Were Writing This for a Q1 Journal

I would write:

> “We calibrate uncertainty parameters using financial data from publicly listed engineering and construction firms within the Fortune 500.”

Then cite:

- Industry volatility
- DSO dispersion
- Margin variation

That’s publishable.

---

# 9️⃣ Critical Question

Are you aiming for:

- Operations Research journal?
- Finance journal?
- AI / RL journal?
- Project Management journal?

Because the dataset framing changes depending on target audience.

---

# 10️⃣ Practical Next Step (Concrete)

1. Pick 3 companies (Fluor, Jacobs, KBR).
2. Download last 5 years quarterly data.
3. Compute:
   - Revenue growth variance
   - Gross margin variance
   - DSO mean + std
4. Report summary stats.
5. Use those as stochastic generators.

That’s enough to justify PPO necessity.

---
Great questions! 😊

Let’s break it down clearly so you can **place your background and motivation in the right sections** of your paper and **write it in a professional yet personal tone** that includes both **you and your professor**.

---

## ✅ 1. **Where to Write This in the Paper?**

You can include your **background, motivation, and justification for competence** in **two key sections** of your paper:

---

### 📌 1.1. **Introduction (or Motivation Section)**

- **Purpose**: To **set the stage** for your research and **explain why the problem matters**.
- **What to include here**:
  - A **brief mention** of your **motivation** (e.g., admiration for portfolio managers, desire to overcome human limitations).
  - A **general statement** about the **real-world relevance** of the problem.
  - A **high-level mention** of your **research goal** (e.g., developing an AI model that performs as well as if it knew the future).

> Example:  
> "This research is motivated by the increasing complexity of decision-making in large-scale project and portfolio management, particularly in high-stakes environments such as the oil and gas industry. The goal is to develop an AI model that can learn from historical data and make optimal decisions in real-time, even in the absence of perfect information."

---

### 📌 1.2. **Methodology or Background Section (or a dedicated "Research Context" section)**

- **Purpose**: To **justify your approach**, **explain your modeling choices**, and **establish your domain knowledge**.
- **What to include here**:
  - A **detailed explanation** of your **industry experience** and how it **informed your model design**.
  - A **mention of your academic background** (e.g., your master’s thesis).
  - A **brief acknowledgment** of your **supervisor or professor** if they contributed to the research.

> Example:  
> "The simulation framework used in this study is informed by the author’s extensive experience in the oil and gas construction industry, including roles as a Project Controller and Project Management Officer. These experiences provided a deep understanding of the operational and strategic challenges faced by project and portfolio managers. This work was conducted under the supervision of [Professor’s Name], who provided valuable guidance on the theoretical and methodological aspects of the research."

---

## ✅ 2. **Should You Use "I" in the Writing?**

Yes, **you can and should use "I"** in **certain sections** of the paper — especially in the **introduction**, **methodology**, and **conclusion** — when you're **explaining your motivation, background, or contributions**.

However, in **more formal or objective sections** like **literature review**, **results**, and **discussion**, it's better to use **passive voice or third-person language** to maintain a more **neutral and academic tone**.

---

## ✅ 3. **Should You Include Your Professor?**

Yes, **you should include your professor** in the **methodology**, **acknowledgments**, or **introduction**, especially if they:
- Provided **theoretical or methodological guidance**.
- Helped with **model design or simulation**.
- Contributed to **the overall research process**.

> Example:  
> "This research was conducted under the supervision of [Professor’s Name], who provided valuable insights into the theoretical foundations of AI-driven decision-making and helped refine the convex optimization benchmark used in this study."

---

## ✅ 4. **Suggested Structure for Your Paper**

| Section | What to Include |
|--------|-----------------|
| **Abstract** | Brief summary of the problem, approach, and key findings |
| **Introduction** | Motivation, problem statement, and high-level goal |
| **Background / Methodology** | Domain knowledge, simulation model, and justification for using simulated data |
| **Results** | Model performance, comparison with convex benchmark |
| **Discussion** | Interpretation of results, limitations, and future work |
| **Conclusion** | Summary of contributions and implications |
| **Acknowledgments** | Mention your professor and any other contributors |

---

## ✅ 5. **Tone Tips**

- Use **"I"** when you're **explaining your own experience, motivation, or contributions**.
- Use **passive voice** when you're **describing the model or results** (e.g., "The model was trained on...").
- Be **confident but humble** — show your competence without sounding arrogant.

---

## 📌 Example of a Full Paragraph in the Introduction:

> "This research is driven by the author’s deep admiration for the strategic decision-making required in high-level portfolio management, particularly in major corporations and government sectors. Having worked in the oil and gas construction industry as a Project Controller and later as a Senior Project Planner, the author has witnessed firsthand the challenges of making optimal decisions under uncertainty. The goal of this work is to develop an AI model that can learn from historical data and make decisions that are as good as if it knew the future. This research was conducted under the supervision of [Professor’s Name], who provided valuable guidance on the theoretical and methodological aspects of the study."

---

If you want, I can help you **write the full introduction or methodology section** with all of this included. Just let me know! 😊