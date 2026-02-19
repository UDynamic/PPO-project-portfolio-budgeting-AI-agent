# Data preparation

## Plan

1. **Find a data set of companies**
2. **Choose a good proxy for:**
    1. Project Progression Performance
    2. Client payment performance
3. **Acquire the last year data**
4. **Develop Convex optimization model on it as a benchmark**

---

## prompt

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

