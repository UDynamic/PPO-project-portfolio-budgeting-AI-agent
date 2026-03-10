
## What Kind of Companies Do You Actually Need?

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



# 8️⃣ If I Were Writing This for a Q1 Journal

I would write:

> “We calibrate uncertainty parameters using financial data from publicly listed engineering and construction firms within the Fortune 500.”

Then cite:

- Industry volatility
- DSO dispersion
- Margin variation

That’s publishable.


## 1. Where to Write background, motivation, and justification for competence?

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
