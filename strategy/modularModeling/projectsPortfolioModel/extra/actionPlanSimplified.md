# Complete Guide to Action Plan Mechanics: From Theory to Paper Calculations

## 1. Core Concept: What Are We Trying to Model?

**Real-world scenario:**
A project is falling behind schedule. The project manager decides to intervene by:
- Adding overtime shifts
- Bringing in additional workers
- Expediting material deliveries
- Increasing supervision intensity

**Key question:** How much does this intervention help?

**Reality:** It helps, but not perfectly. If you're at 70% efficiency, you can't instantly jump to 100% efficiency. The action plan might push you to 85% efficiency.

This is what $\eta$ (eta) captures: **the bounded effectiveness of management intervention**.

---

## 2. The Schedule Performance Index (SPI)

### Definition

$$\text{SPI} = \frac{\text{Earned Value (EV)}}{\text{Planned Value (PV)}}$$

Or in terms of progress:

$$\text{SPI}_i(t) = \frac{P_i^{\text{actual}}(t)}{P_i^{\text{planned}}(t)}$$

### Interpretation

- **SPI = 1.0** → On schedule (actual progress = planned progress)
- **SPI = 0.8** → Behind schedule (only 80% of planned work completed)
- **SPI = 1.2** → Ahead of schedule (120% of planned work completed)

### Simple Example

**Month 3 of a project:**
- Planned progress: 30% of total work
- Actual progress: 24% of total work

$$\text{SPI} = \frac{24\%}{30\%} = \frac{0.24}{0.30} = 0.80$$

**Interpretation:** The project is working at 80% of the planned rate.

---

## 3. The Effective SPI Formula

### The Formula

$$\text{SPI}^{\text{eff}}_i(t) = \text{SPI}_i(t) + \eta_i \left(1 - \text{SPI}_i(t)\right)$$

### Breaking It Down

Let's decompose this step by step:

**Step 1:** Calculate the "performance gap"
$$\text{Gap} = 1 - \text{SPI}_i(t)$$

This is how far you are from perfect performance (SPI = 1.0).

**Step 2:** Calculate the "improvement from action plan"
$$\text{Improvement} = \eta_i \times \text{Gap}$$

The action plan closes $\eta$ fraction of the gap.

**Step 3:** Add the improvement to current SPI
$$\text{SPI}^{\text{eff}} = \text{SPI} + \text{Improvement}$$

### Why This Formula Makes Sense

**Intuition:** The worse your current performance, the more room there is to improve. But you can only close a fraction $\eta$ of that gap.

**Mathematical properties:**
- If SPI = 1.0 (perfect performance), then Gap = 0, so no improvement needed
- If SPI = 0.5 (terrible performance), then Gap = 0.5, so more room to improve
- The parameter $\eta$ controls how much of that gap you can close

---

## 4. Worked Examples: Paper Calculations

### Example 1: Moderate Delay, Average Capability

**Given:**
- Current SPI = 0.75 (project is at 75% of planned rate)
- Action plan effectiveness $\eta = 0.50$ (average organizational capability)

**Step-by-step calculation:**

**Step 1:** Calculate the gap
$$\text{Gap} = 1 - 0.75 = 0.25$$

**Step 2:** Calculate improvement
$$\text{Improvement} = 0.50 \times 0.25 = 0.125$$

**Step 3:** Calculate effective SPI
$$\text{SPI}^{\text{eff}} = 0.75 + 0.125 = 0.875$$

**Interpretation:**
- Without action plan: working at 75% efficiency
- With action plan: working at 87.5% efficiency
- The action plan closed 50% of the gap (from 0.75 to 1.0)
- Remaining gap: 12.5% (still not perfect)

---

### Example 2: Severe Delay, Weak Capability

**Given:**
- Current SPI = 0.60 (project is at 60% of planned rate)
- Action plan effectiveness $\eta = 0.30$ (weak organizational capability)

**Step-by-step calculation:**

**Step 1:** Calculate the gap
$$\text{Gap} = 1 - 0.60 = 0.40$$

**Step 2:** Calculate improvement
$$\text{Improvement} = 0.30 \times 0.40 = 0.12$$

**Step 3:** Calculate effective SPI
$$\text{SPI}^{\text{eff}} = 0.60 + 0.12 = 0.72$$

**Interpretation:**
- Without action plan: working at 60% efficiency
- With action plan: working at 72% efficiency
- The action plan closed only 30% of the gap
- Remaining gap: 28% (still significantly behind)

**Key insight:** Even with an action plan, this project is still struggling. The weak organizational capability ($\eta = 0.30$) limits recovery potential.

---

### Example 3: Severe Delay, Strong Capability

**Given:**
- Current SPI = 0.60 (same as Example 2)
- Action plan effectiveness $\eta = 0.70$ (strong organizational capability)

**Step-by-step calculation:**

**Step 1:** Calculate the gap
$$\text{Gap} = 1 - 0.60 = 0.40$$

**Step 2:** Calculate improvement
$$\text{Improvement} = 0.70 \times 0.40 = 0.28$$

**Step 3:** Calculate effective SPI
$$\text{SPI}^{\text{eff}} = 0.60 + 0.28 = 0.88$$

**Interpretation:**
- Without action plan: working at 60% efficiency
- With action plan: working at 88% efficiency
- The action plan closed 70% of the gap
- Remaining gap: 12% (much better recovery)

**Comparison with Example 2:**
- Same starting point (SPI = 0.60)
- Strong capability ($\eta = 0.70$) achieves SPI = 0.88
- Weak capability ($\eta = 0.30$) achieves only SPI = 0.72
- Difference: 16 percentage points in performance

---

### Example 4: Minor Delay, Average Capability

**Given:**
- Current SPI = 0.92 (project is at 92% of planned rate)
- Action plan effectiveness $\eta = 0.50$

**Step-by-step calculation:**

**Step 1:** Calculate the gap
$$\text{Gap} = 1 - 0.92 = 0.08$$

**Step 2:** Calculate improvement
$$\text{Improvement} = 0.50 \times 0.08 = 0.04$$

**Step 3:** Calculate effective SPI
$$\text{SPI}^{\text{eff}} = 0.92 + 0.04 = 0.96$$

**Interpretation:**
- Without action plan: working at 92% efficiency
- With action plan: working at 96% efficiency
- Small gap means small absolute improvement
- But relative improvement is still 50% of the gap

---

## 5. Converting SPI to Monthly Delay

### The Delay Formula

$$\delta_i(t) = \max\left(0, \frac{1 - \text{SPI}^{\text{eff}}_i(t)}{\text{SPI}^{\text{eff}}_i(t)}\right) \text{ months}$$

### Understanding the Formula

This formula answers: **"If I continue at this SPI, how many extra months do I need per month of planned work?"**

**Derivation:**
- Planned work rate: 1 month of work per 1 month of time
- Actual work rate: $\text{SPI}^{\text{eff}}$ months of work per 1 month of time
- Time needed to complete 1 month of planned work: $\frac{1}{\text{SPI}^{\text{eff}}}$ months
- Extra time needed: $\frac{1}{\text{SPI}^{\text{eff}}} - 1 = \frac{1 - \text{SPI}^{\text{eff}}}{\text{SPI}^{\text{eff}}}$

---

## 6. Complete Example: From SPI to Cumulative Delay

### Scenario Setup

**Project details:**
- Baseline duration: 12 months
- Action plan effectiveness: $\eta = 0.50$
- Action plan triggers at Month 4 when progress gap exceeds 10%

**Monthly performance data:**

| Month | Planned Progress | Actual Progress | SPI | Notes |
|-------|-----------------|-----------------|-----|-------|
| 1 | 8.33% | 8.33% | 1.00 | On track |
| 2 | 16.67% | 15.00% | 0.90 | Slight delay |
| 3 | 25.00% | 21.00% | 0.84 | Worsening |
| 4 | 33.33% | 26.00% | 0.78 | Gap = 7.33% (no trigger yet) |
| 5 | 41.67% | 30.00% | 0.72 | Gap = 11.67% → **ACTION PLAN TRIGGERS** |

---

### Month 1: Perfect Performance

**Given:**
- Planned progress: 8.33%
- Actual progress: 8.33%

**Calculate SPI:**
$$\text{SPI} = \frac{8.33}{8.33} = 1.00$$

**No action plan needed (SPI = 1.00):**
$$\text{SPI}^{\text{eff}} = 1.00$$

**Calculate delay:**
$$\delta(1) = \frac{1 - 1.00}{1.00} = 0 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(1) = 0 \text{ months}$$

---

### Month 2: Slight Delay

**Given:**
- Planned progress: 16.67%
- Actual progress: 15.00%

**Calculate SPI:**
$$\text{SPI} = \frac{15.00}{16.67} = 0.90$$

**No action plan yet (gap = 1.67% < 10%):**
$$\text{SPI}^{\text{eff}} = 0.90$$

**Calculate delay:**
$$\delta(2) = \frac{1 - 0.90}{0.90} = \frac{0.10}{0.90} = 0.111 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(2) = 0 + 0.111 = 0.111 \text{ months} \approx 3.3 \text{ days}$$

---

### Month 3: Worsening Performance

**Given:**
- Planned progress: 25.00%
- Actual progress: 21.00%

**Calculate SPI:**
$$\text{SPI} = \frac{21.00}{25.00} = 0.84$$

**No action plan yet (gap = 4.00% < 10%):**
$$\text{SPI}^{\text{eff}} = 0.84$$

**Calculate delay:**
$$\delta(3) = \frac{1 - 0.84}{0.84} = \frac{0.16}{0.84} = 0.190 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(3) = 0.111 + 0.190 = 0.301 \text{ months} \approx 9 \text{ days}$$

---

### Month 4: Approaching Threshold

**Given:**
- Planned progress: 33.33%
- Actual progress: 26.00%

**Calculate SPI:**
$$\text{SPI} = \frac{26.00}{33.33} = 0.78$$

**Progress gap:**
$$\text{Gap} = 33.33 - 26.00 = 7.33\% < 10\%$$

**No action plan yet:**
$$\text{SPI}^{\text{eff}} = 0.78$$

**Calculate delay:**
$$\delta(4) = \frac{1 - 0.78}{0.78} = \frac{0.22}{0.78} = 0.282 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(4) = 0.301 + 0.282 = 0.583 \text{ months} \approx 17.5 \text{ days}$$

---

### Month 5: Action Plan Triggers

**Given:**
- Planned progress: 41.67%
- Actual progress: 30.00%

**Calculate SPI:**
$$\text{SPI} = \frac{30.00}{41.67} = 0.72$$

**Progress gap:**
$$\text{Gap} = 41.67 - 30.00 = 11.67\% > 10\%$$

**ACTION PLAN ACTIVATES!**

**Calculate effective SPI with $\eta = 0.50$:**

**Step 1:** Performance gap
$$\text{Performance Gap} = 1 - 0.72 = 0.28$$

**Step 2:** Improvement
$$\text{Improvement} = 0.50 \times 0.28 = 0.14$$

**Step 3:** Effective SPI
$$\text{SPI}^{\text{eff}} = 0.72 + 0.14 = 0.86$$

**Calculate delay with improved SPI:**
$$\delta(5) = \frac{1 - 0.86}{0.86} = \frac{0.14}{0.86} = 0.163 \text{ months}$$

**Compare to without action plan:**
$$\delta(5)_{\text{no action}} = \frac{1 - 0.72}{0.72} = 0.389 \text{ months}$$

**Delay reduction:**
$$0.389 - 0.163 = 0.226 \text{ months} \approx 6.8 \text{ days saved this month}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(5) = 0.583 + 0.163 = 0.746 \text{ months} \approx 22.4 \text{ days}$$

---

### Month 6: Action Plan Continues

**Assume performance stabilizes:**
- SPI remains at 0.72 (without action plan)
- Action plan still active (duration = 3 months)

**Effective SPI (same calculation as Month 5):**
$$\text{SPI}^{\text{eff}} = 0.86$$

**Calculate delay:**
$$\delta(6) = \frac{1 - 0.86}{0.86} = 0.163 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(6) = 0.746 + 0.163 = 0.909 \text{ months} \approx 27.3 \text{ days}$$

---

### Month 7: Action Plan Continues

**Same performance:**
$$\text{SPI}^{\text{eff}} = 0.86$$

**Calculate delay:**
$$\delta(7) = 0.163 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(7) = 0.909 + 0.163 = 1.072 \text{ months} \approx 32.2 \text{ days}$$

---

### Month 8: Action Plan Expires

**Action plan duration = 3 months (Months 5, 6, 7).**

**Month 8: Action plan no longer active.**

**Assume SPI improves slightly to 0.78 (some lasting benefit):**
$$\text{SPI}^{\text{eff}} = 0.78$$

**Calculate delay:**
$$\delta(8) = \frac{1 - 0.78}{0.78} = 0.282 \text{ months}$$

**Cumulative delay:**
$$\Delta D^{\text{recovery}}(8) = 1.072 + 0.282 = 1.354 \text{ months} \approx 40.6 \text{ days}$$

---

## 7. Summary Table: Complete Timeline

| Month | Planned % | Actual % | SPI | Action Plan? | SPI_eff | δ (months) | Cumulative Delay |
|-------|-----------|----------|-----|--------------|---------|------------|------------------|
| 1 | 8.33 | 8.33 | 1.00 | No | 1.00 | 0.000 | 0.000 |
| 2 | 16.67 | 15.00 | 0.90 | No | 0.90 | 0.111 | 0.111 |
| 3 | 25.00 | 21.00 | 0.84 | No | 0.84 | 0.190 | 0.301 |
| 4 | 33.33 | 26.00 | 0.78 | No | 0.78 | 0.282 | 0.583 |
| 5 | 41.67 | 30.00 | 0.72 | **YES** | 0.86 | 0.163 | 0.746 |
| 6 | 50.00 | 36.00 | 0.72 | **YES** | 0.86 | 0.163 | 0.909 |
| 7 | 58.33 | 42.00 | 0.72 | **YES** | 0.86 | 0.163 | 1.072 |
| 8 | 66.67 | 52.00 | 0.78 | No | 0.78 | 0.282 | 1.354 |

---

## 8. Key Insights from the Example

### Insight 1: Action Plans Slow Deterioration

**Without action plan (Months 5-7):**
- Monthly delay would be 0.389 months each
- Total delay over 3 months: $3 \times 0.389 = 1.167$ months

**With action plan (Months 5-7):**
- Monthly delay is 0.163 months each
- Total delay over 3 months: $3 \times 0.163 = 0.489$ months

**Delay prevented:**
$$1.167 - 0.489 = 0.678 \text{ months} \approx 20.3 \text{ days}$$

### Insight 2: Action Plans Don't Reverse Past Delays

**At Month 5 start:**
- Cumulative delay: 0.583 months

**At Month 7 end (after 3 months of action plan):**
- Cumulative delay: 1.072 months

**The action plan didn't eliminate the 0.583 months already accumulated. It only slowed the rate of new delay accumulation.**

### Insight 3: The Cost of Delay

**If this project has BAC = \$10M and overhead = \$200K/month:**

**Without action plan:**
- Total delay: ~2.5 months
- Overhead cost: $500K

**With action plan:**
- Total delay: ~1.5 months
- Overhead cost: $300K
- Action plan cost: $\kappa \times \text{BAC} \times \text{Gap} = 0.20 \times 10M \times 0.1167 = $233K$

**Net savings:**
$$500K - (300K + 233K) = -33K$$

**In this case, the action plan is marginally cost-negative, but it provides schedule certainty and reduces risk of further deterioration.**

---

## 9. Practice Problems

### Problem 1
**Given:** SPI = 0.65, $\eta = 0.40$

**Calculate:** $\text{SPI}^{\text{eff}}$ and $\delta$

### Problem 2
**Given:** SPI = 0.88, $\eta = 0.60$

**Calculate:** $\text{SPI}^{\text{eff}}$ and $\delta$

### Problem 3
**Given:** 
- Month 1: SPI = 0.95, no action plan
- Month 2: SPI = 0.82, no action plan
- Month 3: SPI = 0.70, action plan activates with $\eta = 0.50$

**Calculate:** Cumulative delay at end of Month 3

---

## 10. Solutions to Practice Problems

### Solution 1
**Step 1:** Gap = $1 - 0.65 = 0.35$

**Step 2:** Improvement = $0.40 \times 0.35 = 0.14$

**Step 3:** $\text{SPI}^{\text{eff}} = 0.65 + 0.14 = 0.79$

**Step 4:** $\delta = \frac{1 - 0.79}{0.79} = \frac{0.21}{0.79} = 0.266$ months

---

### Solution 2
**Step 1:** Gap = $1 - 0.88 = 0.12$

**Step 2:** Improvement = $0.60 \times 0.12 = 0.072$

**Step 3:** $\text{SPI}^{\text{eff}} = 0.88 + 0.072 = 0.952$

**Step 4:** $\delta = \frac{1 - 0.952}{0.952} = \frac{0.048}{0.952} = 0.050$ months

---

### Solution 3

**Month 1:**
- SPI = 0.95, no action plan
- $\delta(1) = \frac{1 - 0.95}{0.95} = 0.053$ months
- Cumulative: 0.053 months

**Month 2:**
- SPI = 0.82, no action plan
- $\delta(2) = \frac{1 - 0.82}{0.82} = 0.220$ months
- Cumulative: $0.053 + 0.220 = 0.273$ months

**Month 3:**
- SPI = 0.70, action plan with $\eta = 0.50$
- Gap = $1 - 0.70 = 0.30$
- Improvement = $0.50 \times 0.30 = 0.15$
- $\text{SPI}^{\text{eff}} = 0.70 + 0.15 = 0.85$
- $\delta(3) = \frac{1 - 0.85}{0.85} = 0.176$ months
- Cumulative: $0.273 + 0.176 = 0.449$ months $\approx$ **13.5 days**

---

## 11. Final Conceptual Summary

**The action plan model captures three realities:**

1. **Bounded rationality:** Organizations can't instantly fix problems ($\eta < 1$)

2. **Diminishing returns:** The worse your performance, the harder it is to improve (gap-based formula)

3. **Temporal limits:** Recovery efforts can't be sustained indefinitely (3-month duration)

**Mathematical elegance:**
- Simple formula: $\text{SPI}^{\text{eff}} = \text{SPI} + \eta(1 - \text{SPI})$
- Intuitive parameters: $\eta \in [0.3, 0.7]$ maps to organizational capability
- Direct conversion to delay: $\delta = \frac{1 - \text{SPI}^{\text{eff}}}{\text{SPI}^{\text{eff}}}$

**Practical application:**
- Trigger threshold: 10% progress gap
- Cost model: $\kappa \times \text{BAC} \times \text{Gap}$
- Decision framework: Compare action plan cost vs. delay cost

You now have all the tools to calculate action plan effects by hand and understand the underlying mechanics.
