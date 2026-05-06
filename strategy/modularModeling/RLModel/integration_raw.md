
## 5. Integration with RL Framework

### 5.1 State Representation

The payment model enriches the RL state with:

**Project-level state features:**
- **Milestone achievement flags**: $\{m_{i,k}\}_{k=1}^{K_i}$ where $m_{i,k} = \mathbb{1}_{\tau_i(t) \geq \tau_{i,k}}$
- **Payments received flags**: $\{p_{i,k}\}_{k=1}^{K_i}$ where $p_{i,k} = \mathbb{1}_{t \geq t_{i,k}^{\text{cash}}}$
- **Current working capital**: $\text{WC}_i(t)$
- **Remaining contract value**: $R_i^{\text{remaining}}(t) = R_i^{\text{total}} - \text{Cash}_i^{\text{in,cumulative}}(t)$
- **Retention held**: $\text{Retention}_i^{\text{held}}(t)$

**Portfolio-level state features:**
- **Total portfolio WC**: $\text{WC}^{\text{portfolio}}(t) = \sum_{i \in \text{Active}} \text{WC}_i(t)$
- **Total cash inflow (period)**: $\text{Cash}^{\text{in,total}}(t) = \sum_i \text{Cash}_i^{\text{in}}(t)$
- **Total outstanding receivables**: $\sum_i R_i^{\text{remaining}}(t)$
- **Number of pending payments**: $\sum_i \sum_k \mathbb{1}_{m_{i,k}=1, p_{i,k}=0}$
- **Total retention held**: $\sum_i \text{Retention}_i^{\text{held}}(t)$

---

### 5.2 Reward Signal

The payment model directly impacts the RL reward function:

**Base reward (net cash flow):**
$$r_t^{\text{base}} = \sum_i \text{Cash}_i^{\text{in}}(t) - \sum_i \text{Cost}_i(t)$$

**Working capital penalty:**
$$r_t^{\text{WC penalty}} = -\lambda \cdot \text{WC}^{\text{portfolio}}(t)$$

where $\lambda$ is the working capital cost coefficient (e.g., $\lambda = 0.0001$ for 10% annual cost).

**Total reward:**
$$r_t = r_t^{\text{base}} + r_t^{\text{WC penalty}}$$

**Rationale:**
- Penalizes high working capital to incentivize cash-efficient project selection
- Encourages portfolio composition that balances profitability with cash flow timing
- Reflects real-world cost of capital and financing constraints

---

### 5.3 Action Space Impact

Payment structure influences optimal actions:

**Project selection decisions:**
- **High advance projects** (IL, IH): Lower initial WC, attractive for cash-constrained portfolios
- **Low retention projects** (DL): Faster cash recovery, lower long-term WC
- **Low delay projects** (DL): More predictable cash flow, lower WC volatility

**Portfolio composition strategies:**
- **Front-loaded payment projects**: Reduce peak WC
- **Diversification across categories**: Hedge payment delay risk
- **Staggered project starts**: Smooth cash flow profile

---

### 5.4 Value Function Approximation

Payment model features for value function $V(s)$ or $Q(s,a)$:

**Input features:**
1. **Current portfolio WC**: $\text{WC}^{\text{portfolio}}(t)$
2. **Expected future cash inflows** (next 30/60/90 days):
   $$\mathbb{E}\left[\sum_{i,k: t < t_{i,k}^{\text{cash}} \leq t+\Delta t} P_{i,k}^{\text{actual}}\right]$$
3. **Payment delay risk exposure**:
   $$\sum_i \sum_{k: m_{i,k}=1, p_{i,k}=0} P_{i,k}^{\text{net}}$$
4. **Retention release schedule**:
   $$\sum_i \left(\text{Retention}_i^{\text{stage 1}} \cdot \mathbb{1}_{t < t_{\text{retention}_1}^{\text{cash}}} + \text{Retention}_i^{\text{stage 2}} \cdot \mathbb{1}_{t < t_{\text{retention}_2}^{\text{cash}}}\right)$$
5. **Category-specific payment risk**:
   - Fraction of portfolio in high-delay categories (IH, IL)
   - Fraction of portfolio with high default probability (DH, IH)

---
