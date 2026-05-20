## Custom PPO Environment Requirements for Portfolio Budgeting

Based on the documentation, here's what you need to implement:

### 1. Core Environment Components (Gymnasium/Gym Interface)

State Space - Must include:

- Portfolio-level features:
    - Current timestep t (within rolling horizon H=12)
    - Total available budget at time t
    - Portfolio-level working capital WC_portfolio(t)
    - Total cash inflow/outflow for period
    - Number of active projects
    - Total outstanding receivables
    - Total retention held
- Per-project features (for N projects, with masking for inactive):
    - Project BAC (Budget at Completion)
    - Project profit margin (from truncated normal distributions)
    - Current SPI (Schedule Performance Index)
    - Current CPI (Cost Performance Index)
    - Cumulative spending S_i(t) (Beta CDF S-curve)
    - Project duration D_i
    - Time since project start
    - Remaining contract value
    - Working capital per project WC_i(t)
    - Milestone achievement flags
    - Payment received flags
    - Project category (DS/DC/IC/IP - Domestic/International, Standard/Complex)
    - Action plan status (if intervention active)
    - Payment delay parameters

Action Space:

- Continuous allocation vector for each active project
- Budget allocation amounts (subject to constraints)
- Optional: binary intervention decisions (action plans for underperforming projects)

Reward Function:

- Base reward: Net cash flow = Σ Cash_in(t) - Σ Cost(t)
- Working capital penalty: -λ * WC_portfolio(t) (λ ≈ 0.0001 for 10% annual cost)
- Portfolio NPV maximization
- Penalties for budget violations
- Penalties for project failures

———

### 2. Project Generation Module

Portfolio Size:

- Discrete uniform: N ~ U(8, 25) projects per portfolio

Project BAC (Budget at Completion):

- Truncated lognormal distribution
- Base: μ_ln = 5.32, σ_ln = 0.75 → Median ≈ $148M
- Upstream: μ_ln = 5.63, σ_ln = 0.80 → Median ≈ $278M
- Downstream: μ_ln = 5.01, σ_ln = 0.70 → Median ≈ $150M
- Bounds: [$50M, $2B]

Profit Margins (by category):

- DS (Domestic Standard): μ=3.5%, σ=1.8%, bounds=[-2%, 10%], weight=42%
- DC (Domestic Complex): μ=5.5%, σ=2.5%, bounds=[0%, 14%], weight=18%
- IC (International Competitive): μ=2.5%, σ=2.2%, bounds=[-4%, 9%], weight=26%
- IP (International Premium): μ=7.0%, σ=3.0%, bounds=[1%, 16%], weight=14%

Project Duration:

- Correlated with BAC (larger projects take longer)
- Typical range: 12-36 months

Project Start Times:

- Staggered: Uniform distribution over planning horizon
- Allows for rolling portfolio dynamics

———

### 3. S-Curve Cashflow Model (Beta CDF)

Cumulative spending:

S_i(t) = BAC_i * I_x(α, β)
where x = (t - t_start) / D_i

Parameters:

- α = 2.0 (moderate front-loading)
- β = 2.5 (gradual tail-off)
- Peak spending at ~29% of project lifecycle
- Spending profile: 18% @ 25%, 52% @ 50%, 84% @ 75%

Incremental spending rate:

dS_i/dt = (BAC_i / D_i) * Beta(x; α, β)

———

### 4. Payment/Revenue Model

Milestone-based payments:

- Advance payment (Milestone 0): 5-20% depending on category
- Progress milestones (1 to N-1): Tied to completion percentage
- Final payment (Milestone N): Includes retention release
- Retention: 5-10% held until project completion

Payment delays (Log-normal):

- Domestic Low-risk: μ_ln=0.5, σ_ln=0.4 → ~1-2 months
- Domestic High-risk: μ_ln=1.0, σ_ln=0.5 → ~2-4 months
- International Low-risk: μ_ln=1.2, σ_ln=0.6 → ~3-5 months
- International High-risk: μ_ln=1.5, σ_ln=0.7 → ~4-7 months

Working capital dynamics:

WC_i(t) = Cumulative_Cost_i(t) - Cumulative_Cash_Received_i(t)

———

### 5. Uncertainty & Performance Dynamics

SPI (Schedule Performance Index) dynamics:

- Stochastic evolution with drift
- Can trigger action plans when SPI < threshold (e.g., 0.85)
- Action plan effectiveness: η = 0.18 (18% max SPI improvement)
- Action plan duration: 3 months

CPI (Cost Performance Index):

- Linked to profit margin realization
- Stochastic cost growth

Year-end effects:

- Seasonal variance parameter λ_S(t)
- Budget pressure at fiscal year boundaries

———

### 6. Rolling Horizon Architecture

Key mechanism:

- Fixed episode length: H = 12 periods (months)
- At each timestep t:
    1. Observe current state
    2. Plan allocation for next H periods
    3. Execute only allocation for period t
    4. Advance to t+1 and re-plan
- Handles variable portfolio horizons (12, 24, 36+ months)
- Uses masking for projects starting/ending at different times

———

### 7. Constraints & Feasibility

Budget constraints:

- Total allocation ≤ Available budget at time t
- Per-project credit limits based on BAC and category

Working capital constraints:

- Soft penalty for high WC (not hard constraint)
- Credit limit per project: Credit_i = k * BAC_i where k varies by category

Project masking:

- Inactive projects (not yet started or completed) masked in state/action
- Dynamic portfolio composition over time

———

### 8. Implementation Requirements

Must implement:

1. PortfolioEnv(gym.Env) class with:
    - reset() → initial state
    - step(action) → next_state, reward, done, truncated, info
    - render() → visualization (optional)
2. Project generator with all distributions above
3. S-curve simulator (Beta CDF implementation)
4. Payment/revenue simulator with delays and retention
5. Working capital tracker
6. SPI/CPI dynamics with action plan mechanism
7. Reward calculator with multi-objective components
8. State normalizer (critical for PPO training)
9. Action masking for inactive projects
10. Episode termination logic (rolling horizon)

———

### 9. Validation & Baselines

Need to compare against:

- Greedy heuristic: Allocate to highest NPV projects first
- Rolling MIP: Deterministic optimization at each timestep
- Stochastic programming: Scenario-based optimization
- Perfect information upper bound: Oracle with full knowledge

Metrics:

- Expected portfolio NPV
- Budget utilization rate
- Working capital efficiency
- Risk-adjusted return
- Computational time (RL should be ~10,000× faster than MIP)

Simplifications (for Q1 paper):

- Aggregated cashflow (not resource-level)
- Fixed project scope (no scope changes)
- No inter-project dependencies
- Single contractor per project
- No portfolio-level constraints beyond budget

Future extensions:

- Inter-project resource conflicts
- Portfolio-level risk constraints

This is a comprehensive but tractable environment for your Q1 paper. The key innovation is the rolling horizon + RL
combination that handles uncertainty better than classical OR methods while maintaining computational feasibility.