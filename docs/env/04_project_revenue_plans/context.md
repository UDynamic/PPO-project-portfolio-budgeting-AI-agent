# Context: Modular Portfolio Modeling Framework

## Purpose
This document provides compressed context for AI agents working on the **Project Revenue Plans** module (4.projectRevenuePlans). It summarizes completed work in modules 01-03 to enable efficient continuation without re-reading all source files.

---

## Project Overview
**Goal:** Build a synthetic EPC contractor portfolio model for RL-based project portfolio optimization.

**Architecture:** Modular design with independent, literature-calibrated components:
1. Portfolio Size & BAC Distribution (01)
2. Profit Margin Composition (02)
3. Project S-Curves & Duration Dynamics (03)
4. **Project Revenue Plans (04)** ← Current module

**Methodology:** Each module follows strict structure:
- Literature review → Model formulation → Parameter calibration → Validation
- All parameters derived from empirical studies (Merrow 2011, Flyvbjerg et al., Khanzadi et al.)
- Probabilistic distributions with P10/P50/P90 percentiles for Monte Carlo simulation

---

## Module 01: Portfolio Size & BAC Distribution

**Key Outputs:**
- **Portfolio size N:** Discrete Uniform(8, 25) projects per portfolio
  - Calibrated from ENR Top 400 Contractors data
  - Typical EPC contractor: 15-18 active projects

- **BAC distribution:** Lognormal per project category
  - Domestic: μ_ln = 17.27, σ_ln = 0.85 → Median $31.6M, Mean $35.8M
  - International: μ_ln = 18.23, σ_ln = 0.92 → Median $83.7M, Mean $96.4M
  - Range: $10M - $500M (capped at P99)

- **Portfolio composition:** 60% domestic, 40% international (Khanzadi et al. 2018)

**Critical Assumption:** BAC and profit margin are **independent** (validated in Module 02, chunk 05).

---

## Module 02: Profit Margin Composition

**Key Outputs:**
- **Domestic margins:** Triangular(min=4%, mode=8%, max=14%)
  - Mean: 8.67%, Std: 2.04%
  - Reflects competitive local markets

- **International margins:** Triangular(min=8%, mode=12%, max=18%)
  - Mean: 12.67%, Std: 2.04%
  - Higher margins compensate for risk/complexity

- **Margin definition:** π_i = (Revenue_i - Cost_i) / Revenue_i × 100%

**Critical Finding:** No empirical correlation between BAC and margin (ρ ≈ 0.05, p > 0.4 across 5 studies). Allows independent sampling.

**Algorithm:**
1. Sample N from Uniform(8, 25)
2. Assign categories: 60% domestic, 40% international
3. Sample BAC_i from category-specific Lognormal
4. Sample π_i from category-specific Triangular
5. Compute Revenue_i = BAC_i / (1 - π_i)

---

## Module 03: Project S-Curves & Duration Dynamics

**Key Outputs:**

### S-Curve Model (Beta CDF)
- **Cumulative spend:** C(τ) = BAC × I_τ(α, β) where τ ∈ [0,1] is normalized progress
- **Parameters:** α = 2.5, β = 2.0 (calibrated from Barraza & Bueno 2007, Cioffi 2005)
- **Validation:** Spend at τ=0.25: 16.1% (lit: 15-20%), τ=0.50: 50% (lit: 45-55%), τ=0.75: 84% (lit: 80-88%)

### Duration Model
- **Baseline durations:** Gamma distributions
  - Domestic: k=5.0, θ=6.0 → Mean 30 months, Std 13.4 months
  - International: k=4.5, θ=10.7 → Mean 48 months, Std 22.6 months

- **Start time distribution:** Budget cycle pattern (70% weight) + Uniform (30% opportunistic)
  - Q1: 42.5% of starts (January peak: 18-20%)
  - Q2: 17.5%, Q3: 27.5% (July secondary peak), Q4: 12.5% (December trough)

### Action Plan Dynamics (Schedule Recovery)
**Trigger:** SPI < 0.85 for 2 consecutive months

**Effectiveness (3-month intervention):**
- η_SPI: Beta(α=8.2, β=8.2) → Mean 0.50, P10=0.32, P90=0.68
- Interpretation: Action plan closes 50% of schedule gap during intervention

**Post-Action Decay (exponential):**
- SPI(t) = SPI_eq + (1-ρ)·Δ_peak·exp(-λ·t)
- Retention rate ρ: Beta(α=5.8, β=11.0) → Mean 0.35, P10=0.22, P90=0.50
- Decay rate λ: Lognormal(μ_ln=-1.71, σ_ln=0.31) → Median 0.18/month, half-life 3.85 months
- Stabilization: 6-12 months post-intervention

**Key Insight:** Action plans deliver 5-10 percentage point **permanent** SPI improvement (not full recovery). Only 35% of peak gains persist long-term.

### Financial Justification (BCR Framework)
- Cost: κ·BAC·(1-SPI) where κ ~ Triangular(0.12, 0.18, 0.28)
- Benefits: Avoided LD penalties + Avoided cost overrun + Avoided opportunity cost
- Decision rule: Approve if BCR > 1.5-2.0

### Master Parameter Table (Chunk 09)
All 23 parameters documented with:
- Distribution type (Beta, Lognormal, Triangular, Gamma, Uniform)
- Calibration source (Merrow 2011, Love et al. 2016, Barraza & Bueno 2007, etc.)
- P10/P50/P90 percentiles for scenario planning
- Units and physical interpretation

---

## Implementation Notes

### Python Code Structure
- `generate_scurve_profile()`: Beta CDF via scipy.special.betainc
- `sample_baseline_duration()`: Gamma sampling per category
- `sample_start_time_budget_cycle()`: Quarterly pattern with monthly sub-distribution
- `update_duration_dynamics()`: SPI-based delay accumulation

### Validation Framework
- **Analytical checks:** Budget conservation (<0.01% error), non-negativity, monotonicity
- **Literature benchmarks:** All predictions within empirical ranges
- **Edge case testing:** Zero duration, perfect performance (SPI=CPI=1.0), extreme underperformance

---

## Critical Dependencies for Module 04 (Revenue Plans)

**From Module 01:**
- N (portfolio size)
- BAC_i per project
- Category assignments (domestic/international)

**From Module 02:**
- π_i (profit margins) per project
- Revenue_i = BAC_i / (1 - π_i)

**From Module 03:**
- T_i^start (start times)
- D_i^baseline (durations)
- S-curve shape (α=2.5, β=2.0)
- SPI dynamics (action plans, decay)

**Expected Module 04 Scope:**
- Revenue recognition timing (milestone-based vs. PoC)
- Payment terms (advance, progress, retention, final)
- Revenue S-curve alignment with cost S-curve
- Cash inflow modeling (lagged from revenue recognition)
- Working capital dynamics (receivables, retainage)

---

## Key Assumptions Across All Modules

1. **Independence:** BAC ⊥ Margin, Projects are independent (no resource contention)
2. **Time-invariant margins:** Set at project start, no erosion modeling
3. **Homogeneous S-curve shape:** All projects use α=2.5, β=2.0 (industry-calibrated)
4. **Single action plan per project:** No multi-intervention dynamics
5. **EPC oil & gas focus:** Parameters calibrated for process industries, not construction/IT
6. **No external shocks:** Stable environment (no pandemic, regulatory disruption modeling)

---

## Chunking Methodology Applied

All modules follow semantic chunking per `chunkingPrompt.md`:
- **Target:** 500-800 words per chunk, 2-4 subsections max
- **Never:** Place entire section in one chunk (always subdivide large sections)
- **Preserve:** Theorem-proof pairs, equation blocks, derivation chains, notation scope
- **Metadata:** Each chunk has coverage, dependencies, overlap notes
- **Format:** Markdown with LaTeX math, Python code blocks, tables

**Example:** Module 03 (projectSCurves.md, 29,674 words) → 15 chunks
- Chunk 01: Scope Definition
- Chunks 02-03: Literature review
- Chunks 04-06: Action plan models
- Chunk 07: Mathematical S-curve
- Chunk 08: Duration dynamics
- Chunk 09: Master parameter table
- Chunk 10: Financial justification
- Chunks 11-12: Sensitivity & Monte Carlo
- Chunks 13-15: Implementation, validation, references

---

## Next Steps for Module 04

1. **Literature review:** Revenue recognition standards (IFRS 15, ASC 606), EPC payment terms
2. **Model formulation:** Revenue S-curve vs. cost S-curve timing differences
3. **Parameter calibration:** Advance payment %, progress billing frequency, retention %
4. **Integration:** Link Revenue_i and cost S-curve to generate cash inflow profiles
5. **Validation:** Ensure revenue timing aligns with industry practice (30-60 day lag typical)

**Critical question to resolve:** Does revenue follow cost S-curve exactly (PoC method) or milestone-based with step functions?

---

**Document Status:** v1.0 | Last Updated: 2025-05-13 | Covers Modules 01-03 (Complete)
