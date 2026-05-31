# CHUNK 08
## Coverage
Section 4.5-4.6: Duration Model & Post-Intervention SPI Dynamics

## Dependency Notes
Uses BAC-duration correlation from Chunk 03. Implements persistence model from Chunk 06.

## Overlap Notes
References retention rate ρ and decay parameters from Chunk 06.

## Content

---

### 4.5 Project Duration Model

Project duration $D_i$ is modeled as a **lognormal random variable** with conditional mean depending on project size (BAC).

#### 4.5.1 Marginal Distribution

$$\ln(D_i) \sim \mathcal{N}(\mu_{\ln,i}, \sigma_{\ln})$$

where:
- $\mu_{\ln,i}$ = conditional mean (depends on $\text{BAC}_i$)
- $\sigma_{\ln} = 0.40$ = residual standard deviation (after conditioning on BAC)

**Unconditional statistics** (for reference):
- Median duration: $\exp(\mu_{\ln}) = 36$ months
- Mean duration: $\exp(\mu_{\ln} + \sigma_{\ln}^2/2) = 39$ months
- Coefficient of variation: $\sqrt{\exp(\sigma_{\ln}^2) - 1} = 0.42$

#### 4.5.2 BAC-Duration Correlation (Power Law)

The conditional mean follows a **power law scaling**:

$$E[D_i \mid \text{BAC}_i] = \gamma \cdot \text{BAC}_i^\delta$$

**Calibrated parameters** (from Merrow, 2011; AACE, 2020):
- $\gamma = 2.5$ (scaling constant)
- $\delta = 0.35$ (elasticity)

**Interpretation**:
- Doubling project size increases duration by $2^{0.35} = 1.27$ (27% increase)
- Sublinear scaling reflects **economies of scale** in execution (parallel work streams)

**Conditional mean in log-space**:
$$\mu_{\ln,i} = \ln(\gamma) + \delta \cdot \ln(\text{BAC}_i) - \frac{\sigma_{\ln}^2}{2}$$

**Example calibration**:
- Small project ($\text{BAC} = $100M$): $E[D] = 2.5 \times 100^{0.35} = 18$ months
- Mid-size project ($\text{BAC} = $500M$): $E[D] = 2.5 \times 500^{0.35} = 36$ months
- Large project ($\text{BAC} = $2B$): $E[D] = 2.5 \times 2000^{0.35} = 64$ months

#### 4.5.3 Sampling Algorithm

For each project $i$:
1. Sample $\text{BAC}_i$ from lognormal (module 01)
2. Compute conditional mean: $\mu_{\ln,i} = \ln(2.5) + 0.35 \ln(\text{BAC}_i) - 0.08$
3. Sample $\ln(D_i) \sim \mathcal{N}(\mu_{\ln,i}, 0.40)$
4. Transform: $D_i = \exp(\ln(D_i))$

**Correlation check**:
- Theoretical correlation: $\rho(\ln(\text{BAC}), \ln(D)) \approx 0.68$
- Matches empirical evidence (Merrow, 2011; Flyvbjerg et al., 2018)

---

### 4.6 Post-Intervention SPI Dynamics

This section formalizes the mathematical model for SPI evolution after an action plan intervention.

#### 4.6.1 Model Specification

**Phase 1: Action Plan Execution** ($0 \leq t \leq T_{\text{action}}$)

$$\text{SPI}(t) = \text{SPI}_{\text{baseline}} + \eta \cdot \left(1 - e^{-\lambda_{\text{ramp}} t}\right)$$

where:
- $\text{SPI}_{\text{baseline}}$ = pre-intervention SPI (typically 0.80-0.90)
- $\eta = 0.18$ = maximum effectiveness (from Chunk 04)
- $\lambda_{\text{ramp}} = 0.50$ per month = ramp-up rate (fast improvement)
- $T_{\text{action}} = 3$ months = action plan duration (from Chunk 05)

**Interpretation**: SPI improves exponentially toward $\text{SPI}_{\text{baseline}} + \eta$, reaching 95% of peak improvement after $-\ln(0.05)/0.50 = 6$ months (but action plan ends at 3 months).

**Phase 2: Post-Action Decay** ($t > T_{\text{action}}$)

$$\text{SPI}(t) = \text{SPI}_{\text{peak}} - (1-\rho) \cdot \eta \cdot \left(1 - e^{-\lambda_{\text{decay}} (t - T_{\text{action}})}\right)$$

where:
- $\text{SPI}_{\text{peak}} = \text{SPI}_{\text{baseline}} + \eta \cdot (1 - e^{-\lambda_{\text{ramp}} T_{\text{action}}})$ = SPI at end of action plan
- $\rho = 0.45$ = retention rate (from Chunk 06)
- $\lambda_{\text{decay}} = 0.12$ per month = decay rate (slow erosion)

**Asymptotic behavior**:
$$\lim_{t \to \infty} \text{SPI}(t) = \text{SPI}_{\text{baseline}} + \rho \cdot \eta$$

**Interpretation**: Long-term SPI stabilizes at baseline + 45% of peak improvement.

#### 4.6.2 Parameter Relationships and Constraints

**Consistency conditions**:

1. **Continuity at $t = T_{\text{action}}$**:
   - Left limit: $\text{SPI}(T_{\text{action}}^-) = \text{SPI}_{\text{baseline}} + \eta \cdot (1 - e^{-\lambda_{\text{ramp}} T_{\text{action}}})$
   - Right limit: $\text{SPI}(T_{\text{action}}^+) = \text{SPI}_{\text{peak}}$
   - Ensured by definition of $\text{SPI}_{\text{peak}}$

2. **Monotonicity during action plan**:
   - $\frac{d\text{SPI}}{dt} = \eta \lambda_{\text{ramp}} e^{-\lambda_{\text{ramp}} t} > 0$ for $t < T_{\text{action}}$

3. **Monotonicity during decay**:
   - $\frac{d\text{SPI}}{dt} = -(1-\rho) \eta \lambda_{\text{decay}} e^{-\lambda_{\text{decay}} (t - T_{\text{action}})} < 0$ for $t > T_{\text{action}}$

4. **Bounded SPI**:
   - $\text{SPI}(t) \leq 1.0$ for all $t$ (cannot exceed perfect schedule performance)
   - If $\text{SPI}_{\text{baseline}} + \eta > 1.0$, cap at 1.0

**Parameter constraints**:
- $0 < \eta < 0.30$ (effectiveness bounded by empirical evidence)
- $0 < \rho < 1$ (retention rate is a fraction)
- $\lambda_{\text{ramp}} > \lambda_{\text{decay}}$ (improvement faster than decay)
- $T_{\text{action}} \in [2, 4]$ months (practical range)

#### 4.6.3 Numerical Example

**Scenario**: Project with $\text{SPI}_{\text{baseline}} = 0.82$

| Time (months) | Phase | SPI | Calculation |
|---------------|-------|-----|-------------|
| 0 | Baseline | 0.82 | Given |
| 1 | Action plan | 0.89 | $0.82 + 0.18(1 - e^{-0.5})$ |
| 2 | Action plan | 0.93 | $0.82 + 0.18(1 - e^{-1.0})$ |
| 3 | Action plan end | 0.96 | $0.82 + 0.18(1 - e^{-1.5})$ |
| 6 | Decay | 0.92 | $0.96 - 0.55 \times 0.18(1 - e^{-0.36})$ |
| 12 | Decay | 0.90 | $0.96 - 0.55 \times 0.18(1 - e^{-1.08})$ |
| ∞ | Stabilized | 0.90 | $0.82 + 0.45 \times 0.18$ |

**Interpretation**: Action plan improves SPI from 0.82 to 0.96, then decays to long-term level of 0.90 (8% permanent improvement).

---

**End of Chunk 08**

**Next Chunk Preview**: Chunk 09 covers parameter calibration tables (Section 5), consolidating all model parameters with distributions and sensitivity ranges.
