# CHUNK 15
## Coverage
Section 12: Appendices + Section 13: References (Complete Bibliography)

## Dependency Notes
Final chunk. Provides distribution fitting methodology and complete reference list for all prior chunks.

## Overlap Notes
None. Terminal chunk.

## Content

---

## 12. Appendices

### 12.1 Distribution Fitting Methodology

#### 12.1.1 Beta Distribution Parameter Estimation

**Method of moments:**

Given empirical mean $\mu$ and standard deviation $\sigma$:

$$\alpha = \mu \left( \frac{\mu(1-\mu)}{\sigma^2} - 1 \right)$$

$$\beta = (1 - \mu) \left( \frac{\mu(1-\mu)}{\sigma^2} - 1 \right)$$

**Validation:**
- Check that $\alpha > 0$ and $\beta > 0$ (required for valid Beta distribution)
- Verify that theoretical mean and variance match empirical values:
  - $E[X] = \frac{\alpha}{\alpha + \beta}$
  - $\text{Var}[X] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$

---

#### 12.1.2 Lognormal Distribution Parameter Estimation

**Method of moments:**

Given empirical median $\tilde{x}$ and geometric standard deviation $\sigma_g$:

$$\mu_{\ln} = \ln(\tilde{x})$$

$$\sigma_{\ln} = \ln(\sigma_g)$$

**Alternative (from mean and standard deviation):**

Given empirical mean $\mu$ and standard deviation $\sigma$:

$$\sigma_{\ln} = \sqrt{\ln\left(1 + \frac{\sigma^2}{\mu^2}\right)}$$

$$\mu_{\ln} = \ln(\mu) - \frac{\sigma_{\ln}^2}{2}$$

**Validation:**
- Verify that theoretical median matches empirical: $\tilde{x}_{\text{theoretical}} = e^{\mu_{\ln}}$
- Check that 95% confidence interval is reasonable: $[e^{\mu_{\ln} - 1.96\sigma_{\ln}}, e^{\mu_{\ln} + 1.96\sigma_{\ln}}]$

---

#### 12.1.3 Goodness-of-Fit Testing

**Kolmogorov-Smirnov (K-S) test:**

For each fitted distribution, calculate:

$$D = \max_i |F_{\text{empirical}}(x_i) - F_{\text{theoretical}}(x_i)|$$

Where:
- $F_{\text{empirical}}(x_i) = \frac{i}{n}$ (empirical cumulative distribution)
- $F_{\text{theoretical}}(x_i)$ = theoretical CDF evaluated at $x_i$

**Acceptance criterion:**
- $D < D_{\text{critical}}$ at 5% significance level
- For $n = 87$ (Merrow sample size): $D_{\text{critical}} = \frac{1.36}{\sqrt{87}} = 0.146$

**Chi-square test:**

Divide range into $k$ bins and calculate:

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

Where:
- $O_i$ = observed frequency in bin $i$
- $E_i$ = expected frequency based on theoretical distribution

**Acceptance criterion:**
- $\chi^2 < \chi^2_{\text{critical}}$ with $k-3$ degrees of freedom (for 2-parameter distributions)

---

### 12.2 Summary and Key Takeaways

**Core findings:**

1. **Action plans stabilize, not reverse, schedule delays**
   - Peak SPI improvement: 15-20 percentage points (during 3-month intervention)
   - Permanent SPI improvement: 5-8 percentage points (long-term)
   - Retention rate: 25-45% of peak gains persist (median 35%)

2. **Post-intervention decay follows exponential pattern**
   - Half-life of transient gains: 3-4 months for EPC projects
   - Stabilization period: 6-12 months post-intervention
   - Decay rate: 0.10-0.25 per month (median 0.18)

3. **Organizational capability is the dominant factor**
   - Action plan effectiveness ($\eta$) has largest impact on outcomes
   - Retention rate ($\rho$) determines long-term success
   - Projects with mature PMOs retain 2× more gains than weak organizations

4. **Probabilistic modeling is essential**
   - Deterministic estimates underestimate uncertainty
   - Monte Carlo simulation reveals 80% confidence interval: [0.72, 0.78] for equilibrium SPI
   - Only 1% chance of achieving near-perfect recovery (SPI ≥ 0.85)

**Practical recommendations:**

1. **Set realistic expectations:** Action plans deliver 5-10 percentage point permanent SPI improvement, not full recovery
2. **Invest in institutionalization:** Embed improvements into standard procedures to maximize retention rate
3. **Monitor post-intervention performance:** Track decay rate monthly to detect rapid reversion early
4. **Use probabilistic planning:** Incorporate parameter uncertainty into schedule risk analysis
5. **Assess organizational capability:** Adjust parameter values based on PMO maturity and governance strength

**Model applicability:**

- **Primary use case:** EPC oil & gas projects with baseline durations 18-60 months
- **Calibration required for:** Other industries (construction, infrastructure, IT)
- **Not applicable to:** Projects with <12 month duration or >$5B BAC (megaprojects)

---

## 13. References

**Primary empirical sources:**

- Abdel-Hamid, T., & Madnick, S. E. (1991). *Software Project Dynamics: An Integrated Approach*. Prentice Hall.

- Barraza, G. A., & Bueno, R. A. (2007). "Probabilistic control of project performance using control limit curves." *Journal of Construction Engineering and Management*, 133(12), 957-965.

- Cioffi, D. F. (2005). "A tool for managing projects: An analytic parameterization of the S-curve." *International Journal of Project Management*, 23(3), 215-222.

- Flyvbjerg, B., Holm, M. S., & Buhl, S. (2002). "Underestimating costs in public works projects: Error or lie?" *Journal of the American Planning Association*, 68(3), 279-295.

- Flyvbjerg, B., Holm, M. S., & Buhl, S. (2003). "How common and how large are cost overruns in transport infrastructure projects?" *Transport Reviews*, 23(1), 71-88.

- Flyvbjerg, B. (2014). "What you should know about megaprojects and why: An overview." *Project Management Journal*, 45(2), 6-19.

- Flyvbjerg, B., Ansar, A., Budzier, A., Buhl, S., Cantarelli, C., Garbuio, M., ... & van Wee, B. (2018). "Five things you should know about cost overrun." *Transportation Research Part A: Policy and Practice*, 118, 174-190.

- Hanna, A. S., Taylor, C. S., & Sullivan, K. T. (2005). "Impact of extended overtime on construction labor productivity." *Journal of Construction Engineering and Management*, 131(6), 734-739.

- Keil, M., Depledge, G., & Rai, A. (2000). "Escalation: The role of problem recognition and cognitive bias." *Decision Sciences*, 31(2), 455-479.

- Kenley, R., & Wilson, O. D. (1986). "A construction project cash flow model—an idiographic approach." *Construction Management and Economics*, 4(3), 213-232.

- Khanzadi, M., Nasirzadeh, F., & Alipour, M. (2018). "Integrating project portfolio selection and scheduling under uncertainty." *Journal of Construction Engineering and Management*, 144(2), 04017106.

- Kutsch, E., Browning, T. R., & Hall, M. (2015). "Bridging the risk gap: The failure of risk management in information systems projects." *Research-Technology Management*, 57(2), 26-32.

- Lipke, W. (2003). "Schedule is different." *The Measurable News*, Summer 2003, 31-34.

- Love, P. E., Edwards, D. J., & Irani, Z. (2012). "Moving beyond optimism bias and strategic misrepresentation: An explanation for social infrastructure project cost overruns." *IEEE Transactions on Engineering Management*, 59(4), 560-571.

- Love, P. E., Sing, C. P., Ika, L. A., & Newton, S. (2016). "The cost performance of transportation infrastructure projects: The fallacy of the Planning Fallacy account." *Transportation Research Part A: Policy and Practice*, 122, 1-20.

- Merrow, E. W. (2011). *Industrial Megaprojects: Concepts, Strategies, and Practices for Success*. Wiley.

- Miskawi, Z. (1989). "An S-curve equation for project control." *Construction Management and Economics*, 7(2), 115-124.

**Statistical methodology references:**

- AACE International (2020). *Cost Estimate Classification System – As Applied in Engineering, Procurement, and Construction for the Process Industries*. Recommended Practice No. 18R-97.

- Limpert, E., Stahel, W. A., & Abbt, M. (2001). "Log-normal distributions across the sciences: Keys and clues." *BioScience*, 51(5), 341-352.

- Vose, D. (2008). *Risk Analysis: A Quantitative Guide* (3rd ed.). Wiley.

**Earned Value Management references:**

- Fleming, Q. W., & Koppelman, J. M. (2016). *Earned Value Project Management* (4th ed.). Project Management Institute.

- Christensen, D. S., & Heise, S. R. (1993). "Cost performance index stability." *National Contract Management Journal*, 25(1), 7-15.

- Project Management Institute (PMI). (2019). *Practice Standard for Earned Value Management* (2nd ed.). PMI.

- Lipke, W. (2009). "Schedule is different." *The Measurable News*, Summer 2009, 31-34.

- Vanhoucke, M. (2012). "Measuring the efficiency of project control using fictitious and empirical project data." *International Journal of Project Management*, 30(2), 252-263.

- Kim, E., Wells, W. G., & Duffey, M. R. (2003). "A model for effective implementation of Earned Value Management methodology." *International Journal of Project Management*, 21(5), 375-382.

---

**End of Document**

**Chunking Complete:** 15 chunks total covering the complete Project S-Curve Cashflow Model & Duration Dynamics document.
