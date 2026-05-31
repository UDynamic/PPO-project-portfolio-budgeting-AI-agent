# Chunk 04: Category-Specific Calibration - Domestic Categories

**Source**: `deprecated/modularModeling/projectsPortfolioModel/projectRevenuePlans.md`  
**Lines**: 357-392  
**Section**: 1.2.1-1.2.2  
**Dependencies**: Chunks 01-03 (literature foundation)  
**Next**: Chunk 05 (International categories calibration)

---

## 1.2 Category-Specific Calibration Framework

### 1.2.1 Domestic Low-Risk (DL) - Government Clients

**Client characteristics**:
- Iranian government agencies and National Iranian Oil Company (NIOC)
- Structured procurement, bureaucratic payment processes
- High payment reliability but longer processing times

**Literature calibration**:
- **Advance payment**: Khanzadi et al. (2018) reports 68% for Iranian government projects
  - Adjusted to 25% for "low-risk" subset (established contractors, routine projects)
  - Percentage: 8% (lower end of Park et al. 2005 range for government)
- **Milestone count**: Elazouni & Gab-Allah (2004) reports 8-12 for government
  - Calibrated to 7-9 (lower end for domestic, less complex projects)
- **Front-loading**: Park et al. (2005) reports 35-40% in first 30% for Middle East
  - Calibrated λ=0.15 (mild front-loading for government)
- **Payment delay**: Tran & Carmichael (2012) reports 75 days mean for government
  - Calibrated to 75 days (2.5 months)
- **Retention**: FIDIC standard 5%, Boussabaine & Elhag (1999) mean 5.2%
  - Calibrated to 5%

### 1.2.2 Domestic High-Risk (DH) - Government Clients

**Client characteristics**:
- Same government clients but higher project complexity/risk
- More structured oversight, additional milestone checkpoints

**Literature calibration**:
- **Advance payment**: Higher probability (30%) due to mobilization needs
  - Percentage: 10% (Park et al. 2005 mean for government)
- **Milestone count**: 8-10 (upper end of Elazouni & Gab-Allah 2004 government range)
- **Front-loading**: λ=0.20 (moderate, reflecting risk mitigation)
- **Payment delay**: 75 days (same as DL, government bureaucracy)
- **Retention**: 5% (standard government rate)

---

## Key Takeaways

**Domestic Low-Risk (DL)**:
- Conservative advance payment (25% probability, 8% amount)
- Moderate milestone count (7-9)
- Mild front-loading (λ=0.15)
- Standard government payment delay (75 days)
- Standard retention (5%)

**Domestic High-Risk (DH)**:
- Higher advance payment probability (30%, 10% amount)
- More milestones for oversight (8-10)
- Moderate front-loading (λ=0.20)
- Same payment delay as DL (75 days)
- Same retention as DL (5%)

**Key Difference**: DH has higher advance payment and more milestones to manage complexity, but payment delays and retention remain consistent with government client characteristics.

---

**End of Chunk 04**
