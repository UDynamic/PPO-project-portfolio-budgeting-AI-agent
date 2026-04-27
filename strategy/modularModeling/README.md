# Modular Modeling Documentation Structure

## Directory Organization

## Status
- ✅ Completed: 2/12
- 🟨 Under review: 2/12
- ⬜ To be developed: 7/12
- ⬛ Deprecated: 1/12

### projectsPortfolioModel/
Portfolio composition and project-level characteristics
1. ✅ profitMarginsComposistions.md - Profit margin distributions by category
2. ✅ projectCounts&BACsDistributions.md - Portfolio size and BAC distributions
3. 🟨 projectSCurves.md - S-curve cashflow model (Beta distribution)
4. ⬛ (deprecated - merged into the projectSCurves) projectDurations.md - Duration distributions and start time staggering
5. 🟨 projectRevenuePlans.md - Revenue recognition and payment structures

### uncertaintyModel/
Stochastic performance and uncertainty modeling
6. 🟨 projectsPerformances.md - Contractor performance uncertainty (SPI/CPI)
7. ⬜ projectsRevenues.md - Payment delays and revenue uncertainty
8. ⬜ portfolioTemporalStructre.md - Temporal structure and seasonal effects

### instanceGeneratorModel/
Synthetic data generation and validation
9. ⬜ portfolioGenerator.md - Complete instance generation algorithm
10. ⬜ datasetValidation.md - Validation framework and sanity checks

### RLModel/
Reinforcement learning framework
11. ⬜ rewardFunction.md - Reward function design and components
12. ⬜ MDP.md - MDP formulation (state, action, transition, reward)

## Template
All documents follow the structure in `0. promptTemplate.md`

---
