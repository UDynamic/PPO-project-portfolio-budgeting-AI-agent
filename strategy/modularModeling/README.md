# Modular Modeling Documentation Structure

## Directory Organization

## Status
- ✅ Completed: 2/12
- 🟨 Under review
- ⬜ To be developed: 10/12

### projectsPortfolioModel/
Portfolio composition and project-level characteristics
1. ✅ profitMarginsComposistions.md - Profit margin distributions by category
2. ✅ projectCounts&BACsDistributions.md - Portfolio size and BAC distributions
3. 🟨 projectSCurves.md - S-curve cashflow model (Beta distribution)
4. ⬜ projectDurations.md - Duration distributions and start time staggering
5. ⬜ projectRevenuePlans.md - Revenue recognition and payment structures

### uncertaintyModel/
Stochastic performance and uncertainty modeling
6. ⬜ projectsPerformances.md - Contractor performance uncertainty (SPI/CPI)
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

### templatePrompt refinement:

```
I loved your output.
excellent job on the literature review first document.

bring in to the template everything detail neccessary for offline (like online chatbots new converation) quality persistence and context maintenance.

now I want you to update the root/strategy/modularModeling/0. promptTemplate.md for better quality responce
generation such as your recent projectScurves.md.

read all the documents and files mentioned for better context.

---

your previous instruction was:
"ok go on and write the /root/strategy/modularModeling/projectsPortfolioModel/projectSCurves.md.

using root/strategy/scope.md and files next to it.
like strategy.md and literatureCalibratedSyntheticData.md.

use root/strategy/modularModeling/0. promptTemplate.md as your instruction."

```