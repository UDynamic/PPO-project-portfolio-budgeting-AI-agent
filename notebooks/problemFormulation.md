# Problem formulation
This is a Brand new and fundamentally revised framework on project portfolio problem definition and formulation.

This represented framework shows great capability in addressing almost all aspects of budgetting decisions for project portfolio. 

---
### parameters
for parameters and valuables of type list or array, we will use square bracket notation for accessing or assigning to specific index. 

```python
a: List[int]  # list of integers
tenth_element = a[9]  # 10th element (index starts at 0)
```


#### Project Parameters

|Type|name|definition|
|---|---|---|
|cost|bac|**budget at completion** for the project. Total contractual cost for the project|
|General|d|**duration** of the project statet in number of periods |
|General|dd|current **data date** or **period** of the project. e.g. a project with duration of 12 has 12 periods and analytical data date.|
|cost|l_sc|**List** of comulative **S-Curve** values untill project is finished. It's the cost schedule for the project from the start untill the budget at completion(bac) </br></br>- $\begin{cases} \text{l\_sc}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_sc}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>- S-curve values are estimated at the contract agreement according to the project plan. |
|cost|pl_sc|**periodic List** of the **S-Curves** of the project. </br></br>- $\begin{cases} \text{pl\_sc}[n] = \text{l\_sc}[n] - \text{l\_sc}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_sc}[n]} = \text{bac} \end{cases}$ |
|revenue|im|**Inflow Model** will specify the distribution of payments according to contract and directly influences on the payment schedule. <br><br>- **typical inflow models**: <br>&nbsp; (1) Lump-sum: fully advanced or fully final payment. <br>&nbsp; (2) Milestone-based: In milestone-based contracts, there’s almost always an advance payment (15–30%) and a final retention/delivery payment (≥15%), with the rest distributed among intermediate milestones. <br>&nbsp; (3) EV-Based: proportional to the earned progress of the project |
|revenue|tpac|**Total Payment at Completion** to be received from the client of the project after delivering 100% project progress or completion.</br></br>- $ \begin{cases} tpac =(1+roi) \times bac \\ \text{project profit} = tpac - bac = roi \times bac \end{cases}$ <br><br>- It's the total planned and agreeded opun inflow of the project. |
|revenue|l_ic|**List** of comulative **Inflow curve** untill each period. <br><br>- It's the payment schedule agreed upon in the project contract. <br><br>- It's directly calculated from the contractual agreement and the inflow model of the project |
|revenue|pl_ic|**periodic List** of **Inflow curve** for each period. <br><br>- It's directly calculated from the contractual agreement and the inflow model of the project </br></br>- $ pl\_ic = roi * l\_sc\_n $ <br><br>- Advance payment = $pl\_ic[0] \quad \text{or} \quad l\_ic[0] $  </br>- Final payment is calculated with respect to the **inflow model**. |
|actual|pp|current **Project Progress** in percent value.|
|actual|l_pp|**List** of the comulative **project progresses** for all the periods. for a completed project we have: </br>- $\begin{cases} \text{l\_pp}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_pp}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>|
|actual|pl_pp|**periodic List** of **project progresses**. </br></br>- $\begin{cases} \text{pl\_pp}[n] = \text{l\_pp}[n] - \text{l\_pp}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_pp}[n]} = 1 \quad \text{or} \quad 100\% & \text{for a completed project} \end{cases}$|
|EVM|l_bcws|**List** of cumulative **Budgeted Cost of Work Scheduled**. This Earned Value Management (EVM) metric represents the planned value of work that was scheduled to be completed by this period. <br><br>- It serves as a baseline for measuring project performance. <br><br>- A significant deviation from the BCWP indicates potential scheduling issues or delays in project execution. |
|EVM|l_acwp|**List** of cumulative **Actual Cost of Work Performed**. This Earned Value Management (EVM) metric reflects the actual costs incurred for the work completed up to this period. <br><br>- It provides insight into the actual expenditure against the planned budget. <br><br>- This metric accounts for all uncertainties affecting project progress. |
|EVM|l_bcwp|**List** of cumulative **Budgeted Cost of Work Performed**. This Earned Value Management (EVM) metric indicates the value of work that was planned to be completed by this period, based on the budget. <br><br>- A greater deviation from the ACWP signifies poorer performance and potential cost overruns. |

#### Portfolio Parameters

|Type|name|definition|
|---|---|---|
|cost|nbac|**Normalized bac** of all the projects in the portfolio for all them to sum up to *1.000*|
|general|pd|**Portfolio Duration** is the the total periods of the problem. total time steps of the model for taking actions. <br><br>- as a symbol of 12 months of the year, it's set to 12|
|general|pdd|**Portfolio data date** is the current period of the portfolio in it's duration. <br><br>- each episode is consisted of **pd** numbers of **pdd**|

#### Project simulation parameters

to be able to create synthetic projects we need to set some intermidiate parameters making us able to set the **project parameters**

these parameters are directly related to the project parameters and state of the project or portfolio.

**for example:** 
> in real use case of the model we have the **ROI** for the project according to the contractual agreements and legal documentations between client and contractor, but in the simulation we don't have an agreed upon contract as an input. we simulate the parameters based on the observed standards in the market. 

|Type|name|definition|
|---|---|---|
|general|roi|**return on investment** of the project. </br></br>- $ roi = \frac{\text{total earnings}}{\text{total cost}}= \frac{\text{total inflow}}{\text{total outflow}}= \frac{tpac}{bac}$ </br></br>- roi is set according to the industry standards. It is set at the contract negotiation time. |
|revenue|l_pic|**List** of comulative **Proportional inflow curve** for the project.</br> proportional being the assumption of recieving payment proportional to the spent cost with **roi** as the multiplier for adding profit. It's like Earned value based inflow model without advanced or final retention payment. </br></br>- $\begin{cases} \text{l\_pic}[n] = (1+ roi) \times l\_sc[n] & \quad \forall{n} \\ \text{l\_pic}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_pic}[n] = (1+roi) \times bac = tpac & \quad \text{for} \quad n = d\end{cases}$ </br></br>- It's used as base payment shedule for different inflow models. __all other models are different distributions of this simple schedule__.|
|revenue|pl_sei|**periodic List** of **simple inflow curve** for the project.</br></br>- $\begin{cases} \text{pl\_sei}[n] = \text{l\_sei}[n] - \text{l\_sei}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_sei}[n]} = 1 \quad \text{or} \quad 100\% \end{cases}$ |
|EVM|l_cpi|**List** of cumulative **Cost Performance Index (CPI)**. This metric measures the cost efficiency of the work accomplished. It is calculated by dividing the Earned Value (EV) by the Actual Cost (AC). <br><br>- $l\_cpi[n] = \frac{l\_bcwp[n]}{l\_acwp[n]} \rightarrow \begin{cases} l\_cpi[n] > 1 : & \text{project is under budget} \\ l\_cpi[n] < 1 : & \text{project is over budget} \end{cases}$ |
|EVM|l_spi|**List** of cumulative **Schedule Performance Index (SPI)**. This metric assesses the schedule efficiency of the work completed. It is calculated by dividing the Earned Value (EV) by the Planned Value (PV). <br><br>- $l\_spi[n] = \frac{l\_bcwp[n]}{l\_bcws[n]} \rightarrow \begin{cases} l\_spi[n] > 1 : & \text{project is ahead of schedule} \\ l\_spi[n] < 1 : & \text{project is behind schedule} \end{cases}$ |

#### APAM (Action Plan Analysis Method) Parameters
|Type|name|definition|
|---|---|---|
||||

---

**remainings**
* actuals
* performance metrics:
    * Inflation (as a general parameter. future assumed costs will cost more as the time passes)
    * contractor performance: (not gonna go in details for this paper, just a percentage of success or accomplishment rate for the allocated budget. we don't do delay analysis here.)
        * resources:
            * labor
            * machinary
            * cost
        * major forces: 
            * war
            * weather
    * client reliability: 
        * shorted payment with respect to expected payment schedule()
        * delayed payment with respect to expected payment schedule()
---

## Prompt

**responses should be at Q1 journal quality.**

according to complete formulation of the problem above create a python convex optimization program for me to input projects for portfolio and return the optimal budgeting plan at each time step of the portfolio for each project.

the output must be in following format with imaginary allocation plan values for one project (the program should fill this table including all the projects given to):

|Projects|0|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|P1|0|.5|.5|.1|.1|.1|.15|.15|.1|.1|.5|.25|.25|