# Problem formulation
This is a Brand new and fundamentally revised framework on project portfolio problem definition and formulation.

This represented framework shows great capability in addressing almost all aspects of budgetting decisions for project portfolio. 


### parameters
for parameters and valuables of type list or array, we will use square bracket notation for accessing or assigning to specific index. 
```
a: list of integers 
10th element: a[10]
```


#### Project Parameters

|Type|name|definition|
|---|---|---|
|cost|bac|**budget at completion** for the project. Total contractual cost for the project|
|General|d|**duration** of the project statet in number of periods |
|General|dd|current **data date** or **period** of the project. e.g. a project with duration of 12 has 12 periods and analytical data date.|
|cost|l_sc|**List** of comulative **S-Curve** values untill project is finished. It's the cost schedule for the project from the start untill the budget at completion(bac) </br></br>- $\begin{cases} \text{l\_sc}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_sc}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>- S-curve values are estimated at the contract agreement according to the project plan. |
|cost|pl_sc|**periodic List** of the **S-Curves** of the project. </br></br>- $\begin{cases} \text{pl\_sc}[n] = \text{l\_sc}[n] - \text{l\_sc}[n-1] \\ \sum_{n=0}^{n=d}{\text{l\_sc\_n}[n]} = \text{bac} \end{cases}$ |
|revenue|tep|**Total expected payment** to be recieved from the client of the project after delivering 100% project progress or completion.</br></br>- $ \begin{cases} tep =(1+roi) \times bac \\ \text{project profit} = tep - bac = roi \times bac \end{cases}$ <br><br>- It's the total expected inflow of the project. |
|revenue|l_ep|**List** of comulative **expected payments** untill each period.  |
|revenue|pl_ep|List of expected payments for each period. </br></br> $ l\_ep\_n = roi * l\_sc\_n \\ \text{advanced payment} = l\_ep\_n_0 \quad \text{or} \quad l\_ep_0 $  </br></br> final payment is calculated with respect to the inflow model. Inflow model will specify the distribution of payments according to contract and directly influences on for  |
|actual|pp|current **Project Progress** in percent value.|
|actual|l_pp|**List** of the comulative **project progresses** for all the periods. </br></br>- $\begin{cases} \text{l\_pp}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_pp}[n] = \text{bac} & \quad \text{for} \quad n = d \end{cases}$ </br></br>|
|actual|pl_pp|**periodic List** of **project progresses**. </br></br>- $\begin{cases} \text{pl\_pp}[n] = \text{l\_pp}[n] - \text{l\_pp}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_pp}[n]} = 1 \quad \text{or} \quad 100\% \end{cases}$|
|actual|l_acwp||
|actual|l_bcwp||
|actual|l_bcwp||

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
|general|roi|**return on investment** of the project. </br></br>- $ roi = \frac{\text{total earnings}}{\text{total cost}}= \frac{\text{total inflow}}{\text{total outflow}}= \frac{tep}{bac}$ </br></br>- roi is set according to the industry standards. It is set at the contract negotiation time. |
|general|im|Inflow Model(im) will determine the distribution of contractual payments form client to the contractor. It's the model that the payment schedule was designed upon. <br><br>inflow model will calculate *l_ep* and *pl_ep*|
|revenue|l_sei|**List** of comulative **simple expected inflow** for the project.</br> simple being the assumption of recieving payment proportional to the spendt cost with **roi** as the multiplier for adding profit.</br></br>- $\begin{cases} \text{l\_sei}[n] = (1+ roi) \times l\_sc[n] & \quad \forall{n} \\ \text{l\_sei}[n] = 0 & \quad \text{for} \quad n = 0 \\ \text{l\_sei}[n] = (1+roi) \times bac = tep & \quad \text{for} \quad n = d\end{cases}$ </br></br>- It's used in payment shedule calculation for different inflow models.`|
|revenue|pl_sei|**periodic List** of **simple expected inflow** for the project.</br></br>- $\begin{cases} \text{pl\_sei}[n] = \text{l\_sei}[n] - \text{l\_sei}[n-1] \\ \sum_{n=0}^{n=d}{\text{pl\_sei}[n]} = 1 \quad \text{or} \quad 100\% \end{cases}$ |
|general|air|**Annual Inflation Rate** is a general parameter. future assumed costs will cost more as the time passes. <br><br>- It's a transitionary multiplier. meaning that if we were to postpone the cost planned for this period to the last period, we would have to pay the inflated price at that period. <br><br>- it's applied to the remaining costs of each period to the next periods. <br><br>- It's declared as a percentage at the end of the year but the ramp for it is often none linear due to certain macro economical events, increase in material cost, shortage of labor etc. <br><br>- __in Iran inflation rate is very significant and can't be crossed out__  |
|general|pl_ir|**periodic list** of **inflation rates** of each period summing up to the annual iflation rate <br><br>- assuming linear ramp for the inflation we devise this formula for the remaining costs for the future: <br> $\begin{cases} pl\_ir[n] = \frac{n}{pd} \times air & \quad n \in pd \\ pl\_ir[n] = 0 & \quad \text{for} \quad n = 0 \\ pl\_ir[n] = air & \quad \text{for} \quad n = pd\end{cases}$ <br><br>- for the inflated costs we have to remember that this is a based ratio. meaning that if we transition a cost from period **a** to period **b** we have to use this formula: <br> $ \text{inflated cost}_b = \text{cost}_a \times ( 1 + (pl\_ir[b] - pl\_ir[a]) ) $ <br><br>- I would argue that the distribution of the inflation rate in detail has no justification of being percise. instead of being accurat we could be conservativaly genorous with it. meaning that instead of trying to accuratly estimate the distribution, we assume a slightly larger ratio so that the linear ramp covers the exact values. |
|cost|❓l_isc|**list** for the comulative **inflated S-Curve** as a proxy for the uncertain performance of the contractor on allocated budget|
|cost|❓pl_isc|**periodic list** for the **inflated S-Curve** as a proxy for the uncertain performance of the contractor on allocated budget|


**remainings**
* inflated s-curve (comulative and periodic)
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