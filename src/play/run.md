clear
del src\db\database.db
del src\db\exports
python src\db\db_init.py
python src\env\configs\single_project.py
python src\play\play.py
python src\db\export.py 


---

I've noticed something in this run, correct me if I'm wrong:
```
del : Cannot find path 'F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports' because it does not 
exist.
At line:3 char:1
+ del src\db\exports
+ ~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (F:\Career\00. U...\src\db\exports:String) [Remove-Item], ItemNotFoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.RemoveItemCommand
 
Database created: F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\database.db
Schema applied.

Tables found:
  OK  environment_config
  OK  projects_profile
  OK  milestones_profile
  OK  portfolios
  OK  projects_status
  OK  projects_observation
  OK  portfolio_observation
  OK  milestones_status
  OK  training_log

All tables verified.
Database already exists: F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\database.db
Skipping schema creation.
Seeded CFG-SINGLE-001
Database already exists: F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\database.db
Skipping schema creation.

  Available configurations:
    #  Config ID                                 Name
  ──────────────────────────────────────────────────────────────────────
    0  CFG-SINGLE-001                            single_project_baseline

  Select config number (or q to quit): 0

════════════════════════════════════════════════════════════════════════════════
  PERIOD 0 / 15     :                net_cashflow      :         0.00
  reward            :      +0.0000   budget_available  :       300.00
════════════════════════════════════════════════════════════════════════════════
  [ACTIVE]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            0   tolerance_remain  :          2/2
  inflow            :        11.50   outflow           :         0.00
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :         0.00   catchup_next_t    :         0.00
  reach_plan_t      :         0.00   reach_plan_next_t :         8.31
────────────────────────────────────────────────────────────────────────────────
  target            :          j=1   net_payment       :        21.56
  progress_gap      :       0.2500   timestep_gap      :            3
  required_alloc    :        25.00   target_npv        :      -1.3743
════════════════════════════════════════════════════════════════════════════════
  Allocate amount (Enter = all-in, q = quit): 10

════════════════════════════════════════════════════════════════════════════════
  PERIOD 1 / 15     :                net_cashflow      :       -10.00
  reward            :     -10.0000   budget_available  :       290.00
════════════════════════════════════════════════════════════════════════════════
  [ACTIVE]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            1   tolerance_remain  :          2/2
  inflow            :        11.50   outflow           :        10.00
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :         0.00   catchup_next_t    :         6.08
  reach_plan_t      :         0.00   reach_plan_next_t :        16.08
────────────────────────────────────────────────────────────────────────────────
  target            :          j=1   net_payment       :        21.56
  progress_gap      :       0.1476   timestep_gap      :            2
  required_alloc    :        14.76   target_npv        :       8.1598
════════════════════════════════════════════════════════════════════════════════
  Allocate amount (Enter = all-in, q = quit): 8

════════════════════════════════════════════════════════════════════════════════
  PERIOD 2 / 15     :                net_cashflow      :        -8.00
  reward            :     -17.7600   budget_available  :       282.00
════════════════════════════════════════════════════════════════════════════════
  [ACTIVE]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            2   tolerance_remain  :          2/2
  inflow            :        11.50   outflow           :        18.00
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :         0.00   catchup_next_t    :        16.81
  reach_plan_t      :         6.52   reach_plan_next_t :        26.81
────────────────────────────────────────────────────────────────────────────────
  target            :          j=1   net_payment       :        21.56
  progress_gap      :       0.0520   timestep_gap      :            1
  required_alloc    :         5.20   target_npv        :      17.0290
════════════════════════════════════════════════════════════════════════════════
  Allocate amount (Enter = all-in, q = quit): 20

════════════════════════════════════════════════════════════════════════════════
  PERIOD 3 / 15     :                net_cashflow      :       -20.07
  reward            :     -36.6392   budget_available  :       261.93
════════════════════════════════════════════════════════════════════════════════
  [ACTIVE]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            3   tolerance_remain  :          2/2
  inflow            :        11.50   outflow           :        38.07
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :         0.00   catchup_next_t    :        16.61
  reach_plan_t      :         8.33   reach_plan_next_t :        26.61
────────────────────────────────────────────────────────────────────────────────
  target            :          j=1   net_payment       :        21.56
  progress_gap      :       0.0000   timestep_gap      :            0
  required_alloc    :         0.00   target_npv        :      21.5625
════════════════════════════════════════════════════════════════════════════════
  Allocate amount (Enter = all-in, q = quit): 10

════════════════════════════════════════════════════════════════════════════════
  PERIOD 4 / 15     :                net_cashflow      :        11.30
  reward            :     -26.3288   budget_available  :       273.23
════════════════════════════════════════════════════════════════════════════════
  [ACTIVE]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            4   tolerance_remain  :          1/2
  inflow            :        33.06   outflow           :        48.33
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :         8.52   catchup_next_t    :        22.81
  reach_plan_t      :        18.52   reach_plan_next_t :        32.81
────────────────────────────────────────────────────────────────────────────────
  target            :          j=2   net_payment       :        21.56
  progress_gap      :       0.0364   timestep_gap      :            2
  required_alloc    :         3.64   target_npv        :      19.2802
════════════════════════════════════════════════════════════════════════════════
  Allocate amount (Enter = all-in, q = quit): 15

════════════════════════════════════════════════════════════════════════════════
  PERIOD 5 / 15     :                net_cashflow      :        19.83
  reward            :      -8.7721   budget_available  :       293.06
════════════════════════════════════════════════════════════════════════════════
  [TERMINATED]  Project 0
  BAC               :       100.00   Price             :       115.00
  t_proj            :            5   tolerance_remain  :          0/2
  inflow            :        68.05   outflow           :        63.48
────────────────────────────────────────────────────────────────────────────────
  catchup_t         :        10.00   catchup_next_t    :        19.89
  reach_plan_t      :        20.00   reach_plan_next_t :        29.89
────────────────────────────────────────────────────────────────────────────────
  target            :          j=2   net_payment       :        21.56
  progress_gap      :       0.0000   timestep_gap      :            1
  required_alloc    :         0.00   target_npv        :      22.2294
════════════════════════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
  Episode terminated (all projects done).
  Cumulative reward: -8.7721
════════════════════════════════════════════════════════════

  Exporting 9 table(s)
  Format: csv
  Output: F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports

  environment_config             → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\environment_config.csv  (1 rows)
  projects_profile               → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\projects_profile.csv  (1 rows)
  milestones_profile             → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\milestones_profile.csv  (5 rows)
  portfolios                     → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\portfolios.csv  (5 rows)
  projects_status                → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\projects_status.csv  (5 rows)
  projects_observation           → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\projects_observation.csv  (5 rows)
  portfolio_observation          → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\portfolio_observation.csv  (5 rows)
  milestones_status              → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\milestones_status.csv  (2 rows)
  training_log                   → F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\src\db\exports\training_log.csv  (0 rows)

  Done.
```

looking at the run above at the timesteps 4 and 5 I've noticed that we are terminating the project with wrong sequence if it's being terminated only for the plan progress breach.

here's why:
at t_4 we are down by 1 for the tolerance and 1 remains because:
  catchup_t         :         8.52   catchup_next_t    :        22.81
  reach_plan_t      :        18.52   reach_plan_next_t :        32.81

any time the catchup_t is none zero it means we are delayed more than the threshold and the tolerance decremented correctly.
but I've allocated 15 to that project and it must close that timestep with catchup_t = 0 for the t_4.
in conclusion, we might be over delayed at the beginning of the timestep but if we allocate enough to resolve the delay the tolerance must be reset.

this is not happening.
the tolerance 1 moves to the next timestep and again with no opportunity for resolving that timestep it terminates early.

the correct behavior would be to only terminate the timestep if after allocation the termination breach conditions stayed active and not resolved for that timestep.

the early termination check at each timestep before the allocation would decrement the tolerance twice which is not what I intend it to be.

this data from project status confirms this:
```
episode_id	i	t_episode	t_project	method	status	inflow	outflow	termination_settlement	net_cashflow	allocation_action	deficit	interest_cost	allocation	efficiency	spi	cpi	eac	progress_actual	progress_plan_t	progress_delay_t	progress_space_t	min_prog_t	progress_needed_t	catchup_alloc_t	reach_plan_t	progress_plan_next_t	progress_delay_next_t	progress_space_next_t	min_prog_next_t	progress_needed_next_t	catchup_alloc_next_t	reach_plan_next_t	target_milestone_j	target_progress_gap	target_timestep_gap	target_net_payment	target_required_alloc	target_payment_rate	target_npv	projected_cost_overrun	projected_finish	projected_finish_delay	abandoned	over_duration_window	over_progress_delay	over_finish_delay	over_cost_overrun	over_any	tolerance_remain
a46a48bc-2324-4943-8afb-0d5b35c4301b	0	0	1	manual	active	11.5	10.000000521540642	0.0	-10.000000521540642	10.000000521540642	0.0	0.0	10.000000521540642	1.0242882543957565	1.126555326438547	1.0242882543957565	87.78469966307378	0.1024288307816552	0.08309400078624861	-0.01933482999540659	0.1193348299954066	0.0	0.0	0.0	0.0	0.26322523563396494	0.16079640485230973	-0.06079640485230972	0.16322523563396493	0.060796404852309735	6.079640485230973	16.07964048523097	1	0.1475711692183448	3	21.562499999999996	14.757116921834479	1.461159392733166	8.868540898655441	0.8778469966307378	10.651940227326769	-1.3480597726732313	0	0	0	0	0	0	2
a46a48bc-2324-4943-8afb-0d5b35c4301b	0	1	2	manual	active	11.5	18.00000047735099	0.0	-7.99999995581035	7.99999995581035	0.0	0.0	7.99999995581035	1.1945883945179534	0.8340416242535307	1.0999772031358608	105.41886829992573	0.19799590181520704	0.26322523563396494	0.0652293338187579	0.03477066618124211	0.16322523563396493	0.0	0.0	6.52293338187579	0.4660666122418507	0.26807071042664365	-0.16807071042664365	0.36606661224185066	0.16807071042664362	16.80707104266436	26.807071042664365	1	0.05200409818479296	2	21.562499999999996	5.2004098184792955	4.146307839697393	17.716478267395928	1.0541886829992573	14.387771126818796	2.387771126818796	0	0	0	0	0	0	2
a46a48bc-2324-4943-8afb-0d5b35c4301b	0	2	3	manual	active	11.5	38.065000865502334	0.0	-20.065000388151343	20.000000383377834	6.500000477350991	0.06500000477350991	20.065000388151343	0.923750862349398	0.8620948550529359	1.0055065522760016	109.2721994159817	0.3827460778265427	0.4660666122418507	0.08332053441530801	0.016679465584691994	0.36606661224185066	0.0	0.0	8.332053441530801	0.6488381772905428	0.26609209946400014	-0.16609209946400014	0.5488381772905429	0.16609209946400016	16.609209946400018	26.609209946400014	1	0.0	1	21.562499999999996	0.0	0.0	22.229381443298966	1.092721994159817	13.919581969043481	1.919581969043481	0	0	0	0	0	0	2
a46a48bc-2324-4943-8afb-0d5b35c4301b	0	3	4	manual	active	33.0625	48.33065125419353	0.0	11.296849611308797	10.000000380036175	26.565000865502334	0.26565000865502336	10.2656503886912	0.8088749202965275	0.7469262723028449	0.959295107553479	123.18751385661068	0.46363357293021273	0.6488381772905428	0.1852046043603301	-0.08520460436033009	0.5488381772905429	0.08520460436033012	8.520460436033012	18.52046043603301	0.7917472710949173	0.32811369816470454	-0.22811369816470453	0.6917472710949173	0.22811369816470456	22.811369816470457	32.81136981647045	2	0.036366427069787266	3	21.562499999999996	3.6366427069787264	5.929232464498507	19.989015113511194	1.2318751385661069	16.065842700917262	4.065842700917262	0	0	1	1	0	1	1
a46a48bc-2324-4943-8afb-0d5b35c4301b	0	4	5	manual	terminated	68.0466838335759	63.483333214390356	34.9841838335759	19.831501873379075	15.000000447654887	15.268151254193533	0.15268151254193535	15.152681960196823	0.8538447830167812	0.7332850218915385	0.9320718750017988	123.22071891488504	0.5917102942050079	0.7917472710949173	0.2000369768899094	-0.1000369768899094	0.6917472710949173	0.10003697688990942	10.003697688990943	20.00369768899094	0.8906342772967548	0.29892398309174695	-0.19892398309174694	0.7906342772967548	0.19892398309174697	19.8923983091747	29.892398309174695	2	0.0	2	21.562499999999996	0.0	0.0	22.91688808587522	1.2322071891488504	16.364714458568255	4.364714458568255	0	0	1	1	0	1	0
```

also if you haven't noticed the net_cashflow is not correct.
for example for the timestep zero when we receive 11 advance and allocate 10 the net must be positive 1.