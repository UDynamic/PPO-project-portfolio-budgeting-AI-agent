-- schema.sql
--
-- =============================================================
-- PPM Training Database Schema
-- Contractor Portfolio Budgeting — RL + Baselines + MILP
-- =============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode  = WAL;
PRAGMA synchronous   = NORMAL;

-- =============================================================
-- STATIC — PROBLEM DEFINITION
-- =============================================================

CREATE TABLE IF NOT EXISTS environment_config (

    config_id                   TEXT NOT NULL PRIMARY KEY,
    config_name                 TEXT NOT NULL,
    created_at                  TEXT NOT NULL,

    -- Portfolio level
    n_projects_dist             TEXT NOT NULL,
    n_projects_p1               REAL NOT NULL,
    n_projects_p2               REAL,
    n_projects_p3               REAL,
    n_projects_p4               REAL,

    budget_available_dist       TEXT NOT NULL,
    budget_available_p1         REAL NOT NULL,
    budget_available_p2         REAL,
    budget_available_p3         REAL,
    budget_available_p4         REAL,

    budget_tightness_dist       TEXT NOT NULL,
    budget_tightness_p1         REAL NOT NULL,
    budget_tightness_p2         REAL,
    budget_tightness_p3         REAL,
    budget_tightness_p4         REAL,

    discount_dist                TEXT NOT NULL,
    discount_p1                  REAL NOT NULL,
    discount_p2                  REAL,
    discount_p3                  REAL,
    discount_p4                  REAL,

    -- Project level
    bac_dist                    TEXT NOT NULL,
    bac_p1                      REAL NOT NULL,
    bac_p2                      REAL,
    bac_p3                      REAL,
    bac_p4                      REAL,

    profit_percent_dist         TEXT NOT NULL,
    profit_percent_p1           REAL NOT NULL,
    profit_percent_p2           REAL,
    profit_percent_p3           REAL,
    profit_percent_p4           REAL,

    planned_start_dist          TEXT NOT NULL,
    planned_start_p1            REAL NOT NULL,
    planned_start_p2            REAL,
    planned_start_p3            REAL,
    planned_start_p4            REAL,

    planned_duration_dist       TEXT NOT NULL,
    planned_duration_p1         REAL NOT NULL,
    planned_duration_p2         REAL,
    planned_duration_p3         REAL,
    planned_duration_p4         REAL,

    scurve_a_dist                TEXT NOT NULL,
    scurve_a_p1                  REAL NOT NULL,
    scurve_a_p2                  REAL,
    scurve_a_p3                  REAL,
    scurve_a_p4                  REAL,

    scurve_b_dist                TEXT NOT NULL,
    scurve_b_p1                  REAL NOT NULL,
    scurve_b_p2                  REAL,
    scurve_b_p3                  REAL,
    scurve_b_p4                  REAL,

    advance_percent_dist        TEXT NOT NULL,
    advance_percent_p1          REAL NOT NULL,
    advance_percent_p2          REAL,
    advance_percent_p3          REAL,
    advance_percent_p4          REAL,

    advance_trigger_dist        TEXT NOT NULL,
    advance_trigger_p1          REAL NOT NULL,
    advance_trigger_p2          REAL,
    advance_trigger_p3          REAL,
    advance_trigger_p4          REAL,

    advance_recovery_dist       TEXT NOT NULL,
    advance_recovery_p1         REAL NOT NULL,
    advance_recovery_p2         REAL,
    advance_recovery_p3         REAL,
    advance_recovery_p4         REAL,

    retention_rate_dist         TEXT NOT NULL,
    retention_rate_p1           REAL NOT NULL,
    retention_rate_p2           REAL,
    retention_rate_p3           REAL,
    retention_rate_p4           REAL,

    -- progress_delay_cap : max tolerated (planned_progress - actual_progress); dimensionless [0, 1]
    progress_delay_cap_dist     TEXT NOT NULL,
    progress_delay_cap_p1       REAL NOT NULL,
    progress_delay_cap_p2       REAL,
    progress_delay_cap_p3       REAL,
    progress_delay_cap_p4       REAL,

    -- finish_delay_cap : max tolerated (projected_finish - planned_finish); units = periods
    finish_delay_cap_dist       TEXT NOT NULL,
    finish_delay_cap_p1         REAL NOT NULL,
    finish_delay_cap_p2         REAL,
    finish_delay_cap_p3         REAL,
    finish_delay_cap_p4         REAL,

    -- cost_overrun_cap : max tolerated EAC / BAC ratio; e.g. 1.30 = 30% overrun
    cost_overrun_cap_dist       TEXT NOT NULL,
    cost_overrun_cap_p1         REAL NOT NULL,
    cost_overrun_cap_p2         REAL,
    cost_overrun_cap_p3         REAL,
    cost_overrun_cap_p4         REAL,

    termination_tolerance_dist  TEXT NOT NULL,
    termination_tolerance_p1    REAL NOT NULL,
    termination_tolerance_p2    REAL,
    termination_tolerance_p3    REAL,
    termination_tolerance_p4    REAL,

    efficiency_dist              TEXT NOT NULL,
    efficiency_p1                REAL NOT NULL,
    efficiency_p2                REAL,
    efficiency_p3                REAL,
    efficiency_p4                REAL,

    -- Milestone level
    n_milestones_dist           TEXT NOT NULL,
    n_milestones_p1             REAL NOT NULL,
    n_milestones_p2             REAL,
    n_milestones_p3             REAL,
    n_milestones_p4             REAL,

    progress_threshold_dist     TEXT NOT NULL,
    progress_threshold_p1       REAL NOT NULL,
    progress_threshold_p2       REAL,
    progress_threshold_p3       REAL,
    progress_threshold_p4       REAL,

    payment_weight_dist         TEXT NOT NULL,
    payment_weight_p1           REAL NOT NULL,
    payment_weight_p2           REAL,
    payment_weight_p3           REAL,
    payment_weight_p4           REAL,

    timestep_threshold_dist     TEXT NOT NULL,
    timestep_threshold_p1       REAL NOT NULL,
    timestep_threshold_p2       REAL,
    timestep_threshold_p3       REAL,
    timestep_threshold_p4       REAL,

    annual_interest_rate        REAL NOT NULL
);


CREATE TABLE IF NOT EXISTS projects_profile (

    episode_id              TEXT    NOT NULL,
    config_id               TEXT    NOT NULL REFERENCES environment_config(config_id),
    i                        INTEGER NOT NULL,

    -- timeline
    planned_start            INTEGER NOT NULL,
    planned_finish           INTEGER NOT NULL,
    planned_duration         INTEGER NOT NULL,

    -- financials
    bac                      REAL NOT NULL,      -- budget at completion (total cost)
    profit_percent           REAL NOT NULL,
    price                    REAL NOT NULL,      -- total contract value = bac * (1 + profit_percent)

    -- s-curve shape
    scurve_a                 REAL NOT NULL,
    scurve_b                 REAL NOT NULL,

    -- advance
    advance_percent          REAL NOT NULL,
    advance_trigger          REAL NOT NULL,      -- placeholder; not yet wired into certification
    advance_recovery         REAL NOT NULL,      -- fraction of milestone gross recovered per milestone

    -- retention
    retention_rate           REAL NOT NULL,

    -- breach caps
    progress_delay_cap       REAL    NOT NULL,
    finish_delay_cap         INTEGER NOT NULL,
    cost_overrun_cap         REAL    NOT NULL,
    termination_tolerance    INTEGER NOT NULL,

    PRIMARY KEY (episode_id, i)
);


CREATE TABLE IF NOT EXISTS milestones_profile (

    episode_id               TEXT    NOT NULL,
    i                        INTEGER NOT NULL,
    j                        INTEGER NOT NULL,   -- 0 = advance, 1..n = milestones, n+1 = retention release

    -- milestone identity
    progress_threshold       REAL    NOT NULL,   -- progress level that triggers certification
    timestep_threshold       INTEGER NOT NULL,   -- earliest period at which certification is allowed
    payment_weight           REAL    NOT NULL,   -- fraction of price

    gross_payment            REAL NOT NULL,      -- payment_weight * price
    advance_recovery         REAL NOT NULL,      -- advance recovered at this milestone
    advance_recovery_remain  REAL NOT NULL,      -- remaining unrecovered advance after this milestone
    retention_withheld       REAL NOT NULL,      -- gross * retention_rate (0 for final milestone)
    retention_released       REAL NOT NULL,      -- retention released at completion (0 for advance and interim)
    net_payment              REAL NOT NULL,      -- gross - advance_recovery - retention_withheld + retention_released

    PRIMARY KEY (episode_id, i, j),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);


-- =============================================================
-- TEMPORAL — SOLUTION TRACES
-- =============================================================

CREATE TABLE IF NOT EXISTS portfolios (

    episode_id           TEXT    NOT NULL,
    config_id            TEXT    NOT NULL REFERENCES environment_config(config_id),
    t_episode            INTEGER NOT NULL,
    method               TEXT    NOT NULL,

    budget_available     REAL    NOT NULL,
    net_cashflow         REAL    NOT NULL,        -- inflow - outflow over all the project

    reward               REAL    NOT NULL,
    done                 INTEGER NOT NULL,

    PRIMARY KEY (episode_id, t_episode, method)
);

-- Includes everything
CREATE TABLE IF NOT EXISTS projects_status (

    episode_id              TEXT    NOT NULL,
    i                       INTEGER NOT NULL,
    t_episode               INTEGER NOT NULL,
    t_project               INTEGER,
    method                  TEXT    NOT NULL,

    -- lifecycle
    status                   TEXT,

    -- cash flow (numbers are cumulative)
    inflow                   REAL,   -- inflows (This is the cumulative record of all received payments and other incomes for the projects)
    outflow                  REAL,   -- outflows (This is the cumulative record of all allocations and other costs for the projects)
    termination_settlement   REAL,   -- although it's an outflow because of it's special case we track it separately
    net_cashflow             REAL,   -- inflow - outflow + termination_settlement this period

    -- allocation & financing
    allocation_action        REAL,   -- raw allocation input from agent or user
    deficit                  REAL,   -- max(0, outflow - inflows)
    interest_cost            REAL,   -- (annual_rate / 12) * deficit
    allocation                REAL,  -- allocation_action + interest_cost; deducted from budget_available

    -- efficiency
    efficiency                REAL,  -- eta drawn each period; applied to allocation_action only

    -- evm signals
    spi                       REAL,  -- earned schedule / actual time
    cpi                       REAL,  -- BCWP / ACWP
    eac                       REAL,  -- estimated cost at completion

    -- progress
    progress_actual           REAL,

    -- current period (assuming efficiency = 1)
    progress_plan_t           REAL,  -- planned_progress(t_project, duration, a, b)
    progress_delay_t          REAL,  -- progress_plan_t - progress_actual
    progress_space_t          REAL,  -- progress_delay_cap - progress_delay_t
    min_prog_t                REAL,  -- max(0.0, progress_plan_t - progress_delay_cap)
    progress_needed_t         REAL,  -- max(0.0, min_prog_t - progress_actual)
    catchup_alloc_t           REAL,  -- (progress_needed_t) * bac
    reach_plan_t              REAL,  -- max(0.0, progress_plan_t - progress_actual) * bac

    -- next period (assuming efficiency = 1)
    progress_plan_next_t      REAL,
    progress_delay_next_t     REAL,  -- progress_plan_next_t - progress_actual
    progress_space_next_t     REAL,
    min_prog_next_t           REAL,  -- max(0.0, progress_plan_next_t - progress_delay_cap)
    progress_needed_next_t    REAL,  -- max(0.0, min_prog_next_t - progress_actual)
    catchup_alloc_next_t      REAL,  -- (progress_needed_next_t) * bac
    reach_plan_next_t         REAL,  -- max(0.0, progress_plan_next_t - progress_actual) * bac

    -- next milestone target (3 features)
    target_milestone_j        INTEGER,  -- index only, not in obs vector
    target_progress_gap       REAL,     -- threshold - progress_actual
    target_timestep_gap       INTEGER,  -- earliest_t - t_episode
    target_net_payment        REAL,     --
    target_required_alloc     REAL,     -- target_progress_gap x BAC
    target_payment_rate       REAL,     -- target_net_payment / target_required_alloc
    target_npv                REAL,     -- PV(target_net_payment, timestep_gap, discount) - target_required_alloc

    -- finish projections
    projected_cost_overrun    REAL,     -- EAC / BAC; compared against cost_overrun_cap
    projected_finish          REAL,     -- start + duration / SPI(t)
    projected_finish_delay    REAL,     -- projected_finish - planned_finish

    -- breach flags
    abandoned                INTEGER,   -- 1 if allocation_action < epsilon while active
    over_duration_window     INTEGER,   -- 1 if t >= planned_finish + finish_delay_cap
    over_progress_delay      INTEGER,   -- 1 if progress_delay > progress_delay_cap
    over_finish_delay        INTEGER,   -- 1 if projected_finish_delay > finish_delay_cap
    over_cost_overrun        INTEGER,   -- 1 if projected_cost_overrun > cost_overrun_cap
    over_any                 INTEGER,   -- 1 if any breach fired this period

    -- termination
    tolerance_remain         INTEGER,   -- cure periods remaining before termination fires

    PRIMARY KEY (episode_id, i, t_episode, method),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);

CREATE TABLE IF NOT EXISTS projects_observation (

    episode_id              TEXT    NOT NULL,
    i                       INTEGER NOT NULL,
    t_episode               INTEGER NOT NULL,
    t_project               INTEGER,
    method                  TEXT    NOT NULL,

    net_cashflow             REAL,

    -- per-project state (8 base features)
    tolerance_remain         REAL,  -- by termination_tolerance
    catchup_alloc_t          REAL,  -- (progress_needed_t) * bac
    catchup_alloc_next_t     REAL,  -- (progress_needed_next_t) * bac
    reach_plan_t             REAL,  -- max(0, progress_plan_t - progress_actual) * bac
    reach_plan_next_t        REAL,  -- max(0, progress_plan_next_t - progress_actual) * bac

    -- next milestone target (3 features)
    target_progress_gap      REAL,     -- threshold - progress_actual
    target_timestep_gap      INTEGER,  -- earliest_t - t_episode
    target_required_alloc    REAL,     -- target_progress_gap x BAC
    target_npv               REAL,     -- PV(target_net_payment, timestep_gap, discount) - target_required_alloc

    PRIMARY KEY (episode_id, i, t_episode, method),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);

CREATE TABLE IF NOT EXISTS portfolio_observation (

    episode_id              TEXT    NOT NULL,
    t_episode               INTEGER NOT NULL,
    method                  TEXT    NOT NULL,

    -- portfolio-level state (1 feature)
    net_cashflow             REAL, -- over all the projects
    budget_available         REAL, -- budget available from last timestep added net_cashflow of this timestep

    PRIMARY KEY (episode_id, t_episode, method)
);

-- =============================================================
-- EVENT — MILESTONE CERTIFICATION
-- =============================================================

CREATE TABLE IF NOT EXISTS milestones_status (

    episode_id              TEXT    NOT NULL,
    i                       INTEGER NOT NULL,
    j                       INTEGER NOT NULL,
    method                  TEXT    NOT NULL,

    -- certification event
    certified_t              INTEGER,  -- period when certified; NULL = not yet certified
    certification_delay      INTEGER,  -- certified_t - timestep_threshold; 0 if on time; NULL if not certified
    net_payment               REAL,    -- actual net payment released at certification; NULL if not certified

    PRIMARY KEY (episode_id, i, j, method),
    FOREIGN KEY (episode_id, i, j) REFERENCES milestones_profile(episode_id, i, j)
);


-- =============================================================
-- TRAINING — PPO LEARNING METRICS
-- =============================================================

CREATE TABLE IF NOT EXISTS training_log (

    "update"             INTEGER NOT NULL PRIMARY KEY,
    episode_id           TEXT    NOT NULL,
    timestep_global      INTEGER NOT NULL,

    reward_mean          REAL NOT NULL,
    reward_std           REAL NOT NULL,
    policy_loss          REAL NOT NULL,
    value_loss           REAL NOT NULL,
    entropy              REAL NOT NULL,
    kl_divergence        REAL NOT NULL,
    clip_fraction         REAL NOT NULL,
    learning_rate         REAL NOT NULL
);


-- =============================================================
-- INDEXES
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_portfolios_episode
    ON portfolios (episode_id, method);

CREATE INDEX IF NOT EXISTS idx_projects_status_episode_project
    ON projects_status (episode_id, i, method);

CREATE INDEX IF NOT EXISTS idx_projects_status_timestep
    ON projects_status (episode_id, t_episode, method);

CREATE INDEX IF NOT EXISTS idx_milestones_status_episode
    ON milestones_status (episode_id, method);

CREATE INDEX IF NOT EXISTS idx_training_log_global_step
    ON training_log (timestep_global);