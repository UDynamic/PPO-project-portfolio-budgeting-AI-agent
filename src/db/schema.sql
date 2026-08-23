--  schema.sql

-- =============================================================
-- PPM Training Database Schema
-- Contractor Portfolio Budgeting — RL + Baselines + MILP
-- =============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- =============================================================
-- STATIC — PROBLEM DEFINITION
-- =============================================================

CREATE TABLE IF NOT EXISTS environment_config (

    config_id               TEXT PRIMARY KEY,
    config_name             TEXT NOT NULL,
    created_at              TEXT NOT NULL,

    -- Portfolio level
    n_projects_dist         TEXT NOT NULL,
    n_projects_p1           REAL NOT NULL,
    n_projects_p2           REAL,
    n_projects_p3           REAL,
    n_projects_p4           REAL,

    budget_available_dist   TEXT NOT NULL,
    budget_available_p1     REAL NOT NULL,
    budget_available_p2     REAL,
    budget_available_p3     REAL,
    budget_available_p4     REAL,

    budget_tightness_dist   TEXT NOT NULL,
    budget_tightness_p1     REAL NOT NULL,
    budget_tightness_p2     REAL,
    budget_tightness_p3     REAL,
    budget_tightness_p4     REAL,

    discount_dist           TEXT NOT NULL,
    discount_p1             REAL NOT NULL,
    discount_p2             REAL,
    discount_p3             REAL,
    discount_p4             REAL,

    -- Project level
    bac_dist                TEXT NOT NULL,
    bac_p1                  REAL NOT NULL,
    bac_p2                  REAL,
    bac_p3                  REAL,
    bac_p4                  REAL,

    profit_percent_dist     TEXT NOT NULL,
    profit_percent_p1       REAL NOT NULL,
    profit_percent_p2       REAL,
    profit_percent_p3       REAL,
    profit_percent_p4       REAL,

    start_dist              TEXT NOT NULL,
    start_p1                REAL NOT NULL,
    start_p2                REAL,
    start_p3                REAL,
    start_p4                REAL,

    duration_dist           TEXT NOT NULL,
    duration_p1             REAL NOT NULL,
    duration_p2             REAL,
    duration_p3             REAL,
    duration_p4             REAL,

    scurve_a_dist           TEXT NOT NULL,
    scurve_a_p1             REAL NOT NULL,
    scurve_a_p2             REAL,
    scurve_a_p3             REAL,
    scurve_a_p4             REAL,

    scurve_b_dist           TEXT NOT NULL,
    scurve_b_p1             REAL NOT NULL,
    scurve_b_p2             REAL,
    scurve_b_p3             REAL,
    scurve_b_p4             REAL,

    advance_percent_dist    TEXT NOT NULL,
    advance_percent_p1      REAL NOT NULL,
    advance_percent_p2      REAL,
    advance_percent_p3      REAL,
    advance_percent_p4      REAL,

    advance_trigger_dist    TEXT NOT NULL,
    advance_trigger_p1      REAL NOT NULL,
    advance_trigger_p2      REAL,
    advance_trigger_p3      REAL,
    advance_trigger_p4      REAL,

    advance_recovery_dist   TEXT NOT NULL,
    advance_recovery_p1     REAL NOT NULL,
    advance_recovery_p2     REAL,
    advance_recovery_p3     REAL,
    advance_recovery_p4     REAL,

    retention_rate_dist     TEXT NOT NULL,
    retention_rate_p1       REAL NOT NULL,
    retention_rate_p2       REAL,
    retention_rate_p3       REAL,
    retention_rate_p4       REAL,

    -- progress_delay_cap : max tolerated (planned_progress − actual_progress); dimensionless [0, 1]
    progress_delay_cap_dist TEXT NOT NULL,
    progress_delay_cap_p1   REAL NOT NULL,
    progress_delay_cap_p2   REAL,
    progress_delay_cap_p3   REAL,
    progress_delay_cap_p4   REAL,

    -- finish_delay_cap   : max tolerated (projected_finish − planned_finish);  units = periods
    finish_delay_cap_dist   TEXT NOT NULL,
    finish_delay_cap_p1     REAL NOT NULL,
    finish_delay_cap_p2     REAL,
    finish_delay_cap_p3     REAL,
    finish_delay_cap_p4     REAL,

    -- cost_overrun_cap   : max tolerated (EAC / BAC) ratio;     e.g. 1.30 = 30% overrun
    cost_overrun_cap_dist   TEXT NOT NULL,
    cost_overrun_cap_p1     REAL NOT NULL,
    cost_overrun_cap_p2     REAL,
    cost_overrun_cap_p3     REAL,
    cost_overrun_cap_p4     REAL,

    termination_tolerance_dist  TEXT NOT NULL,
    termination_tolerance_p1    REAL NOT NULL,
    termination_tolerance_p2    REAL,
    termination_tolerance_p3    REAL,
    termination_tolerance_p4    REAL,

    efficiency_dist         TEXT NOT NULL,
    efficiency_p1           REAL NOT NULL,
    efficiency_p2           REAL,
    efficiency_p3           REAL,
    efficiency_p4           REAL,

    -- Milestone level
    n_milestones_dist       TEXT NOT NULL,
    n_milestones_p1         REAL NOT NULL,
    n_milestones_p2         REAL,
    n_milestones_p3         REAL,
    n_milestones_p4         REAL,

    progress_threshold_dist TEXT NOT NULL,
    progress_threshold_p1   REAL NOT NULL,
    progress_threshold_p2   REAL,
    progress_threshold_p3   REAL,
    progress_threshold_p4   REAL,

    payment_weight_dist     TEXT NOT NULL,
    payment_weight_p1       REAL NOT NULL,
    payment_weight_p2       REAL,
    payment_weight_p3       REAL,
    payment_weight_p4       REAL,

    timestep_threshold_dist TEXT NOT NULL,
    timestep_threshold_p1   REAL NOT NULL,
    timestep_threshold_p2   REAL,
    timestep_threshold_p3   REAL,
    timestep_threshold_p4   REAL,

    annual_interest_rate    REAL NOT NULL
);


CREATE TABLE IF NOT EXISTS projects_profile (

    episode_id                  TEXT NOT NULL,
    config_id                   TEXT NOT NULL REFERENCES environment_config(config_id),
    i                           INTEGER NOT NULL,

    planned_start               INTEGER NOT NULL,
    planned_finish              INTEGER NOT NULL,
    planned_duration            INTEGER NOT NULL,
    
    bac                         REAL NOT NULL,
    profit_percent              REAL NOT NULL,
    price                       REAL NOT NULL,
    scurve_a                    REAL NOT NULL,
    scurve_b                    REAL NOT NULL,
    
    advance_percent             REAL NOT NULL,
    advance_trigger             REAL NOT NULL,
    advance_recovery            REAL NOT NULL,
    
    retention_rate              REAL NOT NULL,
    
    progress_delay_cap          REAL NOT NULL,
    finish_delay_cap            INTEGER NOT NULL,
    cost_overrun_cap            REAL NOT NULL,
    termination_tolerance       INTEGER NOT NULL,

    PRIMARY KEY (episode_id, i)
);


CREATE TABLE IF NOT EXISTS milestones_profile (

    episode_id                  TEXT NOT NULL,
    i                           INTEGER NOT NULL,
    j                           INTEGER NOT NULL,

    progress_threshold          REAL NOT NULL,
    timestep_threshold          INTEGER NOT NULL,
    payment_weight              REAL NOT NULL,

    PRIMARY KEY (episode_id, i, j),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);


-- =============================================================
-- TEMPORAL — SOLUTION TRACES
-- =============================================================

CREATE TABLE IF NOT EXISTS portfolios (

    episode_id          TEXT NOT NULL,
    config_id           TEXT NOT NULL REFERENCES environment_config(config_id),
    t_episode           INTEGER NOT NULL,
    method              TEXT NOT NULL,

    budget_available    REAL NOT NULL,
    inflow              REAL NOT NULL,
    outflow             REAL NOT NULL,
    net_cashflow        REAL,    -- inflow − outflow
    
    reward              REAL NOT NULL,
    done                INTEGER NOT NULL,

    PRIMARY KEY (episode_id, t_episode, method)
);


CREATE TABLE IF NOT EXISTS projects_status (

    episode_id                  TEXT NOT NULL,
    i                           INTEGER NOT NULL,
    t_episode                   INTEGER NOT NULL,
    t_project                   INTEGER,
    method                      TEXT NOT NULL,

    -- ── lifecycle ────────────────────────────────────────────
    status                      TEXT,

    -- ── cash flow and financing (after and before allocation values are different) ────────────────
    total_inflow                      REAL,    -- total inflow (advance or milestone payments or retention release)
    advance_received
    gross_milestone_payment
    retention_release                 

    total_outflow                     REAL,    -- total outflow (alloc + interest + settlement but not the retention and advance recovery )
    retention_withheld
    advance_recovered
    interest_cost               REAL,
    allocation                  REAL,
    
    net_cashflow                REAL,    -- inflow − outflow
    
    deficit                     REAL,    -- max(0, cumulative_cost − cumulative_inflows)
    treasury_draw               REAL,

    -- ── efficiency ───────────────────────────────
    efficiency                  REAL,

    -- ── progress ─────────────────────────────────────────────
    progress_actual             REAL,
    progress_actual_periodic    REAL,
    
    progress_plan               REAL,
    progress_plan_periodic      REAL,
    
    -- planned_progress − actual_progress
    progress_delay              REAL,    

    -- ── evm signals ──────────────────────────────────────────
    spi                         REAL,    -- earned schedule / actual time
    cpi                         REAL,    -- BCWP / ACWP
    tcpi                        REAL,    -- (BAC − BCWP) / (BAC − ACWP)
    eac                         REAL,    -- estimated cost at completion
    
    projected_cost_overrun      REAL,    -- EAC / BAC; compared against cost_overrun_cap
    
    projected_finish            REAL,    -- start + duration / SPI(t)
    projected_finish_delay      REAL,    -- projected_finish − planned_finish

    -- ── breach flags and termination ─────────────────────────────────────────
    abandoned                   INTEGER, -- 1 if allocation < ε while active
    
    over_duration_window     INTEGER, -- 1 if t >= finish + finish_delay_cap
    
    over_progress_delay         INTEGER, -- 1 if progress_delay > progress_delay_cap
    
    over_finish_delay           INTEGER, -- 1 if projected_finish_delay > finish_delay_cap
    over_cost_overrun           INTEGER, -- 1 if projected_cost_overrun > cost_overrun_cap
    
    over_any                    INTEGER, -- 1 if any breach fired this period

    tolerance_remain            INTEGER,

    -- ── payments ─────────────────────────────────────────────

    
    settlement                  REAL,
    
    milestone_net               REAL,
    

    PRIMARY KEY (episode_id, i, t_episode, method),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);


-- =============================================================
-- EVENT — MILESTONE CERTIFICATION
-- =============================================================

CREATE TABLE IF NOT EXISTS milestones_status (

    episode_id              TEXT NOT NULL,
    i                       INTEGER NOT NULL,
    j                       INTEGER NOT NULL,
    method                  TEXT NOT NULL,

    -- milestone identity (denormalised for query convenience)
    progress_threshold      REAL,
    payment_weight          REAL,
    timestep_threshold      INTEGER,

    -- payment breakdown (matches payment profile display)
    gross                   REAL,
    advance_recovery        REAL,
    cumulative_recovery     REAL,
    retention_withheld      REAL,
    payment_net             REAL,

    -- certification event
    certified               INTEGER NOT NULL DEFAULT 0,
    certified_t             INTEGER,

    PRIMARY KEY (episode_id, i, j, method),
    FOREIGN KEY (episode_id, i, j) REFERENCES milestones_profile(episode_id, i, j)
);


-- =============================================================
-- TRAINING — PPO LEARNING METRICS
-- =============================================================

CREATE TABLE IF NOT EXISTS training_log (

    "update"            INTEGER PRIMARY KEY,
    episode_id          TEXT NOT NULL,
    timestep_global     INTEGER NOT NULL,

    reward_mean         REAL NOT NULL,
    reward_std          REAL NOT NULL,
    policy_loss         REAL NOT NULL,
    value_loss          REAL NOT NULL,
    entropy             REAL NOT NULL,
    kl_divergence       REAL NOT NULL,
    clip_fraction       REAL NOT NULL,
    learning_rate       REAL NOT NULL
);


-- =============================================================
-- INDEXES
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_portfolios_episode
    ON portfolios(episode_id, method);

CREATE INDEX IF NOT EXISTS idx_projects_status_episode_project
    ON projects_status(episode_id, i, method);

CREATE INDEX IF NOT EXISTS idx_projects_status_timestep
    ON projects_status(episode_id, t_episode, method);

CREATE INDEX IF NOT EXISTS idx_milestones_status_episode
    ON milestones_status(episode_id, method);

CREATE INDEX IF NOT EXISTS idx_training_log_global_step
    ON training_log(timestep_global);