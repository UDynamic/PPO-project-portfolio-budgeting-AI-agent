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

    initial_budget_dist     TEXT NOT NULL,
    initial_budget_p1       REAL NOT NULL,
    initial_budget_p2       REAL,
    initial_budget_p3       REAL,
    initial_budget_p4       REAL,

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
    budget_dist             TEXT NOT NULL,
    budget_p1               REAL NOT NULL,
    budget_p2               REAL,
    budget_p3               REAL,
    budget_p4               REAL,

    margin_dist             TEXT NOT NULL,
    margin_p1               REAL NOT NULL,
    margin_p2               REAL,
    margin_p3               REAL,
    margin_p4               REAL,

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

    schedule_cap_dist       TEXT NOT NULL,
    schedule_cap_p1         REAL NOT NULL,
    schedule_cap_p2         REAL,
    schedule_cap_p3         REAL,
    schedule_cap_p4         REAL,

    cost_cap_dist           TEXT NOT NULL,
    cost_cap_p1             REAL NOT NULL,
    cost_cap_p2             REAL,
    cost_cap_p3             REAL,
    cost_cap_p4             REAL,

    cure_length_dist        TEXT NOT NULL,
    cure_length_p1          REAL NOT NULL,
    cure_length_p2          REAL,
    cure_length_p3          REAL,
    cure_length_p4          REAL,

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

    threshold_dist          TEXT NOT NULL,
    threshold_p1            REAL NOT NULL,
    threshold_p2            REAL,
    threshold_p3            REAL,
    threshold_p4            REAL,

    payment_weight_dist     TEXT NOT NULL,
    payment_weight_p1       REAL NOT NULL,
    payment_weight_p2       REAL,
    payment_weight_p3       REAL,
    payment_weight_p4       REAL,

    -- earliest_t_fraction: the fraction of project duration at which each
    -- intermediate milestone becomes eligible for certification.
    -- earliest_t[j] = start + round(threshold[j] * duration * fraction)
    -- fraction = 1.0 → earliest_t aligns exactly with the planned
    --                   completion date for that progress threshold.
    -- fraction < 1.0 → allows early certification for high performers.
    -- The final milestone (threshold = 1.0) always uses proj["finish"]
    -- regardless of this parameter — that is a contract invariant.
    earliest_t_fraction_dist    TEXT NOT NULL,
    earliest_t_fraction_p1      REAL NOT NULL,
    earliest_t_fraction_p2      REAL,
    earliest_t_fraction_p3      REAL,
    earliest_t_fraction_p4      REAL
);


CREATE TABLE IF NOT EXISTS projects_profile (

    episode_id          TEXT NOT NULL,
    config_id           TEXT NOT NULL REFERENCES environment_config(config_id),
    i                   INTEGER NOT NULL,

    budget              REAL NOT NULL,
    price               REAL NOT NULL,
    margin              REAL NOT NULL,
    start               INTEGER NOT NULL,
    finish              INTEGER NOT NULL,
    duration            INTEGER NOT NULL,
    scurve_a            REAL NOT NULL,
    scurve_b            REAL NOT NULL,
    advance_percent     REAL NOT NULL,
    advance_trigger     REAL NOT NULL,
    advance_recovery    REAL NOT NULL,
    retention_rate      REAL NOT NULL,
    schedule_cap        INTEGER NOT NULL,
    plan_deviation_threshold REAL NOT NULL DEFAULT 0.10,
    cost_cap            REAL NOT NULL,
    cure_length         INTEGER NOT NULL,

    PRIMARY KEY (episode_id, i)
);


CREATE TABLE IF NOT EXISTS milestones_profile (

    episode_id          TEXT NOT NULL,
    i                   INTEGER NOT NULL,
    j                   INTEGER NOT NULL,

    threshold           REAL NOT NULL,
    earliest_t          INTEGER NOT NULL,
    payment_weight      REAL NOT NULL,

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

    budget              REAL NOT NULL,
    inflow              REAL NOT NULL,
    outflow             REAL NOT NULL,
    reward              REAL NOT NULL,
    done                INTEGER NOT NULL,

    PRIMARY KEY (episode_id, t_episode, method)
);


CREATE TABLE IF NOT EXISTS projects_status (

    episode_id          TEXT NOT NULL,
    i                   INTEGER NOT NULL,
    t_episode           INTEGER NOT NULL,
    t_project           INTEGER,
    method              TEXT NOT NULL,

    status              TEXT,

    allocation          REAL,
    efficiency          REAL,

    progress            REAL,
    progress_plan       REAL,
    progress_increment  REAL,

    spi                 REAL,
    cpi                 REAL,
    tcpi                REAL,
    eac                 REAL,

    schedule_slip       REAL,
    plan_deviation      REAL,
    cost_overrun        REAL,
    forecast_finish     REAL,

    cure_remaining      INTEGER,

    advance_amount      REAL,
    payment_net         REAL,
    retention_release   REAL,
    settlement          REAL,

    PRIMARY KEY (episode_id, i, t_episode, method),
    FOREIGN KEY (episode_id, i) REFERENCES projects_profile(episode_id, i)
);


-- =============================================================
-- EVENT — MILESTONE CERTIFICATION
-- =============================================================

CREATE TABLE IF NOT EXISTS milestones_status (

    episode_id          TEXT NOT NULL,
    i                   INTEGER NOT NULL,
    j                   INTEGER NOT NULL,
    method              TEXT NOT NULL,

    certified           INTEGER NOT NULL DEFAULT 0,
    certified_t         INTEGER,
    payment_released    REAL,

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