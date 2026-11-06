CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    position    TEXT NOT NULL,
    manager_id  TEXT REFERENCES employees (employee_id)
);

CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id    TEXT NOT NULL REFERENCES employees (employee_id),
    year           INTEGER NOT NULL,
    annual_quota   INTEGER NOT NULL CHECK (annual_quota >= 0),
    annual_used    INTEGER NOT NULL CHECK (annual_used >= 0 AND annual_used <= annual_quota),
    sick_days_used INTEGER NOT NULL DEFAULT 0 CHECK (sick_days_used >= 0),
    PRIMARY KEY (employee_id, year)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    question    TEXT NOT NULL,
    employee_id TEXT,
    answer      TEXT NOT NULL,
    route       TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    sources     JSONB NOT NULL DEFAULT '[]',
    steps       JSONB NOT NULL DEFAULT '[]',
    latency_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS agent_runs_created_at_idx ON agent_runs (created_at DESC);
