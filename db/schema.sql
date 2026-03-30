-- SQLite schema for the swing trading platform
-- Run via: python main.py initdb

-- ── OHLCV CACHE ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv_cache (
    symbol      TEXT    NOT NULL,
    date        DATE    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- ── OPEN POSITIONS ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL UNIQUE,
    sector          TEXT    NOT NULL,
    entry_date      DATE    NOT NULL,
    entry_price     REAL    NOT NULL,
    shares          INTEGER NOT NULL,
    stop_loss       REAL    NOT NULL,
    take_profit     REAL    NOT NULL,
    trailing_stop   REAL    NOT NULL,
    peak_price      REAL    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED
    -- New: risk and ML metadata
    atr_at_entry    REAL,
    ml_confidence   REAL,
    regime          TEXT,
    risk_score      REAL,
    sizing_method   TEXT
);

-- ── CLOSED TRADES ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    sector          TEXT    NOT NULL,
    entry_date      DATE    NOT NULL,
    exit_date       DATE,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    shares          INTEGER NOT NULL,
    gross_pnl       REAL,
    charges         REAL,
    net_pnl         REAL,
    exit_reason     TEXT,   -- STOP_LOSS | TAKE_PROFIT | TRAILING | SIGNAL | MANUAL | END_OF_BACKTEST
    hold_days       INTEGER,
    -- New: execution metadata
    slippage_pct    REAL,
    fill_type       TEXT,   -- FULL | PARTIAL | GAP
    regime          TEXT,
    ml_confidence   REAL
);

-- ── DAILY SIGNALS ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE    NOT NULL,
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,   -- BUY | SELL | HOLD
    score           REAL,
    price           REAL,
    reason          TEXT,
    indicators_json TEXT,
    -- New: ML and regime fields
    ml_win_prob     REAL,
    ml_exp_return   REAL,
    regime          TEXT
);

-- ── PORTFOLIO SNAPSHOTS ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE    NOT NULL UNIQUE,
    cash            REAL    NOT NULL,
    invested        REAL    NOT NULL,
    total_value     REAL    NOT NULL,
    open_positions  INTEGER NOT NULL,
    daily_pnl       REAL    NOT NULL DEFAULT 0,
    cumulative_pnl  REAL    NOT NULL DEFAULT 0,
    -- New: risk state
    drawdown_pct    REAL    DEFAULT 0,
    regime          TEXT,
    kill_switch     INTEGER DEFAULT 0   -- 0=off, 1=on
);

-- ── RISK EVENTS ───────────────────────────────────────────────────────────
-- Logs kill-switch activations, drawdown alerts, API failures, etc.
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type      TEXT     NOT NULL,  -- KILL_SWITCH | DRAWDOWN_ALERT | API_FAILURE | ...
    description     TEXT,
    portfolio_value REAL,
    drawdown_pct    REAL,
    metadata_json   TEXT
);

-- ── STRATEGY PERFORMANCE ──────────────────────────────────────────────────
-- Aggregated performance by symbol and sector for analysis
CREATE TABLE IF NOT EXISTS strategy_performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period          TEXT    NOT NULL,   -- 'ALL' | '2024' | 'Q1-2024' etc.
    symbol          TEXT,               -- NULL = aggregate
    sector          TEXT,               -- NULL = aggregate
    total_trades    INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    total_net_pnl   REAL    NOT NULL DEFAULT 0,
    avg_hold_days   REAL,
    best_trade      REAL,
    worst_trade     REAL,
    last_updated    DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (period, symbol, sector)
);

-- ── ML MODEL RUNS ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_model_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    n_trades        INTEGER,
    win_rate        REAL,
    cv_auc          REAL,
    cv_accuracy     REAL,
    feature_version TEXT,
    notes           TEXT
);

-- ── INDEXES ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol       ON ohlcv_cache (symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date         ON ohlcv_cache (date);
CREATE INDEX IF NOT EXISTS idx_signals_date       ON signals (date);
CREATE INDEX IF NOT EXISTS idx_signals_action     ON signals (action);
CREATE INDEX IF NOT EXISTS idx_trades_symbol      ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date  ON trades (entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades (exit_reason);
CREATE INDEX IF NOT EXISTS idx_risk_events_type   ON risk_events (event_type);
CREATE INDEX IF NOT EXISTS idx_snapshots_date     ON portfolio_snapshots (date);
