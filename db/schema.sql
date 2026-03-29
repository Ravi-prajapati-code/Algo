-- SQLite schema for the swing trading tool
-- Run via: python -c "from db.repository import init_db; init_db()"

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
    status          TEXT    NOT NULL DEFAULT 'OPEN'   -- OPEN | CLOSED
);

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
    exit_reason     TEXT,           -- STOP_LOSS | TAKE_PROFIT | TRAILING | SIGNAL | MANUAL
    hold_days       INTEGER
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE    NOT NULL,
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,   -- BUY | SELL | HOLD
    score           REAL,
    price           REAL,
    reason          TEXT,
    indicators_json TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE    NOT NULL UNIQUE,
    cash            REAL    NOT NULL,
    invested        REAL    NOT NULL,
    total_value     REAL    NOT NULL,
    open_positions  INTEGER NOT NULL,
    daily_pnl       REAL    NOT NULL DEFAULT 0,
    cumulative_pnl  REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol  ON ohlcv_cache (symbol);
CREATE INDEX IF NOT EXISTS idx_signals_date  ON signals (date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
