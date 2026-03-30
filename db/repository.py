"""SQLite repository — all DB reads and writes go through here."""

import sqlite3
import json
import os
from datetime import date, datetime
from typing import List, Optional
import pandas as pd

from db.models import Position, Trade, Signal, PortfolioSnapshot
from config.settings import DATA_CACHE_DB


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATA_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(DATA_CACHE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables and indexes from schema.sql if they don't exist."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    with _connect() as conn:
        conn.executescript(sql)
    _migrate_db()
    print(f"[DB] Initialized: {DATA_CACHE_DB}")


def _migrate_db():
    """
    Apply incremental schema migrations for new columns added in v2.
    Safe to run multiple times (ALTER TABLE IF NOT EXISTS equivalent via try/except).
    """
    migrations = [
        # positions — new risk metadata columns
        "ALTER TABLE positions ADD COLUMN atr_at_entry  REAL",
        "ALTER TABLE positions ADD COLUMN ml_confidence REAL",
        "ALTER TABLE positions ADD COLUMN regime        TEXT",
        "ALTER TABLE positions ADD COLUMN risk_score    REAL",
        "ALTER TABLE positions ADD COLUMN sizing_method TEXT",
        # trades — new execution metadata
        "ALTER TABLE trades ADD COLUMN slippage_pct  REAL",
        "ALTER TABLE trades ADD COLUMN fill_type     TEXT",
        "ALTER TABLE trades ADD COLUMN regime        TEXT",
        "ALTER TABLE trades ADD COLUMN ml_confidence REAL",
        # signals — ML and regime
        "ALTER TABLE signals ADD COLUMN ml_win_prob   REAL",
        "ALTER TABLE signals ADD COLUMN ml_exp_return REAL",
        "ALTER TABLE signals ADD COLUMN regime        TEXT",
        # portfolio_snapshots — risk state
        "ALTER TABLE portfolio_snapshots ADD COLUMN drawdown_pct REAL DEFAULT 0",
        "ALTER TABLE portfolio_snapshots ADD COLUMN regime       TEXT",
        "ALTER TABLE portfolio_snapshots ADD COLUMN kill_switch  INTEGER DEFAULT 0",
    ]
    with _connect() as conn:
        for sql in migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists — skip


# ─── OHLCV CACHE ─────────────────────────────────────────────────────────────

def save_ohlcv(symbol: str, df: pd.DataFrame):
    """Upsert OHLCV rows for a symbol. df must have columns: date,open,high,low,close,volume."""
    rows = [
        (symbol, str(row.date), row.open, row.high, row.low, row.close, int(row.volume))
        for row in df.itertuples()
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv_cache VALUES (?,?,?,?,?,?,?)", rows
        )


def load_ohlcv(symbol: str, start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    sql = "SELECT date,open,high,low,close,volume FROM ohlcv_cache WHERE symbol=?"
    params: list = [symbol]
    if start:
        sql += " AND date >= ?"
        params.append(str(start))
    if end:
        sql += " AND date <= ?"
        params.append(str(end))
    sql += " ORDER BY date"
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df


def latest_cached_date(symbol: str) -> Optional[date]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT MAX(date) as d FROM ohlcv_cache WHERE symbol=?", (symbol,)
        ).fetchone()
    if row and row["d"]:
        return datetime.strptime(row["d"], "%Y-%m-%d").date()
    return None


# ─── POSITIONS ───────────────────────────────────────────────────────────────

def save_position(pos: Position):
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO positions
            (symbol, sector, entry_date, entry_price, shares,
             stop_loss, take_profit, trailing_stop, peak_price, status)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (pos.symbol, pos.sector, str(pos.entry_date), pos.entry_price, pos.shares,
              pos.stop_loss, pos.take_profit, pos.trailing_stop, pos.peak_price, pos.status))


def load_open_positions() -> List[Position]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='OPEN'"
        ).fetchall()
    return [_row_to_position(r) for r in rows]


def close_position(symbol: str):
    with _connect() as conn:
        conn.execute(
            "UPDATE positions SET status='CLOSED' WHERE symbol=?", (symbol,)
        )


def _row_to_position(row: sqlite3.Row) -> Position:
    return Position(
        id=row["id"], symbol=row["symbol"], sector=row["sector"],
        entry_date=datetime.strptime(row["entry_date"], "%Y-%m-%d").date(),
        entry_price=row["entry_price"], shares=row["shares"],
        stop_loss=row["stop_loss"], take_profit=row["take_profit"],
        trailing_stop=row["trailing_stop"], peak_price=row["peak_price"],
        status=row["status"],
    )


# ─── TRADES ──────────────────────────────────────────────────────────────────

def save_trade(trade: Trade):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO trades
            (symbol, sector, entry_date, exit_date, entry_price, exit_price,
             shares, gross_pnl, charges, net_pnl, exit_reason, hold_days)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (trade.symbol, trade.sector,
              str(trade.entry_date), str(trade.exit_date) if trade.exit_date else None,
              trade.entry_price, trade.exit_price, trade.shares,
              trade.gross_pnl, trade.charges, trade.net_pnl,
              trade.exit_reason, trade.hold_days))


def load_trades(symbol: Optional[str] = None) -> List[Trade]:
    sql = "SELECT * FROM trades"
    params: list = []
    if symbol:
        sql += " WHERE symbol=?"
        params.append(symbol)
    sql += " ORDER BY entry_date"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_trade(r) for r in rows]


def _row_to_trade(row: sqlite3.Row) -> Trade:
    return Trade(
        id=row["id"], symbol=row["symbol"], sector=row["sector"],
        entry_date=datetime.strptime(row["entry_date"], "%Y-%m-%d").date(),
        exit_date=datetime.strptime(row["exit_date"], "%Y-%m-%d").date() if row["exit_date"] else None,
        entry_price=row["entry_price"], exit_price=row["exit_price"],
        shares=row["shares"], gross_pnl=row["gross_pnl"],
        charges=row["charges"], net_pnl=row["net_pnl"],
        exit_reason=row["exit_reason"], hold_days=row["hold_days"],
    )


# ─── SIGNALS ─────────────────────────────────────────────────────────────────

def save_signal(sig: Signal):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO signals (date, symbol, action, score, price, reason, indicators_json)
            VALUES (?,?,?,?,?,?,?)
        """, (str(sig.date), sig.symbol, sig.action, sig.score, sig.price,
              sig.reason, json.dumps(sig.indicators)))


def load_signals(on_date: Optional[date] = None) -> List[Signal]:
    sql = "SELECT * FROM signals"
    params: list = []
    if on_date:
        sql += " WHERE date=?"
        params.append(str(on_date))
    sql += " ORDER BY score DESC"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_signal(r) for r in rows]


def _row_to_signal(row: sqlite3.Row) -> Signal:
    return Signal(
        id=row["id"],
        date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
        symbol=row["symbol"], action=row["action"],
        score=row["score"] or 0.0, price=row["price"] or 0.0,
        reason=row["reason"] or "",
        indicators=json.loads(row["indicators_json"] or "{}"),
    )


# ─── RISK EVENTS ─────────────────────────────────────────────────────────────

def save_risk_event(event_type: str, description: str, portfolio_value: float = 0.0,
                    drawdown_pct: float = 0.0, metadata: dict = None):
    """Persist a risk event (kill switch, drawdown alert, API error, etc.)."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO risk_events
               (event_type, description, portfolio_value, drawdown_pct, metadata_json)
               VALUES (?,?,?,?,?)""",
            (event_type, description, portfolio_value, drawdown_pct,
             json.dumps(metadata or {}))
        )


# ─── PORTFOLIO SNAPSHOTS ─────────────────────────────────────────────────────

def save_snapshot(snap: PortfolioSnapshot):
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (date, cash, invested, total_value, open_positions, daily_pnl, cumulative_pnl)
            VALUES (?,?,?,?,?,?,?)
        """, (str(snap.date), snap.cash, snap.invested, snap.total_value,
              snap.open_positions, snap.daily_pnl, snap.cumulative_pnl))


def load_snapshots(start: Optional[date] = None, end: Optional[date] = None) -> List[PortfolioSnapshot]:
    sql = "SELECT * FROM portfolio_snapshots"
    params: list = []
    conditions = []
    if start:
        conditions.append("date >= ?")
        params.append(str(start))
    if end:
        conditions.append("date <= ?")
        params.append(str(end))
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY date"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_snapshot(r) for r in rows]


def _row_to_snapshot(row: sqlite3.Row) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        id=row["id"],
        date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
        cash=row["cash"], invested=row["invested"],
        total_value=row["total_value"], open_positions=row["open_positions"],
        daily_pnl=row["daily_pnl"], cumulative_pnl=row["cumulative_pnl"],
    )
