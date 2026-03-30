"""
Daily runner — the full hybrid pipeline, run once after market close.

Pipeline
--------
  1.  Initialise DB
  2.  Fetch market index (^NSEI) — validated first, blocks trading if insufficient
  3.  Detect market regime (BULL/BEAR/SIDEWAYS/HIGH_VOL/UNKNOWN)
  4.  Fetch stock OHLCV data
  5.  Compute relative strength for all symbols vs Nifty
  6.  Compute technical indicators (enriched with RS + quality fields)
  7.  Load open positions
  8.  Generate SELL/HOLD signals (always runs, even in BEAR/UNKNOWN)
  9.  Generate BUY signals (only when regime allows)
  10. Process signals through portfolio manager
  11. Write output JSON files
  12. Send Telegram alert

Safety guarantees
-----------------
- Index fetched and validated before any strategy code runs.
- UNKNOWN regime → all new BUYs suppressed, exits evaluated normally.
- RS computed once for the full watchlist (efficient batch operation).
- Rich per-stock log shows every action with price and P&L.
"""

import logging
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.repository import (
    init_db, load_open_positions, load_snapshots, save_signal,
)
from data.fetcher import fetch_all, fetch_index
from data.universe import get_all_symbols
from indicators.composite import compute_all
from strategy.signals import generate_signals
from strategy.market_filter import is_market_bullish
from strategy.regime import detect_regime, is_buy_allowed, MIN_INDEX_CANDLES
from strategy.relative_strength import compute_rs_for_all
from portfolio.manager import PortfolioManager
from runner.signal_output import write_signals, write_portfolio_state
from config.settings import INITIAL_CAPITAL, MARKET_INDEX_SYMBOL, MARKET_FILTER_SMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DailyRunner")


def run(today: date = None):
    if today is None:
        today = date.today()

    logger.info("=== Daily Runner started: %s ===", today)

    # ── 1. Initialise DB ──────────────────────────────────────────────────
    init_db()

    # ── 2. Fetch index (mandatory before everything else) ─────────────────
    logger.info(
        "Fetching market index %s (need ≥ %d candles)…",
        MARKET_INDEX_SYMBOL, MIN_INDEX_CANDLES,
    )
    index_df = fetch_index(MARKET_INDEX_SYMBOL, lookback_days=MIN_INDEX_CANDLES + 50)
    index_candles = len(index_df) if index_df is not None and not index_df.empty else 0

    # ── 3. Detect regime ──────────────────────────────────────────────────
    if index_candles < MIN_INDEX_CANDLES:
        logger.warning(
            "Trading disabled due to insufficient market data: "
            "%d candles for %s, need ≥ %d. "
            "New BUY entries suppressed; exits evaluated normally.",
            index_candles, MARKET_INDEX_SYMBOL, MIN_INDEX_CANDLES,
        )
        regime         = "UNKNOWN"
        market_bullish = False
    else:
        regime         = detect_regime(index_df)
        market_bullish = is_buy_allowed(regime)
        logger.info(
            "Market regime: %s | BUY entries %s",
            regime, "ALLOWED" if market_bullish else "BLOCKED",
        )

    # ── 4. Fetch stock data ───────────────────────────────────────────────
    symbols = get_all_symbols()
    logger.info("Fetching data for %d symbols…", len(symbols))
    data = fetch_all(symbols)
    logger.info("Data fetched for %d symbols", len(data))

    if not data:
        logger.error("No stock data fetched — aborting")
        return

    # ── 5. Relative strength (batch, before indicators) ───────────────────
    if market_bullish and index_df is not None and not index_df.empty:
        logger.info("Computing relative strength vs %s…", MARKET_INDEX_SYMBOL)
        rs_data = compute_rs_for_all(data, index_df)
        logger.info(
            "RS computed for %d symbols (%d pass filter)",
            len(rs_data), sum(1 for m in rs_data.values() if m.get("rs_qualified")),
        )
    else:
        rs_data = {}  # RS not needed when buys are blocked

    # ── 6. Compute indicators (enriched with RS + quality) ────────────────
    indicators = compute_all(data, rs_data=rs_data)
    logger.info("Indicators computed for %d symbols", len(indicators))

    # ── 7. Load open positions ────────────────────────────────────────────
    open_positions = load_open_positions()
    held_symbols   = {pos.symbol for pos in open_positions}

    # ── 8–9. Generate signals ─────────────────────────────────────────────
    snapshots   = load_snapshots()
    latest_snap = snapshots[-1] if snapshots else None
    pv          = latest_snap.total_value if latest_snap else INITIAL_CAPITAL
    cash_bal    = latest_snap.cash        if latest_snap else INITIAL_CAPITAL

    signals, updated_positions = generate_signals(
        today, indicators, open_positions, held_symbols,
        market_bullish=market_bullish,
        regime=regime,
        portfolio_value=pv,
        cash=cash_bal,
        initial_capital=INITIAL_CAPITAL,
    )

    # ── 10. Process through portfolio manager ─────────────────────────────
    prices = {sym: ind["close"] for sym, ind in indicators.items()}
    mgr    = PortfolioManager(INITIAL_CAPITAL)
    mgr.process_signals(today, signals, prices)

    # ── 11. Write output files ────────────────────────────────────────────
    snapshots   = load_snapshots()   # Reload after portfolio update
    latest_snap = snapshots[-1] if snapshots else None

    for sig in signals:
        save_signal(sig)
    write_signals(today, signals)
    if latest_snap:
        write_portfolio_state(today, latest_snap, mgr.open_positions, prices)

    # ── 12. Telegram alert ────────────────────────────────────────────────
    try:
        from notifications.telegram import send_daily_summary
        buy_sigs  = [s for s in signals if s.action == "BUY"]
        sell_sigs = [s for s in signals if s.action == "SELL"]
        send_daily_summary(today, buy_sigs, sell_sigs, latest_snap)
    except Exception as e:
        logger.warning("Telegram alert failed (non-fatal): %s", e)

    logger.info("=== Daily Runner complete: %s ===", today)


if __name__ == "__main__":
    run()
