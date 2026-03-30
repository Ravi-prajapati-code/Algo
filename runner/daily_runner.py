"""
Daily runner — the full pipeline, run once after market close each weekday.

Pipeline
--------
  1. Initialise DB
  2. Fetch fresh OHLCV data for all watchlist symbols
  3. Fetch market index (^NSEI) — MANDATORY, validated before continuing
  4. Detect market regime — UNKNOWN regime blocks ALL new BUY entries
  5. Compute technical indicators
  6. Load open positions
  7. Generate SELL/HOLD signals for open positions
  8. Generate BUY signals (only if regime is known and bullish)
  9. Process signals through portfolio manager (size, allocate, charge)
  10. Save signals + portfolio state to JSON
  11. Send Telegram alert

Safety guarantees
-----------------
- Index data is fetched BEFORE indicator computation.
- If index fetch returns empty or < MIN_INDEX_CANDLES bars, regime is set
  to UNKNOWN and ALL new BUY signals are suppressed.
- Existing positions are always evaluated for SELL/HOLD regardless of regime.
"""

import logging
import sys
import os
from datetime import date

# Ensure project root is on path when run from GitHub Actions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.repository import init_db, load_open_positions, load_snapshots, save_signal
from data.fetcher import fetch_all, fetch_index
from data.universe import get_all_symbols
from indicators.composite import compute_all
from strategy.signals import generate_signals
from strategy.market_filter import is_market_bullish
from strategy.regime import detect_regime, is_buy_allowed, MIN_INDEX_CANDLES
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

    # ── 2. Fetch market index FIRST (mandatory before strategy) ───────────
    logger.info("Fetching market index %s (need ≥ %d candles)…", MARKET_INDEX_SYMBOL, MIN_INDEX_CANDLES)
    index_df = fetch_index(MARKET_INDEX_SYMBOL, lookback_days=MIN_INDEX_CANDLES + 50)

    # ── 3. Validate index data before proceeding ──────────────────────────
    index_candles = len(index_df) if index_df is not None and not index_df.empty else 0
    if index_candles < MIN_INDEX_CANDLES:
        logger.warning(
            "Trading disabled due to insufficient market data: "
            "%d candles available for %s, need ≥ %d. "
            "All new BUY signals suppressed. "
            "SELL/HOLD signals for existing positions will still be generated.",
            index_candles, MARKET_INDEX_SYMBOL, MIN_INDEX_CANDLES,
        )
        regime = "UNKNOWN"
        market_bullish = False
    else:
        regime = detect_regime(index_df)
        market_bullish = is_buy_allowed(regime)
        if not market_bullish:
            logger.info(
                "Market regime is %s — new BUY signals suppressed.", regime
            )

    # ── 4. Fetch stock data ───────────────────────────────────────────────
    symbols = get_all_symbols()
    logger.info("Fetching data for %d symbols…", len(symbols))
    data = fetch_all(symbols)
    logger.info("Data fetched for %d symbols", len(data))

    if not data:
        logger.error("No stock data fetched — aborting")
        return

    # ── 5. Compute indicators ─────────────────────────────────────────────
    indicators = compute_all(data)
    logger.info("Indicators computed for %d symbols", len(indicators))

    # ── 6. Load open positions ────────────────────────────────────────────
    open_positions = load_open_positions()
    held_symbols   = {pos.symbol for pos in open_positions}

    # ── 7–8. Generate signals ─────────────────────────────────────────────
    # market_bullish=False blocks BUY screening inside generate_signals
    signals, updated_positions = generate_signals(
        today, indicators, open_positions, held_symbols,
        market_bullish=market_bullish,
    )

    buy_count  = sum(1 for s in signals if s.action == "BUY")
    sell_count = sum(1 for s in signals if s.action == "SELL")
    hold_count = sum(1 for s in signals if s.action == "HOLD")
    logger.info(
        "Signals generated — BUY: %d  SELL: %d  HOLD: %d  (regime=%s)",
        buy_count, sell_count, hold_count, regime,
    )

    # ── 9. Process signals through portfolio manager ──────────────────────
    prices = {sym: ind["close"] for sym, ind in indicators.items()}
    mgr = PortfolioManager(INITIAL_CAPITAL)
    mgr.process_signals(today, signals, prices)

    # ── 10. Write output files ────────────────────────────────────────────
    snapshots   = load_snapshots()
    latest_snap = snapshots[-1] if snapshots else None

    for sig in signals:
        save_signal(sig)

    write_signals(today, signals)

    if latest_snap:
        write_portfolio_state(today, latest_snap, mgr.open_positions, prices)

    # ── 11. Telegram alert ────────────────────────────────────────────────
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
