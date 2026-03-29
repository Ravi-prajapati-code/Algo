"""
Daily runner — the full pipeline, run once after market close each weekday.

Pipeline:
  1. Init DB
  2. Fetch fresh OHLCV data for all watchlist symbols
  3. Compute technical indicators
  4. Load open positions
  5. Generate SELL signals for open positions
  6. Generate BUY signals for watchlist candidates
  7. Process signals through portfolio manager (size, allocate, charge)
  8. Save signals + portfolio state to JSON
  9. Send Telegram alert
"""

import logging
import sys
import os
from datetime import date

# Ensure project root is on path when run from GitHub Actions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.repository import init_db, load_open_positions, load_snapshots, save_signal
from data.fetcher import fetch_all
from data.universe import get_all_symbols
from indicators.composite import compute_all
from strategy.signals import generate_signals
from portfolio.manager import PortfolioManager
from runner.signal_output import write_signals, write_portfolio_state
from config.settings import INITIAL_CAPITAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DailyRunner")


def run(today: date = None):
    if today is None:
        today = date.today()

    logger.info(f"=== Daily Runner started: {today} ===")

    # 1. Initialise DB
    init_db()

    # 2. Fetch data
    symbols = get_all_symbols()
    logger.info(f"Fetching data for {len(symbols)} symbols…")
    data = fetch_all(symbols)
    logger.info(f"Data fetched for {len(data)} symbols")

    if not data:
        logger.error("No data fetched — aborting")
        return

    # 3. Compute indicators
    indicators = compute_all(data)
    logger.info(f"Indicators computed for {len(indicators)} symbols")

    # 4. Load open positions
    open_positions = load_open_positions()
    held_symbols = {pos.symbol for pos in open_positions}

    # 5 & 6. Generate signals
    signals, updated_positions = generate_signals(
        today, indicators, open_positions, held_symbols
    )

    # 7. Process signals through portfolio manager
    prices = {sym: ind["close"] for sym, ind in indicators.items()}
    mgr = PortfolioManager(INITIAL_CAPITAL)
    mgr.process_signals(today, signals, prices)

    # 8. Write output files
    snapshots = load_snapshots()
    latest_snap = snapshots[-1] if snapshots else None

    for sig in signals:
        save_signal(sig)

    write_signals(today, signals)

    if latest_snap:
        write_portfolio_state(today, latest_snap, mgr.open_positions, prices)

    # 9. Telegram alert
    try:
        from notifications.telegram import send_daily_summary
        buy_sigs  = [s for s in signals if s.action == "BUY"]
        sell_sigs = [s for s in signals if s.action == "SELL"]
        send_daily_summary(today, buy_sigs, sell_sigs, latest_snap)
    except Exception as e:
        logger.warning(f"Telegram alert failed (non-fatal): {e}")

    logger.info(f"=== Daily Runner complete: {today} ===")


if __name__ == "__main__":
    run()
