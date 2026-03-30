"""
Market regime filter.
Only allow BUY signals when the broad market index (Nifty 50) is in an uptrend.
This single filter prevents buying stocks during bear markets and
significantly improves strategy performance over full market cycles.

Logic:
  - Fetch Nifty 50 daily prices
  - Compute 200-day SMA
  - BULL if index > SMA200  →  allow BUY signals
  - BEAR if index < SMA200  →  block ALL new BUY signals
    (existing positions are still managed: SELL/HOLD work normally)
"""

import logging
import pandas as pd
from datetime import date
from typing import Optional

from config.settings import MARKET_FILTER_ENABLED, MARKET_INDEX_SYMBOL, MARKET_FILTER_SMA

logger = logging.getLogger(__name__)


def is_market_bullish(index_df: Optional[pd.DataFrame] = None) -> bool:
    """
    Returns True if the market index is in bull regime (above 200-SMA).
    Returns True (allow buys) if filter is disabled or data unavailable.
    """
    if not MARKET_FILTER_ENABLED:
        return True

    if index_df is None or len(index_df) < MARKET_FILTER_SMA:
        logger.warning("[MarketFilter] Insufficient index data — allowing buys by default")
        return True

    close = index_df["close"]
    sma200 = close.rolling(MARKET_FILTER_SMA).mean()
    last_close = float(close.iloc[-1])
    last_sma   = float(sma200.iloc[-1])

    bullish = last_close > last_sma
    regime  = "BULL" if bullish else "BEAR"
    logger.info(
        f"[MarketFilter] {MARKET_INDEX_SYMBOL}: {last_close:.0f} vs SMA{MARKET_FILTER_SMA} "
        f"{last_sma:.0f} → {regime}"
    )
    return bullish


def fetch_and_check() -> bool:
    """Convenience: fetch Nifty 50 and return bullish flag. Used by daily runner."""
    try:
        from data.fetcher import fetch_index
        df = fetch_index(MARKET_INDEX_SYMBOL, lookback_days=MARKET_FILTER_SMA + 50)
        return is_market_bullish(df)
    except Exception as e:
        logger.warning(f"[MarketFilter] Failed to fetch index: {e} — allowing buys")
        return True
