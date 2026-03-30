"""
Market regime filter (simple bull/bear binary).

Used as a backward-compatible wrapper around the full regime detector.
`is_market_bullish()` returns False — never True — when index data is
missing or insufficient.  The old behaviour of defaulting to True (allow
buys) on missing data was a silent safety violation and has been removed.

For the full multi-regime classifier see strategy/regime.py.
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import MARKET_FILTER_ENABLED, MARKET_INDEX_SYMBOL, MARKET_FILTER_SMA
from strategy.regime import detect_regime, MIN_INDEX_CANDLES

logger = logging.getLogger(__name__)


def is_market_bullish(index_df: Optional[pd.DataFrame] = None) -> bool:
    """
    Returns True only if the market index is in a confirmed bull regime.

    Safety rules
    ------------
    - Filter disabled (MARKET_FILTER_ENABLED=False): returns True (opted-out).
    - index_df is None or has < {MIN_INDEX_CANDLES} candles: returns False.
      Previously this returned True ("allow buys by default") which was
      incorrect — insufficient data is NOT a signal to trade.
    - SMA200 is NaN: returns False.
    - Regime is UNKNOWN: returns False.

    Note: this function calls detect_regime() internally so the regime log
    message is emitted once; callers should not call both.
    """
    if not MARKET_FILTER_ENABLED:
        return True

    if index_df is None or len(index_df) < MIN_INDEX_CANDLES:
        available = len(index_df) if index_df is not None else 0
        logger.warning(
            "[MarketFilter] Trading disabled due to insufficient market data: "
            "%d candles available, need ≥ %d. Blocking all new BUY signals.",
            available, MIN_INDEX_CANDLES,
        )
        return False

    regime = detect_regime(index_df)

    # UNKNOWN, BEAR_TREND → not bullish
    bullish = regime in ("BULL_TREND", "SIDEWAYS", "HIGH_VOL")

    if not bullish:
        logger.info(
            "[MarketFilter] %s: regime=%s → BUY signals BLOCKED",
            MARKET_INDEX_SYMBOL, regime,
        )
    return bullish


def fetch_and_check() -> bool:
    """
    Fetch Nifty 50 index data and return bullish flag.

    Returns False (block buys) on any fetch failure — never silently
    allows trading when index data cannot be retrieved.
    """
    try:
        from data.fetcher import fetch_index
        df = fetch_index(MARKET_INDEX_SYMBOL, lookback_days=MIN_INDEX_CANDLES + 50)
        if df is None or df.empty:
            logger.warning(
                "[MarketFilter] fetch_and_check: index fetch returned empty DataFrame. "
                "Trading disabled due to insufficient market data."
            )
            return False
        return is_market_bullish(df)
    except Exception as e:
        logger.error(
            "[MarketFilter] fetch_and_check failed: %s. "
            "Trading disabled due to insufficient market data.", e,
        )
        return False
