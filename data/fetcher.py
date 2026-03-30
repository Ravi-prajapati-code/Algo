"""
yfinance wrapper with retry, rate limiting, and incremental caching.
Always fetches only missing dates to avoid redundant downloads.
"""

import time
import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Suppress yfinance's internal error/warning logs — our fetcher handles
# failures gracefully and logs its own clean warnings instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from config.settings import LOOKBACK_DAYS
from db import repository as repo

logger = logging.getLogger(__name__)

_RATE_LIMIT_DELAY = 0.3   # seconds between requests
_MAX_RETRIES = 3


def _fetch_raw(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download OHLCV from yfinance with retry logic."""
    for attempt in range(_MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=str(start), end=str(end), auto_adjust=True)
            if df.empty:
                logger.warning(f"[Fetcher] No data for {symbol} ({start} → {end})")
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index).date
            df.index.name = "date"
            df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
            df = df[df["volume"] > 0].dropna()
            time.sleep(_RATE_LIMIT_DELAY)
            return df
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"[Fetcher] {symbol} attempt {attempt+1} failed: {e}. Retrying in {wait}s")
            time.sleep(wait)
    logger.error(f"[Fetcher] Failed to fetch {symbol} after {_MAX_RETRIES} attempts")
    return pd.DataFrame()


def fetch_symbol(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch OHLCV for a symbol, using DB cache for already-downloaded dates.
    Returns a DataFrame with at least `lookback_days` of history.
    """
    today = date.today()
    fetch_from = today - timedelta(days=lookback_days + 10)  # +10 for weekends/holidays

    cached_latest = repo.latest_cached_date(symbol)

    if cached_latest is None or cached_latest < fetch_from:
        # No cache or cache is too old — full download
        df_new = _fetch_raw(symbol, fetch_from, today + timedelta(days=1))
    elif cached_latest < today - timedelta(days=1):
        # Incremental: fetch only missing days
        df_new = _fetch_raw(symbol, cached_latest + timedelta(days=1), today + timedelta(days=1))
    else:
        df_new = pd.DataFrame()  # Already up to date

    if not df_new.empty:
        df_new_reset = df_new.reset_index()
        repo.save_ohlcv(symbol, df_new_reset)

    # Load full history from cache
    df = repo.load_ohlcv(symbol, start=fetch_from)
    df.index = pd.to_datetime(df.index).map(lambda x: x.date())
    return df.sort_index()


def fetch_all(symbols: List[str], lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    Fetch OHLCV for all symbols. Returns {symbol: DataFrame}.
    Symbols with no data are excluded from the result.
    """
    result = {}
    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[Fetcher] {i}/{total} {symbol}")
        df = fetch_symbol(symbol, lookback_days)
        if not df.empty and len(df) >= 50:
            result[symbol] = df
        else:
            logger.warning(f"[Fetcher] Skipping {symbol}: insufficient data ({len(df)} rows)")
    return result


def get_latest_price(symbol: str) -> Optional[float]:
    """Return the most recent closing price from cache."""
    df = repo.load_ohlcv(symbol)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def fetch_index(symbol: str = "^NSEI", lookback_days: int = 250) -> pd.DataFrame:
    """
    Fetch index OHLCV (e.g. Nifty 50 ^NSEI) without caching to DB.
    Used only for market regime filter — not a tradeable instrument.
    """
    today = date.today()
    start = today - timedelta(days=lookback_days + 30)
    df = _fetch_raw(symbol, start, today + timedelta(days=1))
    return df.sort_index() if not df.empty else pd.DataFrame()
