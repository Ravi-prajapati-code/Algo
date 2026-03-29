"""
Composite indicator builder.
Takes a single symbol's OHLCV DataFrame and returns a flat dict
of all indicator values — used by the strategy layer.
"""

import logging
import pandas as pd

from indicators.trend import compute_trend
from indicators.momentum import compute_momentum
from indicators.volatility import compute_volatility
from indicators.volume import compute_volume

logger = logging.getLogger(__name__)

MIN_ROWS = 60   # Minimum candles needed for reliable indicators


def compute_indicators(df: pd.DataFrame) -> dict | None:
    """
    Compute all indicators for a symbol's OHLCV data.
    Returns None if data is insufficient.
    The returned dict is JSON-serialisable (no Series objects).
    """
    if df is None or len(df) < MIN_ROWS:
        logger.warning(f"Insufficient data: {len(df) if df is not None else 0} rows (need {MIN_ROWS})")
        return None

    try:
        trend = compute_trend(df)
        momentum = compute_momentum(df)
        volatility = compute_volatility(df)
        volume = compute_volume(df)

        # Latest close price
        last_close = float(df["close"].iloc[-1])
        last_high  = float(df["high"].iloc[-1])
        last_low   = float(df["low"].iloc[-1])

        # Remove non-serialisable Series before returning
        serialisable = {
            "close":          round(last_close, 2),
            "high":           round(last_high, 2),
            "low":            round(last_low, 2),
            # trend
            "ema_fast":       trend["ema_fast"],
            "ema_slow":       trend["ema_slow"],
            "above_ema_fast": trend["above_ema_fast"],
            "uptrend":        trend["uptrend"],
            "golden_cross":   trend["golden_cross"],
            "death_cross":    trend["death_cross"],
            # momentum
            "rsi":            momentum["rsi"],
            "rsi_prev":       momentum["rsi_prev"],
            "macd":           momentum["macd"],
            "macd_signal":    momentum["macd_signal"],
            "macd_hist":      momentum["macd_hist"],
            "macd_hist_prev": momentum["macd_hist_prev"],
            "macd_bullish":   momentum["macd_bullish"],
            "macd_turning_up": momentum["macd_turning_up"],
            # volatility
            "bb_upper":       volatility["bb_upper"],
            "bb_mid":         volatility["bb_mid"],
            "bb_lower":       volatility["bb_lower"],
            "bb_pct":         volatility["bb_pct"],
            "above_bb_lower": volatility["above_bb_lower"],
            "above_bb_mid":   volatility["above_bb_mid"],
            "atr":            volatility["atr"],
            "atr_pct":        volatility["atr_pct"],
            # volume
            "vol_avg":        volume["vol_avg"],
            "vol_today":      volume["vol_today"],
            "vol_ratio":      volume["vol_ratio"],
            "vol_spike":      volume["vol_spike"],
            "vol_increasing": volume["vol_increasing"],
        }
        return serialisable

    except Exception as e:
        logger.error(f"compute_indicators failed: {e}")
        return None


def compute_all(data: dict) -> dict:
    """
    data: {symbol: DataFrame}
    Returns {symbol: indicators_dict}  (symbols with None are excluded)
    """
    result = {}
    for symbol, df in data.items():
        ind = compute_indicators(df)
        if ind is not None:
            result[symbol] = ind
    return result
