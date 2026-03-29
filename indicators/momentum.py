"""Momentum indicators: RSI and MACD."""

import pandas as pd
from config.settings import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_momentum(df: pd.DataFrame) -> dict:
    """
    Returns:
        rsi              — latest RSI value
        rsi_prev         — RSI one bar ago
        macd             — latest MACD line
        macd_signal      — latest MACD signal line
        macd_hist        — latest histogram
        macd_hist_prev   — histogram one bar ago
        macd_bullish     — histogram is positive
        macd_turning_up  — histogram increasing (turning bullish)
    """
    close = df["close"]
    rsi = compute_rsi(close)
    macd_line, signal_line, histogram = compute_macd(close)

    last_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2]) if len(rsi) >= 2 else last_rsi
    last_macd = float(macd_line.iloc[-1])
    last_signal = float(signal_line.iloc[-1])
    last_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2]) if len(histogram) >= 2 else last_hist

    return {
        "rsi": round(last_rsi, 2),
        "rsi_prev": round(prev_rsi, 2),
        "macd": round(last_macd, 4),
        "macd_signal": round(last_signal, 4),
        "macd_hist": round(last_hist, 4),
        "macd_hist_prev": round(prev_hist, 4),
        "macd_bullish": last_hist > 0,
        "macd_turning_up": last_hist > prev_hist,
        "rsi_series": rsi,
        "macd_hist_series": histogram,
    }
