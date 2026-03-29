"""Trend indicators: EMA and crossover detection."""

import pandas as pd
from config.settings import EMA_FAST, EMA_SLOW, EMA_CROSSOVER_LOOKBACK


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_trend(df: pd.DataFrame) -> dict:
    """
    Returns:
        ema_fast        — last EMA(20) value
        ema_slow        — last EMA(50) value
        above_ema_fast  — price above EMA20?
        uptrend         — EMA20 > EMA50?
        golden_cross    — EMA20 crossed above EMA50 within last N days?
        death_cross     — EMA20 crossed below EMA50 within last N days?
        ema_fast_series — full series (for composite use)
        ema_slow_series — full series
    """
    close = df["close"]
    ema_f = ema(close, EMA_FAST)
    ema_s = ema(close, EMA_SLOW)

    last_price = float(close.iloc[-1])
    last_ema_f = float(ema_f.iloc[-1])
    last_ema_s = float(ema_s.iloc[-1])

    # Detect crossovers in the last N candles
    window = min(EMA_CROSSOVER_LOOKBACK, len(ema_f) - 1)
    golden_cross = False
    death_cross = False
    for i in range(-window, 0):
        prev_diff = ema_f.iloc[i - 1] - ema_s.iloc[i - 1]
        curr_diff = ema_f.iloc[i] - ema_s.iloc[i]
        if prev_diff < 0 and curr_diff >= 0:
            golden_cross = True
        if prev_diff > 0 and curr_diff <= 0:
            death_cross = True

    return {
        "ema_fast": round(last_ema_f, 2),
        "ema_slow": round(last_ema_s, 2),
        "above_ema_fast": last_price >= last_ema_f,
        "uptrend": last_ema_f > last_ema_s,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "ema_fast_series": ema_f,
        "ema_slow_series": ema_s,
    }
