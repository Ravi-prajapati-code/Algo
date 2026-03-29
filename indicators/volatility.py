"""Volatility indicators: Bollinger Bands and ATR."""

import pandas as pd


def compute_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    last = float(series.iloc[-1])
    return {
        "bb_upper": round(float(upper.iloc[-1]), 2),
        "bb_mid":   round(float(sma.iloc[-1]), 2),
        "bb_lower": round(float(lower.iloc[-1]), 2),
        "bb_pct":   round((last - float(lower.iloc[-1])) /
                          (float(upper.iloc[-1]) - float(lower.iloc[-1]) + 1e-9), 4),
        "above_bb_lower": last >= float(lower.iloc[-1]),
        "above_bb_mid":   last >= float(sma.iloc[-1]),
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> dict:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    last_atr = float(atr.iloc[-1])
    last_close = float(close.iloc[-1])
    return {
        "atr": round(last_atr, 2),
        "atr_pct": round(last_atr / last_close, 4),   # ATR as % of price
    }


def compute_volatility(df: pd.DataFrame) -> dict:
    bb = compute_bollinger(df["close"])
    atr = compute_atr(df)
    return {**bb, **atr}
