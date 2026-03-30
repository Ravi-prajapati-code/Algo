"""
Feature engineering for the ML prediction layer.

Transforms raw indicator dictionaries (produced by indicators/composite.py)
into a flat, normalised feature vector suitable for XGBoost / LightGBM.

Features are designed to be:
  - Dimensionless (ratios, z-scores, percentages)
  - Forward-looking-free (all computed from data ≤ today)
  - Robust to missing values (filled with column medians at training time)

Feature groups
--------------
  trend      : EMA ratios, golden/death cross flags, trend strength
  momentum   : RSI normalised, MACD normalised, momentum slope
  volatility : BB position, ATR % of price, volatility regime
  volume     : volume ratio, spike flag
  regime     : market regime one-hot encoding (4 columns)
  composite  : derived combinations (e.g. RSI × volume)
"""

from typing import Optional
import numpy as np
import pandas as pd

# ── Canonical feature order (must remain stable across train/predict) ──────
FEATURE_NAMES = [
    # Trend
    "ema_ratio",            # EMA20 / EMA50  (> 1 = uptrend)
    "price_to_ema_fast",    # close / EMA20
    "price_to_ema_slow",    # close / EMA50
    "golden_cross",         # 1 if golden cross recently
    "death_cross",          # 1 if death cross recently
    # Momentum
    "rsi_norm",             # (RSI - 50) / 50  → [-1, +1]
    "rsi_slope",            # RSI - RSI_prev  (momentum of momentum)
    "macd_hist_norm",       # MACD hist / price × 100
    "macd_bullish",         # 1 if MACD hist > 0
    "macd_turning_up",      # 1 if MACD hist > prev hist
    # Volatility
    "bb_pct",               # Bollinger %B  (position within bands)
    "atr_pct",              # ATR / price × 100 (normalised volatility)
    "above_bb_lower",       # 1 if price > lower BB
    "above_bb_mid",         # 1 if price > mid BB
    # Volume
    "vol_ratio",            # today's volume / 20-day avg
    "vol_spike",            # 1 if vol_ratio ≥ 1.5
    "vol_increasing",       # 1 if volume trending up (last 3 days)
    # Regime
    "regime_bull",          # 1 if BULL_TREND
    "regime_sideways",      # 1 if SIDEWAYS
    "regime_highvol",       # 1 if HIGH_VOL
    "regime_bear",          # 1 if BEAR_TREND
    # Composite
    "rsi_x_vol",            # rsi_norm × vol_ratio  (quality × volume)
    "trend_x_macd",         # ema_ratio × macd_bullish
]

_REGIME_CODES = {
    "BULL_TREND": (1, 0, 0, 0),
    "SIDEWAYS":   (0, 1, 0, 0),
    "HIGH_VOL":   (0, 0, 1, 0),
    "BEAR_TREND": (0, 0, 0, 1),
}


def build_feature_vector(ind: dict, regime: str = "BULL_TREND") -> dict:
    """
    Convert a single indicator dict into a named feature dictionary.

    Parameters
    ----------
    ind    : Indicator dict from indicators/composite.py
    regime : Market regime string (from strategy/regime.py)

    Returns
    -------
    dict mapping feature name → float value.
    All values are plain Python floats (JSON-serialisable).
    """
    close     = float(ind.get("close", 1) or 1)
    ema_fast  = float(ind.get("ema_fast", close) or close)
    ema_slow  = float(ind.get("ema_slow", close) or close)
    rsi       = float(ind.get("rsi", 50) or 50)
    rsi_prev  = float(ind.get("rsi_prev", rsi) or rsi)
    macd_hist = float(ind.get("macd_hist", 0) or 0)
    macd_prev = float(ind.get("macd_hist_prev", 0) or 0)
    bb_pct    = float(ind.get("bb_pct", 0.5) or 0.5)
    atr_pct   = float(ind.get("atr_pct", 0) or 0)
    vol_ratio = float(ind.get("vol_ratio", 1) or 1)

    r_bull, r_side, r_hvol, r_bear = _REGIME_CODES.get(regime, (1, 0, 0, 0))

    ema_ratio       = ema_fast / ema_slow if ema_slow > 0 else 1.0
    price_ema_fast  = close / ema_fast if ema_fast > 0 else 1.0
    price_ema_slow  = close / ema_slow if ema_slow > 0 else 1.0
    rsi_norm        = (rsi - 50.0) / 50.0
    rsi_slope       = rsi - rsi_prev
    macd_hist_norm  = (macd_hist / close * 100) if close > 0 else 0.0

    golden_cross  = int(bool(ind.get("golden_cross")))
    death_cross   = int(bool(ind.get("death_cross")))
    macd_bullish  = int(bool(ind.get("macd_bullish")))
    macd_turn_up  = int(bool(ind.get("macd_turning_up")))
    above_bb_low  = int(bool(ind.get("above_bb_lower")))
    above_bb_mid  = int(bool(ind.get("above_bb_mid")))
    vol_spike     = int(bool(ind.get("vol_spike")))
    vol_increasing= int(bool(ind.get("vol_increasing")))

    # Composite
    rsi_x_vol    = rsi_norm * vol_ratio
    trend_x_macd = ema_ratio * macd_bullish

    return {
        "ema_ratio":         round(ema_ratio, 4),
        "price_to_ema_fast": round(price_ema_fast, 4),
        "price_to_ema_slow": round(price_ema_slow, 4),
        "golden_cross":      golden_cross,
        "death_cross":       death_cross,
        "rsi_norm":          round(rsi_norm, 4),
        "rsi_slope":         round(rsi_slope, 4),
        "macd_hist_norm":    round(macd_hist_norm, 4),
        "macd_bullish":      macd_bullish,
        "macd_turning_up":   macd_turn_up,
        "bb_pct":            round(bb_pct, 4),
        "atr_pct":           round(atr_pct, 4),
        "above_bb_lower":    above_bb_low,
        "above_bb_mid":      above_bb_mid,
        "vol_ratio":         round(vol_ratio, 4),
        "vol_spike":         vol_spike,
        "vol_increasing":    vol_increasing,
        "regime_bull":       r_bull,
        "regime_sideways":   r_side,
        "regime_highvol":    r_hvol,
        "regime_bear":       r_bear,
        "rsi_x_vol":         round(rsi_x_vol, 4),
        "trend_x_macd":      round(trend_x_macd, 4),
    }


def build_feature_matrix(
    indicator_records: list[dict],
    regimes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from a list of indicator dicts.

    Parameters
    ----------
    indicator_records : List of indicator dicts (one per trade/day/symbol).
    regimes           : Optional list of regime strings aligned to records.

    Returns
    -------
    pd.DataFrame with FEATURE_NAMES columns, rows = records.
    """
    if regimes is None:
        regimes = ["BULL_TREND"] * len(indicator_records)

    rows = [
        build_feature_vector(ind, regime)
        for ind, regime in zip(indicator_records, regimes)
    ]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)

    # Fill any NaN with column median (robust to outliers)
    df = df.fillna(df.median(numeric_only=True))
    return df


def label_trades(trades: list, min_return_pct: float = 0.03) -> list[int]:
    """
    Create binary labels for supervised learning from a list of Trade objects.

    Label = 1 (WIN) if net_pnl / (entry_price × shares) ≥ min_return_pct
    Label = 0 (LOSS) otherwise.

    Parameters
    ----------
    trades          : List of Trade dataclass objects (from db/models.py).
    min_return_pct  : Minimum net return to classify as a win (default 3 %).

    Returns
    -------
    List of 0/1 labels aligned to `trades`.
    """
    labels = []
    for t in trades:
        if t.net_pnl is None or t.entry_price is None or t.shares is None:
            labels.append(0)
            continue
        cost_basis = t.entry_price * t.shares
        ret = t.net_pnl / cost_basis if cost_basis > 0 else 0.0
        labels.append(1 if ret >= min_return_pct else 0)
    return labels
