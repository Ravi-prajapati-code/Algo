"""
Composite indicator builder.

Takes a single symbol's OHLCV DataFrame and returns a flat dict of all
indicator values — used by the strategy layer.  The returned dict is
JSON-serialisable (no pandas Series objects).

New in hybrid model
-------------------
compute_indicators() now accepts an optional `index_df` parameter and an
optional `symbol` string.  When provided:
  - RS ratio and RS rank fields are computed (strategy/relative_strength.py)
  - Quality score is computed (strategy/quality_filter.py)
  - 52-week high proximity is added

compute_all() has a matching `index_df` / `rs_data` parameter so callers
can pre-compute RS for the whole watchlist in one pass (efficient) and
inject per-symbol results when building the indicator dict.
"""

import logging
import pandas as pd
from typing import Optional

from indicators.trend import compute_trend
from indicators.momentum import compute_momentum
from indicators.volatility import compute_volatility
from indicators.volume import compute_volume

logger = logging.getLogger(__name__)

MIN_ROWS = 60   # Minimum candles needed for reliable indicators


def compute_indicators(
    df: pd.DataFrame,
    symbol: str = "",
    rs_metrics: Optional[dict] = None,
) -> dict | None:
    """
    Compute all indicators for a symbol's OHLCV data.

    Parameters
    ----------
    df         : OHLCV DataFrame indexed by date.
    symbol     : Stock symbol — used for quality score lookup.
    rs_metrics : Pre-computed RS dict from relative_strength.compute_rs_for_all().
                 Keys: rs_ratio, rs_ratio_1m, rs_outperforming,
                       rs_accelerating, rs_rank, rs_qualified.
                 If None, RS fields are set to neutral defaults.

    Returns
    -------
    Flat dict of all indicators, or None if data is insufficient.
    """
    if df is None or len(df) < MIN_ROWS:
        logger.warning(
            "Insufficient data for %s: %d rows (need %d)",
            symbol or "?", len(df) if df is not None else 0, MIN_ROWS,
        )
        return None

    try:
        trend      = compute_trend(df)
        momentum   = compute_momentum(df)
        volatility = compute_volatility(df)
        volume     = compute_volume(df)

        last_close = float(df["close"].iloc[-1])
        last_high  = float(df["high"].iloc[-1])
        last_low   = float(df["low"].iloc[-1])

        # ── 20-day high (breakout reference level) ────────────────────────
        lookback_20  = df["high"].tail(20)
        high_20d     = round(float(lookback_20.max()), 2) if len(lookback_20) >= 20 else round(last_high, 2)

        # ── 52-week high ──────────────────────────────────────────────────
        lookback_252 = df["high"].tail(252)
        week52_high  = round(float(lookback_252.max()), 2) if len(lookback_252) >= 50 else 0.0

        # ── RS fields (injected from pre-computed batch) ──────────────────
        if rs_metrics:
            rs_ratio         = rs_metrics.get("rs_ratio", 0.0)
            rs_ratio_1m      = rs_metrics.get("rs_ratio_1m", 0.0)
            rs_outperforming = bool(rs_metrics.get("rs_outperforming", False))
            rs_accelerating  = bool(rs_metrics.get("rs_accelerating", False))
            rs_rank          = rs_metrics.get("rs_rank", 0.0)
            rs_qualified     = bool(rs_metrics.get("rs_qualified", False))
        else:
            rs_ratio = rs_ratio_1m = rs_rank = 0.0
            rs_outperforming = rs_accelerating = rs_qualified = False

        # ── Quality score ─────────────────────────────────────────────────
        quality_score_val = 0.0
        if symbol:
            from strategy.quality_filter import quality_score as _qs
            # Build a minimal ind dict for quality_score (needs bb + vol fields)
            _mini = {
                "close":      last_close,
                "week52_high": week52_high,
                "vol_avg":    volume["vol_avg"],
                "vol_today":  volume["vol_today"],
                "bb_upper":   volatility["bb_upper"],
                "bb_lower":   volatility["bb_lower"],
                "bb_mid":     volatility["bb_mid"],
            }
            quality_score_val = _qs(symbol, _mini)

        serialisable = {
            "close":              round(last_close, 2),
            "high":               round(last_high, 2),
            "low":                round(last_low, 2),
            # ── trend ──────────────────────────────────────────────────
            "ema_fast":           trend["ema_fast"],
            "ema_slow":           trend["ema_slow"],
            "above_ema_fast":     trend["above_ema_fast"],
            "uptrend":            trend["uptrend"],
            "golden_cross":       trend["golden_cross"],
            "death_cross":        trend["death_cross"],
            # ── momentum ───────────────────────────────────────────────
            "rsi":                momentum["rsi"],
            "rsi_prev":           momentum["rsi_prev"],
            "macd":               momentum["macd"],
            "macd_signal":        momentum["macd_signal"],
            "macd_hist":          momentum["macd_hist"],
            "macd_hist_prev":     momentum["macd_hist_prev"],
            "macd_bullish":       momentum["macd_bullish"],
            "macd_turning_up":    momentum["macd_turning_up"],
            # ── volatility ─────────────────────────────────────────────
            "bb_upper":           volatility["bb_upper"],
            "bb_mid":             volatility["bb_mid"],
            "bb_lower":           volatility["bb_lower"],
            "bb_pct":             volatility["bb_pct"],
            "above_bb_lower":     volatility["above_bb_lower"],
            "above_bb_mid":       volatility["above_bb_mid"],
            "atr":                volatility["atr"],
            "atr_pct":            volatility["atr_pct"],
            # ── volume ─────────────────────────────────────────────────
            "vol_avg":            volume["vol_avg"],
            "vol_today":          volume["vol_today"],
            "vol_ratio":          volume["vol_ratio"],
            "vol_spike":          volume["vol_spike"],
            "vol_increasing":     volume["vol_increasing"],
            # ── price context ──────────────────────────────────────────
            "high_20d":           high_20d,
            "week52_high":        week52_high,
            # ── relative strength ──────────────────────────────────────
            "rs_ratio":           round(rs_ratio, 4),
            "rs_ratio_1m":        round(rs_ratio_1m, 4),
            "rs_outperforming":   rs_outperforming,
            "rs_accelerating":    rs_accelerating,
            "rs_rank":            round(rs_rank, 1),
            "rs_qualified":       rs_qualified,
            # ── quality ────────────────────────────────────────────────
            "quality_score_val":  round(quality_score_val, 1),
        }
        return serialisable

    except Exception as e:
        logger.error("compute_indicators failed for %s: %s", symbol or "?", e)
        return None


def compute_all(
    data: dict,
    rs_data: Optional[dict] = None,
) -> dict:
    """
    Compute indicators for all symbols.

    Parameters
    ----------
    data    : {symbol: ohlcv_DataFrame}
    rs_data : Pre-computed RS dict from relative_strength.compute_rs_for_all().
              Keys are symbols; values are RS metric dicts.

    Returns
    -------
    {symbol: indicators_dict}  — symbols with None indicators are excluded.
    """
    result: dict = {}
    for symbol, df in data.items():
        rs_metrics = (rs_data or {}).get(symbol)
        ind = compute_indicators(df, symbol=symbol, rs_metrics=rs_metrics)
        if ind is not None:
            result[symbol] = ind
    return result
