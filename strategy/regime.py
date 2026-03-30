"""
Multi-regime market detector.

Classifies the current market environment into one of four regimes:
  BULL_TREND   — Nifty above 200-SMA, low volatility, rising
  BEAR_TREND   — Nifty below 200-SMA, downtrend confirmed
  SIDEWAYS     — Range-bound, no clear trend direction
  HIGH_VOL     — Elevated volatility (VIX-proxy or realised vol spike)

The regime drives strategy behaviour:
  BULL_TREND  → full BUY allowance, normal position sizes
  SIDEWAYS    → BUY allowed but reduced size (×0.75), tighter stops
  HIGH_VOL    → BUY allowed only for highest-scored signals (score ≥ 70)
  BEAR_TREND  → no new BUYs (existing positions managed normally)
"""

import logging
from typing import Literal, Optional

import pandas as pd

from config.settings import MARKET_FILTER_SMA, MARKET_FILTER_ENABLED

logger = logging.getLogger(__name__)

Regime = Literal["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOL"]

# ── Thresholds ─────────────────────────────────────────────────────────────
_ADX_TREND_THRESHOLD    = 25.0   # ADX > 25 → trending (ADX < 25 → sideways)
_REALISED_VOL_LOOKBACK  = 20     # Rolling window for realised-vol calculation
_HIGH_VOL_THRESHOLD     = 0.25   # Annualised vol > 25 % → HIGH_VOL regime
_SMA_FAST               = 50     # Short SMA for trend confirmation
_SMA_SLOW               = 200    # Long SMA (same as market filter)


def detect_regime(index_df: Optional[pd.DataFrame]) -> Regime:
    """
    Detect the current market regime from index OHLCV data.

    Logic
    -----
    1. If insufficient data → default to BULL_TREND (allow trading).
    2. Compute 200-SMA and 50-SMA; check if price is above/below each.
    3. Compute 20-day realised volatility (annualised).
    4. Classify:
       - Realised vol > threshold          → HIGH_VOL
       - Price < 200-SMA                   → BEAR_TREND
       - Price > 200-SMA, > 50-SMA         → BULL_TREND
       - Price > 200-SMA, < 50-SMA         → SIDEWAYS (recovering)

    Parameters
    ----------
    index_df : DataFrame with 'close' column indexed by date.
               Should contain at least 220 rows for accurate detection.

    Returns
    -------
    Regime literal string.
    """
    if not MARKET_FILTER_ENABLED:
        return "BULL_TREND"

    if index_df is None or len(index_df) < _SMA_SLOW + 5:
        logger.debug("[Regime] Insufficient index data — defaulting to BULL_TREND")
        return "BULL_TREND"

    close = index_df["close"].astype(float)
    last_close = float(close.iloc[-1])

    sma200 = float(close.rolling(_SMA_SLOW).mean().iloc[-1])
    sma50  = float(close.rolling(_SMA_FAST).mean().iloc[-1])

    # Realised vol: std of log returns × sqrt(252)
    log_returns = close.pct_change().dropna()
    realised_vol = float(log_returns.tail(_REALISED_VOL_LOOKBACK).std() * (252 ** 0.5))

    logger.info(
        "[Regime] Nifty50=%.0f  SMA50=%.0f  SMA200=%.0f  RealVol=%.1f%%",
        last_close, sma50, sma200, realised_vol * 100,
    )

    # ── High-volatility regime (overrides trend) ──────────────────────
    if realised_vol > _HIGH_VOL_THRESHOLD:
        regime: Regime = "HIGH_VOL"

    # ── Bear trend ────────────────────────────────────────────────────
    elif last_close < sma200:
        regime = "BEAR_TREND"

    # ── Bull trend ────────────────────────────────────────────────────
    elif last_close > sma200 and last_close > sma50:
        regime = "BULL_TREND"

    # ── Sideways / recovery ───────────────────────────────────────────
    else:
        regime = "SIDEWAYS"

    logger.info("[Regime] → %s", regime)
    return regime


def regime_position_factor(regime: Regime) -> float:
    """
    Return a multiplier (0.0–1.0) to scale position sizes based on regime.

    BULL_TREND  → 1.0   (full size)
    SIDEWAYS    → 0.75  (reduce exposure in indecisive market)
    HIGH_VOL    → 0.5   (cut size in volatile market)
    BEAR_TREND  → 0.0   (no new positions)
    """
    return {
        "BULL_TREND": 1.0,
        "SIDEWAYS":   0.75,
        "HIGH_VOL":   0.5,
        "BEAR_TREND": 0.0,
    }.get(regime, 1.0)


def regime_min_score(regime: Regime) -> float:
    """
    Minimum signal score required to open a new position in this regime.

    BULL_TREND  → 40   (most signals qualify)
    SIDEWAYS    → 55   (only decent setups)
    HIGH_VOL    → 70   (only highest-conviction setups)
    BEAR_TREND  → 999  (effectively blocks all new buys)
    """
    return {
        "BULL_TREND": 40.0,
        "SIDEWAYS":   55.0,
        "HIGH_VOL":   70.0,
        "BEAR_TREND": 999.0,
    }.get(regime, 40.0)


def is_buy_allowed(regime: Regime) -> bool:
    """Quick check: can we open new positions in this regime?"""
    return regime != "BEAR_TREND"
