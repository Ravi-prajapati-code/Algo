"""
Multi-regime market detector.

Classifies the current market environment into one of five regimes:
  BULL_TREND   — Nifty above 200-SMA, low volatility, rising
  BEAR_TREND   — Nifty below 200-SMA, downtrend confirmed
  SIDEWAYS     — Range-bound, no clear trend direction
  HIGH_VOL     — Elevated volatility (realised vol spike)
  UNKNOWN      — Insufficient index data to determine regime safely

CRITICAL: UNKNOWN is NOT a safe default — trading is BLOCKED when regime
is UNKNOWN. This prevents silent failures where stale/missing index data
allows the strategy to trade without a valid market context.

Minimum data requirement: 210 candles (200 for SMA + 10-bar buffer).

The regime drives strategy behaviour:
  BULL_TREND  → full BUY allowance, normal position sizes
  SIDEWAYS    → BUY allowed but reduced size (×0.75), tighter stops
  HIGH_VOL    → BUY allowed only for highest-scored signals (score ≥ 70)
  BEAR_TREND  → no new BUYs (existing positions managed normally)
  UNKNOWN     → no new BUYs (same as BEAR_TREND; existing positions managed)
"""

import logging
from typing import Literal, Optional

import pandas as pd

from config.settings import MARKET_FILTER_SMA, MARKET_FILTER_ENABLED

logger = logging.getLogger(__name__)

# UNKNOWN added as a first-class regime — never silently allow trading
Regime = Literal["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOL", "UNKNOWN"]

# ── Thresholds ─────────────────────────────────────────────────────────────
_REALISED_VOL_LOOKBACK = 20       # Rolling window for realised-vol calculation
_HIGH_VOL_THRESHOLD    = 0.25     # Annualised vol > 25 % → HIGH_VOL regime
_SMA_FAST              = 50       # Short SMA for trend confirmation
_SMA_SLOW              = 200      # Long SMA (same as market filter)
MIN_INDEX_CANDLES      = _SMA_SLOW + 10   # 210 — hard minimum for safe regime detection


def detect_regime(index_df: Optional[pd.DataFrame]) -> Regime:
    """
    Detect the current market regime from Nifty 50 OHLCV data.

    Returns UNKNOWN (never BULL_TREND) when data is missing or too short.
    Callers MUST check for UNKNOWN and block all new BUY entries.

    Data requirements
    -----------------
    Minimum {MIN_INDEX_CANDLES} candles required.  With fewer rows:
      - SMA200 cannot be reliably computed (rolling mean is NaN).
      - Defaulting to BULL_TREND would silently open positions with zero
        market-context validation — a critical safety violation.

    Classification logic
    --------------------
    1. Realised vol > 25 % (annualised)  → HIGH_VOL
    2. Price < SMA200                    → BEAR_TREND
    3. Price > SMA200 AND > SMA50        → BULL_TREND
    4. Price > SMA200 AND < SMA50        → SIDEWAYS

    Parameters
    ----------
    index_df : DataFrame with 'close' column indexed by date.
               Must contain at least {MIN_INDEX_CANDLES} rows.

    Returns
    -------
    Regime string.  UNKNOWN if data is insufficient or MARKET_FILTER_ENABLED=False.
    """
    if not MARKET_FILTER_ENABLED:
        # When filter is explicitly disabled, caller decides; return BULL as opted-out
        return "BULL_TREND"

    # ── Strict data-sufficiency check ─────────────────────────────────────
    if index_df is None or len(index_df) < MIN_INDEX_CANDLES:
        available = len(index_df) if index_df is not None else 0
        logger.warning(
            "[Regime] UNKNOWN — insufficient index data: %d candles available, "
            "need ≥ %d for SMA%d. Trading disabled due to insufficient market data.",
            available, MIN_INDEX_CANDLES, _SMA_SLOW,
        )
        return "UNKNOWN"

    close = index_df["close"].astype(float)

    # Verify the SMA200 value itself is not NaN (e.g. gaps in data)
    sma200_series = close.rolling(_SMA_SLOW).mean()
    sma50_series  = close.rolling(_SMA_FAST).mean()

    last_close = float(close.iloc[-1])
    last_sma200 = sma200_series.iloc[-1]
    last_sma50  = sma50_series.iloc[-1]

    if pd.isna(last_sma200) or pd.isna(last_sma50):
        logger.warning(
            "[Regime] UNKNOWN — SMA calculation produced NaN "
            "(check for data gaps in index history). "
            "Trading disabled due to insufficient market data."
        )
        return "UNKNOWN"

    last_sma200 = float(last_sma200)
    last_sma50  = float(last_sma50)

    # Realised vol: std of log-returns × √252
    log_returns  = close.pct_change().dropna()
    realised_vol = float(log_returns.tail(_REALISED_VOL_LOOKBACK).std() * (252 ** 0.5))

    logger.info(
        "[Regime] Nifty50=%.0f  SMA50=%.0f  SMA200=%.0f  RealVol=%.1f%%  bars=%d",
        last_close, last_sma50, last_sma200, realised_vol * 100, len(index_df),
    )

    # ── Classification ─────────────────────────────────────────────────────
    if realised_vol > _HIGH_VOL_THRESHOLD:
        regime: Regime = "HIGH_VOL"
    elif last_close < last_sma200:
        regime = "BEAR_TREND"
    elif last_close > last_sma200 and last_close > last_sma50:
        regime = "BULL_TREND"
    else:
        regime = "SIDEWAYS"

    logger.info("[Regime] → %s", regime)
    return regime


def regime_position_factor(regime: Regime) -> float:
    """
    Multiplicative factor (0.0–1.0) applied to position sizes.

    BULL_TREND  → 1.0  (full size)
    SIDEWAYS    → 0.75 (cautious)
    HIGH_VOL    → 0.5  (reduced for volatility)
    BEAR_TREND  → 0.0  (no new positions)
    UNKNOWN     → 0.0  (no new positions — data dependency not met)
    """
    return {
        "BULL_TREND": 1.0,
        "SIDEWAYS":   0.75,
        "HIGH_VOL":   0.5,
        "BEAR_TREND": 0.0,
        "UNKNOWN":    0.0,
    }.get(regime, 0.0)   # Fail-safe: unknown strings → 0


def regime_min_score(regime: Regime) -> float:
    """
    Minimum signal score (0–100) required to open a new position.

    BULL_TREND  → 40   (most signals qualify)
    SIDEWAYS    → 55   (decent setups only)
    HIGH_VOL    → 70   (highest-conviction only)
    BEAR_TREND  → 999  (blocks all new buys)
    UNKNOWN     → 999  (blocks all new buys — safety guard)
    """
    return {
        "BULL_TREND": 40.0,
        "SIDEWAYS":   55.0,
        "HIGH_VOL":   70.0,
        "BEAR_TREND": 999.0,
        "UNKNOWN":    999.0,
    }.get(regime, 999.0)   # Fail-safe: unknown strings → block


def is_buy_allowed(regime: Regime) -> bool:
    """
    Returns True only for regimes where opening new positions is permitted.
    BEAR_TREND and UNKNOWN both return False.
    """
    return regime in ("BULL_TREND", "SIDEWAYS", "HIGH_VOL")
