"""
Multi-layer entry validator.

A trade is only opened when ALL five layers pass.  Each layer targets a
different risk — missing any one of them is a known reason trades fail.

Layer 1 — Market regime (macro filter)
  The broad market must be in BULL, SIDEWAYS, or HIGH_VOL.
  Checked externally (backtest/engine.py, daily_runner.py) before entry.py
  is called; entry.py receives regime as a parameter and enforces the gate.

Layer 2 — Relative Strength (momentum quality)
  Stock must be outperforming Nifty over 3 months (rs_ratio ≥ 1.0)
  AND in the top 60 % of the watchlist by RS rank.
  This prevents buying laggards that look cheap but are being distributed.

Layer 3 — Quality filter (fundamental safety)
  Stock must be in quality Tier 1 or 2 (no speculative names in strict mode).
  Extreme BB width (>25 %) is also blocked regardless of tier.

Layer 4 — Technical setup (timing)
  EMA trend, RSI zone, MACD confirmation, volume spike, BB position.
  Same logic as before, unchanged.

Layer 5 — Breakout / momentum confirmation
  Price must be above the 20-day EMA AND near (within 3 %) a local high,
  or have a recent golden cross.  Prevents buying in the middle of nowhere.
"""

import logging
from config.settings import RSI_BUY_MIN, RSI_BUY_MAX
from strategy.quality_filter import passes_quality_filter, MIN_QUALITY_TIER

logger = logging.getLogger(__name__)


def check_entry(
    ind: dict,
    symbol: str = "",
    regime: str = "BULL_TREND",
) -> tuple[bool, str]:
    """
    Run all five entry layers.

    Parameters
    ----------
    ind    : Indicator dict (from compute_indicators, enriched with RS/quality fields).
    symbol : Stock symbol — used for quality tier lookup and logging.
    regime : Current market regime string.

    Returns
    -------
    (qualified: bool, reason: str)
    reason = first failed layer, or 'All entry conditions met' on pass.
    """

    # ── Layer 1: Market regime gate ───────────────────────────────────────
    if regime in ("BEAR_TREND", "UNKNOWN"):
        return False, f"Market regime {regime} — new entries blocked"

    # ── Layer 2: Relative Strength ────────────────────────────────────────
    rs_ratio = float(ind.get("rs_ratio", 0) or 0)
    rs_rank  = float(ind.get("rs_rank",  0) or 0)

    if rs_ratio > 0:   # Only enforce when RS data is available
        if not ind.get("rs_outperforming", True):
            return False, (
                f"RS below index: rs_ratio={rs_ratio:.3f} < 1.0 — "
                "stock underperforming Nifty"
            )
        if not ind.get("rs_qualified", True):
            return False, (
                f"RS rank too low: {rs_rank:.0f}th percentile "
                "(need ≥ 40th) — relative weakness vs watchlist"
            )

    # ── Layer 3: Quality filter ───────────────────────────────────────────
    if symbol:
        passes, reason = passes_quality_filter(symbol, ind, min_tier=MIN_QUALITY_TIER)
        if not passes:
            return False, f"Quality filter: {reason}"

    # ── Layer 4: Technical setup ──────────────────────────────────────────

    # 4a. Uptrend: EMA20 > EMA50
    if not ind.get("uptrend"):
        return False, (
            f"No uptrend (EMA20={ind.get('ema_fast', 0):.2f} "
            f"< EMA50={ind.get('ema_slow', 0):.2f})"
        )

    # 4b. Recent golden cross OR pullback to EMA20
    price    = ind.get("close", 0)
    ema_fast = ind.get("ema_fast", price) or price
    near_ema = ema_fast > 0 and abs(price - ema_fast) / ema_fast <= 0.02
    if not ind.get("golden_cross") and not near_ema:
        return False, (
            f"No recent golden cross and price not near EMA20 "
            f"(price={price:.2f}, EMA20={ema_fast:.2f})"
        )

    # 4c. RSI in buy zone
    rsi = ind.get("rsi", 0)
    if not (RSI_BUY_MIN <= rsi <= RSI_BUY_MAX):
        return False, f"RSI {rsi:.1f} outside buy zone [{RSI_BUY_MIN}, {RSI_BUY_MAX}]"

    # 4d. MACD bullish OR turning up
    if not ind.get("macd_bullish") and not ind.get("macd_turning_up"):
        return False, (
            f"MACD not bullish "
            f"(hist={ind.get('macd_hist', 0):.4f}, "
            f"prev={ind.get('macd_hist_prev', 0):.4f})"
        )

    # 4e. Volume spike
    if not ind.get("vol_spike"):
        return False, f"No volume spike (ratio={ind.get('vol_ratio', 0):.2f}x)"

    # 4f. Price above lower Bollinger Band
    if not ind.get("above_bb_lower"):
        return False, (
            f"Price below lower BB "
            f"(price={price:.2f}, BB_lower={ind.get('bb_lower', 0):.2f})"
        )

    # ── Layer 5: Breakout / momentum confirmation ─────────────────────────
    # Price must be above EMA20 (not just near it) for momentum confirmation
    if price <= ema_fast * 0.99:
        return False, (
            f"No breakout confirmation: price {price:.2f} not above "
            f"EMA20 {ema_fast:.2f}"
        )

    # 52-week high proximity: stock must be within 25 % of its 52W high
    # (avoids buying deeply broken stocks with a dead-cat bounce)
    week52_hi = ind.get("week52_high", 0)
    if week52_hi and week52_hi > 0:
        proximity = price / week52_hi
        if proximity < 0.75:
            return False, (
                f"Price {price:.2f} is {proximity:.1%} of 52W high "
                f"{week52_hi:.2f} — too far from highs (< 75 %)"
            )

    return True, "All entry conditions met"
