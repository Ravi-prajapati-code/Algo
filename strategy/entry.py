"""
Multi-layer entry validator with three distinct entry strategies.

A trade is opened when regime + RS/quality gates pass AND at least one
of three technical setups is confirmed:

  BREAKOUT       — Golden cross / price breaking above EMA20 with volume spike
  PULLBACK       — Price retesting EMA20 in an uptrend (mean-reversion entry)
  TREND_CONT     — Trend continuation in strong uptrend with MACD confirmation

Separating the three strategies prevents over-filtering: a pullback entry
does not need a volume spike (pullbacks have naturally lower volume), and a
trend-continuation entry does not need a golden cross.

Layer 1 — Market regime gate
  Blocks BEAR_TREND and UNKNOWN.  SIDEWAYS is allowed (reduced position size
  applied by the caller via regime_position_factor).

Layer 2 — Relative Strength (relaxed thresholds)
  Hard-blocks only if stock is > 10 % below index (rs_ratio < 0.90) or
  in the bottom 30th percentile of the watchlist.
  Slightly-underperforming stocks (0.90–1.00) that have strong technical
  setups are allowed through — this captures early RS turnarounds.

Layer 3 — Quality filter
  Tier and extreme-BB-width check (unchanged).

Layer 4 — Entry strategy detection
  One of BREAKOUT / PULLBACK / TREND_CONT must match.

Layer 5 — Not in active breakdown
  Price must be above lower Bollinger Band.
"""

import logging
from config.settings import RSI_BUY_MIN, RSI_BUY_MAX
from strategy.quality_filter import passes_quality_filter, MIN_QUALITY_TIER

logger = logging.getLogger(__name__)

# RS hard-block thresholds (Layer 2)
_RS_HARD_BLOCK_RATIO = 0.90    # Only block if > 10 % below index
_RS_MIN_RANK         = 30.0    # Lowered from 40 for more trade opportunities


def check_entry(
    ind: dict,
    symbol: str = "",
    regime: str = "BULL_TREND",
) -> tuple[bool, str]:
    """
    Run all entry layers.

    Parameters
    ----------
    ind    : Indicator dict enriched with RS/quality fields.
    symbol : Stock symbol — used for quality tier lookup and logging.
    regime : Current market regime string.

    Returns
    -------
    (qualified: bool, reason: str)
    reason = first failed layer, or entry strategy name on pass.
    """

    # ── Layer 1: Market regime gate ───────────────────────────────────────
    if regime in ("BEAR_TREND", "UNKNOWN"):
        return False, f"Market regime {regime} — new entries blocked"

    # ── Layer 2: Relative Strength (relaxed) ─────────────────────────────
    rs_ratio = float(ind.get("rs_ratio", 0) or 0)
    rs_rank  = float(ind.get("rs_rank",  0) or 0)

    if rs_ratio > 0:
        # Hard-block only when stock is significantly lagging the index
        if rs_ratio < _RS_HARD_BLOCK_RATIO:
            return False, (
                f"RS significantly below index: rs_ratio={rs_ratio:.3f} "
                f"< {_RS_HARD_BLOCK_RATIO:.2f} — stock clearly underperforming Nifty"
            )
        if rs_rank < _RS_MIN_RANK:
            return False, (
                f"RS rank too low: {rs_rank:.0f}th percentile "
                f"(need ≥ {_RS_MIN_RANK:.0f}th) — relative weakness vs watchlist"
            )

    # ── Layer 3: Quality filter ───────────────────────────────────────────
    if symbol:
        passes, reason = passes_quality_filter(symbol, ind, min_tier=MIN_QUALITY_TIER)
        if not passes:
            return False, f"Quality filter: {reason}"

    # ── Layer 4: Entry strategy detection ────────────────────────────────
    entry_type, tech_reason = _detect_entry_strategy(ind)
    if not entry_type:
        return False, tech_reason

    # ── Layer 5: Not in active breakdown ─────────────────────────────────
    if not ind.get("above_bb_lower"):
        price  = ind.get("close", 0)
        bb_low = ind.get("bb_lower", 0)
        return False, (
            f"Price {price:.2f} below lower BB {bb_low:.2f} — breakdown risk"
        )

    return True, f"Entry ({entry_type}): conditions met"


def _detect_entry_strategy(ind: dict) -> tuple[str | None, str]:
    """
    Detect which of three entry strategies is active.

    Returns (strategy_name, description) or (None, rejection_reason).

    Strategy 1 — BREAKOUT
    ---------------------
    Fresh golden cross OR strong price breakout above EMA20.
    Requires: volume spike + RSI in buy zone + MACD not strongly bearish.
    Best for: early-trend entries after a crossover event.

    Strategy 2 — PULLBACK
    ----------------------
    Uptrend intact + price pulling back to EMA20 (within 3 %).
    Requires: RSI recovering (40–60) + MACD turning up or bullish.
    No volume spike needed — pullbacks naturally have lower volume.
    Best for: adding to winning trends at lower-risk re-entry points.

    Strategy 3 — TREND CONTINUATION
    --------------------------------
    Strong uptrend + price above EMA20 + RSI in momentum zone (50–65).
    Requires: MACD positive + moderate volume increase (≥ 1.2×).
    Best for: steady uptrending stocks with consistent momentum.
    """
    price     = float(ind.get("close",    0) or 0)
    ema_fast  = float(ind.get("ema_fast", price) or price) or (price or 1)
    rsi       = float(ind.get("rsi",      50) or 50)
    vol_ratio = float(ind.get("vol_ratio", 1.0) or 1.0)

    # Price within 3 % of EMA20 (pullback zone)
    near_ema20 = ema_fast > 0 and abs(price - ema_fast) / ema_fast <= 0.03

    # MACD is not strongly bearish (histogram not deep negative without recovery)
    macd_hist       = float(ind.get("macd_hist", 0) or 0)
    macd_not_tanking = (
        ind.get("macd_bullish")
        or ind.get("macd_turning_up")
        or macd_hist > -0.15
    )

    # ── Strategy 1: BREAKOUT ──────────────────────────────────────────────
    if (
        ind.get("golden_cross")
        and price >= ema_fast * 0.99
        and ind.get("vol_spike")
        and RSI_BUY_MIN <= rsi <= RSI_BUY_MAX
        and macd_not_tanking
    ):
        return "BREAKOUT", "Golden cross breakout with volume confirmation"

    # ── Strategy 2: PULLBACK ──────────────────────────────────────────────
    if (
        ind.get("uptrend")
        and near_ema20
        and RSI_BUY_MIN <= rsi <= 60
        and (ind.get("macd_turning_up") or ind.get("macd_bullish"))
    ):
        return "PULLBACK", "Pullback to EMA20 in uptrend with MACD support"

    # ── Strategy 3: TREND CONTINUATION ───────────────────────────────────
    if (
        ind.get("uptrend")
        and price > ema_fast * 1.005       # Price clearly above EMA20
        and 50 <= rsi <= RSI_BUY_MAX
        and ind.get("macd_bullish")
        and vol_ratio >= 1.2               # Some volume increase (not full spike)
    ):
        return "TREND_CONT", "Trend continuation with MACD and volume"

    # ── No strategy matched — build a useful rejection reason ─────────────
    reasons: list[str] = []
    if not ind.get("uptrend") and not ind.get("golden_cross"):
        reasons.append("no uptrend / golden cross")
    if not (RSI_BUY_MIN <= rsi <= RSI_BUY_MAX):
        reasons.append(f"RSI {rsi:.0f} outside [{RSI_BUY_MIN}–{RSI_BUY_MAX}]")
    if not ind.get("macd_bullish") and not ind.get("macd_turning_up") and macd_hist <= -0.15:
        reasons.append(f"MACD bearish (hist={macd_hist:.3f})")
    if not ind.get("vol_spike") and vol_ratio < 1.2:
        reasons.append(f"low volume ({vol_ratio:.1f}×)")
    if not near_ema20 and price <= ema_fast * 1.005:
        reasons.append("price not near / above EMA20")

    msg = "No entry strategy matched"
    if reasons:
        msg += ": " + ", ".join(reasons)
    return None, msg
