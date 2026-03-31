"""
Multi-layer entry validator with three distinct entry strategies.

A trade is opened when regime + RS/quality gates pass AND at least one
of three technical setups is confirmed:

  BREAKOUT       — Close breaks above 20-day high with volume confirmation
  PULLBACK       — Price pulling back to SMA20 in uptrend with RSI > 50
  MOMENTUM       — Price above SMA50 with RSI > 60 (strong momentum continuation)

Entry strategies are deliberately independent so that over-filtering is avoided:
  - BREAKOUT: captures fresh trend initiations at new highs
  - PULLBACK: captures low-risk re-entries during healthy corrections
  - MOMENTUM: captures stocks already in strong sustained uptrends

Layer 1 — Market regime gate
  Blocks BEAR_TREND and UNKNOWN.  PARTIAL/SIDEWAYS/HIGH_VOL allowed with
  reduced position sizing applied by the caller via regime_position_factor.

Layer 2 — Relative Strength (relaxed thresholds)
  Hard-blocks only if stock is > 10 % below index (rs_ratio < 0.90) or
  in the bottom 30th percentile of the watchlist.

Layer 3 — Quality filter
  Tier and extreme-BB-width check (unchanged).

Layer 4 — Entry strategy detection
  One of BREAKOUT / PULLBACK / MOMENTUM must match.

Layer 5 — Dynamic filter relaxation
  If score threshold is too strict (no candidates), engine auto-relaxes by
  lowering min_score. Entry layer itself does not hold that state — it always
  runs fresh. Relaxation tracked in backtest/engine.py.

Layer 6 — Not in active breakdown
  Price must be above lower Bollinger Band.
"""

import logging
from config.settings import RSI_BUY_MIN, RSI_BUY_MAX
from strategy.quality_filter import passes_quality_filter, MIN_QUALITY_TIER

logger = logging.getLogger(__name__)

# RS hard-block thresholds (Layer 2)
_RS_HARD_BLOCK_RATIO = 0.90    # Only block if > 10 % below index
_RS_MIN_RANK         = 30.0    # Lowered from 40 for more trade opportunities

# Momentum strategy RSI floor — stronger momentum required
_MOMENTUM_RSI_MIN = 60
_PULLBACK_RSI_MIN = 50          # Pullback only needs RSI recovering above 50


def check_entry(
    ind: dict,
    symbol: str = "",
    regime: str = "BULL_TREND",
) -> tuple[bool, str]:
    """
    Run all entry layers.

    Parameters
    ----------
    ind    : Indicator dict enriched with RS/quality/high_20d fields.
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
    Close breaks above 20-day high with volume confirmation.
    Requires: close >= high_20d + vol_spike + RSI in buy zone.
    Best for: fresh trend initiations after consolidation, new momentum highs.
    No MACD requirement — breakouts can start before MACD catches up.

    Strategy 2 — PULLBACK
    ----------------------
    Uptrend intact + price pulling back to SMA20 (within 3 %).
    Requires: RSI > 50 (recovering, not oversold) + MACD turning up.
    No volume spike needed — pullbacks naturally have lower volume.
    Best for: adding to winning trends at lower-risk, higher-probability entries.

    Strategy 3 — MOMENTUM
    ---------------------
    Price above SMA50 with strong RSI (> 60) and MACD confirmation.
    Requires: uptrend + price > SMA50 + RSI > 60 + MACD bullish.
    Best for: stocks already in strong sustained uptrends with proven momentum.
    """
    price     = float(ind.get("close",    0) or 0)
    ema_fast  = float(ind.get("ema_fast", price) or price) or (price or 1)
    ema_slow  = float(ind.get("ema_slow", ema_fast) or ema_fast) or (ema_fast or 1)
    rsi       = float(ind.get("rsi",      50) or 50)
    vol_ratio = float(ind.get("vol_ratio", 1.0) or 1.0)
    high_20d  = float(ind.get("high_20d", price) or price)

    # Price within 3 % of SMA20 (pullback zone)
    near_sma20 = ema_fast > 0 and abs(price - ema_fast) / ema_fast <= 0.03

    # ── Strategy 1: BREAKOUT (20-day high + volume) ───────────────────────
    # Close breaks out above 20-day high with above-average volume
    if (
        price >= high_20d                       # Breaking or at 20-day high
        and ind.get("vol_spike")                # Volume confirms the breakout
        and RSI_BUY_MIN <= rsi <= RSI_BUY_MAX   # RSI in buy zone (not overbought)
        and ind.get("above_bb_lower")           # Not in breakdown territory
    ):
        return "BREAKOUT", f"20-day high breakout: close={price:.2f} >= high_20d={high_20d:.2f} with volume spike"

    # ── Strategy 2: PULLBACK (SMA20 + RSI > 50) ──────────────────────────
    # Healthy pullback to SMA20 in an established uptrend
    if (
        ind.get("uptrend")                          # Stock in uptrend
        and near_sma20                              # Price near SMA20 (pullback zone)
        and rsi >= _PULLBACK_RSI_MIN                # RSI recovering (> 50)
        and rsi <= RSI_BUY_MAX                      # Not overbought (≤ 65)
        and (ind.get("macd_turning_up") or ind.get("macd_bullish"))  # MACD supporting
    ):
        return "PULLBACK", f"Pullback to SMA20: price={price:.2f} near SMA20={ema_fast:.2f}, RSI={rsi:.0f}"

    # ── Strategy 3: MOMENTUM (price > SMA50 + RSI > 60) ──────────────────
    # Strong trend continuation — price above SMA50, momentum confirmed
    if (
        ind.get("uptrend")                          # Uptrend confirmed
        and price > ema_slow * 1.005                # Price clearly above SMA50
        and rsi >= _MOMENTUM_RSI_MIN                # Strong RSI (> 60)
        and rsi <= RSI_BUY_MAX                      # Not overbought
        and ind.get("macd_bullish")                 # MACD positive
        and vol_ratio >= 1.2                        # Above-average volume
    ):
        return "MOMENTUM", f"Momentum: price={price:.2f} above SMA50={ema_slow:.2f}, RSI={rsi:.0f}"

    # ── No strategy matched — build a useful rejection reason ─────────────
    reasons: list[str] = []
    if price < high_20d:
        reasons.append(f"below 20d-high {high_20d:.2f} ({(price/high_20d - 1)*100:.1f}%)")
    if not ind.get("uptrend"):
        reasons.append("no uptrend")
    if not (RSI_BUY_MIN <= rsi <= RSI_BUY_MAX):
        reasons.append(f"RSI {rsi:.0f} outside [{RSI_BUY_MIN}–{RSI_BUY_MAX}]")
    if not ind.get("macd_bullish") and not ind.get("macd_turning_up"):
        reasons.append("MACD not bullish/turning")
    if not ind.get("vol_spike") and vol_ratio < 1.2:
        reasons.append(f"low volume ({vol_ratio:.1f}×)")
    if not near_sma20 and price <= ema_slow * 1.005:
        reasons.append("price not near SMA20 or above SMA50")

    msg = "No entry strategy matched"
    if reasons:
        msg += ": " + ", ".join(reasons)
    return None, msg
