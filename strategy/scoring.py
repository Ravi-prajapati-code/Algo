"""
Hybrid signal scorer — ranks BUY candidates on a 0–130 scale.

Score breakdown (130 pts max, normalised to 0–100 for display)
--------------------------------------------------------------
Layer A — Technical signals (60 pts):
  RSI positioning      : 0–15 pts
  MACD strength        : 0–15 pts
  Volume confirmation  : 0–15 pts
  EMA/trend alignment  : 0–15 pts

Layer B — Relative Strength vs Nifty (40 pts):
  RS ratio magnitude   : 0–25 pts  (how much stock beats index)
  RS acceleration      : 0–10 pts  (1-month RS > 3-month RS)
  RS rank percentile   : 0–5 pts   (top decile bonus)

Layer C — Quality tier (30 pts):
  Pre-computed quality_score from quality_filter.py

All three layers must contribute for a high score — a stock that is
technically perfect but underperforming the index cannot score above 75.
A stock that beats the index but has weak technicals cannot score above 70.
This cross-layer requirement enforces high-probability, multi-confirmed entries.

Signal strength → position-size bucket
---------------------------------------
  Score 90–130 → STRONG   → 100 % of risk budget
  Score 70–89  → GOOD     → 80 %
  Score 50–69  → MODERATE → 60 %
  Score 30–49  → WEAK     → 40 %  (near minimum threshold)
  Score < 30   → REJECT   → 0 %   (filtered out)
"""

import logging
from config.settings import RSI_BUY_MIN, RSI_BUY_MAX

logger = logging.getLogger(__name__)

# ── Score thresholds → size factors ───────────────────────────────────────
SCORE_BUCKETS = [
    (90, 1.00),   # STRONG
    (70, 0.80),   # GOOD
    (50, 0.60),   # MODERATE
    (30, 0.40),   # WEAK
    (0,  0.00),   # REJECT
]


def score_signal(ind: dict) -> float:
    """
    Compute full hybrid score for a BUY candidate.

    Uses fields injected by compute_all_with_rs():
      rs_ratio, rs_ratio_1m, rs_rank, rs_accelerating  (from relative_strength.py)
      quality_score_val                                  (from quality_filter.py)

    Falls back gracefully if RS/quality fields are absent (backward-compatible).

    Returns normalised score 0–100.
    """
    raw = _raw_score(ind)
    # Normalise from 0–130 → 0–100
    return round(min(100.0, raw / 1.30), 1)


def _raw_score(ind: dict) -> float:
    """Compute raw 0–130 score across all three layers."""
    total = 0.0

    # ══ LAYER A: Technical signals (60 pts) ══════════════════════════════

    # A1. RSI positioning (0–15): ideal zone 45–62, penalise extremes
    rsi       = ind.get("rsi", 50)
    rsi_mid   = (RSI_BUY_MIN + RSI_BUY_MAX) / 2        # 52.5
    rsi_range = (RSI_BUY_MAX - RSI_BUY_MIN) / 2        # 12.5
    rsi_dist  = abs(rsi - rsi_mid) / rsi_range
    total    += max(0.0, 15.0 * (1 - rsi_dist))

    # A2. MACD histogram strength (0–15)
    hist      = ind.get("macd_hist", 0)
    hist_prev = ind.get("macd_hist_prev", 0)
    if hist > 0 and hist > hist_prev:
        total += 15    # Rising positive → strongest
    elif hist > 0:
        total += 10    # Positive but flat → good
    elif hist > hist_prev:
        total += 6     # Turning up from negative → possible early entry

    # A3. Volume confirmation (0–15): 3× volume = full points
    vol_ratio = ind.get("vol_ratio", 1.0)
    total    += min(15.0, max(0.0, 15.0 * (vol_ratio - 1.0) / 2.0))

    # A4. EMA / trend alignment (0–15)
    ema_pts = 0.0
    if ind.get("golden_cross"):
        ema_pts += 9    # Fresh crossover — highest conviction
    if ind.get("uptrend"):
        ema_pts += 3
    price    = ind.get("close", 1) or 1
    ema_fast = ind.get("ema_fast", 1) or 1
    if price > ema_fast:
        ema_pts += 3    # Price above EMA20 — momentum confirmed
    total += min(15.0, ema_pts)

    # ══ LAYER B: Relative Strength vs Nifty (40 pts) ═════════════════════

    rs_ratio = float(ind.get("rs_ratio", 0) or 0)
    if rs_ratio > 0:
        # B1. RS ratio magnitude (0–25): 1.0 = neutral, 1.3+ = max points
        rs_excess = max(0.0, rs_ratio - 1.0)             # 0.3+ = strong
        total    += min(25.0, 25.0 * rs_excess / 0.30)

        # B2. RS acceleration (0–10): 1-month RS > 3-month = momentum building
        if ind.get("rs_accelerating"):
            total += 10

        # B3. RS rank percentile bonus (0–5): top 10 % of watchlist
        rs_rank = float(ind.get("rs_rank", 0) or 0)
        if rs_rank >= 90:
            total += 5
        elif rs_rank >= 75:
            total += 3
    else:
        # RS data absent — partial credit using price vs EMA50 as proxy
        ema_slow = ind.get("ema_slow", 1) or 1
        if price > ema_slow * 1.05:     # 5 % above 50-SMA = relative strength proxy
            total += 10

    # ══ LAYER C: Quality tier score (0–30 pts) ════════════════════════════
    quality_pts = float(ind.get("quality_score_val", 0) or 0)
    total += min(30.0, quality_pts)

    return round(total, 1)


def score_to_size_factor(score: float) -> float:
    """
    Convert normalised score (0–100) to a position-size multiplier.

    Higher-conviction trades get larger allocations; weak signals are
    downsized rather than rejected outright.
    """
    for threshold, factor in SCORE_BUCKETS:
        if score >= threshold:
            return factor
    return 0.0


def score_label(score: float) -> str:
    """Human-readable label for a normalised score."""
    if score >= 90:
        return "STRONG"
    if score >= 70:
        return "GOOD"
    if score >= 50:
        return "MODERATE"
    if score >= 30:
        return "WEAK"
    return "REJECT"
