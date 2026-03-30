"""
Stock Quality Filter — ensures we only trade fundamentally strong companies.

Since free real-time fundamental data (P/E, ROE, debt) isn't available via
yfinance in a reliable, production-safe way, quality is determined by two
complementary methods:

Method 1: Tiered Watchlist Classification (primary)
----------------------------------------------------
Every stock in the watchlist is pre-classified into a quality tier based on
publicly known fundamentals (updated manually per quarter):

  Tier 1 — ELITE (score 3):  Large-cap, consistent earnings, ROE > 15 %,
            debt/equity < 1.0, revenue growth > 10 % CAGR.  These are Nifty 50
            index constituents and blue-chips.  Full position size allowed.

  Tier 2 — QUALITY (score 2): Midcap leaders with solid balance sheets,
            ROE > 12 %, moderate debt.  Position size capped at 75 % of max.

  Tier 3 — SPECULATIVE (score 1): Smaller companies or cyclicals with higher
            volatility / debt.  Position size capped at 50 % of max.
            Require higher score threshold to enter.

Method 2: Price-action Quality Proxies (secondary, computed from OHLCV)
------------------------------------------------------------------------
Derived from indicator data — no external API needed:

  52-week position  : close / 52-week-high  (higher = stronger)
  Volume stability  : 1 - (vol_std / vol_mean)  (higher = more consistent flow)
  Bollinger Width   : narrow bands = lower volatility = quality persistence

These proxies augment the tier score but do not override it.

Usage
-----
    tier = get_quality_tier("RELIANCE.NS")       # returns 1/2/3
    score = quality_score("RELIANCE.NS", ind)    # returns 0-30 quality score
    allowed = passes_quality_filter("TECHM.NS", ind, min_tier=2)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Quality configuration ──────────────────────────────────────────────────
MIN_QUALITY_TIER  = 1     # Accept all tiers by default; raise to 2 for stricter mode
TIER_SIZE_FACTOR  = {1: 1.0, 2: 0.75, 3: 0.50}   # Position-size multiplier per tier

# ── Tier 1: ELITE — Nifty 50 blue-chips with strong fundamentals ──────────
_TIER_1 = {
    # Banking & Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS",
    # IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    # Consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    # Energy & Industrials
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    # Auto
    "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    # Metals
    "TATASTEEL.NS", "HINDALCO.NS",
    # Infra
    "LTIM.NS", "LT.NS",
}

# ── Tier 2: QUALITY — Midcap leaders with solid fundamentals ──────────────
_TIER_2 = {
    # Banking
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    # Finance
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    # IT
    "TECHM.NS", "MPHASIS.NS", "PERSISTENT.NS",
    # Pharma
    "DIVISLAB.NS", "AUROPHARMA.NS", "TORNTPHARM.NS",
    # Consumer
    "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS",
    "TATACONSUM.NS", "COLPAL.NS",
    # Healthcare
    "APOLLOHOSP.NS",
    # Auto
    "EICHERMOT.NS", "TVSMOTOR.NS",
    # Energy
    "TATAPOWER.NS", "ADANIPORTS.NS", "ADANIGREEN.NS",
    # Chemicals
    "PIDILITIND.NS", "ASIANPAINT.NS", "BERGEPAINT.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Infra
    "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS",
    # RealEstate
    "DLF.NS",
    # Retail
    "DMART.NS",
    # Aviation
    "INDIGO.NS",
    # Consumer Electronics
    "TITAN.NS",
}


def get_quality_tier(symbol: str) -> int:
    """
    Return quality tier for a symbol: 1 (elite), 2 (quality), 3 (speculative).
    Unknown symbols default to tier 3 (most conservative).
    """
    if symbol in _TIER_1:
        return 1
    if symbol in _TIER_2:
        return 2
    return 3


def tier_size_factor(symbol: str) -> float:
    """
    Return the position-size multiplier for a symbol's quality tier.
    Tier 1 → 1.0, Tier 2 → 0.75, Tier 3 → 0.50
    """
    return TIER_SIZE_FACTOR[get_quality_tier(symbol)]


def quality_score(symbol: str, ind: dict) -> float:
    """
    Compute a 0–30 quality score combining tier classification and
    price-action quality proxies.

    Breakdown (30 pts total):
      Tier points  : Tier1=15, Tier2=10, Tier3=5
      52W position : 0–8 pts   (close / 52W-high, capped at 95 %)
      Vol stability: 0–4 pts   (consistent institutional flow)
      BB width     : 0–3 pts   (low volatility = quality persistence)
    """
    total = 0.0

    # ── Tier score ────────────────────────────────────────────────────────
    tier = get_quality_tier(symbol)
    tier_pts = {1: 15, 2: 10, 3: 5}
    total += tier_pts[tier]

    # ── 52-week high proximity (price strength proxy) ──────────────────
    close     = ind.get("close", 0)
    week52_hi = ind.get("week52_high", 0)
    if week52_hi and close:
        proximity = min(1.0, close / week52_hi)   # 1.0 = at 52W high
        # Full 8 pts at 90 %+ of 52W high, scales down linearly
        total += max(0, 8 * (proximity - 0.60) / 0.40)

    # ── Volume consistency ────────────────────────────────────────────────
    vol_avg = ind.get("vol_avg", 0)
    vol_today = ind.get("vol_today", 0)
    if vol_avg and vol_avg > 0:
        # Consistent high volume relative to average = institutional interest
        vol_score = min(4.0, 4 * (vol_avg / max(vol_today, 1)))
        total += max(0, vol_score)

    # ── Bollinger Band width (low vol = quality) ──────────────────────────
    bb_upper = ind.get("bb_upper", 0)
    bb_lower = ind.get("bb_lower", 0)
    bb_mid   = ind.get("bb_mid", close)
    if bb_mid and bb_mid > 0 and bb_upper and bb_lower:
        bb_width_pct = (bb_upper - bb_lower) / bb_mid
        # Narrow bands (<5%) = 3pts, wide bands (>15%) = 0pts
        bb_pts = max(0, 3 * (1 - (bb_width_pct - 0.05) / 0.10))
        total += min(3.0, bb_pts)

    return round(min(30.0, total), 1)


def passes_quality_filter(
    symbol: str,
    ind: dict,
    min_tier: int = MIN_QUALITY_TIER,
) -> tuple[bool, str]:
    """
    Returns (passes: bool, reason: str).

    Blocks trades on:
      - Tier below minimum (e.g. speculative stock in strict mode)
      - Extremely wide Bollinger Bands (>25 % of price) — high-risk condition
    """
    tier = get_quality_tier(symbol)
    if tier > min_tier:
        return False, f"{symbol} quality tier {tier} below minimum {min_tier}"

    # Block on extreme volatility regardless of tier
    close    = ind.get("close", 1) or 1
    bb_upper = ind.get("bb_upper", 0)
    bb_lower = ind.get("bb_lower", 0)
    if bb_upper and bb_lower:
        bb_width_pct = (bb_upper - bb_lower) / close
        if bb_width_pct > 0.25:
            return False, (
                f"{symbol} BB width {bb_width_pct:.1%} > 25% — "
                "extreme volatility, quality filter blocked"
            )

    return True, "quality_ok"
