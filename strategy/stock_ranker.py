"""
Stock ranking system for selecting top candidates from qualifying signals.

When multiple stocks pass all entry filters on the same day, ranks them
by a composite score and returns only the top N for actual position sizing.

This prevents overtrading, focuses capital on the best opportunities, and
provides full visibility into why each stock was selected or skipped.

Composite Rank Score (0–100):
  ┌─────────────────────────────────────────────────────────┐
  │  RS Rank Component     (0–40 pts)                       │
  │  Momentum Component    (0–30 pts)   RSI + returns       │
  │  Volume Strength       (0–20 pts)                       │
  │  Breakout Proximity    (0–10 pts)   closeness to highs  │
  └─────────────────────────────────────────────────────────┘

A stock scoring 90+ is an elite setup — strong RS, strong momentum,
high volume, and near a breakout level. These get priority allocation.

Usage
-----
    from strategy.stock_ranker import rank_and_select, compute_rank_score

    # candidates: list of (signal_score, symbol, ind_dict)
    top5 = rank_and_select(candidates, max_stocks=5)

    # For visibility logging per stock:
    for score, symbol, ind in all_candidates:
        rank = compute_rank_score(ind)
        print(f"{symbol}: RS={ind['rs_rank']:.0f}, rank_score={rank:.1f}")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_rank_score(ind: dict) -> float:
    """
    Compute composite ranking score for a single stock.

    Parameters
    ----------
    ind : Indicator dict with rs_rank, rsi, rs_ratio_1m, vol_ratio,
          week52_high, close, high_20d fields.

    Returns
    -------
    float in [0, 100].  Higher = better setup, higher priority selection.
    """

    # ── RS Rank Component (0–40 pts) ──────────────────────────────────────
    # Uses percentile rank within watchlist: top-10% gets near-full pts
    rs_rank = float(ind.get("rs_rank", 0) or 0)
    rs_score = min(40.0, rs_rank * 0.4)   # 100th pct → 40 pts

    # ── Momentum Component (0–30 pts) ────────────────────────────────────
    # RSI sub-score (0–20 pts): ideal zone 55–70, penalised at extremes
    rsi = float(ind.get("rsi", 50) or 50)
    if 55 <= rsi <= 70:
        rsi_pts = 20.0
    elif 45 <= rsi < 55:
        rsi_pts = 10.0 + (rsi - 45) * 1.0   # 10–20 pts
    elif 70 < rsi <= 75:
        rsi_pts = max(0.0, 20.0 - (rsi - 70) * 4)  # Declining above 70
    elif 40 <= rsi < 45:
        rsi_pts = (rsi - 40) * 2.0          # 0–10 pts
    else:
        rsi_pts = 0.0

    # RS acceleration sub-score (0–10 pts): 1-month RS > 3-month RS
    rs_ratio    = float(ind.get("rs_ratio",    1.0) or 1.0)
    rs_ratio_1m = float(ind.get("rs_ratio_1m", 1.0) or 1.0)
    if rs_ratio > 0:
        rs_accel = rs_ratio_1m - rs_ratio
        rs_accel_pts = min(10.0, max(0.0, rs_accel * 100))   # +0.10 ratio delta → 10 pts
    else:
        rs_accel_pts = 0.0

    momentum_score = rsi_pts + rs_accel_pts

    # ── Volume Strength Component (0–20 pts) ─────────────────────────────
    # vol_ratio: 2× average = 10 pts, 3× = 20 pts (linear, capped)
    vol_ratio = float(ind.get("vol_ratio", 1.0) or 1.0)
    if vol_ratio >= 3.0:
        vol_score = 20.0
    elif vol_ratio >= 1.0:
        vol_score = (vol_ratio - 1.0) / (3.0 - 1.0) * 20.0
    else:
        vol_score = 0.0

    # ── Breakout Proximity Component (0–10 pts) ───────────────────────────
    # Reward stocks close to their 20-day or 52-week high
    close    = float(ind.get("close", 0) or 0)
    high_20d = float(ind.get("high_20d", 0) or 0)
    week52   = float(ind.get("week52_high", 0) or 0)

    breakout_pts = 0.0
    if high_20d > 0 and close > 0:
        # Close ≥ 20d high → breakout → max pts
        pct_below_20d = max(0.0, (high_20d - close) / high_20d)
        if pct_below_20d <= 0.005:   # Within 0.5% of 20-day high
            breakout_pts = 10.0
        elif pct_below_20d <= 0.03:  # Within 3% → partial pts
            breakout_pts = 10.0 * (1 - pct_below_20d / 0.03)
        elif week52 > 0:
            # Fall back to 52-week proximity if not near 20-day high
            pct_below_52w = max(0.0, (week52 - close) / week52)
            breakout_pts = max(0.0, 5.0 * (1 - pct_below_52w / 0.15))

    total = rs_score + momentum_score + vol_score + breakout_pts
    return round(min(100.0, max(0.0, total)), 2)


def rank_and_select(
    candidates: list,
    max_stocks: int = 5,
    verbose: bool = True,
) -> list:
    """
    Rank qualifying candidates and return the top N for execution.

    Parameters
    ----------
    candidates  : list of (signal_score, symbol, ind_dict) tuples
                  — all have already passed check_entry() and score threshold.
    max_stocks  : Maximum number of stocks to actually trade.
    verbose     : When True, logs the full ranking table with pass/skip status.

    Returns
    -------
    List of (signal_score, symbol, ind_dict) for the top max_stocks entries,
    sorted by rank_score DESC.  Remaining candidates are logged as SKIPPED.
    """
    if not candidates:
        return []

    # Compute rank score for each candidate
    ranked: list[tuple[float, float, str, dict]] = []
    for signal_score, symbol, ind in candidates:
        rank_score = compute_rank_score(ind)
        ranked.append((rank_score, signal_score, symbol, ind))

    # Sort by rank score descending (highest first)
    ranked.sort(key=lambda x: x[0], reverse=True)

    selected = ranked[:max_stocks]
    skipped  = ranked[max_stocks:]

    if verbose and ranked:
        logger.info(
            "[Ranker] %d candidates → selecting top %d",
            len(ranked), len(selected),
        )
        logger.info("[Ranker] %-12s  %6s  %6s  %6s  %7s  %8s",
                    "Symbol", "Rank", "Signal", "RS%", "RSI", "Status")
        logger.info("[Ranker] " + "-" * 60)
        for rank_score, sig_score, symbol, ind in selected:
            rs_rank = ind.get("rs_rank", 0)
            rsi     = ind.get("rsi", 0)
            logger.info(
                "[Ranker] %-12s  %6.1f  %6.1f  %6.0f  %7.1f  SELECTED",
                symbol, rank_score, sig_score, rs_rank, rsi,
            )
        for rank_score, sig_score, symbol, ind in skipped:
            rs_rank = ind.get("rs_rank", 0)
            rsi     = ind.get("rsi", 0)
            logger.info(
                "[Ranker] %-12s  %6.1f  %6.1f  %6.0f  %7.1f  SKIPPED (not in top %d)",
                symbol, rank_score, sig_score, rs_rank, rsi, max_stocks,
            )

    # Return in (signal_score, symbol, ind) format for compatibility with engine
    return [(sig_score, sym, ind) for _, sig_score, sym, ind in selected]


def build_decision_log(
    all_scanned: list[str],
    rs_passed: set[str],
    signal_passed: set[str],
    selected: set[str],
    rank_scores: Optional[dict] = None,
    rs_ranks: Optional[dict] = None,
) -> list[dict]:
    """
    Build per-stock decision log for daily debug output.

    Returns a list of dicts, one per scanned symbol, with:
      symbol, rs_pass, signal, rank_score, selected

    Parameters
    ----------
    all_scanned    : All symbols evaluated today.
    rs_passed      : Symbols that passed RS filter.
    signal_passed  : Symbols that passed entry signal check.
    selected       : Final selected symbols for trading.
    rank_scores    : {symbol: rank_score} — pre-computed scores.
    rs_ranks       : {symbol: rs_rank_pct} — RS percentile ranks.
    """
    rank_scores = rank_scores or {}
    rs_ranks    = rs_ranks    or {}

    log = []
    for symbol in sorted(all_scanned):
        log.append({
            "symbol":     symbol,
            "rs_pass":    "PASS" if symbol in rs_passed    else "FAIL",
            "signal":     "YES"  if symbol in signal_passed else "NO",
            "rank_score": round(rank_scores.get(symbol, 0.0), 1),
            "rs_rank":    round(rs_ranks.get(symbol, 0.0),   1),
            "selected":   "YES"  if symbol in selected       else "NO",
        })
    return log
