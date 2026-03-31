"""
Relative Strength (RS) calculator — measures stock performance vs Nifty 50.

Only stocks outperforming the broad index are eligible for BUY entries.
Buying relative-strength leaders means we trade with institutional momentum,
not against it.

RS Ratio
--------
rs_ratio = (stock_return_N_days) / (index_return_N_days)

  rs_ratio > 1.0  → stock outperforming Nifty  → eligible
  rs_ratio < 1.0  → stock underperforming Nifty → skip
  rs_ratio = 0.0  → insufficient data           → skip

RS Rank (percentile within watchlist)
--------------------------------------
After computing rs_ratio for all symbols, each symbol is ranked on a
0–100 scale against the rest of the watchlist.  Only stocks in the top
60 % (rs_rank ≥ 40) are considered for entry — this ensures we pick
leaders, not laggards.

Lookback periods
----------------
Primary  : 63 trading days (~3 months)  — captures medium-term momentum
Secondary: 21 trading days (~1 month)   — confirms recent acceleration
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
RS_PRIMARY_LOOKBACK   = 63    # ~3-month return comparison
RS_SECONDARY_LOOKBACK = 21    # ~1-month return for recency check
RS_MIN_RATIO          = 1.0   # Stock must match or beat index (used for rs_outperforming flag)
RS_MIN_RANK           = 30.0  # Minimum percentile rank within watchlist (lowered from 40)
RS_MIN_DATA_BARS      = 70    # Minimum bars needed to compute RS reliably


def compute_rs_ratio(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    lookback: int = RS_PRIMARY_LOOKBACK,
) -> float:
    """
    Compute the relative strength ratio of a stock vs the index.

    rs_ratio = stock_return / index_return  over `lookback` bars.

    Returns 0.0 if either series has insufficient data or index return is 0.
    """
    if stock_df is None or index_df is None:
        return 0.0
    if len(stock_df) < lookback + 5 or len(index_df) < lookback + 5:
        return 0.0

    try:
        stock_close = stock_df["close"].astype(float)
        index_close = index_df["close"].astype(float)

        # Align on common dates (stock may have fewer rows than index)
        common_idx = stock_close.index.intersection(index_close.index)
        if len(common_idx) < lookback + 5:
            return 0.0

        s = stock_close.loc[common_idx]
        ix = index_close.loc[common_idx]

        stock_ret = (float(s.iloc[-1]) - float(s.iloc[-lookback])) / float(s.iloc[-lookback])
        index_ret = (float(ix.iloc[-1]) - float(ix.iloc[-lookback])) / float(ix.iloc[-lookback])

        if abs(index_ret) < 1e-9:
            return 1.0   # Index flat → neutral RS

        rs = (1 + stock_ret) / (1 + index_ret)
        return round(rs, 4)

    except Exception as e:
        logger.debug("[RS] compute_rs_ratio error: %s", e)
        return 0.0


def compute_rs_metrics(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> dict:
    """
    Compute full RS metrics for a single stock.

    Returns dict with:
      rs_ratio        : 3-month RS vs index (>1.0 = outperforming)
      rs_ratio_1m     : 1-month RS vs index
      rs_outperforming: True if rs_ratio ≥ RS_MIN_RATIO
      rs_accelerating : True if 1-month RS > 3-month RS (momentum picking up)
    """
    rs_3m = compute_rs_ratio(stock_df, index_df, lookback=RS_PRIMARY_LOOKBACK)
    rs_1m = compute_rs_ratio(stock_df, index_df, lookback=RS_SECONDARY_LOOKBACK)

    return {
        "rs_ratio":         rs_3m,
        "rs_ratio_1m":      rs_1m,
        "rs_outperforming": rs_3m >= RS_MIN_RATIO,
        "rs_accelerating":  rs_1m > rs_3m and rs_3m > 0,
    }


def compute_rs_for_all(
    data: dict,
    index_df: Optional[pd.DataFrame],
) -> dict:
    """
    Compute RS metrics for every symbol in `data`.

    Parameters
    ----------
    data     : {symbol: ohlcv_DataFrame}
    index_df : Nifty 50 OHLCV DataFrame (from fetch_index)

    Returns
    -------
    {symbol: rs_metrics_dict}  where rs_metrics_dict contains:
      rs_ratio, rs_ratio_1m, rs_outperforming, rs_accelerating, rs_rank
    """
    if index_df is None or index_df.empty:
        logger.warning("[RS] No index data — RS metrics unavailable for all symbols")
        return {}

    raw: dict = {}
    for symbol, df in data.items():
        if len(df) < RS_MIN_DATA_BARS:
            continue
        metrics = compute_rs_metrics(df, index_df)
        if metrics["rs_ratio"] > 0:
            raw[symbol] = metrics

    if not raw:
        return {}

    # ── Compute percentile rank within watchlist ───────────────────────────
    ratios = {sym: m["rs_ratio"] for sym, m in raw.items()}
    sorted_syms = sorted(ratios, key=lambda s: ratios[s])
    n = len(sorted_syms)

    for rank_idx, sym in enumerate(sorted_syms):
        percentile = (rank_idx / max(n - 1, 1)) * 100
        raw[sym]["rs_rank"] = round(percentile, 1)
        raw[sym]["rs_qualified"] = (
            raw[sym]["rs_outperforming"] and percentile >= RS_MIN_RANK
        )

    qualified = sum(1 for m in raw.values() if m.get("rs_qualified"))
    logger.info(
        "[RS] Computed for %d symbols — %d qualify (ratio≥%.2f outperforming, rank≥%.0f)",
        len(raw), qualified, RS_MIN_RATIO, RS_MIN_RANK,
    )
    return raw
