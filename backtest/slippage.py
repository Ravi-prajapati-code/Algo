"""
Slippage and market-impact models for realistic backtesting.

Three models available (select via config):
  - 'none'       : 0 % slippage (optimistic baseline)
  - 'fixed_pct'  : flat percentage (e.g. 0.1 % per side)
  - 'volatility' : ATR-proportional (higher volatility → more slippage)

Slippage is applied to the EXECUTION price (next-bar open).
For BUY  orders: execution_price = open × (1 + slippage_rate)
For SELL orders: execution_price = open × (1 − slippage_rate)

Gap simulation
--------------
When next-bar open gaps more than GAP_THRESHOLD_PCT vs prior close,
the gap open is used as-is (no further slippage adjustment) and
the trade is flagged as a "gap fill" for analysis.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

SlippageModel = Literal["none", "fixed_pct", "volatility"]

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL: SlippageModel = "fixed_pct"
FIXED_SLIPPAGE_PCT = 0.001      # 0.10 % per side
VOLATILITY_ATR_MULT = 0.10      # slippage = atr × 0.10  (as % of price)
GAP_THRESHOLD_PCT   = 0.02      # Gap > 2 % treated as gap-open scenario
MAX_SLIPPAGE_PCT    = 0.005     # Hard cap: never apply more than 0.5 % slippage


def apply_slippage(
    side: Literal["buy", "sell"],
    open_price: float,
    prior_close: float,
    atr: float = 0.0,
    model: SlippageModel = DEFAULT_MODEL,
) -> tuple[float, str]:
    """
    Compute execution price after slippage and gap adjustment.

    Parameters
    ----------
    side        : 'buy' or 'sell'
    open_price  : Next bar's opening price (raw).
    prior_close : Previous bar's closing price.
    atr         : 14-day ATR of the instrument (used in volatility model).
    model       : Slippage model to apply.

    Returns
    -------
    (execution_price, note)
    note is a short string describing what happened (for trade logs).
    """
    if open_price <= 0:
        return prior_close, "invalid_open"

    # ── Gap detection ────────────────────────────────────────────────
    gap_pct = (open_price - prior_close) / prior_close if prior_close > 0 else 0.0
    is_gap = abs(gap_pct) > GAP_THRESHOLD_PCT

    if is_gap:
        # Gap open: trade executes AT the gap open price, no extra slippage
        direction = "up" if gap_pct > 0 else "down"
        note = f"gap_{direction}_{abs(gap_pct):.1%}"
        # For a stop-loss triggered sell on gap-down, execution is at gap open
        # which is already worse — we do NOT add extra slippage
        return round(open_price, 2), note

    # ── Compute slippage rate ────────────────────────────────────────
    if model == "none":
        rate = 0.0
    elif model == "fixed_pct":
        rate = FIXED_SLIPPAGE_PCT
    elif model == "volatility":
        if atr > 0 and open_price > 0:
            rate = min(MAX_SLIPPAGE_PCT, (atr / open_price) * VOLATILITY_ATR_MULT)
        else:
            rate = FIXED_SLIPPAGE_PCT   # Fallback to fixed if ATR not available
    else:
        rate = FIXED_SLIPPAGE_PCT

    rate = min(rate, MAX_SLIPPAGE_PCT)  # Hard cap

    # ── Apply slippage in correct direction ──────────────────────────
    if side == "buy":
        execution_price = open_price * (1.0 + rate)
    else:
        execution_price = open_price * (1.0 - rate)

    note = f"slippage_{model}_{rate:.3%}"
    return round(execution_price, 2), note


def simulate_partial_fill(
    requested_shares: int,
    trade_value: float,
    avg_daily_volume: float,
    participation_rate: float = 0.10,
) -> int:
    """
    Simulate partial fills for large orders relative to daily volume.

    For small orders (< 1 % of ADV) → full fill.
    For larger orders → cap at participation_rate × ADV to avoid market impact.

    Parameters
    ----------
    requested_shares    : Number of shares we want to buy/sell.
    trade_value         : Total trade value in ₹.
    avg_daily_volume    : Average daily volume in shares (20-day).
    participation_rate  : Max fraction of ADV we are willing to fill (default 10 %).

    Returns
    -------
    filled_shares (int) — may be less than requested.
    """
    if avg_daily_volume <= 0:
        return requested_shares  # Unknown volume → assume full fill

    max_fillable = int(avg_daily_volume * participation_rate)
    filled = min(requested_shares, max_fillable)

    if filled < requested_shares:
        logger.debug(
            "[Slippage] Partial fill: requested=%d, filled=%d (ADV=%.0f, rate=%.0f%%)",
            requested_shares, filled, avg_daily_volume, participation_rate * 100,
        )

    return max(0, filled)
