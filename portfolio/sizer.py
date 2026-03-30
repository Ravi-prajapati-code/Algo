"""
Position sizer — calculates share count based on risk per trade.

Supports two sizing methods:
  1. ATR-based (primary): stop_distance = atr × ATR_STOP_MULTIPLIER
     More accurate — respects current volatility of each stock.
  2. Fixed-% (fallback):  stop_distance = entry_price - stop_loss_price
     Used when ATR is not available.

Both methods cap shares by:
  - Max stock allocation (20 % of portfolio)
  - Available cash (minus 5 % reserve)
  - Drawdown reduction factor (50 % in moderate drawdown)

For the full production implementation see portfolio/optimizer.py
(PortfolioAllocator), which layers in regime and ML confidence scaling.
"""

import math
from typing import Optional

from config.settings import (
    MAX_RISK_PER_TRADE_PCT,
    MAX_STOCK_ALLOCATION_PCT,
    ATR_STOP_MULTIPLIER,
)


def calculate_shares(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
    atr: Optional[float] = None,
    drawdown_factor: float = 1.0,   # 0.5 in moderate DD, 1.0 normal
) -> int:
    """
    Calculate number of shares to buy.

    Sizing logic
    ------------
    1. risk_budget = portfolio_value × MAX_RISK_PER_TRADE_PCT × drawdown_factor
    2. If ATR provided:
           stop_distance = atr × ATR_STOP_MULTIPLIER
       else:
           stop_distance = entry_price - stop_loss_price
    3. shares_by_risk      = floor(risk_budget / stop_distance)
    4. shares_by_allocation = floor(portfolio × 20 % / entry_price)
    5. shares_by_cash       = floor(available_cash × 0.95 / entry_price)
    6. shares = min(all three)

    Returns 0 if position should not be taken.
    """
    if entry_price <= 0:
        return 0

    # Determine stop distance
    if atr and atr > 0:
        stop_distance = atr * ATR_STOP_MULTIPLIER
    else:
        stop_distance = entry_price - stop_loss_price

    if stop_distance <= 0:
        return 0

    # Risk budget (scaled by drawdown factor)
    effective_risk_pct = MAX_RISK_PER_TRADE_PCT * max(0.0, min(1.0, drawdown_factor))
    risk_budget = portfolio_value * effective_risk_pct
    shares_by_risk = math.floor(risk_budget / stop_distance)

    # Cap by max stock allocation
    max_alloc = portfolio_value * MAX_STOCK_ALLOCATION_PCT
    shares_by_allocation = math.floor(max_alloc / entry_price)

    # Cap by available cash (keep 5 % reserve)
    shares_by_cash = math.floor(available_cash * 0.95 / entry_price)

    shares = min(shares_by_risk, shares_by_allocation, shares_by_cash)
    return max(0, shares)


def position_value(shares: int, price: float) -> float:
    return round(shares * price, 2)
