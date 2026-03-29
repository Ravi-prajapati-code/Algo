"""
Position sizer — calculates share count based on risk per trade.
Uses the classic fixed-risk sizing: risk only 2% of portfolio per trade.
"""

import math
from config.settings import MAX_RISK_PER_TRADE_PCT, STOP_LOSS_PCT, MAX_STOCK_ALLOCATION_PCT


def calculate_shares(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
) -> int:
    """
    Calculate number of shares to buy.

    Logic:
      - Max loss per trade = portfolio_value * MAX_RISK_PER_TRADE_PCT
      - Risk per share = entry_price - stop_loss_price
      - Shares = max_loss / risk_per_share
      - Capped by: available cash & max stock allocation

    Returns 0 if no position should be taken.
    """
    if entry_price <= 0 or stop_loss_price >= entry_price:
        return 0

    # Risk-based sizing
    max_loss = portfolio_value * MAX_RISK_PER_TRADE_PCT
    risk_per_share = entry_price - stop_loss_price
    shares_by_risk = math.floor(max_loss / risk_per_share)

    # Cap by max allocation per stock
    max_allocation = portfolio_value * MAX_STOCK_ALLOCATION_PCT
    shares_by_allocation = math.floor(max_allocation / entry_price)

    # Cap by available cash
    shares_by_cash = math.floor(available_cash / entry_price)

    shares = min(shares_by_risk, shares_by_allocation, shares_by_cash)
    return max(0, shares)


def position_value(shares: int, price: float) -> float:
    return round(shares * price, 2)
