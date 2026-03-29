"""
Portfolio-level risk controls (circuit breakers).
Checked once per day before processing new BUY signals.
"""

from typing import List
from db.models import Position
from config.settings import (
    MAX_NEW_TRADES_PER_DAY,
    MAX_OPEN_POSITIONS,
    INITIAL_CAPITAL,
)

# Max portfolio drawdown before halting new buys
MAX_PORTFOLIO_DRAWDOWN = 0.20   # Halt new buys if portfolio is down 20% from peak


def can_open_new_trades(
    new_trades_today: int,
    open_positions: List[Position],
    portfolio_value: float,
    peak_portfolio_value: float,
) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Checks: daily trade limit, max open positions, drawdown circuit breaker.
    """
    if new_trades_today >= MAX_NEW_TRADES_PER_DAY:
        return False, f"Daily trade limit reached ({MAX_NEW_TRADES_PER_DAY} trades/day)"

    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions reached ({MAX_OPEN_POSITIONS})"

    drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
    if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
        return False, (
            f"Portfolio drawdown circuit breaker: {drawdown:.1%} "
            f"(limit {MAX_PORTFOLIO_DRAWDOWN:.0%})"
        )

    return True, "OK"
