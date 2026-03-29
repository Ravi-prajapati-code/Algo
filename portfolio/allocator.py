"""
Allocation checker — enforces stock-level and sector-level caps.
Called before any new position is opened.
"""

from typing import List
from db.models import Position
from data.universe import get_sector
from config.settings import MAX_STOCK_ALLOCATION_PCT, MAX_SECTOR_ALLOCATION_PCT


def portfolio_invested_value(positions: List[Position], prices: dict) -> float:
    """Current market value of all open positions."""
    return sum(
        prices.get(pos.symbol, pos.entry_price) * pos.shares
        for pos in positions
    )


def sector_allocation(positions: List[Position], prices: dict, portfolio_value: float) -> dict:
    """Return {sector: pct_of_portfolio} for open positions."""
    sector_values: dict = {}
    for pos in positions:
        sector = get_sector(pos.symbol)
        value = prices.get(pos.symbol, pos.entry_price) * pos.shares
        sector_values[sector] = sector_values.get(sector, 0) + value

    return {s: v / portfolio_value for s, v in sector_values.items()} if portfolio_value > 0 else {}


def can_open_position(
    symbol: str,
    trade_value: float,
    portfolio_value: float,
    open_positions: List[Position],
    prices: dict,
) -> tuple[bool, str]:
    """
    Check if a new position can be opened within allocation rules.
    Returns (allowed: bool, reason: str).
    """
    # Check stock already held
    held = {pos.symbol for pos in open_positions}
    if symbol in held:
        return False, f"{symbol} already in portfolio"

    # Stock allocation cap
    stock_pct = trade_value / portfolio_value
    if stock_pct > MAX_STOCK_ALLOCATION_PCT:
        return False, (
            f"{symbol}: trade value {stock_pct:.1%} exceeds stock cap "
            f"{MAX_STOCK_ALLOCATION_PCT:.0%}"
        )

    # Sector allocation cap
    sector = get_sector(symbol)
    sector_alloc = sector_allocation(open_positions, prices, portfolio_value)
    existing_sector_pct = sector_alloc.get(sector, 0.0)
    new_sector_pct = existing_sector_pct + stock_pct
    if new_sector_pct > MAX_SECTOR_ALLOCATION_PCT:
        return False, (
            f"{symbol} ({sector}): sector would reach {new_sector_pct:.1%}, "
            f"cap is {MAX_SECTOR_ALLOCATION_PCT:.0%}"
        )

    return True, "OK"
