"""
Exit condition checker for open positions.
Evaluates stop loss, take profit, trailing stop, and signal reversal.
"""

from db.models import Position
from config.settings import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_PCT, RSI_OVERBOUGHT_EXIT
)


def check_exit(position: Position, current_price: float, ind: dict) -> tuple[bool, str]:
    """
    Returns (should_exit: bool, reason: str).
    Called daily for each open position.
    """
    entry = position.entry_price

    # 1. Hard stop loss
    stop_price = entry * (1 - STOP_LOSS_PCT)
    if current_price <= stop_price:
        return True, f"STOP_LOSS (price={current_price:.2f} ≤ stop={stop_price:.2f})"

    # 2. Take profit target
    target_price = entry * (1 + TAKE_PROFIT_PCT)
    if current_price >= target_price:
        return True, f"TAKE_PROFIT (price={current_price:.2f} ≥ target={target_price:.2f})"

    # 3. Trailing stop (from peak price)
    if current_price <= position.trailing_stop:
        return True, f"TRAILING_STOP (price={current_price:.2f} ≤ trail={position.trailing_stop:.2f})"

    # 4. Death cross — trend reversal
    if ind.get("death_cross"):
        return True, "SIGNAL: Death cross (EMA20 crossed below EMA50)"

    # 5. RSI overbought — take profit on extended move
    if ind.get("rsi", 0) >= RSI_OVERBOUGHT_EXIT:
        return True, f"SIGNAL: RSI overbought ({ind['rsi']:.1f} ≥ {RSI_OVERBOUGHT_EXIT})"

    return False, "HOLD"


def update_trailing_stop(position: Position, current_price: float) -> Position:
    """Update peak price and trailing stop if price moved higher."""
    if current_price > position.peak_price:
        position.peak_price = current_price
        position.trailing_stop = current_price * (1 - TRAILING_STOP_PCT)
    return position


def initial_stops(entry_price: float) -> dict:
    """Calculate initial stop loss, take profit, and trailing stop for a new position."""
    return {
        "stop_loss":     round(entry_price * (1 - STOP_LOSS_PCT), 2),
        "take_profit":   round(entry_price * (1 + TAKE_PROFIT_PCT), 2),
        "trailing_stop": round(entry_price * (1 - TRAILING_STOP_PCT), 2),
        "peak_price":    round(entry_price, 2),
    }
