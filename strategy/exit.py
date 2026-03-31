"""
Exit condition checker for open positions.

Exit hierarchy (checked in order, first trigger wins):
  1. ATR hard stop    — max(STOP_LOSS_PCT=2%, ATR×1.5) below entry
  2. Take profit      — 13 % above entry (configurable)
  3. ATR trailing stop— 1.0 × ATR below peak price (locks in profits)
  4. Trend exit       — price < SMA20 (EMA_FAST) — trend broken
  5. Death cross      — EMA20 crosses below EMA50
  6. RSI overbought   — RSI ≥ 75 (take profit on extended move)
  7. Time-based exit  — held > MAX_HOLD_DAYS=10 calendar days
  8. RS deterioration — stock starts underperforming index while in drawdown

Stop loss uses 2% floor or ATR×1.5 (whichever is larger):
  - Volatile stock (ATR 3 %) → stop placed 4.5 % below entry
  - Calm stock (ATR 0.8 %)   → stop placed max(2 %, 1.2 %) = 2 % below
  This prevents being stopped out by normal noise on volatile stocks,
  while keeping tight stops on low-volatility quality names.

Time exit is 10 days (swing trade horizon) — forces reassessment even if
no other exit triggers, preventing dead-money positions.

Trend exit (price < SMA20) provides an early warning exit when the primary
support level breaks, before the death cross forms (which is a lagging signal).
"""

import logging
from db.models import Position
from config.settings import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_PCT,
    RSI_OVERBOUGHT_EXIT, MAX_HOLD_DAYS, ATR_STOP_MULTIPLIER,
)

logger = logging.getLogger(__name__)

# Trailing stop uses a tighter ATR multiplier than the hard stop
_TRAILING_ATR_MULT = 1.0   # Trailing = 1.0 × ATR from peak (tighter)


def _atr_stop_distance(entry_price: float, atr: float) -> float:
    """
    Compute the stop distance using ATR, with a percentage floor.

    stop_distance = max(STOP_LOSS_PCT × entry, ATR_STOP_MULTIPLIER × ATR)

    This ensures:
    - Minimum 2 % (STOP_LOSS_PCT) stop regardless of ATR
    - Wider stop for volatile stocks to avoid noise-stops
    """
    fixed_stop = entry_price * STOP_LOSS_PCT
    atr_stop   = atr * ATR_STOP_MULTIPLIER if atr and atr > 0 else 0.0
    return max(fixed_stop, atr_stop)


def check_exit(
    position: Position,
    current_price: float,
    ind: dict,
) -> tuple[bool, str]:
    """
    Evaluate all exit conditions for an open position.

    Returns (should_exit: bool, reason: str).
    Called daily for every open position — ALWAYS runs regardless of regime.
    """
    entry = position.entry_price
    atr   = float(ind.get("atr", 0) or 0)

    # ── 1. ATR hard stop (volatility-aware) ──────────────────────────────
    stop_distance = _atr_stop_distance(entry, atr)
    stop_price    = entry - stop_distance
    if current_price <= stop_price:
        stop_type = "ATR" if atr > 0 else "PCT"
        return True, (
            f"STOP_LOSS[{stop_type}] "
            f"price={current_price:.2f} ≤ stop={stop_price:.2f} "
            f"(dist={stop_distance:.2f}, ATR={atr:.2f})"
        )

    # ── 2. Take profit target ─────────────────────────────────────────────
    target_price = entry * (1 + TAKE_PROFIT_PCT)
    if current_price >= target_price:
        pnl_pct = (current_price - entry) / entry * 100
        return True, (
            f"TAKE_PROFIT "
            f"price={current_price:.2f} ≥ target={target_price:.2f} "
            f"(+{pnl_pct:.1f}%)"
        )

    # ── 3. ATR trailing stop (tighter than fixed %) ───────────────────────
    if current_price <= position.trailing_stop:
        peak_gain = (position.peak_price - entry) / entry * 100
        return True, (
            f"TRAILING_STOP "
            f"price={current_price:.2f} ≤ trail={position.trailing_stop:.2f} "
            f"(peak={position.peak_price:.2f}, peak_gain={peak_gain:.1f}%)"
        )

    # ── 4. Trend exit — price below SMA20 (primary support broken) ───────
    ema_fast = float(ind.get("ema_fast", 0) or 0)
    if ema_fast > 0 and current_price < ema_fast:
        return True, (
            f"TREND_EXIT: price={current_price:.2f} < SMA20={ema_fast:.2f} "
            f"({(current_price/ema_fast - 1)*100:+.1f}%) — trend support broken"
        )

    # ── 5. Death cross — trend reversal ───────────────────────────────────
    if ind.get("death_cross"):
        return True, "SIGNAL: Death cross (EMA20 crossed below EMA50)"

    # ── 6. RSI overbought — extended move ─────────────────────────────────
    rsi = ind.get("rsi", 0)
    if rsi >= RSI_OVERBOUGHT_EXIT:
        return True, f"SIGNAL: RSI overbought ({rsi:.1f} ≥ {RSI_OVERBOUGHT_EXIT})"

    # ── 7. RS deterioration (exit if underperforming AND in drawdown) ────────
    # NOTE: Time-based exit is handled by the caller (backtest engine /
    # signals.py) using the correct trading date, not date.today().
    rs_ratio = float(ind.get("rs_ratio", 1.0) or 1.0)
    in_drawdown = current_price < entry * 0.98   # Down ≥ 2 % from entry
    if rs_ratio > 0 and rs_ratio < 0.85 and in_drawdown:
        return True, (
            f"RS_DETERIORATION: "
            f"rs_ratio={rs_ratio:.3f} < 0.85 while in drawdown "
            f"({(current_price/entry - 1)*100:+.1f}%)"
        )

    return False, "HOLD"


def update_trailing_stop(position: Position, current_price: float, atr: float = 0.0) -> Position:
    """
    Update peak price and trailing stop if price moved higher.

    Uses ATR-based trailing when ATR is available (tighter, adaptive);
    falls back to fixed-percentage trailing stop.
    """
    if current_price > position.peak_price:
        position.peak_price = current_price
        if atr and atr > 0:
            # ATR trailing: 1.0× ATR below peak
            position.trailing_stop = current_price - (atr * _TRAILING_ATR_MULT)
        else:
            position.trailing_stop = current_price * (1 - TRAILING_STOP_PCT)
        position.trailing_stop = round(position.trailing_stop, 2)
    return position


def initial_stops(entry_price: float, atr: float = 0.0) -> dict:
    """
    Calculate initial stop loss, take profit, and trailing stop for a new position.

    Uses ATR-based stop when ATR is available; falls back to fixed %.
    """
    stop_distance  = _atr_stop_distance(entry_price, atr)
    stop_loss      = round(entry_price - stop_distance, 2)
    take_profit    = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

    if atr and atr > 0:
        trailing_stop = round(entry_price - atr * _TRAILING_ATR_MULT, 2)
    else:
        trailing_stop = round(entry_price * (1 - TRAILING_STOP_PCT), 2)

    return {
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
        "trailing_stop": trailing_stop,
        "peak_price":    round(entry_price, 2),
        "stop_distance": round(stop_distance, 2),
        "stop_type":     "atr" if (atr and atr > 0) else "pct",
    }
