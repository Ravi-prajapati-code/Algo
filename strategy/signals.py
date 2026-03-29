"""
Daily signal generator.
Produces BUY/SELL/HOLD signals for all symbols.
"""

import logging
from datetime import date
from typing import List

from db.models import Signal, Position
from strategy.entry import check_entry
from strategy.exit import check_exit, update_trailing_stop
from strategy.scoring import score_signal
from db import repository as repo

logger = logging.getLogger(__name__)


def generate_signals(
    today: date,
    indicators: dict,           # {symbol: ind_dict}
    open_positions: List[Position],
    held_symbols: set,
) -> tuple[List[Signal], List[Position]]:
    """
    Returns:
        signals        — list of Signal objects (BUY/SELL for the day)
        updated_positions — positions with updated trailing stops
    """
    signals: List[Signal] = []
    updated_positions: List[Position] = []

    # ── STEP 1: Check exit conditions for all open positions ──────────────
    for pos in open_positions:
        ind = indicators.get(pos.symbol)
        if ind is None:
            # No fresh data — keep holding
            updated_positions.append(pos)
            continue

        current_price = ind["close"]

        # Update trailing stop first
        pos = update_trailing_stop(pos, current_price)

        should_exit, reason = check_exit(pos, current_price, ind)

        if should_exit:
            sig = Signal(
                date=today,
                symbol=pos.symbol,
                action="SELL",
                score=0.0,
                price=current_price,
                reason=reason,
                indicators=ind,
            )
            signals.append(sig)
            logger.info(f"[Signals] SELL {pos.symbol} @ {current_price:.2f} — {reason}")
        else:
            updated_positions.append(pos)
            sig = Signal(
                date=today,
                symbol=pos.symbol,
                action="HOLD",
                score=0.0,
                price=current_price,
                reason="HOLD",
                indicators=ind,
            )
            signals.append(sig)

    # ── STEP 2: Screen candidates for BUY ────────────────────────────────
    buy_candidates: List[Signal] = []

    for symbol, ind in indicators.items():
        if symbol in held_symbols:
            continue  # Already holding this stock

        qualified, reason = check_entry(ind)
        if not qualified:
            continue

        score = score_signal(ind)
        sig = Signal(
            date=today,
            symbol=symbol,
            action="BUY",
            score=score,
            price=ind["close"],
            reason=reason,
            indicators=ind,
        )
        buy_candidates.append(sig)

    # Sort by score descending
    buy_candidates.sort(key=lambda s: s.score, reverse=True)
    signals.extend(buy_candidates)

    logger.info(
        f"[Signals] {today}: "
        f"{sum(1 for s in signals if s.action=='BUY')} BUY, "
        f"{sum(1 for s in signals if s.action=='SELL')} SELL, "
        f"{sum(1 for s in signals if s.action=='HOLD')} HOLD"
    )

    return signals, updated_positions
