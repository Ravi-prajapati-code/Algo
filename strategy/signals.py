"""
Daily signal generator — produces BUY/SELL/HOLD for all symbols.

Logs every single stock action with price, P&L, score, RS, and a running
portfolio summary so the daily log is self-contained and actionable.

Log format examples
-------------------
  [BUY]  RELIANCE.NS  ₹2,500.00 | score=82 GOOD  | RS=1.18 | SL=₹2,338
  [SELL] TCS.NS       ₹3,820.00 | P&L=+₹1,240 (+8.4%) | reason=TAKE_PROFIT
  [HOLD] INFY.NS      ₹1,652.00 | unreal=+₹680 (+5.3%) | trail=₹1,552 | held=12d
  ─────────────────────────────────────────────────────
  PORTFOLIO  ₹82,340  cash=₹31,200  invested=₹51,140  P&L=+₹7,340 (+10.8%)
  REGIME: BULL_TREND | BUY:3  SELL:1  HOLD:5
"""

import logging
from datetime import date
from typing import List

from db.models import Signal, Position
from strategy.entry import check_entry
from strategy.exit import check_exit, update_trailing_stop
from strategy.scoring import score_signal, score_to_size_factor, score_label
from db import repository as repo

logger = logging.getLogger(__name__)


def generate_signals(
    today: date,
    indicators: dict,           # {symbol: ind_dict}
    open_positions: List[Position],
    held_symbols: set,
    market_bullish: bool = True,
    regime: str = "BULL_TREND",
    portfolio_value: float = 0.0,
    cash: float = 0.0,
    initial_capital: float = 0.0,
) -> tuple[List[Signal], List[Position]]:
    """
    Generate BUY/SELL/HOLD signals for today.

    Parameters
    ----------
    today           : Trading date.
    indicators      : {symbol: ind_dict} — enriched with RS + quality fields.
    open_positions  : Current open positions.
    held_symbols    : Set of symbols already in portfolio.
    market_bullish  : Whether broad market allows new BUY entries.
    regime          : Current market regime string (for entry layer 1).
    portfolio_value : Total portfolio value for summary logging (optional).
    cash            : Available cash for summary logging (optional).
    initial_capital : Starting capital for cumulative P&L (optional).

    Returns
    -------
    (signals, updated_positions)
    """
    signals: List[Signal]    = []
    updated_positions: List[Position] = []

    sell_logs: list[str] = []
    hold_logs: list[str] = []
    buy_logs:  list[str] = []

    # ── STEP 1: Exit evaluation for all open positions ────────────────────
    # Exit logic ALWAYS runs — regime does NOT block exits.
    for pos in open_positions:
        ind = indicators.get(pos.symbol)
        if ind is None:
            updated_positions.append(pos)
            continue

        current_price = float(ind["close"])
        atr           = float(ind.get("atr", 0) or 0)
        pos           = update_trailing_stop(pos, current_price, atr)
        should_exit, reason = check_exit(pos, current_price, ind)

        if should_exit:
            gross_pnl  = (current_price - pos.entry_price) * pos.shares
            pnl_pct    = (current_price / pos.entry_price - 1) * 100
            pnl_sign   = "+" if gross_pnl >= 0 else ""
            sell_logs.append(
                f"  [SELL] {pos.symbol:<20} ₹{current_price:>9,.2f}"
                f" | P&L={pnl_sign}₹{gross_pnl:,.0f} ({pnl_sign}{pnl_pct:.1f}%)"
                f" | reason={reason.split('(')[0].strip()}"
            )
            signals.append(Signal(
                date=today, symbol=pos.symbol, action="SELL",
                score=0.0, price=current_price, reason=reason, indicators=ind,
            ))
        else:
            updated_positions.append(pos)
            unreal     = (current_price - pos.entry_price) * pos.shares
            unreal_pct = (current_price / pos.entry_price - 1) * 100
            held_days  = (today - pos.entry_date).days
            trail      = pos.trailing_stop
            sign       = "+" if unreal >= 0 else ""
            hold_logs.append(
                f"  [HOLD] {pos.symbol:<20} ₹{current_price:>9,.2f}"
                f" | unreal={sign}₹{unreal:,.0f} ({sign}{unreal_pct:.1f}%)"
                f" | trail=₹{trail:,.2f} | held={held_days}d"
            )
            signals.append(Signal(
                date=today, symbol=pos.symbol, action="HOLD",
                score=0.0, price=current_price, reason="HOLD", indicators=ind,
            ))

    # ── STEP 2: BUY screening ─────────────────────────────────────────────
    if not market_bullish:
        logger.info(
            "[Signals] %s: regime=%s — new BUY entries blocked",
            today, regime,
        )
    else:
        buy_candidates: List[Signal] = []

        for symbol, ind in indicators.items():
            if symbol in held_symbols:
                continue

            qualified, reason = check_entry(ind, symbol=symbol, regime=regime)
            if not qualified:
                continue

            score = score_signal(ind)
            ind["_score_factor"] = score_to_size_factor(score)
            buy_candidates.append(Signal(
                date=today, symbol=symbol, action="BUY",
                score=score, price=ind["close"], reason=reason, indicators=ind,
            ))

        buy_candidates.sort(key=lambda s: s.score, reverse=True)

        for sig in buy_candidates:
            ind       = sig.indicators
            rs_ratio  = ind.get("rs_ratio", 0) or 0
            rs_rank   = ind.get("rs_rank", 0) or 0
            atr       = ind.get("atr", 0) or 0
            stop_dist = atr * 1.5 if atr else sig.price * 0.06
            sl_price  = sig.price - stop_dist
            label     = score_label(sig.score)
            buy_logs.append(
                f"  [BUY]  {sig.symbol:<20} ₹{sig.price:>9,.2f}"
                f" | score={sig.score:.0f} {label:<8}"
                f" | RS={rs_ratio:.2f} (rank={rs_rank:.0f})"
                f" | SL=₹{sl_price:,.2f}"
            )
            signals.append(sig)

    # ── STEP 3: Rich log output ───────────────────────────────────────────
    _log_daily_summary(
        today, sell_logs, hold_logs, buy_logs,
        regime, portfolio_value, cash, initial_capital,
    )

    return signals, updated_positions


def _log_daily_summary(
    today: date,
    sell_logs: list[str],
    hold_logs: list[str],
    buy_logs:  list[str],
    regime: str,
    portfolio_value: float,
    cash: float,
    initial_capital: float,
) -> None:
    """Emit structured daily summary to the log."""
    sep = "  " + "─" * 65

    lines: list[str] = [
        "",
        f"  ╔══ DAILY SIGNALS: {today}  [regime={regime}] ══",
    ]

    if sell_logs:
        lines.append(f"  ── EXITS ({len(sell_logs)}) ──────────────────────────────")
        lines.extend(sell_logs)

    if hold_logs:
        lines.append(f"  ── HOLDING ({len(hold_logs)}) ─────────────────────────────")
        lines.extend(hold_logs)

    if buy_logs:
        lines.append(f"  ── BUY CANDIDATES ({len(buy_logs)}) ──────────────────────")
        lines.extend(buy_logs)

    if not sell_logs and not hold_logs and not buy_logs:
        lines.append("  (no open positions and no signals today)")

    # Portfolio summary line
    if portfolio_value > 0:
        invested = portfolio_value - cash
        pnl      = portfolio_value - initial_capital if initial_capital else 0.0
        pnl_pct  = pnl / initial_capital * 100 if initial_capital else 0.0
        sign     = "+" if pnl >= 0 else ""
        lines.append(sep)
        lines.append(
            f"  PORTFOLIO  ₹{portfolio_value:>10,.2f}"
            f"  cash=₹{cash:,.0f}"
            f"  invested=₹{invested:,.0f}"
            f"  P&L={sign}₹{pnl:,.0f} ({sign}{pnl_pct:.1f}%)"
        )

    lines.append(
        f"  SUMMARY  BUY:{len(buy_logs)}  "
        f"SELL:{len(sell_logs)}  HOLD:{len(hold_logs)}"
    )
    lines.append("  ╚" + "═" * 60)

    logger.info("\n".join(lines))
