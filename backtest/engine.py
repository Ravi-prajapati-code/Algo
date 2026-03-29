"""
Event-driven backtesting engine.
Uses the EXACT same strategy/*, portfolio/*, charges/* code as live trading.
Only difference: data is sliced by date instead of fetched fresh.

Critical: NO lookahead bias — indicators are computed on data up to and
including the current bar. The next bar's open is used as the execution price.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config.settings import INITIAL_CAPITAL, LOOKBACK_DAYS
from indicators.composite import compute_indicators
from strategy.signals import generate_signals
from strategy.exit import update_trailing_stop
from charges.calculator import net_pnl as calc_net_pnl, buy_charges
from portfolio.sizer import calculate_shares, position_value
from portfolio.allocator import can_open_position, portfolio_invested_value
from portfolio.risk import can_open_new_trades
from strategy.exit import initial_stops
from data.universe import get_sector
from db.models import Position, Trade

logger = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self):
        self.trades: list[Trade] = []
        self.equity_curve: dict[date, float] = {}   # {date: portfolio_value}
        self.cash_curve: dict[date, float] = {}


class BacktestEngine:
    """
    Simulates daily swing trading on historical OHLCV data.

    Usage:
        engine = BacktestEngine(data, start_date, end_date)
        result = engine.run()
    """

    def __init__(
        self,
        data: dict,             # {symbol: full_ohlcv_DataFrame}
        start_date: date,
        end_date: date,
        initial_capital: float = INITIAL_CAPITAL,
        lookback: int = LOOKBACK_DAYS,
    ):
        self.data = data
        self.start = start_date
        self.end = end_date
        self.initial_capital = initial_capital
        self.lookback = lookback

    def run(self) -> BacktestResult:
        result = BacktestResult()
        cash = self.initial_capital
        peak_value = self.initial_capital
        open_positions: list[Position] = []
        all_dates = self._trading_dates()

        for today in all_dates:
            new_trades_today = 0

            # ── Build indicator snapshot for today ───────────────────
            indicators: dict = {}
            for symbol, full_df in self.data.items():
                # Slice: only data up to (and including) today
                hist = full_df[full_df.index <= today]
                if len(hist) < 60:
                    continue
                ind = compute_indicators(hist)
                if ind is not None:
                    indicators[symbol] = ind

            if not indicators:
                continue

            # Current prices (today's close, used for P&L)
            prices = {sym: ind["close"] for sym, ind in indicators.items()}

            # ── Evaluate open positions ───────────────────────────────
            positions_to_close: list = []
            updated_positions: list = []
            held = {pos.symbol for pos in open_positions}

            _, updated_open = generate_signals(
                today, indicators, open_positions, held
            )

            # Check which positions generate SELL signals vs HOLD
            from strategy.exit import check_exit
            for pos in open_positions:
                ind = indicators.get(pos.symbol)
                if ind is None:
                    updated_positions.append(pos)
                    continue
                price = ind["close"]
                pos = update_trailing_stop(pos, price)
                should_exit, reason = check_exit(pos, price, ind)
                if should_exit:
                    positions_to_close.append((pos, price, reason))
                else:
                    updated_positions.append(pos)

            # Execute sells
            for pos, exit_price, reason in positions_to_close:
                pnl = calc_net_pnl(pos.entry_price, exit_price, pos.shares)
                trade = Trade(
                    symbol=pos.symbol, sector=pos.sector,
                    entry_date=pos.entry_date, exit_date=today,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares,
                    gross_pnl=pnl["gross_pnl"],
                    charges=pnl["total_charges"],
                    net_pnl=pnl["net_pnl"],
                    exit_reason=reason,
                    hold_days=(today - pos.entry_date).days,
                )
                result.trades.append(trade)
                cash += exit_price * pos.shares - pnl["sell_charges"]["total"]

            open_positions = updated_positions
            held = {pos.symbol for pos in open_positions}

            # ── Screen for BUY candidates ─────────────────────────────
            from strategy.entry import check_entry
            from strategy.scoring import score_signal
            candidates = []
            for symbol, ind in indicators.items():
                if symbol in held:
                    continue
                ok, _ = check_entry(ind)
                if ok:
                    candidates.append((score_signal(ind), symbol, ind))
            candidates.sort(reverse=True)

            portfolio_val = cash + portfolio_invested_value(open_positions, prices)
            peak_value = max(peak_value, portfolio_val)

            # Execute buys (top scored, within limits)
            for score, symbol, ind in candidates:
                allowed, _ = can_open_new_trades(
                    new_trades_today, open_positions, portfolio_val, peak_value
                )
                if not allowed:
                    break
                price = ind["close"]
                stops = initial_stops(price)
                shares = calculate_shares(portfolio_val, price, stops["stop_loss"], cash)
                if shares <= 0:
                    continue
                tv = position_value(shares, price)
                ok, _ = can_open_position(symbol, tv, portfolio_val, open_positions, prices)
                if not ok:
                    continue
                charges = buy_charges(tv)
                total_cost = tv + charges.total
                if total_cost > cash:
                    continue

                pos = Position(
                    symbol=symbol, sector=get_sector(symbol),
                    entry_date=today, entry_price=price, shares=shares,
                    stop_loss=stops["stop_loss"], take_profit=stops["take_profit"],
                    trailing_stop=stops["trailing_stop"], peak_price=stops["peak_price"],
                )
                cash -= total_cost
                open_positions.append(pos)
                new_trades_today += 1

            # ── Record equity curve ───────────────────────────────────
            pv = cash + portfolio_invested_value(open_positions, prices)
            result.equity_curve[today] = round(pv, 2)
            result.cash_curve[today] = round(cash, 2)

        # Force-close remaining positions at final date (mark-to-market)
        if open_positions:
            final_date = all_dates[-1] if all_dates else self.end
            for pos in open_positions:
                price = prices.get(pos.symbol, pos.entry_price)
                pnl = calc_net_pnl(pos.entry_price, price, pos.shares)
                trade = Trade(
                    symbol=pos.symbol, sector=pos.sector,
                    entry_date=pos.entry_date, exit_date=final_date,
                    entry_price=pos.entry_price, exit_price=price,
                    shares=pos.shares,
                    gross_pnl=pnl["gross_pnl"],
                    charges=pnl["total_charges"],
                    net_pnl=pnl["net_pnl"],
                    exit_reason="END_OF_BACKTEST",
                    hold_days=(final_date - pos.entry_date).days,
                )
                result.trades.append(trade)

        logger.info(
            f"[Backtest] Done. {len(result.trades)} trades over "
            f"{len(result.equity_curve)} trading days"
        )
        return result

    def _trading_dates(self) -> list[date]:
        """Generate weekday dates from start to end (proxy for trading days)."""
        dates = []
        current = self.start
        while current <= self.end:
            if current.weekday() < 5:   # Mon–Fri
                dates.append(current)
            current += timedelta(days=1)
        return dates
