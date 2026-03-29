"""
Portfolio manager — the single source of truth for portfolio state.
Processes signals → executes trades (paper) → updates DB.
"""

import logging
from datetime import date
from typing import List

from db.models import Position, Trade, Signal, PortfolioSnapshot
from db import repository as repo
from portfolio.sizer import calculate_shares, position_value
from portfolio.allocator import can_open_position, portfolio_invested_value
from portfolio.risk import can_open_new_trades
from charges.calculator import net_pnl as calc_net_pnl
from strategy.exit import initial_stops
from data.universe import get_sector
from config.settings import INITIAL_CAPITAL

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages paper portfolio state: cash, positions, trade log."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self._load_state()

    def _load_state(self):
        self.open_positions: List[Position] = repo.load_open_positions()
        snapshots = repo.load_snapshots()
        if snapshots:
            latest = snapshots[-1]
            self.cash = latest.cash
            self.peak_value = max(s.total_value for s in snapshots)
        else:
            self.cash = self.initial_capital
            self.peak_value = self.initial_capital
        self.new_trades_today = 0

    @property
    def held_symbols(self) -> set:
        return {pos.symbol for pos in self.open_positions}

    def portfolio_value(self, prices: dict) -> float:
        invested = portfolio_invested_value(self.open_positions, prices)
        return self.cash + invested

    def process_signals(self, today: date, signals: List[Signal], prices: dict):
        """
        Process today's signals:
        1. Execute SELL signals first (free up cash)
        2. Execute BUY signals (up to daily limit)
        3. Save snapshot
        """
        self.new_trades_today = 0
        prev_value = self.portfolio_value(prices)

        # ── SELLs ──────────────────────────────────────────────────────
        sell_signals = [s for s in signals if s.action == "SELL"]
        for sig in sell_signals:
            self._execute_sell(today, sig, prices)

        # ── BUYs ───────────────────────────────────────────────────────
        buy_signals = [s for s in signals if s.action == "BUY"]
        portfolio_val = self.portfolio_value(prices)
        current_peak = max(self.peak_value, portfolio_val)

        for sig in buy_signals:
            allowed, reason = can_open_new_trades(
                self.new_trades_today, self.open_positions,
                portfolio_val, current_peak
            )
            if not allowed:
                logger.info(f"[Portfolio] Buy halted: {reason}")
                break
            self._execute_buy(today, sig, portfolio_val, prices)

        # ── SNAPSHOT ───────────────────────────────────────────────────
        final_value = self.portfolio_value(prices)
        invested = portfolio_invested_value(self.open_positions, prices)
        cumulative_pnl = final_value - self.initial_capital
        self.peak_value = max(self.peak_value, final_value)

        snap = PortfolioSnapshot(
            date=today,
            cash=round(self.cash, 2),
            invested=round(invested, 2),
            total_value=round(final_value, 2),
            open_positions=len(self.open_positions),
            daily_pnl=round(final_value - prev_value, 2),
            cumulative_pnl=round(cumulative_pnl, 2),
        )
        repo.save_snapshot(snap)
        logger.info(
            f"[Portfolio] {today}: cash=₹{self.cash:.0f}, "
            f"invested=₹{invested:.0f}, total=₹{final_value:.0f}, "
            f"P&L=₹{cumulative_pnl:.0f}"
        )

    def _execute_buy(self, today: date, sig: Signal, portfolio_val: float, prices: dict):
        price = sig.price
        stops = initial_stops(price)
        shares = calculate_shares(
            portfolio_value=portfolio_val,
            entry_price=price,
            stop_loss_price=stops["stop_loss"],
            available_cash=self.cash,
        )
        if shares <= 0:
            logger.info(f"[Portfolio] Skip BUY {sig.symbol}: insufficient cash/sizing")
            return

        trade_value = position_value(shares, price)
        allowed, reason = can_open_position(
            sig.symbol, trade_value, portfolio_val, self.open_positions, prices
        )
        if not allowed:
            logger.info(f"[Portfolio] Skip BUY {sig.symbol}: {reason}")
            return

        from charges.calculator import buy_charges
        charges = buy_charges(trade_value)
        total_cost = trade_value + charges.total
        if total_cost > self.cash:
            logger.info(f"[Portfolio] Skip BUY {sig.symbol}: not enough cash "
                        f"(need ₹{total_cost:.0f}, have ₹{self.cash:.0f})")
            return

        sector = get_sector(sig.symbol)
        pos = Position(
            symbol=sig.symbol, sector=sector,
            entry_date=today, entry_price=price, shares=shares,
            stop_loss=stops["stop_loss"], take_profit=stops["take_profit"],
            trailing_stop=stops["trailing_stop"], peak_price=stops["peak_price"],
        )
        self.cash -= total_cost
        self.open_positions.append(pos)
        repo.save_position(pos)
        self.new_trades_today += 1
        logger.info(
            f"[Portfolio] BUY {shares}×{sig.symbol} @ ₹{price:.2f} "
            f"(cost=₹{total_cost:.0f}, charges=₹{charges.total:.2f})"
        )

    def _execute_sell(self, today: date, sig: Signal, prices: dict):
        price = sig.price
        pos = next((p for p in self.open_positions if p.symbol == sig.symbol), None)
        if pos is None:
            return

        result = calc_net_pnl(pos.entry_price, price, pos.shares)
        trade = Trade(
            symbol=pos.symbol, sector=pos.sector,
            entry_date=pos.entry_date, exit_date=today,
            entry_price=pos.entry_price, exit_price=price,
            shares=pos.shares,
            gross_pnl=result["gross_pnl"],
            charges=result["total_charges"],
            net_pnl=result["net_pnl"],
            exit_reason=sig.reason,
            hold_days=(today - pos.entry_date).days,
        )
        self.cash += result["sell_value"] - result["sell_charges"]["total"]
        self.open_positions = [p for p in self.open_positions if p.symbol != sig.symbol]
        repo.close_position(pos.symbol)
        repo.save_trade(trade)
        logger.info(
            f"[Portfolio] SELL {pos.shares}×{sig.symbol} @ ₹{price:.2f} "
            f"net P&L=₹{result['net_pnl']:.2f} ({result['net_pct']:+.2f}%)"
        )
