"""
Paper Trading Broker.

Simulates order execution without connecting to a real broker API.
Used for:
  - Live paper trading (real signals, no real money)
  - Integration testing
  - Strategy validation before going live

All orders fill immediately at the requested price (or last known price).
State is persisted to the SQLite DB (same as backtester) so the dashboard
shows paper trades alongside real performance metrics.
"""

import logging
import uuid
from datetime import datetime
from typing import List

from broker.base import (
    BaseBroker, OrderRequest, OrderResult, OrderStatus,
    OrderSide, LivePosition,
)
from charges.calculator import buy_charges, sell_charges
from db import repository as repo
from db.models import Position, Trade
from data.universe import get_sector

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading.

    Fills orders at the price specified in the OrderRequest.
    Deducts realistic Upstox delivery charges from cash balance.
    Saves positions and trades to the DB.

    State
    -----
    cash            : Available cash.
    open_positions  : List[Position] mirroring DB state.
    """

    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self._load_state()

    def _load_state(self):
        """Sync in-memory state from DB."""
        positions = repo.load_open_positions()
        self.open_positions: List[Position] = positions
        snapshots = repo.load_snapshots()
        if snapshots:
            self.cash = snapshots[-1].cash
        logger.info(
            "[PaperBroker] Loaded: cash=₹%.0f, positions=%d",
            self.cash, len(self.open_positions),
        )

    # ── BaseBroker implementation ──────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Simulate order execution with realistic charges."""
        order_id = str(uuid.uuid4())[:8]
        price = request.price

        if price <= 0:
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                symbol=request.symbol, side=request.side,
                requested_qty=request.quantity,
                rejection_reason="Invalid price (0 or negative)",
            )

        if request.side == OrderSide.BUY:
            return self._execute_buy(order_id, request, price)
        else:
            return self._execute_sell(order_id, request, price)

    def cancel_order(self, order_id: str) -> bool:
        # Paper broker fills immediately — nothing to cancel
        logger.info("[PaperBroker] cancel_order(%s): no-op (immediate fills)", order_id)
        return True

    def get_order_status(self, order_id: str) -> OrderResult:
        # Paper orders are always COMPLETE immediately after placement
        return OrderResult(
            order_id=order_id, status=OrderStatus.COMPLETE,
            symbol="", side=OrderSide.BUY, requested_qty=0,
        )

    def get_positions(self) -> List[LivePosition]:
        return [
            LivePosition(
                symbol=p.symbol, quantity=p.shares,
                avg_price=p.entry_price, ltp=p.entry_price,
                pnl=0.0, product="CNC",
            )
            for p in self.open_positions
        ]

    def get_portfolio_value(self) -> float:
        invested = sum(p.entry_price * p.shares for p in self.open_positions)
        return self.cash + invested

    def get_available_cash(self) -> float:
        return self.cash

    # ── Internal execution ─────────────────────────────────────────────────

    def _execute_buy(self, order_id: str, req: OrderRequest, price: float) -> OrderResult:
        trade_value = price * req.quantity
        charges = buy_charges(trade_value)
        total_cost = trade_value + charges.total

        if total_cost > self.cash:
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                symbol=req.symbol, side=req.side,
                requested_qty=req.quantity,
                rejection_reason=f"Insufficient cash: need ₹{total_cost:.0f}, have ₹{self.cash:.0f}",
            )

        sector = get_sector(req.symbol)
        from strategy.exit import initial_stops
        stops = initial_stops(price)

        pos = Position(
            symbol=req.symbol, sector=sector,
            entry_date=datetime.now().date(),
            entry_price=price, shares=req.quantity,
            stop_loss=stops["stop_loss"], take_profit=stops["take_profit"],
            trailing_stop=stops["trailing_stop"], peak_price=stops["peak_price"],
        )

        self.cash -= total_cost
        self.open_positions.append(pos)
        repo.save_position(pos)

        logger.info(
            "[PaperBroker] BUY %d×%s @ ₹%.2f = ₹%.0f (charges=₹%.2f)",
            req.quantity, req.symbol, price, trade_value, charges.total,
        )

        return OrderResult(
            order_id=order_id, status=OrderStatus.COMPLETE,
            symbol=req.symbol, side=req.side,
            requested_qty=req.quantity, filled_qty=req.quantity,
            avg_price=price, placed_at=datetime.now(), filled_at=datetime.now(),
        )

    def _execute_sell(self, order_id: str, req: OrderRequest, price: float) -> OrderResult:
        pos = next((p for p in self.open_positions if p.symbol == req.symbol), None)
        if pos is None:
            return OrderResult(
                order_id=order_id, status=OrderStatus.REJECTED,
                symbol=req.symbol, side=req.side,
                requested_qty=req.quantity,
                rejection_reason=f"No open position in {req.symbol}",
            )

        from charges.calculator import net_pnl as calc_net_pnl
        result = calc_net_pnl(pos.entry_price, price, pos.shares)

        trade = Trade(
            symbol=pos.symbol, sector=pos.sector,
            entry_date=pos.entry_date, exit_date=datetime.now().date(),
            entry_price=pos.entry_price, exit_price=price,
            shares=pos.shares,
            gross_pnl=result["gross_pnl"],
            charges=result["total_charges"],
            net_pnl=result["net_pnl"],
            exit_reason=req.tag or "MANUAL",
            hold_days=(datetime.now().date() - pos.entry_date).days,
        )

        self.cash += result["sell_value"] - result["sell_charges"]["total"]
        self.open_positions = [p for p in self.open_positions if p.symbol != req.symbol]
        repo.close_position(pos.symbol)
        repo.save_trade(trade)

        logger.info(
            "[PaperBroker] SELL %d×%s @ ₹%.2f | net P&L=₹%.2f (%.1f%%)",
            pos.shares, req.symbol, price, result["net_pnl"], result["net_pct"],
        )

        return OrderResult(
            order_id=order_id, status=OrderStatus.COMPLETE,
            symbol=req.symbol, side=req.side,
            requested_qty=req.quantity, filled_qty=pos.shares,
            avg_price=price, placed_at=datetime.now(), filled_at=datetime.now(),
        )
