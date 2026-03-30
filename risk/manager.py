"""
Production-grade Risk Management Engine.

Implements per-trade risk, portfolio constraints, drawdown protection,
and volatility-based position sizing in a unified RiskManager class.

Integrates with: portfolio/sizer.py, backtest/engine.py, runner/daily_runner.py
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from config.settings import (
    MAX_RISK_PER_TRADE_PCT,
    MAX_STOCK_ALLOCATION_PCT,
    MAX_SECTOR_ALLOCATION_PCT,
    MAX_OPEN_POSITIONS,
    MAX_NEW_TRADES_PER_DAY,
    INITIAL_CAPITAL,
)
from db.models import Position
from portfolio.allocator import sector_allocation

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a pre-trade risk check."""
    allowed: bool
    reason: str
    risk_score: float = 0.0     # 0–100, higher = more risky (informational)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    shares: int
    trade_value: float
    risk_amount: float          # Capital at risk in ₹
    risk_pct: float             # Risk as fraction of portfolio (e.g. 0.02 = 2%)
    sizing_method: str          # 'atr' | 'fixed_pct' | 'kill_switch' | 'invalid'


class RiskManager:
    """
    Unified risk management engine.

    Responsibilities
    ----------------
    * Per-trade risk limits  — max 1–2 % capital at risk
    * Portfolio exposure caps — stock 20 %, sector 30 % (configurable)
    * Drawdown kill-switch    — halt new buys if DD ≥ 15 %
    * Volatility-based sizing — ATR method (primary) / fixed-% (fallback)
    * Daily trade-count limit — max 5 new trades per day
    * Confidence-scaling      — shrink size when ML model is uncertain

    Usage
    -----
    >>> rm = RiskManager(portfolio_value=80_000, peak_value=85_000)
    >>> size = rm.size_position(entry=500, stop_loss_price=470, atr=12)
    >>> check = rm.check_pre_trade("RELIANCE.NS", size.trade_value, positions, prices, 0, "Energy")
    """

    # ── Configurable class-level limits (override via config if needed) ────
    MAX_RISK_PER_TRADE  = MAX_RISK_PER_TRADE_PCT        # 0.02
    MAX_STOCK_PCT       = MAX_STOCK_ALLOCATION_PCT       # 0.20
    MAX_SECTOR_PCT      = MAX_SECTOR_ALLOCATION_PCT      # 0.30
    MAX_POSITIONS       = MAX_OPEN_POSITIONS             # 15
    MAX_DAILY_TRADES    = MAX_NEW_TRADES_PER_DAY         # 5
    DRAWDOWN_HALT_PCT   = 0.15   # 15 % from peak → stop all new buys
    DRAWDOWN_REDUCE_PCT = 0.10   # 10 % from peak → cut position size by 50 %
    MIN_CASH_RESERVE    = 0.05   # Always keep 5 % of portfolio as cash buffer
    ATR_STOP_MULTIPLIER = 1.5    # Stop distance = ATR × this multiplier

    def __init__(
        self,
        portfolio_value: float,
        peak_value: float,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.portfolio_value = portfolio_value
        self.peak_value = max(peak_value, portfolio_value)
        self.initial_capital = initial_capital

    # ── Drawdown metrics ───────────────────────────────────────────────────

    @property
    def current_drawdown(self) -> float:
        """Drawdown from peak as a fraction (0.15 = 15 %)."""
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.portfolio_value) / self.peak_value)

    @property
    def is_kill_switch_active(self) -> bool:
        """True → drawdown threshold breached; block all new entries."""
        active = self.current_drawdown >= self.DRAWDOWN_HALT_PCT
        if active:
            logger.warning(
                "[RiskManager] KILL SWITCH ACTIVE — drawdown %.1f%% ≥ halt threshold %.0f%%",
                self.current_drawdown * 100,
                self.DRAWDOWN_HALT_PCT * 100,
            )
        return active

    @property
    def size_reduction_factor(self) -> float:
        """
        Multiplicative factor applied to position size.
        - DD  < 10 %  → 1.0  (full size)
        - DD 10–15 %  → 0.5  (half size)
        - DD ≥ 15 %   → 0.0  (kill switch — no new positions)
        """
        dd = self.current_drawdown
        if dd >= self.DRAWDOWN_HALT_PCT:
            return 0.0
        if dd >= self.DRAWDOWN_REDUCE_PCT:
            return 0.5
        return 1.0

    # ── Position sizing ────────────────────────────────────────────────────

    def size_position(
        self,
        entry_price: float,
        stop_loss_price: float,
        available_cash: float,
        atr: Optional[float] = None,
        confidence: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate how many shares to buy.

        Sizing algorithm
        ----------------
        1. Apply drawdown kill-switch / reduction factor.
        2. Scale base risk by ML confidence (clamped to [0.5, 1.0]).
        3. If ATR is available → use ATR stop method:
               stop_distance = atr × ATR_STOP_MULTIPLIER
           else → use fixed-% stop:
               stop_distance = entry_price − stop_loss_price
        4. shares = floor(risk_budget / stop_distance)
        5. Cap by: max stock allocation, available cash (minus reserve).

        Parameters
        ----------
        entry_price      : Execution price per share.
        stop_loss_price  : Hard stop-loss level (fallback sizing).
        available_cash   : Cash currently available for deployment.
        atr              : 14-day ATR value (enables ATR method).
        confidence       : ML win-probability score 0–1 (scales size).
        """
        if entry_price <= 0:
            return PositionSizeResult(0, 0.0, 0.0, 0.0, "invalid")

        reduction = self.size_reduction_factor
        if reduction == 0.0:
            return PositionSizeResult(0, 0.0, 0.0, 0.0, "kill_switch")

        # Confidence scaling: clamp to [0.5, 1.0] so we never bet less than half
        confidence_factor = max(0.5, min(1.0, float(confidence)))
        effective_risk_pct = self.MAX_RISK_PER_TRADE * reduction * confidence_factor
        risk_budget = self.portfolio_value * effective_risk_pct

        # Determine stop distance
        if atr and atr > 0:
            stop_distance = atr * self.ATR_STOP_MULTIPLIER
            sizing_method = "atr"
        else:
            stop_distance = entry_price - stop_loss_price
            sizing_method = "fixed_pct"

        if stop_distance <= 0:
            return PositionSizeResult(0, 0.0, 0.0, 0.0, "invalid_stop")

        # Risk-based shares
        shares_by_risk = math.floor(risk_budget / stop_distance)

        # Cap by max stock allocation
        max_alloc_value = self.portfolio_value * self.MAX_STOCK_PCT
        shares_by_allocation = math.floor(max_alloc_value / entry_price)

        # Cap by available cash (keep cash reserve)
        spendable = available_cash * (1.0 - self.MIN_CASH_RESERVE)
        shares_by_cash = math.floor(spendable / entry_price)

        shares = max(0, min(shares_by_risk, shares_by_allocation, shares_by_cash))
        trade_value = round(shares * entry_price, 2)
        actual_risk = shares * stop_distance

        logger.debug(
            "[RiskManager] Sizing %s: method=%s shares=%d value=₹%.0f risk=₹%.0f (%.2f%%)",
            sizing_method, sizing_method, shares, trade_value,
            actual_risk, actual_risk / self.portfolio_value * 100 if self.portfolio_value else 0,
        )

        return PositionSizeResult(
            shares=shares,
            trade_value=trade_value,
            risk_amount=round(actual_risk, 2),
            risk_pct=round(actual_risk / self.portfolio_value, 4) if self.portfolio_value > 0 else 0.0,
            sizing_method=sizing_method,
        )

    # ── Pre-trade gate ─────────────────────────────────────────────────────

    def check_pre_trade(
        self,
        symbol: str,
        trade_value: float,
        open_positions: List[Position],
        prices: dict,
        new_trades_today: int,
        sector: str,
    ) -> RiskCheck:
        """
        Run all pre-trade risk checks.

        Checks (in order)
        -----------------
        1. Drawdown kill-switch
        2. Daily trade limit
        3. Max open positions
        4. Duplicate symbol
        5. Stock allocation cap (per-stock ≤ MAX_STOCK_PCT)
        6. Sector allocation cap (sector ≤ MAX_SECTOR_PCT)

        Returns
        -------
        RiskCheck with allowed=True/False, reason, and a 0–100 risk score.
        """
        # 1. Kill switch
        if self.is_kill_switch_active:
            return RiskCheck(
                False,
                f"Kill switch: drawdown {self.current_drawdown:.1%} ≥ {self.DRAWDOWN_HALT_PCT:.0%}",
            )

        # 2. Daily trade limit
        if new_trades_today >= self.MAX_DAILY_TRADES:
            return RiskCheck(
                False,
                f"Daily limit reached: {new_trades_today}/{self.MAX_DAILY_TRADES} trades",
            )

        # 3. Max open positions
        if len(open_positions) >= self.MAX_POSITIONS:
            return RiskCheck(
                False,
                f"Position limit reached: {len(open_positions)}/{self.MAX_POSITIONS}",
            )

        # 4. Duplicate
        if any(p.symbol == symbol for p in open_positions):
            return RiskCheck(False, f"{symbol} already in portfolio")

        # 5. Stock allocation cap
        if self.portfolio_value > 0:
            stock_pct = trade_value / self.portfolio_value
        else:
            stock_pct = 0.0
        if stock_pct > self.MAX_STOCK_PCT:
            return RiskCheck(
                False,
                f"{symbol}: {stock_pct:.1%} of portfolio > stock cap {self.MAX_STOCK_PCT:.0%}",
            )

        # 6. Sector cap
        sector_alloc = sector_allocation(open_positions, prices, self.portfolio_value)
        current_sector = sector_alloc.get(sector, 0.0)
        projected_sector = current_sector + stock_pct
        if projected_sector > self.MAX_SECTOR_PCT:
            return RiskCheck(
                False,
                f"{sector} sector: projected {projected_sector:.1%} > cap {self.MAX_SECTOR_PCT:.0%}",
            )

        risk_score = self._risk_score(stock_pct, current_sector)
        return RiskCheck(True, "OK", risk_score=risk_score)

    # ── Portfolio health snapshot ──────────────────────────────────────────

    def health_report(self) -> dict:
        """Return a dictionary of key risk metrics for monitoring/logging."""
        return {
            "portfolio_value":          round(self.portfolio_value, 2),
            "peak_value":               round(self.peak_value, 2),
            "drawdown_pct":             round(self.current_drawdown * 100, 2),
            "drawdown_halt_pct":        self.DRAWDOWN_HALT_PCT * 100,
            "kill_switch_active":       self.is_kill_switch_active,
            "size_reduction_factor":    self.size_reduction_factor,
            "return_from_start_pct":    round(
                (self.portfolio_value - self.initial_capital) / self.initial_capital * 100, 2
            ) if self.initial_capital else 0.0,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _risk_score(self, stock_pct: float, sector_pct: float) -> float:
        """Compute a 0–100 risk score (informational, not a gate)."""
        stock_component  = (stock_pct  / self.MAX_STOCK_PCT)  * 50
        sector_component = (sector_pct / self.MAX_SECTOR_PCT) * 30
        dd_component     = (self.current_drawdown / max(self.DRAWDOWN_HALT_PCT, 1e-9)) * 20
        return round(min(100.0, stock_component + sector_component + dd_component), 1)
