"""
Dynamic Portfolio Allocator.

Upgrades the fixed 2%-risk / 20%-stock-cap model with:
  1. Confidence-weighted allocation — higher ML confidence → larger position
  2. Regime-aware sizing          — scale down in sideways/volatile markets
  3. Sector diversification guard — enforce sector caps with soft warnings
  4. Kelly-fraction position sizing — optional; uses win rate + avg win/loss

PortfolioAllocator is a higher-level wrapper that combines:
  - RiskManager (pre-trade gates + kill switch)
  - Strategy regime factor
  - ML confidence
  - Kelly criterion (optional)

Usage
-----
>>> allocator = PortfolioAllocator(portfolio_value=80000, peak_value=82000)
>>> allocation = allocator.compute_allocation(
...     symbol="RELIANCE.NS",
...     entry_price=2500,
...     stop_loss=2350,
...     atr=45,
...     available_cash=15000,
...     open_positions=positions,
...     prices=prices,
...     sector="Energy",
...     regime="BULL_TREND",
...     ml_confidence=0.72,
...     new_trades_today=2,
... )
>>> if allocation.approved:
...     buy(allocation.shares, allocation.trade_value)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import INITIAL_CAPITAL
from db.models import Position
from risk.manager import RiskManager, PositionSizeResult
from strategy.regime import regime_position_factor, Regime

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Complete result of a portfolio allocation decision."""
    approved:       bool
    shares:         int
    trade_value:    float
    risk_amount:    float
    risk_pct:       float
    sizing_method:  str
    regime_factor:  float
    ml_confidence:  float
    risk_score:     float
    reject_reason:  str = ""
    notes:          list = field(default_factory=list)


class PortfolioAllocator:
    """
    Determines how much capital to allocate to each trade.

    Layers
    ------
    Layer 1 — RiskManager pre-trade gate (hard limits)
    Layer 2 — ATR-based position sizing
    Layer 3 — Regime factor (scale by market environment)
    Layer 4 — ML confidence factor (scale by model conviction)
    Layer 5 — Kelly fraction (optional; nudge size up/down)
    Layer 6 — Final cap checks (cash, stock %, sector %)
    """

    # Kelly is applied as a fractional adjustment: size × kelly_fraction
    # Full Kelly is notoriously aggressive; use 0.25× (quarter-Kelly)
    KELLY_FRACTION = 0.25
    USE_KELLY      = False   # Disabled by default; enable when WR/PF is stable

    def __init__(
        self,
        portfolio_value: float,
        peak_value: float,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.portfolio_value = portfolio_value
        self.peak_value = peak_value
        self.initial_capital = initial_capital
        self._risk_manager = RiskManager(portfolio_value, peak_value, initial_capital)

    def compute_allocation(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        available_cash: float,
        open_positions: List[Position],
        prices: dict,
        sector: str,
        regime: Regime = "BULL_TREND",
        atr: float = 0.0,
        ml_confidence: float = 1.0,
        new_trades_today: int = 0,
        win_rate: float = 0.5,      # For Kelly calculation
        profit_factor: float = 1.5, # For Kelly calculation
    ) -> AllocationResult:
        """
        Compute full allocation for a BUY candidate.

        Returns AllocationResult with approved=True/False and position details.
        """
        notes = []

        # ── Layer 1: Pre-trade risk gate ──────────────────────────────
        risk_check = self._risk_manager.check_pre_trade(
            symbol=symbol,
            trade_value=entry_price * 1,   # Placeholder; actual size computed below
            open_positions=open_positions,
            prices=prices,
            new_trades_today=new_trades_today,
            sector=sector,
        )
        if not risk_check.allowed:
            return AllocationResult(
                approved=False, shares=0, trade_value=0.0,
                risk_amount=0.0, risk_pct=0.0, sizing_method="blocked",
                regime_factor=0.0, ml_confidence=ml_confidence,
                risk_score=risk_check.risk_score,
                reject_reason=risk_check.reason,
            )

        # ── Layer 2: ATR-based position sizing ────────────────────────
        size: PositionSizeResult = self._risk_manager.size_position(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            available_cash=available_cash,
            atr=atr,
            confidence=ml_confidence,
        )
        if size.shares <= 0:
            return AllocationResult(
                approved=False, shares=0, trade_value=0.0,
                risk_amount=0.0, risk_pct=0.0, sizing_method=size.sizing_method,
                regime_factor=1.0, ml_confidence=ml_confidence,
                risk_score=risk_check.risk_score,
                reject_reason=f"Zero shares from {size.sizing_method} sizing",
            )

        # ── Layer 3: Regime factor ────────────────────────────────────
        r_factor = regime_position_factor(regime)
        if r_factor == 0.0:
            return AllocationResult(
                approved=False, shares=0, trade_value=0.0,
                risk_amount=0.0, risk_pct=0.0, sizing_method="regime_blocked",
                regime_factor=r_factor, ml_confidence=ml_confidence,
                risk_score=risk_check.risk_score,
                reject_reason=f"Regime {regime} blocks new positions",
            )

        adjusted_shares = max(1, math.floor(size.shares * r_factor))
        if r_factor < 1.0:
            notes.append(f"Regime {regime}: size reduced to {r_factor:.0%}")

        # ── Layer 4: Kelly fraction (optional) ───────────────────────
        if self.USE_KELLY:
            kelly = self._kelly_fraction(win_rate, profit_factor)
            kelly_shares = max(1, math.floor(adjusted_shares * kelly))
            if kelly_shares != adjusted_shares:
                notes.append(f"Kelly {kelly:.2f}×: {adjusted_shares}→{kelly_shares} shares")
            adjusted_shares = kelly_shares

        # ── Layer 5: Final cap validation ─────────────────────────────
        trade_value = round(adjusted_shares * entry_price, 2)

        # Re-run risk check with actual trade value
        final_check = self._risk_manager.check_pre_trade(
            symbol=symbol,
            trade_value=trade_value,
            open_positions=open_positions,
            prices=prices,
            new_trades_today=new_trades_today,
            sector=sector,
        )
        if not final_check.allowed:
            # Try with a smaller allocation (halve until it fits or give up)
            for divisor in (2, 4):
                adjusted_shares = max(0, math.floor(adjusted_shares / divisor))
                if adjusted_shares == 0:
                    break
                trade_value = round(adjusted_shares * entry_price, 2)
                final_check = self._risk_manager.check_pre_trade(
                    symbol=symbol, trade_value=trade_value,
                    open_positions=open_positions, prices=prices,
                    new_trades_today=new_trades_today, sector=sector,
                )
                if final_check.allowed:
                    notes.append(f"Size halved ÷{divisor} to fit caps")
                    break

            if not final_check.allowed:
                return AllocationResult(
                    approved=False, shares=0, trade_value=0.0,
                    risk_amount=0.0, risk_pct=0.0, sizing_method="cap_exceeded",
                    regime_factor=r_factor, ml_confidence=ml_confidence,
                    risk_score=final_check.risk_score,
                    reject_reason=final_check.reason,
                )

        # Cash check
        if trade_value > available_cash * 0.95:
            return AllocationResult(
                approved=False, shares=0, trade_value=0.0,
                risk_amount=0.0, risk_pct=0.0, sizing_method="insufficient_cash",
                regime_factor=r_factor, ml_confidence=ml_confidence,
                risk_score=final_check.risk_score,
                reject_reason=f"Need ₹{trade_value:.0f}, available ₹{available_cash:.0f}",
            )

        stop_distance = (entry_price - stop_loss_price) if stop_loss_price < entry_price else (atr * 1.5 if atr else entry_price * 0.06)
        actual_risk = adjusted_shares * max(stop_distance, 0)

        logger.info(
            "[Allocator] APPROVED %s: %d shares @ ₹%.2f = ₹%.0f | "
            "risk=₹%.0f (%.1f%%) | regime=%s | ML=%.0f%%",
            symbol, adjusted_shares, entry_price, trade_value,
            actual_risk, actual_risk / self.portfolio_value * 100,
            regime, ml_confidence * 100,
        )

        return AllocationResult(
            approved=True,
            shares=adjusted_shares,
            trade_value=trade_value,
            risk_amount=round(actual_risk, 2),
            risk_pct=round(actual_risk / self.portfolio_value, 4) if self.portfolio_value else 0,
            sizing_method=size.sizing_method,
            regime_factor=r_factor,
            ml_confidence=ml_confidence,
            risk_score=final_check.risk_score,
            notes=notes,
        )

    @staticmethod
    def _kelly_fraction(win_rate: float, profit_factor: float) -> float:
        """
        Compute quarter-Kelly fraction.

        Kelly formula: f* = (p × b - q) / b
          p = win rate
          q = 1 - p (loss rate)
          b = profit_factor (avg_win / avg_loss)

        Clamped to [0.25, 1.5] to prevent aggressive over-betting.
        Quarter-Kelly = full_kelly × KELLY_FRACTION.
        """
        p = max(0.01, min(0.99, win_rate))
        q = 1.0 - p
        b = max(0.01, profit_factor)
        full_kelly = (p * b - q) / b
        quarter_kelly = full_kelly * PortfolioAllocator.KELLY_FRACTION
        return max(0.25, min(1.5, quarter_kelly))
