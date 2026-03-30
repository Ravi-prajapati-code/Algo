"""
Position sizer — calculates share count based on risk and signal quality.

Sizing layers (applied in order, each can only reduce shares):
  1. Risk budget    — portfolio × MAX_RISK_PCT × drawdown_factor
  2. ATR stop dist  — volatility-aware stop distance
  3. Score factor   — signal conviction (STRONG=100%, GOOD=80%, MODERATE=60%)
  4. Quality factor — stock tier (Tier1=100%, Tier2=75%, Tier3=50%)
  5. Hard caps      — max stock allocation (20%), available cash

This multi-factor sizing ensures that:
  - High-conviction trades in elite stocks get full allocation
  - Speculative / moderate signals get smaller positions automatically
  - We never risk more than MAX_RISK_PCT on any single trade
"""

import math
from typing import Optional

from config.settings import (
    MAX_RISK_PER_TRADE_PCT,
    MAX_STOCK_ALLOCATION_PCT,
    ATR_STOP_MULTIPLIER,
)


def calculate_shares(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    available_cash: float,
    atr: Optional[float] = None,
    drawdown_factor: float = 1.0,
    score_factor: float = 1.0,
    quality_factor: float = 1.0,
) -> int:
    """
    Calculate how many shares to buy.

    Parameters
    ----------
    portfolio_value : Total portfolio value (cash + invested).
    entry_price     : Execution price per share.
    stop_loss_price : Hard stop-loss level (used in fixed-% fallback).
    available_cash  : Cash available for this trade.
    atr             : 14-day ATR — enables volatility-aware stop sizing.
    drawdown_factor : 0.5 in moderate drawdown, 1.0 normal (from RiskManager).
    score_factor    : Signal conviction factor from score_to_size_factor() [0-1].
    quality_factor  : Stock tier multiplier from tier_size_factor() [0.5-1.0].

    Returns 0 if no position should be taken.
    """
    if entry_price <= 0:
        return 0

    # ── Stop distance ──────────────────────────────────────────────────────
    if atr and atr > 0:
        stop_distance = atr * ATR_STOP_MULTIPLIER
    else:
        stop_distance = entry_price - stop_loss_price

    if stop_distance <= 0:
        return 0

    # ── Risk budget (all factors combined) ────────────────────────────────
    combined_factor = (
        max(0.0, min(1.0, drawdown_factor))
        * max(0.0, min(1.0, score_factor))
        * max(0.0, min(1.0, quality_factor))
    )
    risk_budget = portfolio_value * MAX_RISK_PER_TRADE_PCT * combined_factor

    # ── Risk-based share count ────────────────────────────────────────────
    shares_by_risk = math.floor(risk_budget / stop_distance)

    # ── Cap by max stock allocation ───────────────────────────────────────
    max_alloc_value  = portfolio_value * MAX_STOCK_ALLOCATION_PCT
    shares_by_alloc  = math.floor(max_alloc_value / entry_price)

    # ── Cap by available cash (keep 5 % reserve) ─────────────────────────
    shares_by_cash   = math.floor(available_cash * 0.95 / entry_price)

    shares = min(shares_by_risk, shares_by_alloc, shares_by_cash)
    return max(0, shares)


def position_value(shares: int, price: float) -> float:
    return round(shares * price, 2)
