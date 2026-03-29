"""Dataclasses for core domain objects."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, Any


@dataclass
class Position:
    symbol: str
    sector: str
    entry_date: date
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    peak_price: float
    status: str = "OPEN"
    id: Optional[int] = None

    @property
    def current_value(self) -> float:
        return self.entry_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares

    def unrealized_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class Trade:
    symbol: str
    sector: str
    entry_date: date
    entry_price: float
    shares: int
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    charges: Optional[float] = None
    net_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    hold_days: Optional[int] = None
    id: Optional[int] = None


@dataclass
class Signal:
    date: date
    symbol: str
    action: str           # BUY | SELL | HOLD
    score: float = 0.0
    price: float = 0.0
    reason: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None


@dataclass
class PortfolioSnapshot:
    date: date
    cash: float
    invested: float
    total_value: float
    open_positions: int
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    id: Optional[int] = None
