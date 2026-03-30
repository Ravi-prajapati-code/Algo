"""
Real-time performance tracker.

Computes and logs rolling metrics during live trading:
  - Daily P&L, cumulative P&L, win rate, profit factor
  - Sharpe ratio (rolling 30-day)
  - Max drawdown (rolling + all-time)
  - Sector P&L breakdown

Also exposes `PerformanceTracker` as a class that can be embedded in the
daily runner and dashboard for live metrics display.

Usage
-----
    tracker = PerformanceTracker(initial_capital=75000)
    tracker.record_day(date.today(), portfolio_value=81234, trades_closed=[t1, t2])
    report = tracker.daily_report()
    log_performance(report)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from monitoring.logger import log_performance

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Metrics snapshot for a single trading day."""
    date:             date
    portfolio_value:  float
    cash:             float
    daily_pnl:        float
    cumulative_pnl:   float
    cumulative_pct:   float
    drawdown_pct:     float
    open_positions:   int
    trades_closed:    int
    wins_today:       int
    losses_today:     int
    win_rate_rolling: float    # Rolling 30-day win rate
    sharpe_rolling:   float    # Rolling 30-day annualised Sharpe
    profit_factor:    float    # All-time profit factor


class PerformanceTracker:
    """
    Tracks portfolio performance in real-time.

    Maintains an in-memory equity curve and rolling stats.
    Persists daily summaries to monitoring/performance.jsonl via log_performance().
    """

    ROLLING_WINDOW = 30   # Days for rolling metrics

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self._equity_curve:  list[tuple[date, float]] = []   # (date, portfolio_value)
        self._daily_returns: list[float] = []                  # day-over-day returns
        self._trades:        list        = []                  # All closed Trade objects
        self._peak_value     = initial_capital

    # ── Public API ─────────────────────────────────────────────────────────

    def record_day(
        self,
        today: date,
        portfolio_value: float,
        cash: float = 0.0,
        open_positions: int = 0,
        trades_closed: Optional[list] = None,
    ) -> DailyStats:
        """
        Record end-of-day state and compute metrics.

        Parameters
        ----------
        today           : Trading date.
        portfolio_value : Total portfolio value (cash + invested).
        cash            : Cash component of portfolio.
        open_positions  : Number of open positions.
        trades_closed   : List of Trade objects closed today.

        Returns
        -------
        DailyStats for the day.
        """
        trades_closed = trades_closed or []
        self._trades.extend(trades_closed)

        # Daily return
        prev_value = self._equity_curve[-1][1] if self._equity_curve else self.initial_capital
        daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        self._daily_returns.append(daily_return)
        self._equity_curve.append((today, portfolio_value))

        # Drawdown
        self._peak_value = max(self._peak_value, portfolio_value)
        drawdown = (self._peak_value - portfolio_value) / self._peak_value if self._peak_value > 0 else 0.0

        # Today's wins/losses
        wins_today   = sum(1 for t in trades_closed if (t.net_pnl or 0) > 0)
        losses_today = sum(1 for t in trades_closed if (t.net_pnl or 0) <= 0)

        # Rolling window metrics
        recent_returns = self._daily_returns[-self.ROLLING_WINDOW:]
        win_rate_rolling = self._rolling_win_rate()
        sharpe_rolling   = self._rolling_sharpe(recent_returns)
        profit_factor    = self._all_time_profit_factor()

        stats = DailyStats(
            date            = today,
            portfolio_value = round(portfolio_value, 2),
            cash            = round(cash, 2),
            daily_pnl       = round((portfolio_value - prev_value), 2),
            cumulative_pnl  = round(portfolio_value - self.initial_capital, 2),
            cumulative_pct  = round((portfolio_value - self.initial_capital) / self.initial_capital * 100, 2),
            drawdown_pct    = round(drawdown * 100, 2),
            open_positions  = open_positions,
            trades_closed   = len(trades_closed),
            wins_today      = wins_today,
            losses_today    = losses_today,
            win_rate_rolling= round(win_rate_rolling * 100, 1),
            sharpe_rolling  = round(sharpe_rolling, 2),
            profit_factor   = round(profit_factor, 2),
        )

        log_performance({
            "date":             str(today),
            "portfolio_value":  stats.portfolio_value,
            "daily_pnl":        stats.daily_pnl,
            "cumulative_pct":   stats.cumulative_pct,
            "drawdown_pct":     stats.drawdown_pct,
            "sharpe_30d":       stats.sharpe_rolling,
            "win_rate_30d":     stats.win_rate_rolling,
            "profit_factor":    stats.profit_factor,
        })

        logger.info(
            "[Performance] %s | ₹%.0f (+%.1f%% cum) | DD=%.1f%% | "
            "Sharpe=%.2f | WR=%.0f%%",
            today, portfolio_value, stats.cumulative_pct,
            stats.drawdown_pct, stats.sharpe_rolling, stats.win_rate_rolling,
        )
        return stats

    def daily_report(self) -> dict:
        """Return a dict suitable for dashboard display or Telegram alert."""
        if not self._equity_curve:
            return {"error": "No data recorded yet"}

        last_date, last_value = self._equity_curve[-1]
        prev_value = self._equity_curve[-2][1] if len(self._equity_curve) > 1 else self.initial_capital

        return {
            "date":             str(last_date),
            "portfolio_value":  round(last_value, 2),
            "daily_pnl":        round(last_value - prev_value, 2),
            "cumulative_pnl":   round(last_value - self.initial_capital, 2),
            "cumulative_pct":   round((last_value - self.initial_capital) / self.initial_capital * 100, 2),
            "peak_value":       round(self._peak_value, 2),
            "drawdown_pct":     round(
                (self._peak_value - last_value) / self._peak_value * 100, 2
            ) if self._peak_value > 0 else 0.0,
            "win_rate_30d":     round(self._rolling_win_rate() * 100, 1),
            "sharpe_30d":       round(self._rolling_sharpe(self._daily_returns[-30:]), 2),
            "profit_factor":    round(self._all_time_profit_factor(), 2),
            "total_trades":     len(self._trades),
        }

    def max_drawdown(self) -> float:
        """All-time maximum drawdown as a fraction."""
        if not self._equity_curve:
            return 0.0
        peak = self.initial_capital
        max_dd = 0.0
        for _, val in self._equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def cagr(self) -> float:
        """Compute CAGR from recorded equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0
        first_date, first_val = self._equity_curve[0]
        last_date, last_val   = self._equity_curve[-1]
        years = (last_date - first_date).days / 365.25
        if years <= 0 or first_val <= 0:
            return 0.0
        return (last_val / first_val) ** (1.0 / years) - 1.0

    # ── Private helpers ────────────────────────────────────────────────────

    def _rolling_win_rate(self) -> float:
        recent = [t for t in self._trades[-self.ROLLING_WINDOW:] if t.net_pnl is not None]
        if not recent:
            return 0.5
        return sum(1 for t in recent if t.net_pnl > 0) / len(recent)

    def _rolling_sharpe(self, returns: list[float]) -> float:
        if len(returns) < 5:
            return 0.0
        import statistics
        mean = statistics.mean(returns)
        std  = statistics.stdev(returns)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    def _all_time_profit_factor(self) -> float:
        gross_profit = sum(t.net_pnl for t in self._trades if t.net_pnl and t.net_pnl > 0)
        gross_loss   = sum(abs(t.net_pnl) for t in self._trades if t.net_pnl and t.net_pnl < 0)
        return round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
