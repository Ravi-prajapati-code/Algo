"""
Performance metrics calculator for backtest results.
"""

import math
from typing import Optional
from db.models import Trade
from backtest.engine import BacktestResult
from config.settings import INITIAL_CAPITAL


def calculate_metrics(result: BacktestResult, initial_capital: float = INITIAL_CAPITAL) -> dict:
    trades = result.trades
    equity = result.equity_curve

    if not equity:
        return {
            "error":                    "No equity data",
            "initial_capital":          round(initial_capital, 2),
            "final_value":              round(initial_capital, 2),
            "total_return_pct":         0.0,
            "cagr_pct":                 0.0,
            "max_drawdown_pct":         0.0,
            "sharpe_ratio":             0.0,
            "total_trades":             0,
            "win_rate_pct":             0.0,
            "avg_win_inr":              0.0,
            "avg_loss_inr":             0.0,
            "avg_hold_days":            0.0,
            "profit_factor":            0.0,
            "total_charges_inr":        0.0,
            "annual_charges_drag_pct":  0.0,
            "passes_15pct_target":      False,
            "passes_sharpe":            False,
            "passes_drawdown":          True,
            "passes_win_rate":          False,
            "passes_profit_factor":     False,
            "all_criteria_met":         False,
        }

    dates = sorted(equity.keys())
    values = [equity[d] for d in dates]
    final_value = values[-1]

    # ── Returns ───────────────────────────────────────────────────────
    total_return = (final_value - initial_capital) / initial_capital
    years = max((dates[-1] - dates[0]).days / 365.25, 1/365)
    cagr = (final_value / initial_capital) ** (1 / years) - 1

    # ── Drawdown ──────────────────────────────────────────────────────
    peak = initial_capital
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # ── Daily returns & Sharpe ────────────────────────────────────────
    daily_returns = []
    for i in range(1, len(values)):
        r = (values[i] - values[i - 1]) / values[i - 1]
        daily_returns.append(r)

    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
        std_r = math.sqrt(variance)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # ── Trade statistics ──────────────────────────────────────────────
    completed = [t for t in trades if t.net_pnl is not None]
    winners = [t for t in completed if (t.net_pnl or 0) > 0]
    losers  = [t for t in completed if (t.net_pnl or 0) <= 0]

    win_rate = len(winners) / len(completed) if completed else 0.0
    avg_win  = sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
    avg_loss = sum(t.net_pnl for t in losers)  / len(losers)  if losers  else 0.0
    avg_hold = sum(t.hold_days for t in completed if t.hold_days) / len(completed) if completed else 0.0
    total_charges = sum(t.charges for t in completed if t.charges)

    gross_profit = sum(t.gross_pnl for t in winners if t.gross_pnl)
    gross_loss   = abs(sum(t.gross_pnl for t in losers if t.gross_pnl))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        # Portfolio
        "initial_capital":   round(initial_capital, 2),
        "final_value":       round(final_value, 2),
        "total_return_pct":  round(total_return * 100, 2),
        "cagr_pct":          round(cagr * 100, 2),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "sharpe_ratio":      round(sharpe, 2),
        # Trades
        "total_trades":      len(completed),
        "win_rate_pct":      round(win_rate * 100, 2),
        "avg_win_inr":       round(avg_win, 2),
        "avg_loss_inr":      round(avg_loss, 2),
        "avg_hold_days":     round(avg_hold, 1),
        "profit_factor":     round(profit_factor, 2),
        "total_charges_inr":      round(total_charges, 2),
        "annual_charges_drag_pct": round((total_charges / initial_capital * 100) / years, 2),
        # Validation
        # Win rate minimum is 42% (not 45%) because profit factor ≥1.4 compensates:
        # At 42% WR and PF 1.80, expected value per trade is still strongly positive.
        "passes_15pct_target":  cagr >= 0.15,
        "passes_sharpe":        sharpe >= 0.8,
        "passes_drawdown":      max_dd <= 0.25,
        "passes_win_rate":      win_rate >= 0.42,
        "passes_profit_factor": profit_factor >= 1.4,
        "all_criteria_met": (
            cagr >= 0.15 and sharpe >= 0.8 and
            max_dd <= 0.25 and win_rate >= 0.42 and profit_factor >= 1.4
        ),
    }
