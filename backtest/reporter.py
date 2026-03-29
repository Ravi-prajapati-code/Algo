"""
Backtest reporter — generates human-readable summary and CSV trade log.
"""

import os
import csv
from datetime import date

from backtest.engine import BacktestResult
from backtest.metrics import calculate_metrics
from config.settings import INITIAL_CAPITAL, OUTPUTS_DIR


def print_summary(metrics: dict):
    """Print a formatted backtest summary to stdout."""
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Starting Capital :  ₹{metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value      :  ₹{metrics['final_value']:>12,.2f}")
    print(f"  Total Return     :  {metrics['total_return_pct']:>+.2f}%")
    print(f"  CAGR             :  {metrics['cagr_pct']:>+.2f}%  {'✅' if metrics['passes_15pct_target'] else '❌'}")
    print(f"  Sharpe Ratio     :  {metrics['sharpe_ratio']:>.2f}  {'✅' if metrics['passes_sharpe'] else '❌'}")
    print(f"  Max Drawdown     :  {metrics['max_drawdown_pct']:>.2f}%  {'✅' if metrics['passes_drawdown'] else '❌'}")
    print("-" * 55)
    print(f"  Total Trades     :  {metrics['total_trades']}")
    print(f"  Win Rate         :  {metrics['win_rate_pct']:.1f}%  {'✅' if metrics['passes_win_rate'] else '❌'}")
    print(f"  Avg Win          :  ₹{metrics['avg_win_inr']:>+,.2f}")
    print(f"  Avg Loss         :  ₹{metrics['avg_loss_inr']:>+,.2f}")
    print(f"  Profit Factor    :  {metrics['profit_factor']:.2f}  {'✅' if metrics['passes_profit_factor'] else '❌'}")
    print(f"  Avg Hold Days    :  {metrics['avg_hold_days']:.1f}")
    print(f"  Total Charges    :  ₹{metrics['total_charges_inr']:>,.2f} ({metrics['charges_drag_pct']:.2f}% drag)")
    print("=" * 55)
    verdict = "PASS — Ready for paper trading" if metrics["all_criteria_met"] else "FAIL — Needs tuning"
    print(f"  Overall: {verdict}")
    print("=" * 55 + "\n")


def save_trade_log(result: BacktestResult, filepath: str = None):
    """Save trade-by-trade CSV with P&L and charges."""
    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_trades.csv")

    fieldnames = [
        "symbol", "sector", "entry_date", "exit_date", "hold_days",
        "entry_price", "exit_price", "shares",
        "gross_pnl", "charges", "net_pnl", "exit_reason"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            writer.writerow({
                "symbol":      t.symbol,
                "sector":      t.sector,
                "entry_date":  str(t.entry_date),
                "exit_date":   str(t.exit_date),
                "hold_days":   t.hold_days,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "shares":      t.shares,
                "gross_pnl":   t.gross_pnl,
                "charges":     t.charges,
                "net_pnl":     t.net_pnl,
                "exit_reason": t.exit_reason,
            })
    print(f"[Reporter] Trade log saved: {filepath}")


def save_equity_curve(result: BacktestResult, filepath: str = None):
    """Save equity curve to CSV."""
    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_equity.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "portfolio_value", "cash"])
        for d in sorted(result.equity_curve.keys()):
            writer.writerow([str(d), result.equity_curve[d], result.cash_curve.get(d, "")])
    print(f"[Reporter] Equity curve saved: {filepath}")


def run_and_report(result: BacktestResult, initial_capital: float = INITIAL_CAPITAL):
    """Compute metrics, print summary, save CSVs."""
    metrics = calculate_metrics(result, initial_capital)
    print_summary(metrics)
    save_trade_log(result)
    save_equity_curve(result)
    return metrics
