"""
Backtest reporter — generates human-readable summary and CSV logs.

Outputs:
  backtest_trades.csv    — Detailed trade log (every completed trade)
  backtest_rejected.csv  — Rejected trade log (stocks scanned but not traded)
  backtest_decisions.csv — Per-stock decision log (RS, Signal, Rank, Selected)
  backtest_daily_scan.csv— Daily scan summary (scanned/passed/selected counts)
  backtest_equity.csv    — Equity curve

Trade log fields:
  symbol, sector, entry_date, exit_date, hold_days,
  entry_price, exit_price, shares,
  position_size_inr, position_size_pct,
  stop_loss, pnl_pct,
  gross_pnl, charges, net_pnl, exit_reason
"""

import os
import csv
from datetime import date

from backtest.engine import BacktestResult
from backtest.metrics import calculate_metrics
from config.settings import INITIAL_CAPITAL, OUTPUTS_DIR


def print_summary(metrics: dict):
    """Print a formatted backtest summary to stdout."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Starting Capital :  ₹{metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value      :  ₹{metrics['final_value']:>12,.2f}")
    print(f"  Total Return     :  {metrics['total_return_pct']:>+.2f}%")
    print(f"  CAGR             :  {metrics['cagr_pct']:>+.2f}%  {'✅' if metrics['passes_15pct_target'] else '❌'}")
    print(f"  Sharpe Ratio     :  {metrics['sharpe_ratio']:>.2f}  {'✅' if metrics['passes_sharpe'] else '❌'}")
    print(f"  Max Drawdown     :  {metrics['max_drawdown_pct']:>.2f}%  {'✅' if metrics['passes_drawdown'] else '❌'}")
    print("-" * 60)
    print(f"  Total Trades     :  {metrics['total_trades']}")
    print(f"  Win Rate         :  {metrics['win_rate_pct']:.1f}%  {'✅' if metrics['passes_win_rate'] else '❌'}")
    print(f"  Avg Win          :  ₹{metrics['avg_win_inr']:>+,.2f}")
    print(f"  Avg Loss         :  ₹{metrics['avg_loss_inr']:>+,.2f}")
    print(f"  Profit Factor    :  {metrics['profit_factor']:.2f}  {'✅' if metrics['passes_profit_factor'] else '❌'}")
    print(f"  Avg Hold Days    :  {metrics['avg_hold_days']:.1f}")
    print(f"  Total Charges    :  ₹{metrics['total_charges_inr']:>,.2f} ({metrics['annual_charges_drag_pct']:.2f}%/yr drag)")
    print("=" * 60)
    verdict = "PASS — Ready for paper trading" if metrics["all_criteria_met"] else "FAIL — Needs tuning"
    print(f"  Overall: {verdict}")
    print("=" * 60 + "\n")


def save_trade_log(result: BacktestResult, filepath: str = None, initial_capital: float = INITIAL_CAPITAL):
    """
    Save detailed trade-by-trade CSV.

    Fields include:
      - Date, Symbol, Entry/Exit price, Quantity
      - Position size (₹ and % of initial capital)
      - Stop loss (computed as entry * 0.98 fallback if not stored)
      - Exit reason (SL / trailing / time / signal / trend)
      - PnL (absolute and %)
      - Hold period (days)
      - Charges
    """
    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_trades.csv")

    fieldnames = [
        "symbol", "sector",
        "entry_date", "exit_date", "hold_days",
        "entry_price", "exit_price", "shares",
        "position_size_inr", "position_size_pct",
        "stop_loss_approx", "exit_reason",
        "gross_pnl", "net_pnl", "pnl_pct", "charges",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            position_size_inr = round(t.entry_price * t.shares, 2)
            position_size_pct = round(position_size_inr / initial_capital * 100, 2) if initial_capital > 0 else 0.0
            pnl_pct = round((t.exit_price - t.entry_price) / t.entry_price * 100, 2) if t.entry_price > 0 else 0.0
            # Approximate stop loss (2% below entry — matches STOP_LOSS_PCT)
            stop_loss_approx = round(t.entry_price * 0.98, 2)
            writer.writerow({
                "symbol":            t.symbol,
                "sector":            t.sector,
                "entry_date":        str(t.entry_date),
                "exit_date":         str(t.exit_date),
                "hold_days":         t.hold_days,
                "entry_price":       t.entry_price,
                "exit_price":        t.exit_price,
                "shares":            t.shares,
                "position_size_inr": position_size_inr,
                "position_size_pct": position_size_pct,
                "stop_loss_approx":  stop_loss_approx,
                "exit_reason":       t.exit_reason,
                "gross_pnl":         t.gross_pnl,
                "net_pnl":           t.net_pnl,
                "pnl_pct":           pnl_pct,
                "charges":           t.charges,
            })
    print(f"[Reporter] Trade log saved: {filepath}  ({len(result.trades)} trades)")


def save_rejected_log(result: BacktestResult, filepath: str = None):
    """Save per-day rejected trade reasons to CSV."""
    if not result.daily_scan_log:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_rejected.csv")

    # Extract rejected decisions (signal=NO or rs_pass=FAIL)
    fieldnames = ["date", "symbol", "rs_pass", "signal", "rank_score", "rs_rank", "selected"]
    rejected = [d for d in result.decision_log if d.get("selected") == "NO"]

    if not rejected:
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected)
    print(f"[Reporter] Rejected log saved: {filepath}  ({len(rejected)} entries)")


def save_decision_log(result: BacktestResult, filepath: str = None):
    """
    Save per-stock per-day decision table to CSV.

    Shows for each stock scanned:
      Symbol | RS | Signal | Rank | Selected

    Example:
      RELIANCE  PASS  YES  87.2  YES
      TCS       PASS  YES  76.1  YES
      WIPRO     PASS  YES  65.4  NO (not in top 5)
      HDFCBANK  FAIL  NO   12.0  NO (RS failed)
    """
    if not result.decision_log:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_decisions.csv")

    fieldnames = ["date", "symbol", "rs_pass", "signal", "rank_score", "rs_rank", "selected"]
    # Add 'date' field from daily scan if not already in decision_log entries
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.decision_log)
    print(f"[Reporter] Decision log saved: {filepath}  ({len(result.decision_log)} entries)")


def save_daily_scan_log(result: BacktestResult, filepath: str = None):
    """Save daily scan summary (scanned/passed/selected/regime) to CSV."""
    if not result.daily_scan_log:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_daily_scan.csv")

    fieldnames = [
        "date", "regime", "portfolio_value",
        "total_scanned", "rs_passed", "signals", "selected",
        "open_positions", "relaxed_filters",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.daily_scan_log)
    print(f"[Reporter] Daily scan log saved: {filepath}  ({len(result.daily_scan_log)} days)")


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
    """Compute metrics, print summary, save all CSVs."""
    metrics = calculate_metrics(result, initial_capital)
    print_summary(metrics)
    save_trade_log(result, initial_capital=initial_capital)
    save_rejected_log(result)
    save_decision_log(result)
    save_daily_scan_log(result)
    save_equity_curve(result)
    return metrics
