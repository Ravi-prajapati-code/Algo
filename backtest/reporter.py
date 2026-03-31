"""
Backtest reporter — generates human-readable summary and CSV/console reports.

Outputs (all saved to outputs/ directory):
  backtest_trades.csv          — Full trade log with capital details
  backtest_portfolio_snapshots.csv — Daily portfolio state (total/cash/invested/PnL)
  backtest_capital_flow.csv    — Daily capital flows (start/buys/sells/end)
  backtest_decisions.csv       — Per-stock decision log (RS/Signal/Rank/Selected)
  backtest_daily_scan.csv      — Daily scan summary (scanned/passed/selected/regime)
  backtest_equity.csv          — Equity curve

Answers these questions at any time:
  • How much money is invested?     → portfolio_snapshots.invested_capital
  • How much cash is free?          → portfolio_snapshots.cash_available
  • Which trades are running?       → portfolio_snapshots.open_positions (JSON)
  • Which trades made/lost money?   → backtest_trades.net_pnl / pnl_pct
  • What is total portfolio growth? → print_summary() → CAGR, total_return_pct
"""

import os
import csv
import json
from datetime import date

from backtest.engine import BacktestResult
from backtest.metrics import calculate_metrics
from config.settings import INITIAL_CAPITAL, OUTPUTS_DIR


# ── Performance summary ────────────────────────────────────────────────────

def print_summary(metrics: dict):
    """Print a formatted backtest performance summary to stdout."""
    w = 62
    print("\n" + "=" * w)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * w)
    print(f"  Starting Capital :  ₹{metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value      :  ₹{metrics['final_value']:>12,.2f}")
    print(f"  Total Return     :  {metrics['total_return_pct']:>+.2f}%")
    print(f"  CAGR             :  {metrics['cagr_pct']:>+.2f}%  {'✅' if metrics['passes_15pct_target'] else '❌'}")
    print(f"  Sharpe Ratio     :  {metrics['sharpe_ratio']:>.2f}  {'✅' if metrics['passes_sharpe'] else '❌'}")
    print(f"  Max Drawdown     :  {metrics['max_drawdown_pct']:>.2f}%  {'✅' if metrics['passes_drawdown'] else '❌'}")
    print("-" * w)
    print(f"  Total Trades     :  {metrics['total_trades']}")
    print(f"  Win Rate         :  {metrics['win_rate_pct']:.1f}%  {'✅' if metrics['passes_win_rate'] else '❌'}")
    print(f"  Avg Win          :  ₹{metrics['avg_win_inr']:>+,.2f}")
    print(f"  Avg Loss         :  ₹{metrics['avg_loss_inr']:>+,.2f}")
    print(f"  Risk/Reward      :  {abs(metrics['avg_win_inr'] / metrics['avg_loss_inr']):.2f}x" if metrics['avg_loss_inr'] != 0 else "  Risk/Reward      :  N/A")
    print(f"  Profit Factor    :  {metrics['profit_factor']:.2f}  {'✅' if metrics['passes_profit_factor'] else '❌'}")
    print(f"  Avg Hold Days    :  {metrics['avg_hold_days']:.1f}")
    print(f"  Total Charges    :  ₹{metrics['total_charges_inr']:>,.2f} ({metrics['annual_charges_drag_pct']:.2f}%/yr drag)")
    print("=" * w)
    verdict = "PASS ✅ — Ready for paper trading" if metrics["all_criteria_met"] else "FAIL ❌ — Needs tuning"
    print(f"  Overall: {verdict}")
    print("=" * w + "\n")


# ── Portfolio state display ────────────────────────────────────────────────

def print_portfolio_state(result: BacktestResult, initial_capital: float = INITIAL_CAPITAL):
    """
    Print the final-day portfolio state in a rich table format.

    Shows:
      - Total value, cash, invested capital (with %)
      - Unrealized and realized PnL
      - Open positions table (symbol, entry, current, PnL%, hold days)
    """
    if not result.portfolio_snapshots:
        print("[Portfolio] No snapshot data available.")
        return

    snap = result.portfolio_snapshots[-1]   # Last trading day
    w    = 66

    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print(f"║  PORTFOLIO STATE — {snap['date']:<20}  regime={snap['regime']:<12}║")
    print("╠" + "═" * (w - 2) + "╣")

    total  = snap["total_portfolio_value"]
    cash   = snap["cash_available"]
    inv    = snap["invested_capital"]
    unreal = snap["unrealized_pnl"]
    real   = snap["realized_pnl_cumulative"]
    daily  = snap["daily_pnl"]
    ret_pct = (total - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0

    print(f"║  Total Portfolio Value  :  ₹{total:>12,.2f}  ({ret_pct:>+.1f}% from start)  ║")
    print(f"║  Cash Available         :  ₹{cash:>12,.2f}  ({snap['cash_pct']:>5.1f}% of total)   ║")
    print(f"║  Invested Capital       :  ₹{inv:>12,.2f}  ({snap['invested_pct']:>5.1f}% of total)   ║")
    print(f"║  Unrealized PnL         :  ₹{unreal:>+12,.2f}                          ║")
    print(f"║  Realized PnL (total)   :  ₹{real:>+12,.2f}                          ║")
    print(f"║  Daily PnL              :  ₹{daily:>+12,.2f}                          ║")
    print(f"║  Open Positions         :  {snap['open_positions_count']:<5}                             ║")

    positions = snap.get("open_positions", [])
    if positions:
        print("╠" + "═" * (w - 2) + "╣")
        hdr = f"{'Symbol':<12} {'Entry':>8} {'Current':>8} {'PnL%':>7} {'Invested':>10} {'Value':>10} {'Days':>5}"
        print(f"║  {hdr:<{w-4}}║")
        print("║  " + "-" * (w - 4) + "║")
        for p in positions:
            pnl_sign = "+" if p["pnl_pct"] >= 0 else ""
            row = (
                f"{p['symbol']:<12} "
                f"₹{p['entry_price']:>7,.0f} "
                f"₹{p['current_price']:>7,.0f} "
                f"{pnl_sign}{p['pnl_pct']:>5.1f}% "
                f"₹{p['invested_inr']:>9,.0f} "
                f"₹{p['current_value_inr']:>9,.0f} "
                f"{p['hold_days']:>4}d"
            )
            print(f"║  {row:<{w-4}}║")
    print("╚" + "═" * (w - 2) + "╝\n")


# ── Trade log ─────────────────────────────────────────────────────────────

def save_trade_log(
    result: BacktestResult,
    filepath: str = None,
    initial_capital: float = INITIAL_CAPITAL,
):
    """
    Save detailed trade-by-trade CSV.

    Fields:
      Date, Symbol, Entry price, Exit price, Quantity,
      Capital used (₹ and % of portfolio at entry),
      Stop loss, Exit reason, PnL (absolute and %), Hold days, Charges.
    """
    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_trades.csv")

    fieldnames = [
        "symbol", "sector",
        "entry_date", "exit_date", "hold_days",
        "entry_price", "exit_price", "shares",
        "capital_used_inr", "capital_used_pct",
        "portfolio_value_at_entry",
        "stop_loss_approx",
        "exit_reason",
        "gross_pnl", "net_pnl", "pnl_pct", "charges",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            pnl_pct = (
                round((t.exit_price - t.entry_price) / t.entry_price * 100, 2)
                if t.entry_price and t.entry_price > 0 else 0.0
            )
            writer.writerow({
                "symbol":                  t.symbol,
                "sector":                  t.sector,
                "entry_date":              str(t.entry_date),
                "exit_date":               str(t.exit_date),
                "hold_days":               t.hold_days,
                "entry_price":             t.entry_price,
                "exit_price":              t.exit_price,
                "shares":                  t.shares,
                "capital_used_inr":        t.capital_used_inr or round(t.entry_price * t.shares, 2),
                "capital_used_pct":        t.capital_used_pct or 0.0,
                "portfolio_value_at_entry": t.portfolio_value_at_entry or "",
                "stop_loss_approx":        round(t.entry_price * 0.98, 2),   # 2% floor
                "exit_reason":             t.exit_reason,
                "gross_pnl":               t.gross_pnl,
                "net_pnl":                 t.net_pnl,
                "pnl_pct":                 pnl_pct,
                "charges":                 t.charges,
            })
    print(f"[Reporter] Trade log saved: {filepath}  ({len(result.trades)} trades)")


# ── Portfolio snapshots ────────────────────────────────────────────────────

def save_portfolio_snapshots(result: BacktestResult, filepath: str = None):
    """
    Save daily portfolio state to CSV.

    Answers:
      • How much is invested / cash free each day?
      • What are the unrealized / realized PnL each day?
      • How did portfolio value evolve day by day?

    The 'open_positions' column is JSON-encoded for detail.
    """
    if not result.portfolio_snapshots:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_portfolio_snapshots.csv")

    flat_fields = [
        "date", "regime",
        "total_portfolio_value", "cash_available", "cash_pct",
        "invested_capital", "invested_pct",
        "unrealized_pnl", "realized_pnl_cumulative", "daily_pnl",
        "open_positions_count",
        "open_positions_json",   # JSON string with per-position detail
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for snap in result.portfolio_snapshots:
            writer.writerow({
                "date":                    snap["date"],
                "regime":                  snap["regime"],
                "total_portfolio_value":   snap["total_portfolio_value"],
                "cash_available":          snap["cash_available"],
                "cash_pct":                snap["cash_pct"],
                "invested_capital":        snap["invested_capital"],
                "invested_pct":            snap["invested_pct"],
                "unrealized_pnl":          snap["unrealized_pnl"],
                "realized_pnl_cumulative": snap["realized_pnl_cumulative"],
                "daily_pnl":               snap["daily_pnl"],
                "open_positions_count":    snap["open_positions_count"],
                "open_positions_json":     json.dumps(snap.get("open_positions", []), default=str),
            })
    print(f"[Reporter] Portfolio snapshots saved: {filepath}  ({len(result.portfolio_snapshots)} days)")


def save_open_positions_log(result: BacktestResult, filepath: str = None):
    """
    Save a flat CSV of the LAST DAY's open positions (position tracking table).

    Fields: Symbol, Entry price, Qty, Current price, Invested, Current value,
            PnL (₹ and %), % of portfolio, Hold days, Stop loss.
    """
    if not result.portfolio_snapshots:
        return

    last_snap = result.portfolio_snapshots[-1]
    positions = last_snap.get("open_positions", [])
    if not positions:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_open_positions.csv")

    fieldnames = [
        "symbol", "entry_date", "entry_price", "qty",
        "current_price", "invested_inr", "current_value_inr",
        "pnl_inr", "pnl_pct", "pct_of_portfolio",
        "hold_days", "stop_loss",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(positions)
    print(f"[Reporter] Open positions saved: {filepath}  ({len(positions)} positions)")


# ── Capital flow ───────────────────────────────────────────────────────────

def save_capital_flow_log(result: BacktestResult, filepath: str = None):
    """
    Save daily capital flow to CSV.

    Shows how cash moved each day:
      Starting capital → capital used for buys → capital released from sells → ending capital.
    """
    if not result.capital_flow_log:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_capital_flow.csv")

    fieldnames = [
        "date",
        "starting_capital", "capital_used_buys",
        "capital_released_sells", "ending_capital", "net_capital_change",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.capital_flow_log)
    print(f"[Reporter] Capital flow log saved: {filepath}  ({len(result.capital_flow_log)} days)")


# ── Decision / scan logs ───────────────────────────────────────────────────

def save_decision_log(result: BacktestResult, filepath: str = None):
    """Save per-stock per-day decision table (Symbol/RS/Signal/Rank/Selected)."""
    if not result.decision_log:
        return

    if filepath is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUTS_DIR, "backtest_decisions.csv")

    fieldnames = ["date", "symbol", "rs_pass", "signal", "rank_score", "rs_rank", "selected"]
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


# ── Equity curve ───────────────────────────────────────────────────────────

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


# ── Main entry point ───────────────────────────────────────────────────────

def run_and_report(result: BacktestResult, initial_capital: float = INITIAL_CAPITAL):
    """
    Compute metrics, print full report, save all CSVs.

    Console output includes:
      1. Performance summary (CAGR, Sharpe, drawdown, win rate, etc.)
      2. Portfolio state (last day: total/cash/invested/PnL + position table)

    CSV files saved:
      backtest_trades.csv              — Every closed trade with capital details
      backtest_portfolio_snapshots.csv — Daily portfolio state
      backtest_open_positions.csv      — Last-day open positions
      backtest_capital_flow.csv        — Daily capital flows
      backtest_decisions.csv           — Per-stock decision table
      backtest_daily_scan.csv          — Daily scan summary
      backtest_equity.csv              — Equity curve
    """
    metrics = calculate_metrics(result, initial_capital)
    print_summary(metrics)
    print_portfolio_state(result, initial_capital)
    save_trade_log(result, initial_capital=initial_capital)
    save_portfolio_snapshots(result)
    save_open_positions_log(result)
    save_capital_flow_log(result)
    save_decision_log(result)
    save_daily_scan_log(result)
    save_equity_curve(result)
    return metrics
