"""
CLI entry point for the Algo swing trading platform.

Usage
-----
  python main.py run                          # Run today's signal pipeline
  python main.py backtest                     # Backtest with default dates
  python main.py backtest --start 2022-01-01 --end 2024-12-31
  python main.py charges                      # Show charges example
  python main.py initdb                       # Initialise / migrate database
  python main.py train_ml                     # Train ML prediction model
  python main.py train_ml --start 2022-01-01 --end 2024-12-31
  python main.py risk_report                  # Show current portfolio risk
"""

import argparse
import sys
import logging
from datetime import date

from monitoring.logger import setup_logging
setup_logging()


def cmd_run(_args):
    from runner.daily_runner import run
    run()


def cmd_backtest(args):
    from datetime import datetime
    from data.fetcher import fetch_all, fetch_index
    from data.universe import get_all_symbols
    from backtest.engine import BacktestEngine
    from backtest.reporter import run_and_report
    from db.repository import init_db
    from config.settings import INITIAL_CAPITAL, MARKET_INDEX_SYMBOL

    init_db()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    print(f"Fetching historical data ({start} → {end})…")
    symbols  = get_all_symbols()
    lookback = (end - start).days + 60
    data     = fetch_all(symbols, lookback_days=lookback)

    print(f"Fetching market index {MARKET_INDEX_SYMBOL}…")
    index_df = fetch_index(MARKET_INDEX_SYMBOL, lookback_days=lookback)
    if not index_df.empty:
        data[MARKET_INDEX_SYMBOL] = index_df

    slippage = getattr(args, "slippage", "fixed_pct")
    print(f"Running backtest on {len(data)-1} symbols + index  [slippage={slippage}]…")
    engine = BacktestEngine(data, start, end, INITIAL_CAPITAL, slippage_model=slippage)
    result = engine.run()
    run_and_report(result, INITIAL_CAPITAL)


def cmd_charges(_args):
    from charges.calculator import net_pnl
    print("\n=== Upstox Delivery Charges Example ===")
    print("Trade: Buy 10 shares @ ₹1,000, Sell @ ₹1,130 (13% gain)\n")
    result = net_pnl(1000, 1130, 10)
    print(f"  Buy value:       ₹{result['buy_value']:,.2f}")
    print(f"  Sell value:      ₹{result['sell_value']:,.2f}")
    print(f"  Gross P&L:       ₹{result['gross_pnl']:+,.2f}  ({result['gross_pct']:+.2f}%)")
    print(f"  Total Charges:   ₹{result['total_charges']:,.2f}")
    print(f"  Net P&L:         ₹{result['net_pnl']:+,.2f}  ({result['net_pct']:+.2f}%)")
    print("\n  Buy charges breakdown:")
    for k, v in result["buy_charges"].items():
        print(f"    {k:<15}: ₹{v:.2f}")
    print("  Sell charges breakdown:")
    for k, v in result["sell_charges"].items():
        print(f"    {k:<15}: ₹{v:.2f}")


def cmd_initdb(_args):
    from db.repository import init_db
    init_db()


def cmd_train_ml(args):
    """Train the ML prediction model on historical trade data."""
    from datetime import datetime
    from ml.trainer import train, print_feature_importance
    from ml.model import get_model_handler

    start = None
    end   = None
    if hasattr(args, "start") and args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if hasattr(args, "end") and args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("Training ML model on historical trade data…")
    success = train(start_date=start, end_date=end)
    if success:
        handler = get_model_handler()
        meta = handler.metadata
        print(f"\nML Model trained successfully!")
        print(f"  Trades used   : {meta.get('n_trades', 'N/A')}")
        print(f"  Win rate      : {meta.get('win_rate', 0)*100:.1f}%")
        print(f"  Train period  : {meta.get('train_start')} → {meta.get('train_end')}")
        print(f"\nEnable ML in live trading by setting: ML_ENABLED=true")
    else:
        print("Training failed. Run a backtest first to generate trade history.")
        sys.exit(1)


def cmd_risk_report(_args):
    """Print current portfolio risk metrics."""
    from db import repository as repo
    from portfolio.allocator import portfolio_invested_value
    from risk.manager import RiskManager
    from config.settings import INITIAL_CAPITAL

    positions  = repo.load_open_positions()
    snapshots  = repo.load_snapshots()

    if snapshots:
        cash       = snapshots[-1].cash
        total_val  = snapshots[-1].total_value
        peak_val   = max(s.total_value for s in snapshots)
    else:
        cash       = INITIAL_CAPITAL
        total_val  = INITIAL_CAPITAL
        peak_val   = INITIAL_CAPITAL

    rm = RiskManager(total_val, peak_val, INITIAL_CAPITAL)
    report = rm.health_report()

    print("\n=== Portfolio Risk Report ===")
    print(f"  Portfolio Value    : ₹{report['portfolio_value']:,.2f}")
    print(f"  Peak Value         : ₹{report['peak_value']:,.2f}")
    print(f"  Drawdown           : {report['drawdown_pct']:.2f}%  (halt at {report['drawdown_halt_pct']:.0f}%)")
    print(f"  Kill Switch        : {'ACTIVE ⚠️' if report['kill_switch_active'] else 'OFF ✓'}")
    print(f"  Size Reduction     : {report['size_reduction_factor']:.0%}")
    print(f"  Return from Start  : {report['return_from_start_pct']:+.2f}%")
    print(f"\n  Open Positions     : {len(positions)}")
    print(f"  Cash               : ₹{cash:,.2f}")

    if positions:
        print("\n  Positions:")
        for pos in positions:
            print(f"    {pos.symbol:<20} {pos.shares:>4} shares @ ₹{pos.entry_price:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Algo Swing Trading Platform — Production-Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run",         help="Run today's signal pipeline")
    sub.add_parser("initdb",      help="Initialise / migrate the SQLite database")
    sub.add_parser("charges",     help="Show charges calculation example")
    sub.add_parser("risk_report", help="Print current portfolio risk metrics")

    bt = sub.add_parser("backtest", help="Run backtesting engine")
    bt.add_argument("--start",    default="2022-01-01",   help="Start date YYYY-MM-DD")
    bt.add_argument("--end",      default=str(date.today()), help="End date YYYY-MM-DD")
    bt.add_argument("--slippage", default="fixed_pct",    choices=["none", "fixed_pct", "volatility"],
                    help="Slippage model (default: fixed_pct)")

    ml = sub.add_parser("train_ml", help="Train ML prediction model on trade history")
    ml.add_argument("--start", default=None, help="Training start date YYYY-MM-DD")
    ml.add_argument("--end",   default=None, help="Training end date YYYY-MM-DD")

    commands = {
        "run":         cmd_run,
        "backtest":    cmd_backtest,
        "charges":     cmd_charges,
        "initdb":      cmd_initdb,
        "train_ml":    cmd_train_ml,
        "risk_report": cmd_risk_report,
    }

    args = parser.parse_args()
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
