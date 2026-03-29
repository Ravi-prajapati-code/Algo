"""
CLI entry point for the Algo swing trading tool.

Usage:
  python main.py run              # Run today's signal pipeline
  python main.py backtest         # Run backtest with default dates
  python main.py backtest --start 2022-01-01 --end 2024-12-31
  python main.py charges          # Show charges example calculation
  python main.py initdb           # Initialise the database
"""

import argparse
import sys
import logging
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_run(_args):
    from runner.daily_runner import run
    run()


def cmd_backtest(args):
    from datetime import datetime
    from data.fetcher import fetch_all
    from data.universe import get_all_symbols
    from backtest.engine import BacktestEngine
    from backtest.reporter import run_and_report
    from db.repository import init_db
    from config.settings import INITIAL_CAPITAL

    init_db()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    print(f"Fetching historical data ({start} → {end})…")
    symbols = get_all_symbols()
    data = fetch_all(symbols, lookback_days=(end - start).days + 60)

    print(f"Running backtest on {len(data)} symbols…")
    engine = BacktestEngine(data, start, end, INITIAL_CAPITAL)
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


def main():
    parser = argparse.ArgumentParser(description="Algo Swing Trading Tool")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run",     help="Run today's signal pipeline")
    sub.add_parser("initdb",  help="Initialise the SQLite database")
    sub.add_parser("charges", help="Show charges calculation example")

    bt = sub.add_parser("backtest", help="Run backtesting engine")
    bt.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    bt.add_argument("--end",   default=str(date.today()), help="End date YYYY-MM-DD")

    args = parser.parse_args()

    commands = {
        "run":      cmd_run,
        "backtest": cmd_backtest,
        "charges":  cmd_charges,
        "initdb":   cmd_initdb,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
