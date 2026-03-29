"""
Serialise daily signals and portfolio state to JSON files in outputs/.
These files are committed back to the repo by GitHub Actions
so the Streamlit dashboard can read them without a live DB connection.
"""

import json
import os
from datetime import date
from typing import List

from db.models import Signal, PortfolioSnapshot, Position
from config.settings import OUTPUTS_DIR


def _ensure_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def write_signals(today: date, signals: List[Signal]):
    _ensure_dir()
    data = {
        "generated_at": str(today),
        "signals": [
            {
                "symbol":     s.symbol,
                "action":     s.action,
                "score":      s.score,
                "price":      s.price,
                "reason":     s.reason,
                "indicators": s.indicators,
            }
            for s in signals
        ],
    }
    path = os.path.join(OUTPUTS_DIR, "signals.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Output] signals.json written ({len(signals)} signals)")


def write_portfolio_state(
    today: date,
    snapshot: PortfolioSnapshot,
    open_positions: List[Position],
    prices: dict,
):
    _ensure_dir()
    positions_list = []
    for pos in open_positions:
        price = prices.get(pos.symbol, pos.entry_price)
        unreal_pnl = round((price - pos.entry_price) * pos.shares, 2)
        unreal_pct = round((price - pos.entry_price) / pos.entry_price * 100, 2)
        positions_list.append({
            "symbol":       pos.symbol,
            "sector":       pos.sector,
            "entry_date":   str(pos.entry_date),
            "entry_price":  pos.entry_price,
            "current_price": round(price, 2),
            "shares":       pos.shares,
            "stop_loss":    pos.stop_loss,
            "take_profit":  pos.take_profit,
            "trailing_stop": pos.trailing_stop,
            "unrealized_pnl": unreal_pnl,
            "unrealized_pct": unreal_pct,
        })

    data = {
        "date":            str(today),
        "cash":            snapshot.cash,
        "invested":        snapshot.invested,
        "total_value":     snapshot.total_value,
        "open_positions":  len(open_positions),
        "daily_pnl":       snapshot.daily_pnl,
        "cumulative_pnl":  snapshot.cumulative_pnl,
        "positions":       positions_list,
    }
    path = os.path.join(OUTPUTS_DIR, "portfolio_state.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Output] portfolio_state.json written")
