"""
Event-driven backtesting engine.

Uses the EXACT same strategy/*, portfolio/*, charges/* code as live trading.
The only differences from live trading:
  - Data is sliced by date instead of fetched fresh.
  - Execution price = next bar's open (not current close) — no lookahead bias.
  - Slippage and partial-fill simulation applied to execution prices.

Critical design rules:
  NO lookahead bias   — indicators computed on data ≤ today
  NEXT candle open    — buy/sell executes at tomorrow's open
  SLIPPAGE applied    — realistic execution prices via backtest/slippage.py
  MARKET REGIME       — Nifty 50 filter blocks buys in bear markets
  REGIME DETECTION    — multi-regime (BULL/BEAR/SIDEWAYS/HIGH_VOL) modulation
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config.settings import (
    INITIAL_CAPITAL, LOOKBACK_DAYS, MARKET_INDEX_SYMBOL,
    SLIPPAGE_MODEL, PARTIAL_FILL_ENABLED,
)
from indicators.composite import compute_indicators
from strategy.signals import generate_signals
from strategy.regime import detect_regime, regime_min_score, regime_position_factor, is_buy_allowed
from strategy.exit import update_trailing_stop, check_exit, initial_stops
from backtest.slippage import apply_slippage, simulate_partial_fill
from charges.calculator import net_pnl as calc_net_pnl, buy_charges
from portfolio.sizer import calculate_shares, position_value
from portfolio.allocator import can_open_position, portfolio_invested_value
from portfolio.risk import can_open_new_trades
from strategy.entry import check_entry
from strategy.scoring import score_signal
from data.universe import get_sector
from db.models import Position, Trade

logger = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self):
        self.trades: list[Trade]            = []
        self.equity_curve: dict[date, float] = {}
        self.cash_curve:   dict[date, float] = {}
        self.regime_log:   dict[date, str]   = {}   # {date: regime}


class BacktestEngine:
    """
    Simulates daily swing trading on historical OHLCV data.

    Execution model
    ---------------
    - Signals generated using today's close data.
    - Orders executed at tomorrow's open (next bar) to avoid lookahead.
    - Slippage applied on top of next-bar open (configurable model).
    - Partial fills simulated based on average daily volume.

    Usage
    -----
        engine = BacktestEngine(data, start_date, end_date)
        result = engine.run()
    """

    def __init__(
        self,
        data: dict,                 # {symbol: full_ohlcv_DataFrame}
        start_date: date,
        end_date: date,
        initial_capital: float = INITIAL_CAPITAL,
        lookback: int = LOOKBACK_DAYS,
        slippage_model: str = SLIPPAGE_MODEL,
        use_partial_fills: bool = PARTIAL_FILL_ENABLED,
    ):
        self.data            = data
        self.start           = start_date
        self.end             = end_date
        self.initial_capital = initial_capital
        self.lookback        = lookback
        self.slippage_model  = slippage_model
        self.use_partial_fills = use_partial_fills

    def run(self) -> BacktestResult:
        result      = BacktestResult()
        cash        = self.initial_capital
        peak_value  = self.initial_capital
        open_positions: list[Position] = []
        all_dates   = self._trading_dates()

        # Pre-load index data for market regime filter
        index_df_full = self.data.get(MARKET_INDEX_SYMBOL, pd.DataFrame())

        # Build ordered date list per symbol for next-bar open lookup
        symbol_dates = {}
        for sym, df in self.data.items():
            if sym != MARKET_INDEX_SYMBOL:
                symbol_dates[sym] = sorted(df.index.tolist())

        for i, today in enumerate(all_dates):
            new_trades_today = 0

            # ── Build indicator snapshot for today ───────────────────
            indicators: dict = {}
            for symbol, full_df in self.data.items():
                if symbol == MARKET_INDEX_SYMBOL:
                    continue
                hist = full_df[full_df.index <= today]
                if len(hist) < 60:
                    continue
                ind = compute_indicators(hist)
                if ind is not None:
                    indicators[symbol] = ind

            if not indicators:
                continue

            # ── Market regime detection ───────────────────────────────
            index_hist = (
                index_df_full[index_df_full.index <= today]
                if not index_df_full.empty else pd.DataFrame()
            )
            regime = detect_regime(index_hist)
            result.regime_log[today] = regime

            # UNKNOWN regime → block buys (same as BEAR_TREND).
            # Never default to allowing trades when index data is insufficient.
            market_bullish = is_buy_allowed(regime)
            if regime == "UNKNOWN":
                logger.debug(
                    "[Backtest] %s: regime=UNKNOWN (insufficient index data) "
                    "— trading disabled for new entries.", today,
                )

            prices = {sym: ind["close"] for sym, ind in indicators.items()}

            # ── Evaluate open positions → check exits ─────────────────
            positions_to_close: list = []
            updated_positions:  list = []

            for pos in open_positions:
                ind = indicators.get(pos.symbol)
                if ind is None:
                    updated_positions.append(pos)
                    continue
                price = ind["close"]
                pos = update_trailing_stop(pos, price)
                should_exit, reason = check_exit(pos, price, ind)

                # Time-based exit
                hold_days = (today - pos.entry_date).days
                if not should_exit and hold_days >= 60:
                    should_exit, reason = True, f"MAX_HOLD ({hold_days}d)"

                if should_exit:
                    # Execution price = next bar's open with slippage
                    exec_price, slip_note = self._get_execution_price(
                        pos.symbol, today, all_dates, i, "sell", ind
                    )
                    positions_to_close.append((pos, exec_price, f"{reason}|{slip_note}"))
                else:
                    updated_positions.append(pos)

            # Execute sells
            for pos, exit_price, reason in positions_to_close:
                pnl = calc_net_pnl(pos.entry_price, exit_price, pos.shares)
                trade = Trade(
                    symbol=pos.symbol, sector=pos.sector,
                    entry_date=pos.entry_date, exit_date=today,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares,
                    gross_pnl=pnl["gross_pnl"],
                    charges=pnl["total_charges"],
                    net_pnl=pnl["net_pnl"],
                    exit_reason=reason,
                    hold_days=(today - pos.entry_date).days,
                )
                result.trades.append(trade)
                cash += exit_price * pos.shares - pnl["sell_charges"]["total"]

            open_positions = updated_positions
            held = {pos.symbol for pos in open_positions}

            # ── Screen BUY candidates ─────────────────────────────────
            if not market_bullish:
                # Bear regime: no new buys, record equity and continue
                pv = cash + portfolio_invested_value(open_positions, prices)
                result.equity_curve[today] = round(pv, 2)
                result.cash_curve[today]   = round(cash, 2)
                continue

            min_score = regime_min_score(regime)
            size_factor = regime_position_factor(regime)

            candidates = []
            for symbol, ind in indicators.items():
                if symbol in held:
                    continue
                ok, _ = check_entry(ind)
                if not ok:
                    continue
                score = score_signal(ind)
                if score < min_score:
                    continue
                candidates.append((score, symbol, ind))
            candidates.sort(reverse=True)

            portfolio_val = cash + portfolio_invested_value(open_positions, prices)
            peak_value    = max(peak_value, portfolio_val)

            # ── Execute buys ──────────────────────────────────────────
            for score, symbol, ind in candidates:
                allowed, _ = can_open_new_trades(
                    new_trades_today, open_positions, portfolio_val, peak_value
                )
                if not allowed:
                    break

                # Use next bar's open as execution price
                exec_price, slip_note = self._get_execution_price(
                    symbol, today, all_dates, i, "buy", ind
                )
                if exec_price <= 0:
                    exec_price = ind["close"]

                stops = initial_stops(exec_price)
                atr   = ind.get("atr", 0)

                # Drawdown factor: reduce size if in moderate drawdown
                drawdown = (peak_value - portfolio_val) / peak_value if peak_value > 0 else 0.0
                dd_factor = 0.5 if drawdown >= 0.10 else 1.0
                combined_factor = size_factor * dd_factor

                shares = calculate_shares(
                    portfolio_val, exec_price, stops["stop_loss"],
                    cash, atr=atr, drawdown_factor=combined_factor,
                )
                if shares <= 0:
                    continue

                # Partial fill simulation
                if self.use_partial_fills:
                    adv = ind.get("vol_avg", 0)
                    shares = simulate_partial_fill(
                        requested_shares=shares,
                        trade_value=shares * exec_price,
                        avg_daily_volume=adv,
                    )
                if shares <= 0:
                    continue

                tv = position_value(shares, exec_price)
                ok, _ = can_open_position(symbol, tv, portfolio_val, open_positions, prices)
                if not ok:
                    continue

                charges = buy_charges(tv)
                total_cost = tv + charges.total
                if total_cost > cash:
                    continue

                pos = Position(
                    symbol=symbol, sector=get_sector(symbol),
                    entry_date=today, entry_price=exec_price, shares=shares,
                    stop_loss=stops["stop_loss"], take_profit=stops["take_profit"],
                    trailing_stop=stops["trailing_stop"], peak_price=stops["peak_price"],
                )
                cash -= total_cost
                open_positions.append(pos)
                new_trades_today += 1

            # ── Record equity curve ───────────────────────────────────
            pv = cash + portfolio_invested_value(open_positions, prices)
            result.equity_curve[today] = round(pv, 2)
            result.cash_curve[today]   = round(cash, 2)

        # Force-close remaining positions at backtest end
        if open_positions:
            final_date = all_dates[-1] if all_dates else self.end
            for pos in open_positions:
                price = prices.get(pos.symbol, pos.entry_price)
                pnl   = calc_net_pnl(pos.entry_price, price, pos.shares)
                trade = Trade(
                    symbol=pos.symbol, sector=pos.sector,
                    entry_date=pos.entry_date, exit_date=final_date,
                    entry_price=pos.entry_price, exit_price=price,
                    shares=pos.shares,
                    gross_pnl=pnl["gross_pnl"], charges=pnl["total_charges"],
                    net_pnl=pnl["net_pnl"],
                    exit_reason="END_OF_BACKTEST",
                    hold_days=(final_date - pos.entry_date).days,
                )
                result.trades.append(trade)

        logger.info(
            "[Backtest] Done. %d trades over %d trading days",
            len(result.trades), len(result.equity_curve),
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_execution_price(
        self,
        symbol: str,
        today: date,
        all_dates: list,
        today_idx: int,
        side: str,
        ind: dict,
    ) -> tuple[float, str]:
        """
        Return the next trading day's open price as execution price.
        Falls back to today's close if no next bar exists.
        Applies slippage model on top.
        """
        today_close = ind.get("close", 0)
        today_high  = ind.get("high", today_close)
        today_low   = ind.get("low", today_close)
        atr         = ind.get("atr", 0)

        # Get next bar data
        df = self.data.get(symbol)
        if df is not None and today_idx + 1 < len(all_dates):
            next_date = all_dates[today_idx + 1]
            next_bars = df[df.index == next_date]
            if not next_bars.empty:
                open_price = float(next_bars["open"].iloc[0])
                exec_price, note = apply_slippage(
                    side=side,
                    open_price=open_price,
                    prior_close=today_close,
                    atr=atr,
                    model=self.slippage_model,
                )
                return exec_price, note

        # Fallback: use today's close
        return today_close, "no_next_bar"

    def _trading_dates(self) -> list[date]:
        """Generate weekday dates from start to end (proxy for trading days)."""
        dates = []
        current = self.start
        while current <= self.end:
            if current.weekday() < 5:   # Mon–Fri
                dates.append(current)
            current += timedelta(days=1)
        return dates
