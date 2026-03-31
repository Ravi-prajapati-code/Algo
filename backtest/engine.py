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
  REGIME DETECTION    — multi-regime (BULL/BEAR/SIDEWAYS/HIGH_VOL/PARTIAL) modulation
  STOCK RANKING       — only top N ranked stocks selected per day
  PORTFOLIO TRACKING  — daily snapshot of total value, cash, invested, unrealized PnL

Daily Portfolio Snapshot (logged every trading day):
  ┌─────────────────────────────────────────────────────────────┐
  │ [Portfolio] 2024-03-15  regime=BULL_TREND                   │
  │ Total: ₹82,350  Cash: ₹25,100 (30.5%)  Invested: ₹57,250   │
  │ Unrealized PnL: +₹2,150  Realized PnL: +₹5,800  Daily: +₹350│
  │ Open (3): RELIANCE +3.3%  TCS -0.6%  HDFCBANK +2.5%        │
  └─────────────────────────────────────────────────────────────┘

Capital Flow (per day):
  Start: ₹82,000 → Buys: -₹16,000 → Sells: +₹0 → End: ₹82,350

Dynamic Filter Relaxation:
  If zero candidates pass score threshold, min_score is reduced by 10pts
  (floor: 30) and the scan retries once. Ensures minimum trade frequency
  without permanently lowering quality standards.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config.settings import (
    INITIAL_CAPITAL, LOOKBACK_DAYS, MARKET_INDEX_SYMBOL,
    SLIPPAGE_MODEL, PARTIAL_FILL_ENABLED, MAX_SELECTED_STOCKS,
)
from indicators.composite import compute_indicators
from strategy.signals import generate_signals
from strategy.regime import detect_regime, regime_min_score, regime_position_factor, is_buy_allowed
from strategy.exit import update_trailing_stop, check_exit, initial_stops
from strategy.relative_strength import compute_rs_for_all
from strategy.scoring import score_signal, score_to_size_factor
from strategy.quality_filter import tier_size_factor
from strategy.stock_ranker import rank_and_select, compute_rank_score, build_decision_log
from backtest.slippage import apply_slippage, simulate_partial_fill
from charges.calculator import net_pnl as calc_net_pnl, buy_charges
from portfolio.sizer import calculate_shares, position_value
from portfolio.allocator import can_open_position, portfolio_invested_value
from portfolio.risk import can_open_new_trades
from strategy.entry import check_entry
from data.universe import get_sector
from db.models import Position, Trade

logger = logging.getLogger(__name__)

# Minimum score floor for dynamic relaxation (never go below this)
_DYNAMIC_RELAX_MIN_SCORE = 30.0
_DYNAMIC_RELAX_STEP      = 10.0


class BacktestResult:
    def __init__(self):
        self.trades: list[Trade]              = []
        self.equity_curve: dict[date, float]  = {}
        self.cash_curve:   dict[date, float]  = {}
        self.regime_log:   dict[date, str]    = {}   # {date: regime}
        self.daily_scan_log: list[dict]       = []   # Daily scan summaries
        self.decision_log:   list[dict]       = []   # Per-stock decisions
        # Portfolio tracking (new)
        self.portfolio_snapshots: list[dict]  = []   # Daily portfolio state
        self.capital_flow_log:   list[dict]   = []   # Daily capital flows


class BacktestEngine:
    """
    Simulates daily swing trading on historical OHLCV data.

    Portfolio Tracking
    ------------------
    Every day the engine records:
      - Total portfolio value (cash + invested)
      - Cash available (uninvested)
      - Invested capital (sum of position cost bases)
      - Unrealized PnL (open positions at current prices)
      - Realized PnL cumulative (all closed trades)
      - Daily PnL (today vs yesterday)
      - Open position details (symbol, entry, current, PnL%, hold days)
      - Capital flow (how much was deployed/released today)

    Usage
    -----
        engine = BacktestEngine(data, start_date, end_date)
        result = engine.run()
    """

    def __init__(
        self,
        data: dict,
        start_date: date,
        end_date: date,
        initial_capital: float = INITIAL_CAPITAL,
        lookback: int = LOOKBACK_DAYS,
        slippage_model: str = SLIPPAGE_MODEL,
        use_partial_fills: bool = PARTIAL_FILL_ENABLED,
        max_selected: int = MAX_SELECTED_STOCKS,
    ):
        self.data              = data
        self.start             = start_date
        self.end               = end_date
        self.initial_capital   = initial_capital
        self.lookback          = lookback
        self.slippage_model    = slippage_model
        self.use_partial_fills = use_partial_fills
        self.max_selected      = max_selected

    def run(self) -> BacktestResult:
        result         = BacktestResult()
        cash           = self.initial_capital
        peak_value     = self.initial_capital
        prev_pv        = self.initial_capital    # Previous day's portfolio value (for daily PnL)
        open_positions: list[Position] = []
        all_dates      = self._trading_dates()

        # Pre-load index data for market regime filter
        index_df_full = self.data.get(MARKET_INDEX_SYMBOL, pd.DataFrame())

        # Build ordered date list per symbol for next-bar open lookup
        symbol_dates = {}
        for sym, df in self.data.items():
            if sym != MARKET_INDEX_SYMBOL:
                symbol_dates[sym] = sorted(df.index.tolist())

        for i, today in enumerate(all_dates):
            new_trades_today      = 0
            capital_used_buys     = 0.0    # Cash deployed in buys today
            capital_released_sells = 0.0   # Cash received from sells today

            # ── Build indicator snapshot for today ───────────────────
            today_stock_data: dict = {}
            for symbol, full_df in self.data.items():
                if symbol == MARKET_INDEX_SYMBOL:
                    continue
                hist = full_df[full_df.index <= today]
                if len(hist) >= 60:
                    today_stock_data[symbol] = hist

            if not today_stock_data:
                continue

            # Batch RS computation for today
            index_hist_for_rs = (
                index_df_full[index_df_full.index <= today]
                if not index_df_full.empty else pd.DataFrame()
            )
            rs_data = compute_rs_for_all(today_stock_data, index_hist_for_rs)

            # Build enriched indicators with RS + quality
            indicators: dict = {}
            for symbol, hist in today_stock_data.items():
                rs_metrics = rs_data.get(symbol)
                ind = compute_indicators(hist, symbol=symbol, rs_metrics=rs_metrics)
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

            market_bullish = is_buy_allowed(regime)

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
                atr   = float(ind.get("atr", 0) or 0)
                pos   = update_trailing_stop(pos, price, atr)

                # Time-based exit: checked here using backtest date
                hold_days = (today - pos.entry_date).days
                from config.settings import MAX_HOLD_DAYS
                if hold_days >= MAX_HOLD_DAYS:
                    exec_price = self._get_execution_price(pos.symbol, today, all_dates, i, "sell", ind)[0]
                    positions_to_close.append((
                        pos, exec_price,
                        f"TIME_EXIT: held {hold_days} days ≥ {MAX_HOLD_DAYS}|slippage_applied",
                    ))
                    continue

                should_exit, reason = check_exit(pos, price, ind)
                if should_exit:
                    exec_price, slip_note = self._get_execution_price(
                        pos.symbol, today, all_dates, i, "sell", ind
                    )
                    positions_to_close.append((pos, exec_price, f"{reason}|{slip_note}"))
                else:
                    updated_positions.append(pos)

            # Execute sells
            for pos, exit_price, reason in positions_to_close:
                pnl = calc_net_pnl(pos.entry_price, exit_price, pos.shares)
                capital_used_inr = round(pos.entry_price * pos.shares, 2)
                pv_at_entry      = pos.portfolio_value_at_entry or self.initial_capital
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
                    # Capital tracking
                    portfolio_value_at_entry=round(pv_at_entry, 2),
                    capital_used_inr=capital_used_inr,
                    capital_used_pct=round(capital_used_inr / pv_at_entry * 100, 2) if pv_at_entry > 0 else 0.0,
                )
                result.trades.append(trade)
                proceeds = exit_price * pos.shares - pnl["sell_charges"]["total"]
                cash += proceeds
                capital_released_sells += proceeds

                logger.info(
                    "[Trade] SELL %-12s  qty=%d  exit=₹%.2f  pnl=%+.0f  reason=%s",
                    pos.symbol, pos.shares, exit_price, pnl["net_pnl"],
                    reason.split("|")[0][:30],
                )

            open_positions = updated_positions
            held = {pos.symbol for pos in open_positions}

            portfolio_val = cash + portfolio_invested_value(open_positions, prices)
            peak_value    = max(peak_value, portfolio_val)

            # ── Screen BUY candidates ─────────────────────────────────
            if not market_bullish:
                pv = cash + portfolio_invested_value(open_positions, prices)
                self._record_portfolio_snapshot(
                    result, today, regime, cash, pv, prev_pv,
                    open_positions, prices,
                    capital_used_buys, capital_released_sells,
                )
                result.equity_curve[today] = round(pv, 2)
                result.cash_curve[today]   = round(cash, 2)
                prev_pv = pv
                continue

            min_score   = regime_min_score(regime)
            size_factor = regime_position_factor(regime)

            # ── Full scan with per-symbol decision logging ────────────
            total_scanned   = len(indicators) - len(held)
            rs_passed_syms: set[str]     = set()
            signal_passed_syms: set[str] = set()
            rejected_log: list[dict]     = []
            rank_scores_map: dict        = {}
            rs_ranks_map: dict           = {}

            raw_candidates = []
            for symbol, ind in indicators.items():
                if symbol in held:
                    continue

                rs_rank    = float(ind.get("rs_rank", 0) or 0)
                rs_ranks_map[symbol]    = rs_rank
                rank_score = compute_rank_score(ind)
                rank_scores_map[symbol] = rank_score

                ok, reason = check_entry(ind, symbol=symbol, regime=regime)
                if not ok:
                    rejected_log.append({"date": str(today), "symbol": symbol, "reason": reason})
                    continue

                rs_passed_syms.add(symbol)
                score = score_signal(ind)
                if score < min_score:
                    rejected_log.append({
                        "date": str(today), "symbol": symbol,
                        "reason": f"Score {score:.1f} < min_score {min_score:.0f}",
                    })
                    continue

                signal_passed_syms.add(symbol)
                raw_candidates.append((score, symbol, ind))

            # ── Dynamic filter relaxation (if zero candidates) ────────
            relaxed = False
            if not raw_candidates and not signal_passed_syms:
                relaxed_min = max(_DYNAMIC_RELAX_MIN_SCORE, min_score - _DYNAMIC_RELAX_STEP)
                if relaxed_min < min_score:
                    for symbol, ind in indicators.items():
                        if symbol in held or symbol in {r["symbol"] for r in rejected_log if "Score" not in r.get("reason", "")}:
                            continue
                        score = score_signal(ind)
                        if score >= relaxed_min:
                            ok, _ = check_entry(ind, symbol=symbol, regime=regime)
                            if ok and symbol not in signal_passed_syms:
                                signal_passed_syms.add(symbol)
                                raw_candidates.append((score, symbol, ind))
                    if raw_candidates:
                        relaxed = True
                        logger.info(
                            "[Day] %s  RELAXED filters: min_score %.0f→%.0f — found %d candidates",
                            today, min_score, relaxed_min, len(raw_candidates),
                        )

            raw_candidates.sort(reverse=True)

            # ── Rank candidates and select top N ─────────────────────
            candidates    = rank_and_select(raw_candidates, max_stocks=self.max_selected, verbose=True)
            selected_syms = {sym for _, sym, _ in candidates}

            # ── Daily scan summary log ────────────────────────────────
            scan_summary = {
                "date":            str(today),
                "regime":          regime,
                "portfolio_value": round(portfolio_val, 2),
                "total_scanned":   total_scanned,
                "rs_passed":       len(rs_passed_syms),
                "signals":         len(signal_passed_syms),
                "selected":        len(selected_syms),
                "open_positions":  len(open_positions),
                "relaxed_filters": relaxed,
            }
            result.daily_scan_log.append(scan_summary)

            logger.info(
                "[Day] %s  regime=%-10s  capital=₹%,.0f  "
                "scanned=%d  rs_pass=%d  signals=%d  selected=%d",
                today, regime, portfolio_val,
                total_scanned, len(rs_passed_syms),
                len(signal_passed_syms), len(selected_syms),
            )

            # ── Per-stock decision log ────────────────────────────────
            decision_entries = build_decision_log(
                all_scanned=list(set(indicators.keys()) - held),
                rs_passed=rs_passed_syms,
                signal_passed=signal_passed_syms,
                selected=selected_syms,
                rank_scores=rank_scores_map,
                rs_ranks=rs_ranks_map,
            )
            result.decision_log.extend(decision_entries)

            for rej in rejected_log:
                logger.debug("[Rejected] %-12s  %s", rej["symbol"], rej["reason"])

            # ── Execute buys ──────────────────────────────────────────
            for score, symbol, ind in candidates:
                allowed, reason = can_open_new_trades(
                    new_trades_today, open_positions, portfolio_val, peak_value
                )
                if not allowed:
                    logger.debug("[Day] %s: buy limit — %s", today, reason)
                    break

                # Validate: insufficient cash → skip and log
                if cash <= 0:
                    logger.info("[Rejected] %-12s  Insufficient cash (₹%.0f)", symbol, cash)
                    continue

                exec_price, slip_note = self._get_execution_price(
                    symbol, today, all_dates, i, "buy", ind
                )
                if exec_price <= 0:
                    exec_price = ind["close"]

                atr   = float(ind.get("atr", 0) or 0)
                stops = initial_stops(exec_price, atr=atr)

                drawdown  = (peak_value - portfolio_val) / peak_value if peak_value > 0 else 0.0
                dd_factor = 0.5 if drawdown >= 0.10 else 1.0
                sc_factor = score_to_size_factor(score)
                qt_factor = tier_size_factor(symbol)

                shares = calculate_shares(
                    portfolio_val, exec_price, stops["stop_loss"],
                    cash, atr=atr,
                    drawdown_factor=dd_factor * size_factor,
                    score_factor=sc_factor,
                    quality_factor=qt_factor,
                )
                if shares <= 0:
                    continue

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

                charges    = buy_charges(tv)
                total_cost = tv + charges.total
                if total_cost > cash:
                    logger.info(
                        "[Rejected] %-12s  Insufficient cash: need ₹%.0f, have ₹%.0f",
                        symbol, total_cost, cash,
                    )
                    continue

                pos = Position(
                    symbol=symbol, sector=get_sector(symbol),
                    entry_date=today, entry_price=exec_price, shares=shares,
                    stop_loss=stops["stop_loss"], take_profit=stops["take_profit"],
                    trailing_stop=stops["trailing_stop"], peak_price=stops["peak_price"],
                    portfolio_value_at_entry=round(portfolio_val, 2),  # Track capital at entry
                )
                cash -= total_cost
                capital_used_buys += total_cost
                open_positions.append(pos)
                new_trades_today += 1

                pos_size_pct = tv / portfolio_val * 100 if portfolio_val > 0 else 0
                logger.info(
                    "[Trade] BUY  %-12s  qty=%d  price=₹%.2f  "
                    "stop=₹%.2f  size=₹%.0f (%.1f%%)  score=%.1f",
                    symbol, shares, exec_price,
                    stops["stop_loss"], tv, pos_size_pct, score,
                )

            # ── Record equity curve ───────────────────────────────────
            pv = cash + portfolio_invested_value(open_positions, prices)
            result.equity_curve[today] = round(pv, 2)
            result.cash_curve[today]   = round(cash, 2)

            # ── Portfolio snapshot + capital flow ─────────────────────
            self._record_portfolio_snapshot(
                result, today, regime, cash, pv, prev_pv,
                open_positions, prices,
                capital_used_buys, capital_released_sells,
            )
            prev_pv = pv

        # Force-close remaining positions at backtest end
        if open_positions:
            final_date = all_dates[-1] if all_dates else self.end
            for pos in open_positions:
                price = prices.get(pos.symbol, pos.entry_price)
                pnl   = calc_net_pnl(pos.entry_price, price, pos.shares)
                capital_used_inr = round(pos.entry_price * pos.shares, 2)
                pv_at_entry      = pos.portfolio_value_at_entry or self.initial_capital
                trade = Trade(
                    symbol=pos.symbol, sector=pos.sector,
                    entry_date=pos.entry_date, exit_date=final_date,
                    entry_price=pos.entry_price, exit_price=price,
                    shares=pos.shares,
                    gross_pnl=pnl["gross_pnl"], charges=pnl["total_charges"],
                    net_pnl=pnl["net_pnl"],
                    exit_reason="END_OF_BACKTEST",
                    hold_days=(final_date - pos.entry_date).days,
                    portfolio_value_at_entry=round(pv_at_entry, 2),
                    capital_used_inr=capital_used_inr,
                    capital_used_pct=round(capital_used_inr / pv_at_entry * 100, 2) if pv_at_entry > 0 else 0.0,
                )
                result.trades.append(trade)

        logger.info(
            "[Backtest] Done. %d trades over %d trading days",
            len(result.trades), len(result.equity_curve),
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _record_portfolio_snapshot(
        self,
        result: BacktestResult,
        today: date,
        regime: str,
        cash: float,
        pv: float,
        prev_pv: float,
        open_positions: list,
        prices: dict,
        capital_used_buys: float,
        capital_released_sells: float,
    ) -> None:
        """Record daily portfolio state snapshot and capital flow log."""

        invested_capital = portfolio_invested_value(open_positions, prices)
        unrealized_pnl   = sum(
            (prices.get(pos.symbol, pos.entry_price) - pos.entry_price) * pos.shares
            for pos in open_positions
        )
        realized_pnl_cum = sum(t.net_pnl or 0.0 for t in result.trades)
        daily_pnl        = pv - prev_pv

        # Build per-position snapshot
        pos_details = []
        for pos in open_positions:
            cur_price    = prices.get(pos.symbol, pos.entry_price)
            invested_inr = round(pos.entry_price * pos.shares, 2)
            cur_val_inr  = round(cur_price * pos.shares, 2)
            pnl_inr      = round((cur_price - pos.entry_price) * pos.shares, 2)
            pnl_pct      = round((cur_price - pos.entry_price) / pos.entry_price * 100, 2) if pos.entry_price > 0 else 0.0
            pct_of_port  = round(cur_val_inr / pv * 100, 2) if pv > 0 else 0.0
            pos_details.append({
                "symbol":       pos.symbol,
                "entry_date":   str(pos.entry_date),
                "entry_price":  pos.entry_price,
                "qty":          pos.shares,
                "current_price": round(cur_price, 2),
                "invested_inr": invested_inr,
                "current_value_inr": cur_val_inr,
                "pnl_inr":      pnl_inr,
                "pnl_pct":      pnl_pct,
                "pct_of_portfolio": pct_of_port,
                "hold_days":    (today - pos.entry_date).days,
                "stop_loss":    pos.stop_loss,
            })

        result.portfolio_snapshots.append({
            "date":                    str(today),
            "regime":                  regime,
            "total_portfolio_value":   round(pv, 2),
            "cash_available":          round(cash, 2),
            "cash_pct":                round(cash / pv * 100, 2) if pv > 0 else 0.0,
            "invested_capital":        round(invested_capital, 2),
            "invested_pct":            round(invested_capital / pv * 100, 2) if pv > 0 else 0.0,
            "unrealized_pnl":          round(unrealized_pnl, 2),
            "realized_pnl_cumulative": round(realized_pnl_cum, 2),
            "daily_pnl":               round(daily_pnl, 2),
            "open_positions_count":    len(open_positions),
            "open_positions":          pos_details,
        })

        result.capital_flow_log.append({
            "date":                    str(today),
            "starting_capital":        round(prev_pv, 2),
            "capital_used_buys":       round(capital_used_buys, 2),
            "capital_released_sells":  round(capital_released_sells, 2),
            "ending_capital":          round(pv, 2),
            "net_capital_change":      round(pv - prev_pv, 2),
        })

        # Console log
        cash_pct     = cash / pv * 100 if pv > 0 else 0
        invested_pct = invested_capital / pv * 100 if pv > 0 else 0
        pos_summary  = "  ".join(
            f"{p['symbol']} {p['pnl_pct']:+.1f}%"
            for p in pos_details
        ) or "none"
        logger.info(
            "[Portfolio] %s  total=₹%,.0f  cash=₹%,.0f(%.0f%%)  "
            "invested=₹%,.0f(%.0f%%)  unreal=%+.0f  real=%+.0f  daily=%+.0f",
            today, pv, cash, cash_pct,
            invested_capital, invested_pct,
            unrealized_pnl, realized_pnl_cum, daily_pnl,
        )
        if pos_details:
            logger.info("[Positions] %s", pos_summary)

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
        atr         = ind.get("atr", 0)

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

        return today_close, "no_next_bar"

    def _trading_dates(self) -> list[date]:
        """Generate weekday dates from start to end (proxy for trading days)."""
        dates = []
        current = self.start
        while current <= self.end:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)
        return dates
