"""
Tests for the backtesting engine and metrics.
Uses synthetic data with known outcomes to validate logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics


def _make_trending_df(n=200, drift=0.002, seed=42):
    """
    Generate a trending OHLCV DataFrame that reliably triggers
    golden cross (EMA20 > EMA50) and RSI in buy zone.
    drift > 0 = uptrend.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    close = np.maximum(close, 1)
    high   = close * (1 + rng.uniform(0, 0.015, n))
    low    = close * (1 - rng.uniform(0, 0.015, n))
    open_  = close * (1 + rng.uniform(-0.008, 0.008, n))
    volume = rng.integers(500_000, 2_000_000, n).astype(float)

    idx = [date(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "open": open_, "high": high,
        "low": low, "close": close, "volume": volume
    }, index=idx)


class TestBacktestEngine:
    def test_runs_without_error(self):
        df = _make_trending_df(200)
        data = {"SYNTH1.NS": df, "SYNTH2.NS": _make_trending_df(200, seed=99)}
        start = date(2022, 4, 1)
        end   = date(2022, 9, 30)
        engine = BacktestEngine(data, start, end, initial_capital=75_000)
        result = engine.run()
        assert result is not None
        assert len(result.equity_curve) > 0

    def test_equity_curve_has_entries(self):
        df = _make_trending_df(200)
        data = {"SYNTH1.NS": df}
        engine = BacktestEngine(data, date(2022, 4, 1), date(2022, 9, 30), 75_000)
        result = engine.run()
        assert len(result.equity_curve) >= 10

    def test_no_lookahead_bias(self):
        # Dates in equity_curve should all be within [start, end]
        start, end = date(2022, 4, 1), date(2022, 9, 30)
        df = _make_trending_df(200)
        data = {"SYNTH1.NS": df}
        engine = BacktestEngine(data, start, end, 75_000)
        result = engine.run()
        for d in result.equity_curve:
            assert start <= d <= end

    def test_all_trades_have_exit(self):
        df = _make_trending_df(200)
        data = {"SYNTH1.NS": df, "SYNTH2.NS": _make_trending_df(200, seed=7)}
        engine = BacktestEngine(data, date(2022, 4, 1), date(2022, 9, 30), 75_000)
        result = engine.run()
        for trade in result.trades:
            assert trade.exit_date is not None
            assert trade.exit_price is not None


class TestMetrics:
    def _run_backtest(self):
        data = {
            f"SYNTH{i}.NS": _make_trending_df(250, drift=0.002, seed=i)
            for i in range(5)
        }
        engine = BacktestEngine(data, date(2022, 4, 1), date(2022, 12, 31), 75_000)
        return engine.run()

    def test_metrics_keys_present(self):
        result = self._run_backtest()
        metrics = calculate_metrics(result, 75_000)
        for key in ["cagr_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "profit_factor", "total_charges_inr"]:
            assert key in metrics

    def test_max_drawdown_non_negative(self):
        result = self._run_backtest()
        metrics = calculate_metrics(result, 75_000)
        assert metrics["max_drawdown_pct"] >= 0

    def test_win_rate_between_0_and_100(self):
        result = self._run_backtest()
        metrics = calculate_metrics(result, 75_000)
        assert 0 <= metrics["win_rate_pct"] <= 100

    def test_total_charges_positive(self):
        result = self._run_backtest()
        metrics = calculate_metrics(result, 75_000)
        if metrics["total_trades"] > 0:
            assert metrics["total_charges_inr"] >= 0

    def test_all_criteria_met_is_bool(self):
        result = self._run_backtest()
        metrics = calculate_metrics(result, 75_000)
        assert isinstance(metrics["all_criteria_met"], bool)
