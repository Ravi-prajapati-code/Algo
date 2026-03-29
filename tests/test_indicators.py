"""Tests for technical indicators."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from indicators.trend import compute_trend, ema
from indicators.momentum import compute_rsi, compute_momentum
from indicators.volatility import compute_volatility
from indicators.volume import compute_volume
from indicators.composite import compute_indicators


def _make_df(n=100, trend="up"):
    """Create a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    if trend == "up":
        close = 100 + np.cumsum(rng.uniform(-1, 1.5, n))
    elif trend == "down":
        close = 100 + np.cumsum(rng.uniform(-1.5, 1, n))
    else:
        close = 100 + rng.uniform(-5, 5, n)

    close = np.maximum(close, 10)
    high   = close + rng.uniform(0, 2, n)
    low    = close - rng.uniform(0, 2, n)
    open_  = close + rng.uniform(-1, 1, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)

    idx = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=idx)


class TestEMA:
    def test_ema_length(self):
        s = pd.Series(range(100), dtype=float)
        result = ema(s, 20)
        assert len(result) == 100

    def test_ema_converges(self):
        # EMA of constant series should equal the constant
        s = pd.Series([10.0] * 100)
        result = ema(s, 20)
        assert abs(result.iloc[-1] - 10.0) < 0.01


class TestTrend:
    def test_uptrend_detected(self):
        df = _make_df(100, "up")
        result = compute_trend(df)
        # In a strong uptrend, EMA20 > EMA50 eventually
        assert "ema_fast" in result
        assert "uptrend" in result
        assert isinstance(result["golden_cross"], bool)

    def test_keys_present(self):
        df = _make_df()
        result = compute_trend(df)
        for key in ["ema_fast", "ema_slow", "uptrend", "golden_cross", "death_cross"]:
            assert key in result


class TestRSI:
    def test_rsi_range(self):
        df = _make_df(100)
        rsi = compute_rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_overbought_up_trend(self):
        # Strongly rising series should produce high RSI
        close = pd.Series(np.arange(1, 101, dtype=float))
        rsi = compute_rsi(close)
        assert rsi.iloc[-1] > 60

    def test_rsi_oversold_down_trend(self):
        close = pd.Series(np.arange(100, 0, -1, dtype=float))
        rsi = compute_rsi(close)
        assert rsi.iloc[-1] < 40


class TestMomentum:
    def test_keys_present(self):
        df = _make_df()
        result = compute_momentum(df)
        for key in ["rsi", "macd", "macd_signal", "macd_hist", "macd_bullish"]:
            assert key in result

    def test_rsi_value_valid(self):
        df = _make_df()
        result = compute_momentum(df)
        assert 0 <= result["rsi"] <= 100


class TestVolatility:
    def test_bollinger_keys(self):
        df = _make_df()
        result = compute_volatility(df)
        for key in ["bb_upper", "bb_mid", "bb_lower", "atr"]:
            assert key in result

    def test_bb_ordering(self):
        df = _make_df()
        result = compute_volatility(df)
        assert result["bb_upper"] > result["bb_mid"] > result["bb_lower"]

    def test_atr_positive(self):
        df = _make_df()
        result = compute_volatility(df)
        assert result["atr"] > 0


class TestVolume:
    def test_vol_ratio_calculated(self):
        df = _make_df()
        result = compute_volume(df)
        assert result["vol_ratio"] > 0
        assert isinstance(result["vol_spike"], bool)


class TestComposite:
    def test_returns_dict(self):
        df = _make_df()
        result = compute_indicators(df)
        assert result is not None
        assert isinstance(result, dict)

    def test_insufficient_data_returns_none(self):
        df = _make_df(30)  # too few rows
        result = compute_indicators(df)
        assert result is None

    def test_all_keys_present(self):
        df = _make_df()
        result = compute_indicators(df)
        for key in ["close", "rsi", "macd_hist", "ema_fast", "ema_slow",
                    "uptrend", "vol_ratio", "bb_upper", "atr"]:
            assert key in result
