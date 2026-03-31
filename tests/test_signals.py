"""Tests for strategy entry/exit/scoring and signal generation."""

import pytest
from datetime import date

from strategy.entry import check_entry
from strategy.exit import check_exit, initial_stops, update_trailing_stop
from strategy.scoring import score_signal
from db.models import Position


def _make_ind(**overrides) -> dict:
    """Create a baseline 'perfect buy' indicator dict."""
    base = {
        "close":          100.0,
        "high":           102.0,
        "low":            98.0,
        "ema_fast":       98.0,
        "ema_slow":       95.0,
        "above_ema_fast": True,
        "uptrend":        True,
        "golden_cross":   True,
        "death_cross":    False,
        "rsi":            52.0,
        "rsi_prev":       48.0,
        "macd":           0.5,
        "macd_signal":    0.3,
        "macd_hist":      0.2,
        "macd_hist_prev": 0.1,
        "macd_bullish":   True,
        "macd_turning_up": True,
        "bb_upper":       110.0,
        "bb_mid":         100.0,
        "bb_lower":       90.0,
        "bb_pct":         0.5,
        "above_bb_lower": True,
        "above_bb_mid":   True,
        "atr":            2.0,
        "atr_pct":        0.02,
        "vol_avg":        500_000,
        "vol_today":      800_000,
        "vol_ratio":      1.6,
        "vol_spike":      True,
        "vol_increasing": True,
        # RS fields (hybrid model)
        "rs_ratio":         1.20,
        "rs_ratio_1m":      1.25,
        "rs_outperforming": True,
        "rs_accelerating":  True,
        "rs_rank":          85.0,
        "rs_qualified":     True,
        # Quality field (hybrid model)
        "quality_score_val": 18.0,
        # Price context (20-day high above close → not yet at breakout)
        "high_20d":         105.0,
        # 52-week context
        "week52_high":      110.0,
    }
    base.update(overrides)
    return base


def _make_position(**overrides) -> Position:
    base = dict(
        symbol="TEST.NS", sector="IT",
        entry_date=date(2024, 1, 1), entry_price=100.0, shares=10,
        stop_loss=94.0, take_profit=113.0,
        trailing_stop=95.0, peak_price=100.0,
    )
    base.update(overrides)
    return Position(**base)


class TestEntryConditions:
    def test_all_conditions_met(self):
        ok, reason = check_entry(_make_ind())
        assert ok is True

    def test_no_uptrend_fails(self):
        ok, _ = check_entry(_make_ind(uptrend=False, golden_cross=False))
        assert ok is False

    def test_rsi_too_high_fails(self):
        ok, _ = check_entry(_make_ind(rsi=70.0))
        assert ok is False

    def test_rsi_too_low_fails(self):
        ok, _ = check_entry(_make_ind(rsi=30.0))
        assert ok is False

    def test_rsi_at_boundary_min(self):
        # RSI=40 (RSI_BUY_MIN) qualifies for BREAKOUT when price is at 20-day high
        ok, _ = check_entry(_make_ind(rsi=40.0, high_20d=100.0))
        assert ok is True

    def test_rsi_at_boundary_max(self):
        ok, _ = check_entry(_make_ind(rsi=65.0))
        assert ok is True

    def test_no_momentum_fails(self):
        # No golden cross, no MACD support, no volume — all three strategies fail
        ok, _ = check_entry(_make_ind(
            golden_cross=False,
            macd_bullish=False, macd_turning_up=False,
            macd_hist=-0.5, macd_hist_prev=0.0,
            vol_spike=False, vol_ratio=0.8,
        ))
        assert ok is False

    def test_macd_not_bullish_no_turn_fails(self):
        # Deep-negative MACD (hist=-0.5) with no golden cross — BREAKOUT blocked,
        # PULLBACK blocked (no macd support), TREND_CONT blocked (no macd_bullish)
        ok, _ = check_entry(_make_ind(
            golden_cross=False,
            macd_bullish=False, macd_turning_up=False,
            macd_hist=-0.5, macd_hist_prev=0.0,
        ))
        assert ok is False

    def test_macd_bullish_not_rising_still_passes(self):
        # OR logic: positive histogram even if not rising is acceptable
        ok, _ = check_entry(_make_ind(
            macd_bullish=True, macd_turning_up=False,
            macd_hist=0.2, macd_hist_prev=0.5
        ))
        assert ok is True

    def test_macd_rising_but_negative_passes(self):
        # OR logic: turning up from negative is acceptable (momentum shift)
        ok, _ = check_entry(_make_ind(
            macd_bullish=False, macd_turning_up=True,
            macd_hist=-0.1, macd_hist_prev=-0.5
        ))
        assert ok is True

    def test_price_below_bb_lower_fails(self):
        ok, _ = check_entry(_make_ind(above_bb_lower=False))
        assert ok is False


class TestExitConditions:
    def test_stop_loss_triggers(self):
        pos = _make_position(stop_loss=94.0)
        ok, reason = check_exit(pos, 93.0, _make_ind())
        assert ok is True
        assert "STOP_LOSS" in reason

    def test_take_profit_triggers(self):
        pos = _make_position(take_profit=113.0)
        ok, reason = check_exit(pos, 114.0, _make_ind())
        assert ok is True
        assert "TAKE_PROFIT" in reason

    def test_trailing_stop_triggers(self):
        # entry=90 → 2% hard stop = 90-max(1.8,3.0)=87 (well below price=96)
        # trailing_stop=97 > price=96 → trailing stop triggers, not hard stop
        pos = _make_position(entry_price=90.0, peak_price=100.0, trailing_stop=97.0)
        ok, reason = check_exit(pos, 96.0, _make_ind())
        assert ok is True
        assert "TRAILING" in reason

    def test_death_cross_triggers(self):
        ok, reason = check_exit(_make_position(), 100.0, _make_ind(death_cross=True))
        assert ok is True

    def test_rsi_overbought_triggers(self):
        ok, reason = check_exit(_make_position(), 100.0, _make_ind(rsi=76.0))
        assert ok is True

    def test_hold_when_fine(self):
        ok, reason = check_exit(_make_position(), 105.0, _make_ind())
        assert ok is False
        assert reason == "HOLD"


class TestInitialStops:
    def test_stop_loss_below_entry(self):
        stops = initial_stops(100.0)
        assert stops["stop_loss"] < 100.0

    def test_take_profit_above_entry(self):
        stops = initial_stops(100.0)
        assert stops["take_profit"] > 100.0

    def test_trailing_stop_below_entry(self):
        stops = initial_stops(100.0)
        assert stops["trailing_stop"] < 100.0


class TestTrailingStop:
    def test_updates_on_new_high(self):
        pos = _make_position(peak_price=100.0, trailing_stop=95.0)
        pos = update_trailing_stop(pos, 110.0)
        assert pos.peak_price == 110.0
        assert pos.trailing_stop > 95.0

    def test_no_update_on_lower_price(self):
        pos = _make_position(peak_price=100.0, trailing_stop=95.0)
        pos = update_trailing_stop(pos, 98.0)
        assert pos.peak_price == 100.0
        assert pos.trailing_stop == 95.0


class TestScoring:
    def test_score_range(self):
        score = score_signal(_make_ind())
        assert 0 <= score <= 100

    def test_perfect_signal_high_score(self):
        score = score_signal(_make_ind())
        assert score >= 60   # Should score well with all conditions met

    def test_weak_signal_lower_score(self):
        weak = _make_ind(
            vol_ratio=1.0,     # no volume spike
            macd_hist=-0.1,    # bearish MACD
            golden_cross=False,
        )
        strong_score = score_signal(_make_ind())
        weak_score   = score_signal(weak)
        assert strong_score > weak_score
