"""Tests for portfolio sizer, allocator, and risk controls."""

import pytest
from datetime import date

from portfolio.sizer import calculate_shares, position_value
from portfolio.allocator import can_open_position, sector_allocation
from portfolio.risk import can_open_new_trades
from db.models import Position


def _pos(symbol, sector, price=100.0, shares=10):
    return Position(
        symbol=symbol, sector=sector,
        entry_date=date(2024, 1, 1), entry_price=price, shares=shares,
        stop_loss=price * 0.94, take_profit=price * 1.13,
        trailing_stop=price * 0.95, peak_price=price,
    )


class TestPositionSizer:
    def test_basic_sizing(self):
        # ₹75,000 portfolio, 2% risk, entry 100, stop 94 → risk/share=6
        # max_loss=₹1500 → 250 shares by risk
        # but stock allocation cap = 20% of 75k = ₹15k → 150 shares
        # min(250, 150, 750) = 150
        shares = calculate_shares(75_000, 100, 94, 75_000)
        assert shares == 150

    def test_capped_by_allocation(self):
        # If shares_by_risk > max_stock_allocation cap, should be capped
        # max allocation = 75000 * 0.20 = 15000 → 15000/100 = 150 shares
        shares = calculate_shares(75_000, 100, 1, 75_000)  # risk/share = 99, need only 1.5 shares
        # But capped by allocation
        assert shares <= 150

    def test_zero_shares_no_cash(self):
        shares = calculate_shares(75_000, 100, 94, 0)
        assert shares == 0

    def test_zero_shares_bad_stop(self):
        shares = calculate_shares(75_000, 100, 100, 75_000)  # stop == entry
        assert shares == 0

    def test_position_value(self):
        assert position_value(10, 100.5) == pytest.approx(1005.0)


class TestAllocator:
    def test_can_open_new_position(self):
        positions = [_pos("TCS.NS", "IT", 100, 100)]
        ok, _ = can_open_position("INFY.NS", 5000, 75000, positions, {"TCS.NS": 100})
        assert ok is True

    def test_rejects_duplicate_symbol(self):
        positions = [_pos("TCS.NS", "IT")]
        ok, reason = can_open_position("TCS.NS", 5000, 75000, positions, {"TCS.NS": 100})
        assert ok is False
        assert "already" in reason.lower()

    def test_rejects_over_stock_cap(self):
        # Trade value = ₹20,000 on ₹75,000 portfolio = 26.7% > 20% cap
        ok, reason = can_open_position("INFY.NS", 20_000, 75_000, [], {})
        assert ok is False

    def test_rejects_over_sector_cap(self):
        # IT sector: TCS ₹15k, HCL ₹7.5k = ₹22.5k on ₹75k = 30% (at cap)
        # Adding another IT stock should fail
        positions = [
            _pos("TCS.NS",    "IT", 100, 150),   # ₹15,000
            _pos("HCLTECH.NS","IT", 100, 75),    # ₹7,500
        ]
        prices = {"TCS.NS": 100, "HCLTECH.NS": 100}
        ok, reason = can_open_position("INFY.NS", 5_000, 75_000, positions, prices)
        assert ok is False
        assert "sector" in reason.lower()


class TestRiskControls:
    def test_daily_limit_reached(self):
        ok, reason = can_open_new_trades(5, [], 75_000, 75_000)
        assert ok is False
        assert "limit" in reason.lower()

    def test_max_positions_reached(self):
        positions = [_pos(f"STOCK{i}.NS", "IT") for i in range(15)]
        ok, reason = can_open_new_trades(0, positions, 75_000, 75_000)
        assert ok is False

    def test_drawdown_circuit_breaker(self):
        # Portfolio dropped from 100k to 75k = 25% drawdown → halt
        ok, reason = can_open_new_trades(0, [], 75_000, 100_000)
        assert ok is False
        assert "drawdown" in reason.lower()

    def test_within_limits_allowed(self):
        ok, _ = can_open_new_trades(2, [], 75_000, 75_000)
        assert ok is True
