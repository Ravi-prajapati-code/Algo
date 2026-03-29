"""Tests for Upstox charges calculator."""

import pytest
from charges.calculator import buy_charges, sell_charges, round_trip_charges, net_pnl


class TestBuyCharges:
    def test_zero_brokerage(self):
        c = buy_charges(10000)
        assert c.brokerage == 0.0   # Upstox free delivery

    def test_stamp_duty_positive(self):
        c = buy_charges(10000)
        assert c.stamp_duty > 0

    def test_no_stt_on_buy(self):
        c = buy_charges(10000)
        assert c.stt == 0.0

    def test_gst_applied(self):
        c = buy_charges(10000)
        assert c.gst > 0

    def test_total_reasonable(self):
        # Buy charges should be << 0.1% for delivery
        c = buy_charges(10000)
        assert c.total < 10   # Less than ₹10 on ₹10,000 buy

    def test_stamp_duty_cap(self):
        # Very large trade — stamp duty should be capped at ₹1500
        c = buy_charges(100_000_000)
        assert c.stamp_duty <= 1500.0


class TestSellCharges:
    def test_stt_on_sell(self):
        c = sell_charges(10000)
        assert c.stt == pytest.approx(10.0, abs=0.01)   # 0.1% of 10000

    def test_dp_charge_flat(self):
        c = sell_charges(10000)
        assert c.dp_charges == 18.5

    def test_no_stamp_on_sell(self):
        c = sell_charges(10000)
        assert c.stamp_duty == 0.0

    def test_total_dominated_by_stt(self):
        c = sell_charges(10000)
        assert c.stt > c.exchange   # STT is the biggest charge on sell


class TestRoundTrip:
    def test_total_charges_positive(self):
        rt = round_trip_charges(10000, 11000)
        assert rt["total_charges"] > 0

    def test_charges_pct_under_half_percent(self):
        # For delivery, total round trip charges should be < 0.5% of buy value
        rt = round_trip_charges(10000, 10000)
        assert rt["charges_pct"] < 0.5


class TestNetPnL:
    def test_profitable_trade(self):
        # Buy @ 1000, sell @ 1130 (13% gain)
        result = net_pnl(1000, 1130, 10)
        assert result["gross_pnl"] == pytest.approx(1300.0, abs=0.01)
        assert result["net_pnl"] < result["gross_pnl"]   # charges reduce profit
        assert result["net_pnl"] > 0                     # still profitable

    def test_losing_trade(self):
        # Buy @ 1000, sell @ 940 (6% loss — stop loss)
        result = net_pnl(1000, 940, 10)
        assert result["gross_pnl"] < 0
        assert result["net_pnl"] < result["gross_pnl"]   # charges make loss worse

    def test_charges_structure(self):
        result = net_pnl(1000, 1130, 10)
        assert "buy_charges" in result
        assert "sell_charges" in result
        assert result["total_charges"] == pytest.approx(
            result["buy_charges"]["total"] + result["sell_charges"]["total"], abs=0.01
        )

    def test_consistency(self):
        # net_pnl = gross_pnl - total_charges
        result = net_pnl(500, 600, 20)
        expected_net = result["gross_pnl"] - result["total_charges"]
        assert result["net_pnl"] == pytest.approx(expected_net, abs=0.01)
