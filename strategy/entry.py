"""
Entry condition checker.
Given a symbol's indicator dict, determine if it qualifies for a BUY signal.
All conditions must pass for a BUY.
"""

from config.settings import RSI_BUY_MIN, RSI_BUY_MAX


def check_entry(ind: dict) -> tuple[bool, str]:
    """
    Returns (qualified: bool, reason: str).
    reason explains which condition failed, or 'All conditions met' on pass.
    """
    # 1. Uptrend: EMA20 > EMA50
    if not ind.get("uptrend"):
        return False, f"No uptrend (EMA20={ind['ema_fast']} < EMA50={ind['ema_slow']})"

    # 2. Recent golden cross OR price near EMA20 (pullback entry)
    # Allow entry if golden cross happened recently OR price is within 2% of EMA20
    price = ind["close"]
    ema_fast = ind["ema_fast"]
    near_ema_fast = abs(price - ema_fast) / ema_fast <= 0.02
    if not ind.get("golden_cross") and not near_ema_fast:
        return False, f"No recent golden cross and price not near EMA20 (price={price}, EMA20={ema_fast})"

    # 3. RSI in buy zone
    rsi = ind["rsi"]
    if not (RSI_BUY_MIN <= rsi <= RSI_BUY_MAX):
        return False, f"RSI {rsi:.1f} outside buy zone [{RSI_BUY_MIN}, {RSI_BUY_MAX}]"

    # 4. MACD bullish confirmation — histogram must be BOTH positive AND rising
    # (OR was too loose: accepted declining-but-positive or negative-but-rising setups)
    if not (ind.get("macd_bullish") and ind.get("macd_turning_up")):
        return False, f"MACD not bullish+rising (hist={ind['macd_hist']:.4f}, prev={ind['macd_hist_prev']:.4f})"

    # 5. Volume confirmation
    if not ind.get("vol_spike"):
        return False, f"No volume spike (ratio={ind['vol_ratio']:.2f}x)"

    # 6. Price above lower Bollinger Band (not oversold crash)
    if not ind.get("above_bb_lower"):
        return False, f"Price below lower Bollinger Band (price={price}, BB_lower={ind['bb_lower']})"

    return True, "All entry conditions met"
