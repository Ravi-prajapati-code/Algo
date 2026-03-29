"""
Signal strength scorer — ranks candidate BUY signals 0–100.
Higher score = stronger conviction.
"""

from config.settings import RSI_BUY_MIN, RSI_BUY_MAX


def score_signal(ind: dict) -> float:
    """
    Score a BUY signal candidate on 4 dimensions (25 pts each = 100 max):
      1. RSI position within the ideal buy zone
      2. MACD histogram strength
      3. Volume spike magnitude
      4. EMA crossover confirmation (golden cross recency + price/EMA relationship)
    """
    total = 0.0

    # 1. RSI score: ideal zone is 45–60, full points; outside penalised
    rsi = ind.get("rsi", 50)
    rsi_mid = (RSI_BUY_MIN + RSI_BUY_MAX) / 2           # 52.5
    rsi_range = (RSI_BUY_MAX - RSI_BUY_MIN) / 2         # 12.5
    rsi_dist = abs(rsi - rsi_mid) / rsi_range            # 0 = perfect, 1 = edge
    rsi_score = max(0, 25 * (1 - rsi_dist))
    total += rsi_score

    # 2. MACD histogram score
    hist = ind.get("macd_hist", 0)
    hist_prev = ind.get("macd_hist_prev", 0)
    if hist > 0 and hist > hist_prev:        # rising positive histogram = 25
        total += 25
    elif hist > 0:                           # positive but not rising = 15
        total += 15
    elif hist > hist_prev:                   # turning up from negative = 10
        total += 10

    # 3. Volume score
    vol_ratio = ind.get("vol_ratio", 1.0)
    vol_score = min(25, 25 * (vol_ratio - 1.0) / 2.0)  # 3x volume = full 25 pts
    total += max(0, vol_score)

    # 4. EMA / trend score
    ema_score = 0
    if ind.get("golden_cross"):
        ema_score += 15
    if ind.get("uptrend"):
        ema_score += 5
    price = ind.get("close", 1)
    ema_fast = ind.get("ema_fast", 1)
    if price > ema_fast:
        ema_score += 5                       # price above EMA20
    total += min(25, ema_score)

    return round(total, 1)
