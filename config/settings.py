"""
Central configuration for the swing trading tool.
All tunable parameters are here — change these to adjust strategy behaviour.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# MARKET / BROKER
# ──────────────────────────────────────────────
MARKET = "NSE"                      # NSE | BSE
BROKER = "upstox"                   # upstox (0% delivery brokerage)
CURRENCY = "INR"

# ──────────────────────────────────────────────
# PORTFOLIO
# ──────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL") or 75000)  # ₹75,000 default
MAX_OPEN_POSITIONS = 15
MAX_NEW_TRADES_PER_DAY = 5
MAX_STOCK_ALLOCATION_PCT = 0.20     # Max 20% portfolio in any single stock
MAX_SECTOR_ALLOCATION_PCT = 0.30    # Max 30% portfolio in any single sector
MAX_RISK_PER_TRADE_PCT = 0.02       # Risk 2% of portfolio per trade (position sizing)
CASH_RESERVE_PCT = 0.10             # Keep 10% cash as buffer

# ──────────────────────────────────────────────
# STRATEGY — ENTRY SIGNALS
# ──────────────────────────────────────────────
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
RSI_BUY_MIN = 40                    # RSI must be above this to buy
RSI_BUY_MAX = 65                    # RSI must be below this to buy (not overbought)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_SPIKE_MULTIPLIER = 1.5       # Volume must be 1.5x 20-day average
EMA_CROSSOVER_LOOKBACK = 5          # Crossover must have happened within last N days

# ──────────────────────────────────────────────
# STRATEGY — EXIT / RISK
# ──────────────────────────────────────────────
STOP_LOSS_PCT = 0.06                # 6% below entry price
TAKE_PROFIT_PCT = 0.13              # 13% above entry price
TRAILING_STOP_PCT = 0.05            # 5% trailing stop from peak price
RSI_OVERBOUGHT_EXIT = 75            # Force exit if RSI exceeds this

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
LOOKBACK_DAYS = 200                 # Days of history to fetch for indicators
DATA_CACHE_DB = "db/trading.db"     # SQLite path
OUTPUTS_DIR = "outputs"

# ──────────────────────────────────────────────
# BACKTEST ACCEPTANCE CRITERIA
# ──────────────────────────────────────────────
BACKTEST_MIN_CAGR = 0.15
BACKTEST_MIN_SHARPE = 0.8
BACKTEST_MAX_DRAWDOWN = 0.25
BACKTEST_MIN_WIN_RATE = 0.45
BACKTEST_MIN_PROFIT_FACTOR = 1.4

# ──────────────────────────────────────────────
# NOTIFICATIONS
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# UPSTOX API (for live trading — optional)
# ──────────────────────────────────────────────
UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
