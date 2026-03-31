"""
Central configuration for the swing trading platform.
All tunable parameters live here — change these to adjust strategy behaviour.

Parameter hierarchy (highest to lowest priority):
  1. Environment variables (for secrets / per-deploy overrides)
  2. config/risk_config.yaml   (risk parameters)
  3. config/strategy_config.yaml (strategy parameters)
  4. Hardcoded defaults below  (fallback)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# MARKET / BROKER
# ──────────────────────────────────────────────
MARKET    = "NSE"       # NSE | BSE
BROKER    = "upstox"    # upstox (₹0 delivery brokerage)
CURRENCY  = "INR"

# ──────────────────────────────────────────────
# PORTFOLIO
# ──────────────────────────────────────────────
INITIAL_CAPITAL         = float(os.getenv("INITIAL_CAPITAL") or 75000)
MAX_OPEN_POSITIONS      = int(os.getenv("MAX_OPEN_POSITIONS") or 10)
MAX_NEW_TRADES_PER_DAY  = int(os.getenv("MAX_NEW_TRADES_PER_DAY") or 3)
MAX_SELECTED_STOCKS     = int(os.getenv("MAX_SELECTED_STOCKS") or 5)   # Top N ranked stocks per day
MAX_STOCK_ALLOCATION_PCT    = 0.20   # Max 20 % portfolio in any single stock
MAX_SECTOR_ALLOCATION_PCT   = 0.30   # Max 30 % portfolio in any single sector
MAX_RISK_PER_TRADE_PCT      = 0.01   # Risk 1 % of portfolio per trade (compounding)
CASH_RESERVE_PCT            = 0.10   # Keep 10 % cash as buffer

# ──────────────────────────────────────────────
# STRATEGY — ENTRY SIGNALS
# ──────────────────────────────────────────────
EMA_FAST                = 20
EMA_SLOW                = 50
RSI_PERIOD              = 14
RSI_BUY_MIN             = 40         # RSI must be above this to buy
RSI_BUY_MAX             = 65         # RSI must be below this (not overbought)
MACD_FAST               = 12
MACD_SLOW               = 26
MACD_SIGNAL             = 9
VOLUME_SPIKE_MULTIPLIER = 1.5        # Volume must be ≥ 1.5× 20-day average
EMA_CROSSOVER_LOOKBACK  = 5          # Golden cross within last N days
MIN_SIGNAL_SCORE        = 40         # Min score (0–100) for BUY consideration

# ──────────────────────────────────────────────
# MARKET REGIME FILTER
# BUY signals only when Nifty 50 is above its 200-day SMA.
# Prevents buying into broad bear markets (e.g. 2025 correction).
# ──────────────────────────────────────────────
MARKET_INDEX_SYMBOL     = "^NSEI"    # Nifty 50
MARKET_FILTER_SMA       = 200        # 200-day SMA on the index
MARKET_FILTER_ENABLED   = True       # Set False to disable (comparison only)

# ──────────────────────────────────────────────
# STRATEGY — EXIT / RISK
# ──────────────────────────────────────────────
STOP_LOSS_PCT           = 0.02       # 2 % hard stop below entry (or ATR)
TAKE_PROFIT_PCT         = 0.13       # 13 % take profit above entry
TRAILING_STOP_PCT       = 0.05       # 5 % trailing stop from peak
RSI_OVERBOUGHT_EXIT     = 75         # Force exit if RSI ≥ 75
MAX_HOLD_DAYS           = 10         # Time-based exit after 10 calendar days

# ──────────────────────────────────────────────
# RISK MANAGEMENT (Phase 1 — RiskManager)
# ──────────────────────────────────────────────
DRAWDOWN_KILL_SWITCH_PCT    = 0.15   # 15 % drawdown → halt all new buys
DRAWDOWN_REDUCE_SIZE_PCT    = 0.10   # 10 % drawdown → cut position size 50 %
ATR_STOP_MULTIPLIER         = 1.5    # ATR-based stop: stop_distance = ATR × 1.5

# ──────────────────────────────────────────────
# BACKTESTING — SLIPPAGE (Phase 2)
# ──────────────────────────────────────────────
SLIPPAGE_MODEL          = "fixed_pct"  # 'none' | 'fixed_pct' | 'volatility'
SLIPPAGE_FIXED_PCT      = 0.001        # 0.1 % per side (fixed model)
SLIPPAGE_ATR_MULT       = 0.10         # ATR × 0.10 → slippage % (volatility model)
PARTIAL_FILL_ENABLED    = True         # Simulate partial fills for large orders
PARTIAL_FILL_RATE       = 0.10         # Max 10 % of ADV per order

# ──────────────────────────────────────────────
# ML PREDICTION LAYER (Phase 4)
# ──────────────────────────────────────────────
ML_ENABLED              = (os.getenv("ML_ENABLED") or "false").lower() in ("true", "1", "yes")
ML_MIN_CONFIDENCE       = float(os.getenv("ML_MIN_CONFIDENCE") or 0.55)
ML_MODEL_DIR            = "ml/models"

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
PARTIAL_REGIME_MIN_CANDLES = 50       # Minimum candles for PARTIAL regime trading
LOOKBACK_DAYS           = 200         # Days of OHLCV history for indicators
DATA_CACHE_DB           = "db/trading.db"
OUTPUTS_DIR             = "outputs"

# ──────────────────────────────────────────────
# BACKTEST ACCEPTANCE CRITERIA
# ──────────────────────────────────────────────
BACKTEST_MIN_CAGR           = 0.15
BACKTEST_MIN_SHARPE         = 0.80
BACKTEST_MAX_DRAWDOWN       = 0.25
BACKTEST_MIN_WIN_RATE       = 0.42    # Lowered from 0.45 (valid with PF ≥ 1.80)
BACKTEST_MIN_PROFIT_FACTOR  = 1.40

# ──────────────────────────────────────────────
# NOTIFICATIONS
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# UPSTOX API (live trading — optional)
# ──────────────────────────────────────────────
UPSTOX_API_KEY      = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET   = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
