"""
Structured logging setup for the trading platform.

Provides:
  - Rotating file handlers (separate files for trades, errors, performance)
  - JSON-structured log records for machine-readable audit trail
  - Console handler with colour (INFO+ only)
  - Utility functions for logging trades, signals, and errors

Usage
-----
    from monitoring.logger import setup_logging, log_trade, log_error

    setup_logging()   # Call once at startup (main.py / daily_runner.py)
    log_trade(trade)
    log_error("data_fetch", exception, context={"symbol": "RELIANCE.NS"})
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

# ── Log directories ────────────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR") or "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_FILE       = LOG_DIR / "trades.jsonl"
ERROR_LOG_FILE       = LOG_DIR / "errors.jsonl"
PERFORMANCE_LOG_FILE = LOG_DIR / "performance.jsonl"
APP_LOG_FILE         = LOG_DIR / "app.log"

MAX_BYTES   = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5


# ── JSON log formatter ─────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.utcnow().isoformat() + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Colour console formatter ───────────────────────────────────────────────

_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
}
_RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    """Coloured console output."""

    FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        formatter = logging.Formatter(self.FORMAT, datefmt="%H:%M:%S")
        return formatter.format(record)


# ── Setup ──────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with:
      - Rotating app.log (plain text, all levels)
      - Console handler (coloured, INFO+)
    Call once at application startup.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    root.setLevel(logging.DEBUG)

    # Rotating file handler — full logs
    fh = logging.handlers.RotatingFileHandler(
        APP_LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColourFormatter())
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    for noisy in ("yfinance", "peewee", "urllib3", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)


# ── Domain-specific structured loggers ────────────────────────────────────

def log_trade(trade_dict: dict) -> None:
    """Append a completed trade record to trades.jsonl."""
    _append_jsonl(TRADE_LOG_FILE, {"event": "TRADE", **trade_dict})


def log_signal(signal_dict: dict) -> None:
    """Append a generated signal to the trade log."""
    _append_jsonl(TRADE_LOG_FILE, {"event": "SIGNAL", **signal_dict})


def log_error(
    context: str,
    exc: Optional[Exception] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append a structured error record to errors.jsonl."""
    payload: dict[str, Any] = {
        "event":   "ERROR",
        "ts":      datetime.utcnow().isoformat() + "Z",
        "context": context,
    }
    if exc is not None:
        payload["exception"] = type(exc).__name__
        payload["message"]   = str(exc)
        payload["traceback"] = traceback.format_exc()
    if extra:
        payload.update(extra)
    _append_jsonl(ERROR_LOG_FILE, payload)
    logging.getLogger("monitoring").error("[%s] %s: %s", context, type(exc).__name__ if exc else "error", exc)


def log_performance(metrics: dict) -> None:
    """Append a daily performance snapshot to performance.jsonl."""
    _append_jsonl(PERFORMANCE_LOG_FILE, {"event": "PERFORMANCE", **metrics})


def log_risk_event(event_type: str, details: dict) -> None:
    """Log risk events (kill switch, drawdown alert, etc.)."""
    payload = {
        "event":      "RISK",
        "event_type": event_type,
        "ts":         datetime.utcnow().isoformat() + "Z",
        **details,
    }
    _append_jsonl(ERROR_LOG_FILE, payload)
    logging.getLogger("monitoring.risk").warning("[Risk:%s] %s", event_type, details)


def log_api_failure(broker: str, operation: str, exc: Exception, attempt: int) -> None:
    """Log broker API failures for retry monitoring."""
    log_error(
        context=f"broker.{broker}.{operation}",
        exc=exc,
        extra={"attempt": attempt, "broker": broker, "operation": operation},
    )


# ── Helper ─────────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a .jsonl file (one JSON object per line)."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        logging.getLogger("monitoring").warning("Failed to write %s: %s", path, e)
