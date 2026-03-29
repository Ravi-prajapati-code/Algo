"""
Telegram Bot alert sender.
Sends a formatted daily signal summary to your Telegram chat.

Setup:
  1. Message @BotFather on Telegram → create bot → get BOT_TOKEN
  2. Message your bot once → get CHAT_ID from:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
  3. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env or GitHub Secrets
"""

import logging
import requests
from datetime import date
from typing import List, Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from db.models import Signal, PortfolioSnapshot

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def _send(text: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Token/Chat ID not configured — skipping")
        return False
    try:
        url = _API_BASE.format(token=TELEGRAM_BOT_TOKEN)
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"[Telegram] Failed to send message: {e}")
        return False


def send_daily_summary(
    today: date,
    buy_signals: List[Signal],
    sell_signals: List[Signal],
    snapshot: Optional[PortfolioSnapshot],
):
    """Compose and send the end-of-day summary."""
    lines = [f"<b>Algo Swing Trader — {today}</b>"]

    if snapshot:
        pnl_emoji = "📈" if snapshot.cumulative_pnl >= 0 else "📉"
        lines.append(
            f"\n{pnl_emoji} <b>Portfolio</b>\n"
            f"  Total Value: ₹{snapshot.total_value:,.0f}\n"
            f"  Cash:        ₹{snapshot.cash:,.0f}\n"
            f"  Day P&L:     ₹{snapshot.daily_pnl:+,.0f}\n"
            f"  Total P&L:   ₹{snapshot.cumulative_pnl:+,.0f}"
        )

    if buy_signals:
        lines.append("\n🟢 <b>BUY Signals</b>")
        for sig in buy_signals[:5]:   # Top 5
            lines.append(
                f"  • <b>{sig.symbol}</b> @ ₹{sig.price:.2f}  "
                f"score={sig.score:.0f}  RSI={sig.indicators.get('rsi', '?'):.1f}"
            )

    if sell_signals:
        lines.append("\n🔴 <b>SELL Signals</b>")
        for sig in sell_signals:
            lines.append(f"  • <b>{sig.symbol}</b> @ ₹{sig.price:.2f}  {sig.reason}")

    if not buy_signals and not sell_signals:
        lines.append("\n⏸ No new signals today — holding positions")

    lines.append("\n<i>Paper trading — not financial advice</i>")
    _send("\n".join(lines))


def send_error_alert(message: str):
    """Send an error notification."""
    _send(f"⚠️ <b>Algo Error</b>\n\n{message}")
