"""
Upstox v2 API Broker Integration.

Implements the BaseBroker interface for Upstox delivery (CNC) trading.

Prerequisites
-------------
1. Set environment variables:
     UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN
2. Generate access token via Upstox OAuth2 flow (done once per day).
3. Use NSE instrument keys (e.g. "NSE_EQ|INE002A01018" for RELIANCE).

Symbol mapping
--------------
The yfinance symbols we use (e.g. "RELIANCE.NS") must be mapped to
Upstox instrument keys. The mapping file is loaded lazily from
config/upstox_instruments.csv (downloaded from Upstox API once per day).

Order types supported:
  - MARKET (immediate fill at best available price)
  - LIMIT  (fill at specified price or better)
  - SL-M   (stop-loss market: triggers at trigger_price, fills at market)

⚠️ WARNING: This module places REAL orders. Only use in production
    when you have thoroughly tested with PaperBroker first.
"""

import logging
import os
import time
from datetime import datetime, date
from typing import List, Optional

import requests

from broker.base import (
    BaseBroker, OrderRequest, OrderResult, OrderStatus,
    OrderSide, OrderType, LivePosition,
)
from config.settings import UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN
from monitoring.logger import log_api_failure

logger = logging.getLogger(__name__)

UPSTOX_BASE_URL = "https://api.upstox.com/v2"
UPSTOX_SANDBOX  = "https://api-hft.upstox.com/v2"   # Sandbox for testing


class UpstoxBroker(BaseBroker):
    """
    Upstox v2 REST API broker.

    Only enabled when UPSTOX_ACCESS_TOKEN is set in environment.
    Raises UpstoxAuthError on startup if token is missing.

    Usage
    -----
        broker = UpstoxBroker()
        result = broker.buy("RELIANCE.NS", quantity=5, price=0)  # market order
    """

    def __init__(self, sandbox: bool = False):
        if not UPSTOX_ACCESS_TOKEN:
            raise UpstoxAuthError(
                "UPSTOX_ACCESS_TOKEN not set. Generate token via Upstox OAuth2 flow."
            )
        self._base_url = UPSTOX_SANDBOX if sandbox else UPSTOX_BASE_URL
        self._headers  = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        self._instrument_cache: dict = {}
        logger.info("[Upstox] Initialised (sandbox=%s)", sandbox)

    # ── BaseBroker implementation ──────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order via Upstox v2 /order/place endpoint."""
        instrument_key = self._resolve_instrument(request.symbol)
        if not instrument_key:
            return OrderResult(
                order_id="", status=OrderStatus.REJECTED,
                symbol=request.symbol, side=request.side,
                requested_qty=request.quantity,
                rejection_reason=f"Cannot resolve instrument key for {request.symbol}",
            )

        payload = {
            "quantity":        request.quantity,
            "product":         request.product,
            "validity":        "DAY",
            "price":           request.price,
            "tag":             request.tag or "algo_swing",
            "instrument_token": instrument_key,
            "order_type":      request.order_type.value,
            "transaction_type": request.side.value,
            "disclosed_quantity": 0,
            "trigger_price":   request.trigger_price,
            "is_amo":          False,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/order/place",
                json=payload,
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            order_id = data.get("data", {}).get("order_id", "")
            logger.info("[Upstox] Order placed: %s %s×%s → order_id=%s",
                        request.side.value, request.quantity, request.symbol, order_id)

            return OrderResult(
                order_id=order_id,
                status=OrderStatus.OPEN,
                symbol=request.symbol,
                side=request.side,
                requested_qty=request.quantity,
                placed_at=datetime.now(),
                raw_response=data,
            )

        except requests.HTTPError as e:
            reason = self._parse_error(e.response)
            logger.error("[Upstox] HTTP error placing order for %s: %s", request.symbol, reason)
            return OrderResult(
                order_id="", status=OrderStatus.REJECTED,
                symbol=request.symbol, side=request.side,
                requested_qty=request.quantity,
                rejection_reason=reason,
            )

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = requests.delete(
                f"{self._base_url}/order/cancel",
                params={"order_id": order_id},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("[Upstox] Cancelled order %s", order_id)
            return True
        except Exception as e:
            logger.error("[Upstox] Failed to cancel order %s: %s", order_id, e)
            return False

    def get_order_status(self, order_id: str) -> OrderResult:
        try:
            resp = requests.get(
                f"{self._base_url}/order/details",
                params={"order_id": order_id},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return self._parse_order_response(data)
        except Exception as e:
            logger.error("[Upstox] get_order_status(%s) failed: %s", order_id, e)
            return OrderResult(
                order_id=order_id, status=OrderStatus.PENDING,
                symbol="", side=OrderSide.BUY, requested_qty=0,
            )

    def get_positions(self) -> List[LivePosition]:
        try:
            resp = requests.get(
                f"{self._base_url}/portfolio/short-term-positions",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [
                LivePosition(
                    symbol=pos["tradingsymbol"] + ".NS",
                    quantity=int(pos.get("quantity", 0)),
                    avg_price=float(pos.get("average_price", 0)),
                    ltp=float(pos.get("last_price", 0)),
                    pnl=float(pos.get("realised_profit", 0)),
                    product=pos.get("product", "CNC"),
                )
                for pos in data
                if int(pos.get("quantity", 0)) > 0
            ]
        except Exception as e:
            logger.error("[Upstox] get_positions() failed: %s", e)
            return []

    def get_portfolio_value(self) -> float:
        try:
            resp = requests.get(
                f"{self._base_url}/user/get-funds-and-margin",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            equity = data.get("equity", {})
            return float(equity.get("net", 0))
        except Exception as e:
            logger.error("[Upstox] get_portfolio_value() failed: %s", e)
            return 0.0

    def get_available_cash(self) -> float:
        try:
            resp = requests.get(
                f"{self._base_url}/user/get-funds-and-margin",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            equity = data.get("equity", {})
            return float(equity.get("available_margin", 0))
        except Exception as e:
            logger.error("[Upstox] get_available_cash() failed: %s", e)
            return 0.0

    # ── Instrument key resolution ──────────────────────────────────────────

    def _resolve_instrument(self, yfinance_symbol: str) -> Optional[str]:
        """
        Map yfinance symbol (e.g. "RELIANCE.NS") to Upstox instrument key.
        Uses a locally cached instrument CSV or falls back to a simple lookup.
        """
        # Strip .NS suffix
        nse_symbol = yfinance_symbol.replace(".NS", "").replace(".BO", "")

        # Check in-memory cache first
        if nse_symbol in self._instrument_cache:
            return self._instrument_cache[nse_symbol]

        # Try loading from CSV
        import os
        csv_path = "config/upstox_instruments.csv"
        if os.path.exists(csv_path):
            try:
                import csv
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sym = row.get("tradingsymbol", "")
                        key = row.get("instrument_key", "")
                        if sym and key:
                            self._instrument_cache[sym] = key
                if nse_symbol in self._instrument_cache:
                    return self._instrument_cache[nse_symbol]
            except Exception as e:
                logger.warning("[Upstox] Failed to load instruments CSV: %s", e)

        # Last resort: construct key using standard NSE format
        # NSE_EQ|<ISIN> — but we don't have ISIN; use tradingsymbol directly
        # Upstox also accepts NSE_EQ|<tradingsymbol> for many instruments
        constructed = f"NSE_EQ|{nse_symbol}"
        logger.debug("[Upstox] Using constructed key for %s: %s", yfinance_symbol, constructed)
        return constructed

    def _parse_order_response(self, data: dict) -> OrderResult:
        status_map = {
            "complete":   OrderStatus.COMPLETE,
            "rejected":   OrderStatus.REJECTED,
            "cancelled":  OrderStatus.CANCELLED,
            "open":       OrderStatus.OPEN,
            "pending":    OrderStatus.PENDING,
        }
        raw_status = data.get("status", "").lower()
        status = status_map.get(raw_status, OrderStatus.PENDING)
        side = OrderSide.BUY if data.get("transaction_type") == "BUY" else OrderSide.SELL
        return OrderResult(
            order_id=data.get("order_id", ""),
            status=status,
            symbol=data.get("tradingsymbol", "") + ".NS",
            side=side,
            requested_qty=int(data.get("quantity", 0)),
            filled_qty=int(data.get("filled_quantity", 0)),
            avg_price=float(data.get("average_price", 0)),
            rejection_reason=data.get("status_message", ""),
            raw_response=data,
        )

    @staticmethod
    def _parse_error(response) -> str:
        try:
            return response.json().get("errors", [{}])[0].get("message", response.text)
        except Exception:
            return response.text[:200]


class UpstoxAuthError(Exception):
    """Raised when Upstox credentials are missing or invalid."""
    pass
