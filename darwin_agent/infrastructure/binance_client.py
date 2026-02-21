from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger("darwin.binance_client")


@dataclass(slots=True)
class BinanceCredentials:
    api_key: str
    api_secret: str


@dataclass(slots=True)
class ServerSideOrder:
    """Tracks a server-side protective order living on Binance."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""          # "BUY" or "SELL"
    order_type: str = ""    # "STOP_MARKET" or "TAKE_PROFIT_MARKET"
    stop_price: float = 0.0
    quantity: float = 0.0
    status: str = ""        # "NEW", "FILLED", "CANCELED", "EXPIRED"
    placed_at: float = 0.0  # time.time()


class BinanceFuturesClient:
    def __init__(self, credentials: BinanceCredentials, timeout_s: float = 10.0) -> None:
        self._credentials = credentials
        self._timeout_s = timeout_s
        self._base_url = "https://fapi.binance.com"
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": credentials.api_key})
        # Cache: {symbol: {step_size, tick_size, min_qty}}
        self._symbol_filters: dict[str, dict[str, float]] = {}

    def _sign(self, params: dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        params["signature"] = self._sign(params)
        url = f"{self._base_url}{path}"
        response = self._session.request(method, url, params=params, timeout=self._timeout_s)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._session.close()

    def ping_futures(self) -> None:
        response = self._session.get(f"{self._base_url}/fapi/v1/time", timeout=self._timeout_s)
        response.raise_for_status()

    def get_wallet_balance(self) -> float:
        payload = self._signed_request("GET", "/fapi/v2/account")
        return float(payload.get("totalWalletBalance", 0.0))

    def get_open_positions(self) -> list[dict[str, Any]]:
        payload = self._signed_request("GET", "/fapi/v2/positionRisk")
        if not isinstance(payload, list):
            return []
        return [p for p in payload if float(p.get("positionAmt", 0.0)) != 0.0]

    def get_unrealized_pnl(self) -> float:
        payload = self._signed_request("GET", "/fapi/v2/account")
        return float(payload.get("totalUnrealizedProfit", 0.0))

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            payload = self._signed_request(
                "POST",
                "/fapi/v1/leverage",
                params={"symbol": symbol, "leverage": leverage},
            )
            return int(payload.get("leverage", 0)) == leverage
        except Exception as exc:
            # Binance code -4046: "No need to change leverage." — already set correctly
            if "-4046" in str(exc) or "No need to change leverage" in str(exc):
                return True
            raise



    def get_current_price(self, symbol: str) -> float:
        """Fetch current mark price for a symbol (single lightweight call)."""
        try:
            response = self._session.get(
                f"{self._base_url}/fapi/v1/ticker/price",
                params={"symbol": symbol},
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            return float(response.json().get("price", 0.0))
        except Exception:
            return 0.0

    def get_account_snapshot(self) -> tuple[float, float]:
        """
        Return (wallet_balance, unrealized_pnl) in a single API call.
        Avoids calling /fapi/v2/account twice per tick.
        """
        payload = self._signed_request("GET", "/fapi/v2/account")
        wallet = float(payload.get("totalWalletBalance", 0.0))
        upnl = float(payload.get("totalUnrealizedProfit", 0.0))
        return wallet, upnl

    def get_exchange_info(self) -> dict[str, Any]:
        """Fetch exchange info including symbol step sizes."""
        response = self._session.get(
            f"{self._base_url}/fapi/v1/exchangeInfo",
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def get_symbol_step_sizes(self, symbols: list[str]) -> dict[str, float]:
        """
        Return {symbol: step_size} for given symbols.
        Falls back to 0.001 if not found.
        """
        try:
            info = self.get_exchange_info()
            result: dict[str, float] = {}
            for sym_info in info.get("symbols", []):
                sym = sym_info.get("symbol", "")
                if sym not in symbols:
                    continue
                for f in sym_info.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        step = float(f.get("stepSize", 0.001))
                        result[sym] = step
                        break
            # Fill missing with default
            for s in symbols:
                if s not in result:
                    result[s] = 0.001
            return result
        except Exception:
            return {s: 0.001 for s in symbols}

    def validate_startup(self, symbols: list[str], leverage: int) -> dict[str, Any]:
        self.ping_futures()
        wallet = self.get_wallet_balance()
        _ = self.get_unrealized_pnl()
        positions = self.get_open_positions()
        leverage_result: dict[str, bool] = {}
        for symbol in symbols:
            leverage_result[symbol] = self.set_leverage(symbol, leverage)
        return {
            "wallet_balance": wallet,
            "open_positions": positions,
            "leverage_result": leverage_result,
        }

    # ── Server-Side Protective Orders (VISA) ─────────────────────

    def place_stop_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> ServerSideOrder:
        """
        Place a STOP_MARKET order on Binance that lives on the exchange.

        This is the server-side stop loss: executes even if our bot is down.
        Uses closePosition=true when reduce_only to handle partial fills
        and avoid quantity mismatches.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": self._format_price(symbol, stop_price),
            "workingType": "MARK_PRICE",  # trigger on mark price, not last price (avoids wick manipulation)
        }
        if reduce_only:
            # closePosition=true closes the entire position regardless of quantity.
            # This avoids the race condition where position size changes between
            # our read and the SL trigger.
            params["closePosition"] = "true"
        else:
            params["quantity"] = self._format_quantity(symbol, quantity)

        payload = self._signed_request_with_retry("POST", "/fapi/v1/order", params)
        order_id = str(payload.get("orderId", ""))
        if not order_id:
            raise RuntimeError(f"STOP_MARKET failed: {payload}")

        result = ServerSideOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            stop_price=stop_price,
            quantity=quantity,
            status=payload.get("status", "NEW"),
            placed_at=time.time(),
        )
        logger.info(
            "SERVER-SIDE SL placed: %s %s stop=%.4f id=%s",
            symbol, side, stop_price, order_id,
        )
        return result

    def place_take_profit_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> ServerSideOrder:
        """
        Place a TAKE_PROFIT_MARKET order on Binance that lives on the exchange.

        Server-side take profit: executes even if our bot is down.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self._format_price(symbol, stop_price),
            "workingType": "MARK_PRICE",
        }
        if reduce_only:
            params["closePosition"] = "true"
        else:
            params["quantity"] = self._format_quantity(symbol, quantity)

        payload = self._signed_request_with_retry("POST", "/fapi/v1/order", params)
        order_id = str(payload.get("orderId", ""))
        if not order_id:
            raise RuntimeError(f"TAKE_PROFIT_MARKET failed: {payload}")

        result = ServerSideOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="TAKE_PROFIT_MARKET",
            stop_price=stop_price,
            quantity=quantity,
            status=payload.get("status", "NEW"),
            placed_at=time.time(),
        )
        logger.info(
            "SERVER-SIDE TP placed: %s %s target=%.4f id=%s",
            symbol, side, stop_price, order_id,
        )
        return result

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order. Returns True if cancelled or already gone."""
        try:
            self._signed_request(
                "DELETE", "/fapi/v1/order",
                params={"symbol": symbol, "orderId": order_id},
            )
            return True
        except Exception as exc:
            # -2011 = "Unknown order" — already cancelled/filled
            if "-2011" in str(exc) or "Unknown order" in str(exc):
                return True
            logger.warning("cancel_order failed %s/%s: %s", symbol, order_id, exc)
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol. Nuclear option for emergency close."""
        try:
            self._signed_request(
                "DELETE", "/fapi/v1/allOpenOrders",
                params={"symbol": symbol},
            )
            return True
        except Exception as exc:
            logger.warning("cancel_all_orders failed %s: %s", symbol, exc)
            return False

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch all open orders for a symbol. Used to verify server-side orders exist."""
        try:
            payload = self._signed_request(
                "GET", "/fapi/v1/openOrders",
                params={"symbol": symbol},
            )
            if not isinstance(payload, list):
                return []
            return payload
        except Exception as exc:
            logger.warning("get_open_orders failed %s: %s", symbol, exc)
            return []

    def get_order_status(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Check status of a specific order."""
        try:
            return self._signed_request(
                "GET", "/fapi/v1/order",
                params={"symbol": symbol, "orderId": order_id},
            )
        except Exception as exc:
            logger.warning("get_order_status failed %s/%s: %s", symbol, order_id, exc)
            return {}

    def get_order_book_depth(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """
        Fetch order book for pre-trade liquidity check.
        Returns {bids: [[price, qty], ...], asks: [[price, qty], ...]}.
        """
        try:
            response = self._session.get(
                f"{self._base_url}/fapi/v1/depth",
                params={"symbol": symbol, "limit": limit},
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"bids": [], "asks": []}

    def emergency_close_position(
        self, symbol: str, side: str, quantity: float, max_retries: int = 5
    ) -> dict[str, Any]:
        """
        Emergency position close with aggressive retry.

        Used when a client-side SL/TP triggers and we MUST close the position.
        Retries up to max_retries times with exponential backoff.
        On final retry, cancels all orders first (in case a server-side order
        is blocking with reduceOnly conflicts).
        """
        last_error = ""
        for attempt in range(max_retries):
            try:
                # On last attempt, cancel all open orders for this symbol
                # to avoid reduceOnly conflicts with existing SL/TP
                if attempt == max_retries - 1:
                    self.cancel_all_orders(symbol)
                    time.sleep(0.2)

                params: dict[str, Any] = {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": self._format_quantity(symbol, quantity),
                    "reduceOnly": "true",
                }
                payload = self._signed_request("POST", "/fapi/v1/order", params)
                order_id = str(payload.get("orderId", ""))
                if order_id:
                    logger.info(
                        "EMERGENCY CLOSE success: %s %s qty=%.6f attempt=%d id=%s",
                        symbol, side, quantity, attempt + 1, order_id,
                    )
                    return payload
                last_error = f"no orderId in response: {payload}"
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "EMERGENCY CLOSE attempt %d/%d failed %s: %s",
                    attempt + 1, max_retries, symbol, exc,
                )

            if attempt < max_retries - 1:
                backoff = min(0.5 * (2 ** attempt), 8.0)
                time.sleep(backoff)

        logger.critical(
            "EMERGENCY CLOSE FAILED after %d attempts: %s %s — %s",
            max_retries, symbol, side, last_error,
        )
        raise RuntimeError(
            f"EMERGENCY_CLOSE_FAILED: {symbol} {side} after {max_retries} attempts: {last_error}"
        )

    # ── Retry-hardened signed request ────────────────────────────

    def _signed_request_with_retry(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Any:
        """
        Signed request with exponential backoff for transient errors.

        Retries on: 502, 503, 429, connection errors, timeout.
        Does NOT retry on: 400 (bad params), 401 (auth), 403 (forbidden).
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._signed_request(method, path, params)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (502, 503, 429):
                    last_exc = exc
                    backoff = min(1.0 * (2 ** attempt), 16.0)
                    if status == 429:
                        backoff = max(backoff, 10.0)
                    logger.warning(
                        "transient %d on %s %s, retry %d/%d in %.1fs",
                        status, method, path, attempt + 1, max_retries, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                backoff = min(1.0 * (2 ** attempt), 16.0)
                logger.warning(
                    "connection error on %s %s, retry %d/%d in %.1fs: %s",
                    method, path, attempt + 1, max_retries, backoff, exc,
                )
                time.sleep(backoff)
                continue

        raise last_exc or RuntimeError(f"max retries on {method} {path}")

    # ── Price/Quantity formatting ────────────────────────────────

    def load_symbol_filters(self, symbols: list[str]) -> None:
        """Pre-load tick_size and step_size for price/quantity formatting."""
        try:
            info = self.get_exchange_info()
            for sym_info in info.get("symbols", []):
                sym = sym_info.get("symbol", "")
                if sym not in symbols:
                    continue
                filters: dict[str, float] = {}
                for f in sym_info.get("filters", []):
                    ft = f.get("filterType", "")
                    if ft == "LOT_SIZE":
                        filters["step_size"] = float(f.get("stepSize", 0.001))
                        filters["min_qty"] = float(f.get("minQty", 0.001))
                    elif ft == "PRICE_FILTER":
                        filters["tick_size"] = float(f.get("tickSize", 0.01))
                self._symbol_filters[sym] = filters
            logger.info("loaded symbol filters for %d symbols", len(self._symbol_filters))
        except Exception as exc:
            logger.warning("failed to load symbol filters: %s", exc)

    def _format_quantity(self, symbol: str, qty: float) -> str:
        """Format quantity to correct step_size precision."""
        filters = self._symbol_filters.get(symbol, {})
        step = filters.get("step_size", 0.001)
        if step > 0:
            qty = math.floor(qty / step) * step
            precision = max(0, round(-math.log10(step)))
        else:
            precision = 3
        return f"{qty:.{precision}f}"

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price to correct tick_size precision."""
        filters = self._symbol_filters.get(symbol, {})
        tick = filters.get("tick_size", 0.01)
        if tick > 0:
            price = round(price / tick) * tick
            precision = max(0, round(-math.log10(tick)))
        else:
            precision = 2
        return f"{price:.{precision}f}"
