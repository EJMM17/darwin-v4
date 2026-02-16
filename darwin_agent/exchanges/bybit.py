"""
Darwin v4 — Bybit V5 API adapter.

Layer 2 (exchanges). Implements IExchangeAdapter.
USDⓈ-M Perpetual Futures via V5 unified API.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlencode

from darwin_agent.interfaces.enums import (
    ExchangeID, OrderSide, OrderType, TimeFrame,
)
from darwin_agent.interfaces.types import (
    Candle, ExchangeStatus, OrderRequest, OrderResult,
    Position, Ticker,
)

logger = logging.getLogger("darwin.bybit")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_TF_MAP = {
    TimeFrame.M1: "1", TimeFrame.M5: "5", TimeFrame.M15: "15",
    TimeFrame.M30: "30", TimeFrame.H1: "60", TimeFrame.H4: "240",
    TimeFrame.D1: "D",
}


class BybitAdapter:
    """
    Production Bybit V5 adapter.
    
    Features:
      - HMAC-SHA256 authentication
      - Rate limiting with backoff
      - Automatic retry on 5xx
      - Position management
      - Leverage control per symbol
    """

    __slots__ = (
        "_api_key", "_api_secret", "_base_url", "_session",
        "_rate_limiter", "_last_request_time", "_testnet",
    )

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._base_url = (
            "https://api-testnet.bybit.com" if testnet
            else "https://api.bybit.com"
        )
        self._session = None
        self._last_request_time = 0.0

    @property
    def exchange_id(self) -> ExchangeID:
        return ExchangeID.BYBIT

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()

    # ── Auth ─────────────────────────────────────────────────

    def _sign(self, params: Dict) -> Dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        param_str = ts + self._api_key + recv_window
        if params:
            param_str += urlencode(sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode(), param_str.encode(), hashlib.sha256,
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, params: Dict = None) -> Dict:
        await self._ensure_session()
        # Rate limit: min 50ms between requests
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 0.05:
            await asyncio.sleep(0.05 - elapsed)
        self._last_request_time = time.monotonic()

        url = f"{self._base_url}{path}"
        headers = self._sign(params or {})

        for attempt in range(3):
            try:
                if method == "GET":
                    async with self._session.get(url, params=params, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with self._session.post(url, json=params, headers=headers) as resp:
                        data = await resp.json()
                if data.get("retCode") == 0:
                    return data.get("result", {})
                logger.warning("bybit error: %s", data.get("retMsg"))
                return data
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    raise

    # ── IExchangeAdapter ─────────────────────────────────────

    async def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]:
        data = await self._request("GET", "/v5/market/kline", {
            "category": "linear", "symbol": symbol,
            "interval": _TF_MAP.get(timeframe, "15"), "limit": str(limit),
        })
        candles = []
        for row in reversed(data.get("list", [])):
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
                open=float(row[1]), high=float(row[2]),
                low=float(row[3]), close=float(row[4]),
                volume=float(row[5]), timeframe=timeframe,
            ))
        return candles

    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._request("GET", "/v5/market/tickers", {
            "category": "linear", "symbol": symbol,
        })
        item = data.get("list", [{}])[0]
        return Ticker(
            symbol=symbol,
            last_price=float(item.get("lastPrice", 0)),
            bid=float(item.get("bid1Price", 0)),
            ask=float(item.get("ask1Price", 0)),
            volume_24h=float(item.get("volume24h", 0)),
        )

    async def get_positions(self) -> List[Position]:
        data = await self._request("GET", "/v5/position/list", {
            "category": "linear", "settleCoin": "USDT",
        })
        positions = []
        for item in data.get("list", []):
            size = float(item.get("size", 0))
            if size == 0:
                continue
            positions.append(Position(
                symbol=item["symbol"],
                side=OrderSide.BUY if item["side"] == "Buy" else OrderSide.SELL,
                size=size,
                entry_price=float(item.get("avgPrice", 0)),
                current_price=float(item.get("markPrice", 0)),
                unrealized_pnl=float(item.get("unrealisedPnl", 0)),
                leverage=int(item.get("leverage", 1)),
                exchange_id=ExchangeID.BYBIT,
            ))
        return positions

    async def place_order(self, request: OrderRequest) -> OrderResult:
        params = {
            "category": "linear",
            "symbol": request.symbol,
            "side": "Buy" if request.side == OrderSide.BUY else "Sell",
            "orderType": "Market" if request.order_type == OrderType.MARKET else "Limit",
            "qty": str(request.quantity),
        }
        if request.price and request.order_type == OrderType.LIMIT:
            params["price"] = str(request.price)
        if request.stop_loss:
            params["stopLoss"] = str(round(request.stop_loss, 2))
        if request.take_profit:
            params["takeProfit"] = str(round(request.take_profit, 2))

        data = await self._request("POST", "/v5/order/create", params)
        order_id = data.get("orderId", "")
        if not order_id:
            return OrderResult(
                success=False, error=data.get("retMsg", "unknown"),
                exchange_id=ExchangeID.BYBIT,
            )
        return OrderResult(
            order_id=order_id, symbol=request.symbol,
            side=request.side, filled_qty=request.quantity,
            filled_price=0.0,  # filled async, check via WS
            exchange_id=ExchangeID.BYBIT,
        )

    async def close_position(self, symbol: str, side: OrderSide) -> OrderResult:
        close_side = "Sell" if side == OrderSide.BUY else "Buy"
        # Get current position size
        positions = await self.get_positions()
        pos = next((p for p in positions if p.symbol == symbol), None)
        if pos is None:
            return OrderResult(success=False, error="no_position", exchange_id=ExchangeID.BYBIT)
        params = {
            "category": "linear", "symbol": symbol,
            "side": close_side, "orderType": "Market",
            "qty": str(pos.size), "reduceOnly": True,
        }
        data = await self._request("POST", "/v5/order/create", params)
        return OrderResult(
            order_id=data.get("orderId", ""), symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
            filled_qty=pos.size, filled_price=pos.current_price,
            exchange_id=ExchangeID.BYBIT,
        )

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            await self._request("POST", "/v5/position/set-leverage", {
                "category": "linear", "symbol": symbol,
                "buyLeverage": str(leverage), "sellLeverage": str(leverage),
            })
            return True
        except Exception:
            return False

    async def get_balance(self) -> float:
        data = await self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED",
        })
        for acct in data.get("list", []):
            for coin in acct.get("coin", []):
                if coin["coin"] == "USDT":
                    return float(coin.get("walletBalance", 0))
        return 0.0

    async def get_status(self) -> ExchangeStatus:
        try:
            start = time.monotonic()
            await self._request("GET", "/v5/market/time", {})
            latency = (time.monotonic() - start) * 1000
            return ExchangeStatus(
                exchange_id=ExchangeID.BYBIT, connected=True,
                latency_ms=latency,
            )
        except Exception as exc:
            return ExchangeStatus(
                exchange_id=ExchangeID.BYBIT, connected=False,
                last_error=str(exc),
            )
