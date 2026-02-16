"""
Darwin v4 — Binance USDⓈ-M Futures adapter.

Layer 2 (exchanges). Implements IExchangeAdapter.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set
from urllib.parse import urlencode

from darwin_agent.interfaces.enums import (
    ExchangeID, OrderSide, OrderType, TimeFrame,
)
from darwin_agent.interfaces.types import (
    Candle, ExchangeStatus, OrderRequest, OrderResult,
    Position, Ticker,
)

logger = logging.getLogger("darwin.binance")

_TF_MAP = {
    TimeFrame.M1: "1m", TimeFrame.M5: "5m", TimeFrame.M15: "15m",
    TimeFrame.M30: "30m", TimeFrame.H1: "1h", TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
}


def _normalize_symbol(symbol: str) -> str:
    # Accept formats like BTC/USDT, btcusdt, BTC-USDT
    return symbol.replace("/", "").replace("-", "").replace("_", "").upper()


class BinanceAdapter:
    """
    Production Binance USDⓈ-M Futures adapter.

    - HMAC-SHA256 auth
    - Rate limiting with retry
    - Hedge mode awareness
    """

    __slots__ = ("_api_key", "_api_secret", "_base_url", "_session", "_testnet", "_symbols_cache", "_symbols_cache_at")

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._base_url = (
            "https://testnet.binancefuture.com" if testnet
            else "https://fapi.binance.com"
        )
        self._session = None
        self._symbols_cache: Set[str] = set()
        self._symbols_cache_at: datetime | None = None

    @property
    def exchange_id(self) -> ExchangeID:
        return ExchangeID.BINANCE

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()

    def _sign(self, params: Dict) -> str:
        params["timestamp"] = str(int(time.time() * 1000))
        query = urlencode(sorted(params.items()))
        sig = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256,
        ).hexdigest()
        return query + "&signature=" + sig

    async def _public(self, path: str, params: Dict = None) -> Dict:
        await self._ensure_session()
        url = f"{self._base_url}{path}"
        async with self._session.get(url, params=params or {}) as resp:
            return await resp.json()

    async def _signed(self, method: str, path: str, params: Dict = None) -> Dict:
        await self._ensure_session()
        params = params or {}
        signed_qs = self._sign(params)
        url = f"{self._base_url}{path}?{signed_qs}"
        headers = {"X-MBX-APIKEY": self._api_key}
        for attempt in range(3):
            try:
                if method == "GET":
                    async with self._session.get(url, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with self._session.post(url, headers=headers) as resp:
                        data = await resp.json()
                if "code" in data and data["code"] < 0:
                    logger.warning("binance error: %s", data.get("msg"))
                return data
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    raise

    async def get_exchange_symbols(self, force_refresh: bool = False) -> Set[str]:
        """Fetch Binance futures tradable symbols and cache briefly."""
        now = datetime.now(timezone.utc)
        cache_ttl = timedelta(minutes=5)
        if (
            not force_refresh
            and self._symbols_cache
            and self._symbols_cache_at is not None
            and (now - self._symbols_cache_at) <= cache_ttl
        ):
            return set(self._symbols_cache)

        data = await self._public("/fapi/v1/exchangeInfo")
        symbols: Set[str] = set()
        for item in data.get("symbols", []):
            if item.get("status") != "TRADING":
                continue
            name = item.get("symbol")
            if name:
                symbols.add(_normalize_symbol(name))
        self._symbols_cache = symbols
        self._symbols_cache_at = now
        return set(symbols)

    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Return list of symbols not present/tradable on Binance futures."""
        available = await self.get_exchange_symbols()
        missing = []
        for raw in symbols:
            sym = _normalize_symbol(raw)
            if sym not in available:
                missing.append(raw)
        return missing

    # ── IExchangeAdapter ─────────────────────────────────────

    async def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]:
        data = await self._public("/fapi/v1/klines", {
            "symbol": _normalize_symbol(symbol),
            "interval": _TF_MAP.get(timeframe, "15m"),
            "limit": limit,
        })
        return [
            Candle(
                timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                open=float(row[1]), high=float(row[2]),
                low=float(row[3]), close=float(row[4]),
                volume=float(row[5]), timeframe=timeframe,
            )
            for row in data
        ]

    async def get_ticker(self, symbol: str) -> Ticker:
        symbol = _normalize_symbol(symbol)
        data = await self._public("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        price_data = await self._public("/fapi/v1/ticker/price", {"symbol": symbol})
        return Ticker(
            symbol=symbol,
            last_price=float(price_data.get("price", 0)),
            bid=float(data.get("bidPrice", 0)),
            ask=float(data.get("askPrice", 0)),
            volume_24h=0.0,
        )

    async def get_positions(self) -> List[Position]:
        data = await self._signed("GET", "/fapi/v2/positionRisk")
        positions = []
        if not isinstance(data, list):
            return positions
        for item in data:
            amt = float(item.get("positionAmt", 0))
            if amt == 0:
                continue
            positions.append(Position(
                symbol=item["symbol"],
                side=OrderSide.BUY if amt > 0 else OrderSide.SELL,
                size=abs(amt),
                entry_price=float(item.get("entryPrice", 0)),
                current_price=float(item.get("markPrice", 0)),
                unrealized_pnl=float(item.get("unRealizedProfit", 0)),
                leverage=int(item.get("leverage", 1)),
                exchange_id=ExchangeID.BINANCE,
            ))
        return positions

    async def place_order(self, request: OrderRequest) -> OrderResult:
        params = {
            "symbol": _normalize_symbol(request.symbol),
            "side": "BUY" if request.side == OrderSide.BUY else "SELL",
            "type": "MARKET" if request.order_type == OrderType.MARKET else "LIMIT",
            "quantity": str(request.quantity),
        }
        if request.price and request.order_type == OrderType.LIMIT:
            params["price"] = str(request.price)
            params["timeInForce"] = "GTC"
        if request.reduce_only:
            params["reduceOnly"] = "true"
        data = await self._signed("POST", "/fapi/v1/order", params)
        if "orderId" not in data:
            return OrderResult(
                success=False, error=data.get("msg", "unknown"),
                exchange_id=ExchangeID.BINANCE,
            )
        return OrderResult(
            order_id=str(data["orderId"]), symbol=request.symbol,
            side=request.side,
            filled_qty=float(data.get("executedQty", request.quantity)),
            filled_price=float(data.get("avgPrice", 0)),
            exchange_id=ExchangeID.BINANCE,
        )

    async def close_position(self, symbol: str, side: OrderSide) -> OrderResult:
        symbol = _normalize_symbol(symbol)
        positions = await self.get_positions()
        pos = next((p for p in positions if p.symbol == symbol and p.side == side), None)
        if not pos:
            return OrderResult(
                success=False, error="no_position",
                exchange_id=ExchangeID.BINANCE,
            )
        close_side = "SELL" if side == OrderSide.BUY else "BUY"
        params = {
            "symbol": _normalize_symbol(symbol), "side": close_side,
            "type": "MARKET", "quantity": str(pos.size),
            "reduceOnly": "true",
        }
        data = await self._signed("POST", "/fapi/v1/order", params)
        if "orderId" not in data:
            return OrderResult(
                success=False, error=data.get("msg", "unknown"),
                exchange_id=ExchangeID.BINANCE,
            )
        return OrderResult(
            order_id=str(data.get("orderId", "")), symbol=symbol,
            side=OrderSide(close_side),
            filled_qty=pos.size, filled_price=pos.current_price,
            exchange_id=ExchangeID.BINANCE,
        )

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            data = await self._signed("POST", "/fapi/v1/leverage", {
                "symbol": _normalize_symbol(symbol), "leverage": str(leverage),
            })
            if isinstance(data, dict) and data.get("code", 0) < 0:
                return False
            return True
        except Exception:
            return False

    async def get_balance(self) -> float:
        data = await self._signed("GET", "/fapi/v2/balance")
        if not isinstance(data, list):
            return 0.0
        for item in data:
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0))
        return 0.0

    async def get_status(self) -> ExchangeStatus:
        try:
            start = time.monotonic()
            await self._public("/fapi/v1/time")
            latency = (time.monotonic() - start) * 1000
            return ExchangeStatus(
                exchange_id=ExchangeID.BINANCE, connected=True,
                latency_ms=latency,
            )
        except Exception as exc:
            return ExchangeStatus(
                exchange_id=ExchangeID.BINANCE, connected=False,
                last_error=str(exc),
            )
