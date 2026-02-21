"""
Darwin v5 — Binance WebSocket User Data Stream.

Provides real-time notifications for:
    - ORDER_TRADE_UPDATE: fills, cancellations, expirations
    - ACCOUNT_UPDATE: margin changes, position updates, balance changes
    - MARGIN_CALL: pre-liquidation warning

Without this, the bot relies on REST polling every 5 seconds, creating:
    - 5+ second latency on fill detection
    - Orphaned positions if server-side SL triggers between polls
    - Missed margin calls during fast moves
    - No awareness of account changes from external sources

Architecture:
    1. POST /fapi/v1/listenKey → get listen key
    2. Connect wss://fstream.binance.com/ws/<listenKey>
    3. PUT /fapi/v1/listenKey every 30 min to keep alive
    4. Parse events and invoke registered callbacks
    5. Auto-reconnect with exponential backoff on disconnect

Usage:
    stream = BinanceUserDataStream(binance_client)
    stream.on_order_update(handle_fill)
    stream.on_account_update(handle_account)
    stream.on_margin_call(handle_margin)
    await stream.start()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("darwin.v5.websocket")

# Binance Futures WebSocket base URL
WS_BASE_URL = "wss://fstream.binance.com/ws/"
WS_TESTNET_URL = "wss://stream.binancefuture.com/ws/"

# Keep-alive interval (Binance requires PUT every 60 min, we do 30 min)
KEEPALIVE_INTERVAL_S = 1800.0

# Reconnect backoff parameters
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 60.0
RECONNECT_MAX_ATTEMPTS = 50  # ~24 min of retrying before giving up


@dataclass(slots=True)
class OrderUpdate:
    """Parsed ORDER_TRADE_UPDATE event."""
    symbol: str = ""
    order_id: str = ""
    client_order_id: str = ""
    side: str = ""          # BUY / SELL
    order_type: str = ""    # MARKET / LIMIT / STOP_MARKET / TAKE_PROFIT_MARKET
    status: str = ""        # NEW / PARTIALLY_FILLED / FILLED / CANCELED / EXPIRED
    price: float = 0.0
    avg_price: float = 0.0
    stop_price: float = 0.0
    quantity: float = 0.0
    filled_qty: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    trade_time: int = 0
    is_reduce_only: bool = False
    is_close_position: bool = False


@dataclass(slots=True)
class AccountUpdate:
    """Parsed ACCOUNT_UPDATE event."""
    event_reason: str = ""   # DEPOSIT / WITHDRAW / ORDER / FUNDING_FEE / etc.
    balances: List[Dict[str, float]] = field(default_factory=list)
    positions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MarginCallUpdate:
    """Parsed MARGIN_CALL event."""
    positions: List[Dict[str, Any]] = field(default_factory=list)


# Callback types
OrderCallback = Callable[[OrderUpdate], None]
AccountCallback = Callable[[AccountUpdate], None]
MarginCallback = Callable[[MarginCallUpdate], None]


class BinanceUserDataStream:
    """
    WebSocket client for Binance Futures user data stream.

    Manages listen key lifecycle, WebSocket connection, reconnection,
    and dispatches parsed events to registered callbacks.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        For listen key management (REST calls).
    testnet : bool
        Use testnet WebSocket URL.
    """

    def __init__(
        self,
        binance_client: Any,
        testnet: bool = False,
    ) -> None:
        self._client = binance_client
        self._ws_base = WS_TESTNET_URL if testnet else WS_BASE_URL
        self._listen_key: str = ""
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._session: Any = None  # aiohttp.ClientSession
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._last_message_ts: float = 0.0

        # Callbacks
        self._order_callbacks: List[OrderCallback] = []
        self._account_callbacks: List[AccountCallback] = []
        self._margin_callbacks: List[MarginCallback] = []

        # Tasks
        self._ws_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # Stats
        self._messages_received: int = 0
        self._reconnections: int = 0

    # ── Callback registration ─────────────────────────────────────

    def on_order_update(self, callback: OrderCallback) -> None:
        self._order_callbacks.append(callback)

    def on_account_update(self, callback: AccountCallback) -> None:
        self._account_callbacks.append(callback)

    def on_margin_call(self, callback: MarginCallback) -> None:
        self._margin_callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket connection and keepalive loop."""
        if self._running:
            return
        self._running = True
        self._ws_task = asyncio.create_task(self._connection_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info("WebSocket user data stream started")

    async def stop(self) -> None:
        """Gracefully stop the WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._ws_task:
            self._ws_task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False
        logger.info("WebSocket user data stream stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_diagnostics(self) -> dict:
        return {
            "connected": self._connected,
            "messages_received": self._messages_received,
            "reconnections": self._reconnections,
            "last_message_age_s": round(time.time() - self._last_message_ts, 1)
            if self._last_message_ts > 0 else -1,
            "listen_key": self._listen_key[:8] + "..." if self._listen_key else "",
        }

    # ── Listen Key Management ─────────────────────────────────────

    def _create_listen_key(self) -> str:
        """POST /fapi/v1/listenKey → returns listen key string."""
        session = self._client._session
        base_url = self._client._base_url
        response = session.post(
            f"{base_url}/fapi/v1/listenKey",
            timeout=self._client._timeout_s,
        )
        response.raise_for_status()
        key = response.json().get("listenKey", "")
        if not key:
            raise RuntimeError("Empty listenKey from Binance")
        logger.info("listen key created: %s...", key[:8])
        return key

    def _keepalive_listen_key(self) -> None:
        """PUT /fapi/v1/listenKey → extend listen key expiry."""
        session = self._client._session
        base_url = self._client._base_url
        response = session.put(
            f"{base_url}/fapi/v1/listenKey",
            timeout=self._client._timeout_s,
        )
        response.raise_for_status()
        logger.debug("listen key keepalive sent")

    # ── Connection Loop ───────────────────────────────────────────

    async def _connection_loop(self) -> None:
        """Main connection loop with auto-reconnect."""
        import aiohttp

        while self._running:
            try:
                # Get listen key (blocking REST call)
                self._listen_key = await asyncio.to_thread(self._create_listen_key)
                ws_url = f"{self._ws_base}{self._listen_key}"

                # Connect WebSocket
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()

                async with self._session.ws_connect(
                    ws_url,
                    heartbeat=20.0,  # aiohttp will send ping every 20s
                    receive_timeout=60.0,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_count = 0
                    logger.info("WebSocket connected to %s...%s", ws_url[:40], ws_url[-8:])

                    # Read messages until disconnect
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._messages_received += 1
                            self._last_message_ts = time.time()
                            self._dispatch(msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.warning("WebSocket error: %s", ws.exception())
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                            logger.warning("WebSocket close frame received")
                            break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WebSocket connection error: %s", exc)

            # Reconnect with backoff
            self._connected = False
            self._reconnections += 1
            if not self._running:
                break
            self._reconnect_count += 1
            if self._reconnect_count > RECONNECT_MAX_ATTEMPTS:
                logger.critical(
                    "WebSocket reconnect exhausted after %d attempts",
                    RECONNECT_MAX_ATTEMPTS,
                )
                break
            backoff = min(RECONNECT_BASE_S * (2 ** (self._reconnect_count - 1)), RECONNECT_MAX_S)
            logger.info("WebSocket reconnecting in %.0fs (attempt %d)", backoff, self._reconnect_count)
            await asyncio.sleep(backoff)

    async def _keepalive_loop(self) -> None:
        """Periodically extend listen key expiry."""
        while self._running:
            try:
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                if self._listen_key:
                    await asyncio.to_thread(self._keepalive_listen_key)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("listen key keepalive failed: %s", exc)

    # ── Event Dispatch ────────────────────────────────────────────

    def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("invalid WebSocket JSON: %s", raw[:200])
            return

        event_type = data.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_update(data)
        elif event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(data)
        elif event_type == "MARGIN_CALL":
            self._handle_margin_call(data)
        elif event_type == "listenKeyExpired":
            logger.warning("listen key expired — forcing reconnect")
            self._connected = False
            if self._ws and not self._ws.closed:
                asyncio.create_task(self._ws.close())

    def _handle_order_update(self, data: dict) -> None:
        """Parse ORDER_TRADE_UPDATE and invoke callbacks."""
        o = data.get("o", {})
        update = OrderUpdate(
            symbol=o.get("s", ""),
            order_id=str(o.get("i", "")),
            client_order_id=o.get("c", ""),
            side=o.get("S", ""),
            order_type=o.get("o", ""),
            status=o.get("X", ""),
            price=float(o.get("p", 0)),
            avg_price=float(o.get("ap", 0)),
            stop_price=float(o.get("sp", 0)),
            quantity=float(o.get("q", 0)),
            filled_qty=float(o.get("z", 0)),
            realized_pnl=float(o.get("rp", 0)),
            commission=float(o.get("n", 0)),
            commission_asset=o.get("N", ""),
            trade_time=int(o.get("T", 0)),
            is_reduce_only=o.get("R", False),
            is_close_position=o.get("cp", False),
        )

        log_level = logging.INFO if update.status in ("FILLED", "CANCELED", "EXPIRED") else logging.DEBUG
        logger.log(
            log_level,
            "WS ORDER_UPDATE: %s %s %s qty=%.6f avg=%.4f status=%s rpnl=%.4f",
            update.symbol, update.side, update.order_type,
            update.filled_qty, update.avg_price, update.status, update.realized_pnl,
        )

        for cb in self._order_callbacks:
            try:
                cb(update)
            except Exception as exc:
                logger.error("order callback error: %s", exc)

    def _handle_account_update(self, data: dict) -> None:
        """Parse ACCOUNT_UPDATE and invoke callbacks."""
        a = data.get("a", {})
        update = AccountUpdate(
            event_reason=a.get("m", ""),
            balances=[
                {"asset": b.get("a", ""), "balance": float(b.get("wb", 0)),
                 "cross_wallet": float(b.get("cw", 0))}
                for b in a.get("B", [])
            ],
            positions=[
                {"symbol": p.get("s", ""), "amount": float(p.get("pa", 0)),
                 "entry_price": float(p.get("ep", 0)),
                 "unrealized_pnl": float(p.get("up", 0)),
                 "margin_type": p.get("mt", "")}
                for p in a.get("P", [])
            ],
        )

        logger.info(
            "WS ACCOUNT_UPDATE: reason=%s balances=%d positions=%d",
            update.event_reason, len(update.balances), len(update.positions),
        )

        for cb in self._account_callbacks:
            try:
                cb(update)
            except Exception as exc:
                logger.error("account callback error: %s", exc)

    def _handle_margin_call(self, data: dict) -> None:
        """Parse MARGIN_CALL and invoke callbacks. THIS IS CRITICAL."""
        update = MarginCallUpdate(
            positions=[
                {"symbol": p.get("s", ""), "side": p.get("ps", ""),
                 "amount": float(p.get("pa", 0)),
                 "margin_type": p.get("mt", ""),
                 "unrealized_pnl": float(p.get("up", 0)),
                 "maintenance_margin": float(p.get("mm", 0))}
                for p in data.get("p", [])
            ]
        )

        logger.critical(
            "WS MARGIN_CALL: %d positions at risk — LIQUIDATION IMMINENT",
            len(update.positions),
        )

        for cb in self._margin_callbacks:
            try:
                cb(update)
            except Exception as exc:
                logger.error("margin call callback error: %s", exc)
