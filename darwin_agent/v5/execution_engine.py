"""
Darwin v5 — Enhanced Execution Engine.

Handles order placement with:
    - Order validation (min notional, step size, leverage enforcement)
    - Retry logic with exponential backoff on API failures
    - Time re-sync on 400 errors
    - Fill verification before proceeding
    - Slippage tolerance and metrics logging

Usage:
    engine = ExecutionEngine(binance_client, time_sync, telemetry)
    result = await engine.place_order(order_request)
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("darwin.v5.execution")


@dataclass
class ExecutionConfig:
    """Configuration for the execution engine."""
    leverage: int = 5
    max_retries: int = 2               # 2 retries (was 3) — scalping can't wait 14s total
    base_backoff_s: float = 2.0        # 2s base (was 1s) — gives Binance time to clear rate limits
    max_backoff_s: float = 30.0
    slippage_tolerance_pct: float = 0.5    # base tolerance; auto-scales for high-vol assets
    fill_timeout_s: float = 10.0
    min_notional_usdt: float = 5.0
    # Binance step sizes (common ones; fetched dynamically in production)
    default_step_size: float = 0.001
    # Pre-trade liquidity check: reject if order > X% of top-5 book depth
    max_book_impact_pct: float = 30.0   # order notional < 30% of visible book
    # Emergency close retries (more aggressive than normal orders)
    emergency_max_retries: int = 5
    # Maker vs Taker order preference
    # "LIMIT_ENTRY": Use LIMIT for entries (maker rebate), MARKET for exits (guaranteed fill)
    # "MARKET": Always MARKET (current behavior)
    #
    # Fee impact at VIP0:
    #   MARKET round-trip: 0.04% + 0.04% = 0.08% cost per trade
    #   LIMIT entry + MARKET exit: -0.02% + 0.04% = 0.02% cost per trade
    #   Savings per trade: 0.06% — at 20 trades/day = 1.2%/day recaptured
    preferred_entry_type: str = "LIMIT_ENTRY"
    # For LIMIT orders: how many basis points inside best bid/ask
    limit_offset_bps: float = 1.0
    # Max seconds to wait for LIMIT fill before falling back to MARKET
    limit_fill_timeout_s: float = 5.0


@dataclass(slots=True)
class OrderRequest:
    """Validated order request."""
    symbol: str = ""
    side: str = ""  # "BUY" or "SELL"
    quantity: float = 0.0
    price: float = 0.0  # mark price at signal time
    leverage: int = 5
    order_type: str = "MARKET"
    reduce_only: bool = False
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    """Result of order execution."""
    success: bool = False
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "filled_qty": round(self.filled_qty, 8),
            "avg_price": round(self.avg_price, 8),
            "fee": round(self.fee, 8),
            "slippage_bps": round(self.slippage_bps, 2),
            "latency_ms": round(self.latency_ms, 1),
            "retries": self.retries,
            "error": self.error,
        }


class ExecutionEngine:
    """
    Handles order placement, validation, and fill verification.

    Integrates with TimeSyncManager for 400-error recovery and
    TelemetryReporter for observability.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client.
    time_sync : TimeSyncManager, optional
        For timestamp re-sync on errors.
    telemetry : TelemetryReporter, optional
        For logging execution events.
    config : ExecutionConfig, optional
        Tunable parameters.
    """

    def __init__(
        self,
        binance_client: Any,
        time_sync: Any = None,
        telemetry: Any = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self._client = binance_client
        self._time_sync = time_sync
        self._telemetry = telemetry
        self._config = config or ExecutionConfig()
        self._total_orders: int = 0
        self._total_fills: int = 0
        self._total_rejects: int = 0
        # Symbol info cache
        self._symbol_info: Dict[str, Dict[str, Any]] = {}

    def validate_order(self, order: OrderRequest) -> tuple[bool, str]:
        """
        Validate an order before placement.

        Returns (valid, reason).
        """
        cfg = self._config

        if not order.symbol:
            return False, "missing symbol"
        if order.side not in ("BUY", "SELL"):
            return False, f"invalid side: {order.side}"
        if order.quantity <= 0:
            return False, f"quantity must be > 0, got {order.quantity}"
        if order.price <= 0:
            return False, f"price must be > 0, got {order.price}"

        # Min notional check
        notional = order.quantity * order.price
        if notional < cfg.min_notional_usdt:
            return False, f"notional ${notional:.2f} < min ${cfg.min_notional_usdt}"

        # Leverage enforcement
        if order.leverage > cfg.leverage:
            return False, f"leverage {order.leverage}x > max {cfg.leverage}x"

        # Step size
        step = self._get_step_size(order.symbol)
        if step > 0:
            order.quantity = self._round_step(order.quantity, step)
            if order.quantity <= 0:
                return False, f"quantity rounds to 0 with step size {step}"

        return True, ""

    def check_liquidity(self, symbol: str, side: str, notional_usdt: float) -> tuple[bool, str]:
        """
        Pre-trade liquidity check against order book depth.

        Prevents sending a MARKET order that would eat through the book
        and cause catastrophic slippage (e.g. a $500 order on a $200 book).

        Uses 20 levels (not 5) to detect liquidity cliffs on thin memecoin
        books where top-5 looks fine but levels 6-20 have nothing.

        Returns (safe, reason).
        """
        cfg = self._config
        try:
            book = self._client.get_order_book_depth(symbol, limit=20)
        except Exception:
            # If we can't check the book, allow the trade (fail-open for availability)
            return True, "book_check_failed"

        levels = book.get("asks" if side == "BUY" else "bids", [])
        if not levels:
            return True, "empty_book"

        # Sum available liquidity across all visible levels
        total_liquidity = sum(
            float(level[0]) * float(level[1])
            for level in levels
            if len(level) >= 2
        )

        if total_liquidity <= 0:
            return True, "zero_liquidity"

        impact_pct = (notional_usdt / total_liquidity) * 100.0
        if impact_pct > cfg.max_book_impact_pct:
            return False, (
                f"book_impact_too_high: order ${notional_usdt:.2f} = "
                f"{impact_pct:.1f}% of visible book ${total_liquidity:.2f} "
                f"(max {cfg.max_book_impact_pct}%)"
            )

        # Liquidity cliff detection: if top-5 has >80% of total-20 liquidity,
        # the book drops off sharply and slippage risk is elevated.
        if len(levels) > 5:
            top5_liq = sum(
                float(level[0]) * float(level[1])
                for level in levels[:5]
                if len(level) >= 2
            )
            if total_liquidity > 0 and top5_liq / total_liquidity > 0.80:
                # Reduce impact threshold to 15% on thin books
                if impact_pct > cfg.max_book_impact_pct * 0.5:
                    return False, (
                        f"liquidity_cliff: top5={top5_liq:.0f} is "
                        f"{top5_liq/total_liquidity*100:.0f}% of total={total_liquidity:.0f} "
                        f"(order impact {impact_pct:.1f}%)"
                    )

        return True, ""

    async def place_order(
        self, order: OrderRequest, dry_run: bool = False
    ) -> ExecutionResult:
        """
        Place an order with validation, retry, and fill verification.

        Parameters
        ----------
        order : OrderRequest
            The order to place.
        dry_run : bool
            If True, validate but don't place the order.

        Returns
        -------
        ExecutionResult
            Execution outcome.
        """
        self._total_orders += 1

        # Validate
        valid, reason = self.validate_order(order)
        if not valid:
            self._total_rejects += 1
            return ExecutionResult(
                success=False,
                symbol=order.symbol,
                side=order.side,
                error=f"validation_failed: {reason}",
            )

        if dry_run:
            return ExecutionResult(
                success=True,
                symbol=order.symbol,
                side=order.side,
                filled_qty=order.quantity,
                avg_price=order.price,
                order_id="DRY_RUN",
            )

        # Pre-trade liquidity check
        if not order.reduce_only:
            notional = order.quantity * order.price
            liq_ok, liq_reason = self.check_liquidity(order.symbol, order.side, notional)
            if not liq_ok:
                self._total_rejects += 1
                logger.warning("liquidity check failed for %s: %s", order.symbol, liq_reason)
                return ExecutionResult(
                    success=False,
                    symbol=order.symbol,
                    side=order.side,
                    error=f"liquidity_check_failed: {liq_reason}",
                )

        # Log pre-trade
        if self._telemetry:
            self._telemetry.log_order_placed(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                leverage=order.leverage,
            )

        # Execute with retry
        result = await self._execute_with_retry(order)

        # Log result
        if result.success and self._telemetry:
            self._total_fills += 1
            self._telemetry.log_order_filled(
                symbol=result.symbol,
                side=result.side,
                filled_qty=result.filled_qty,
                avg_price=result.avg_price,
                fee=result.fee,
                order_id=result.order_id,
                slippage_bps=result.slippage_bps,
            )
        elif not result.success:
            self._total_rejects += 1

        return result

    async def _execute_with_retry(self, order: OrderRequest) -> ExecutionResult:
        """Execute order with exponential backoff retry.

        For new entries (not reduceOnly), uses LIMIT POST_ONLY when configured
        to capture maker rebates. Falls back to MARKET if LIMIT doesn't fill
        within limit_fill_timeout_s.

        Fee savings at VIP0:
            MARKET round-trip: 0.04% + 0.04% = 0.08% per trade
            LIMIT entry + MARKET exit: -0.02% + 0.04% = 0.02% per trade
            Net savings: 0.06% per trade = 0.6% per 10 trades
            At 20 trades/day on $80 capital: ~$0.96/day recaptured
        """
        cfg = self._config

        # Attempt Post-Only LIMIT for entries (not reduceOnly, not exits)
        if (
            cfg.preferred_entry_type == "LIMIT_ENTRY"
            and not order.reduce_only
            and order.order_type == "MARKET"
        ):
            limit_result = await self._attempt_limit_entry(order)
            if limit_result is not None:
                return limit_result
            # LIMIT didn't fill — fall through to MARKET

        last_error = ""

        for attempt in range(cfg.max_retries + 1):
            t_start = time.time()

            try:
                result = await asyncio.to_thread(
                    self._place_on_exchange, order
                )
                latency_ms = (time.time() - t_start) * 1000.0
                result.latency_ms = latency_ms
                result.retries = attempt

                # Compute slippage
                if result.success and order.price > 0:
                    slippage_pct = abs(result.avg_price - order.price) / order.price * 100.0
                    result.slippage_bps = slippage_pct * 100.0

                    # Adaptive slippage tolerance: for high-vol assets (memecoins),
                    # use max(base_tolerance, 50% of ATR%) to avoid false alarms.
                    # ATR info is passed via order metadata from the engine.
                    atr_pct = order.metadata.get("atr_pct", 0.0)
                    effective_tolerance = max(
                        cfg.slippage_tolerance_pct,
                        atr_pct * 50.0,  # 50% of ATR% (e.g., ATR=5% → tolerance=2.5%)
                    )

                    if slippage_pct > effective_tolerance:
                        logger.warning(
                            "high slippage on %s: %.2f%% (tolerance: %.2f%%)",
                            order.symbol,
                            slippage_pct,
                            effective_tolerance,
                        )

                return result

            except Exception as exc:
                last_error = str(exc)
                latency_ms = (time.time() - t_start) * 1000.0

                # Binance-specific error handling based on parsed code.
                # NOTE: if response.json() fails (empty body), requests raises a plain
                # "400 Client Error: Bad Request" without a Binance code.
                # We treat any 400 as a potential timestamp issue and always re-sync.
                # code=-1021: timestamp outside recvWindow → re-sync time
                # code=-1003: rate limit hit → long backoff, don't hammer API
                # code=-2010: insufficient balance → don't retry (won't fix itself)
                is_400 = (
                    "code=-1021" in last_error
                    or "timestamp" in last_error.lower()
                    or "400 Client Error" in last_error   # generic Bad Request fallback
                    or "Bad Request" in last_error
                )
                # 502/503: Binance server-side issue — always retry with longer backoff
                is_502_503 = any(
                    code in last_error for code in ("502", "503", "binance_502", "binance_503")
                )
                if is_502_503:
                    logger.warning("binance 502/503 on %s, retrying with extended backoff", order.symbol)
                    # Extended backoff for server errors
                    if attempt < cfg.max_retries:
                        await asyncio.sleep(min(3.0 * (2 ** attempt), 30.0))
                    continue
                elif is_400:
                    if self._time_sync:
                        logger.warning("time sync triggered on 400 error, re-syncing clock")
                        self._time_sync.force_sync_on_error(400)
                elif "code=-1003" in last_error:
                    logger.warning("rate limit hit (-1003), backing off 30s")
                    await asyncio.sleep(30.0)
                    break  # don't retry further, let next tick handle it
                elif "code=-2010" in last_error:
                    logger.error("insufficient balance (-2010), aborting retries")
                    break  # no point retrying, balance won't change
                elif "code=-4131" in last_error or "PERCENT_PRICE" in last_error:
                    logger.error("price filter violation (-4131) on %s, aborting", order.symbol)
                    break  # price moved too far, retrying same price is futile
                else:
                    # Any other Binance error — log it clearly and still retry
                    logger.warning("binance error: %s", last_error)

                if attempt < cfg.max_retries:
                    backoff = min(
                        cfg.base_backoff_s * (2 ** attempt),
                        cfg.max_backoff_s,
                    )
                    logger.warning(
                        "order failed (attempt %d/%d): %s, retrying in %.1fs",
                        attempt + 1,
                        cfg.max_retries + 1,
                        last_error,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

        return ExecutionResult(
            success=False,
            symbol=order.symbol,
            side=order.side,
            error=f"max_retries_exceeded: {last_error}",
            retries=cfg.max_retries,
        )

    def _place_on_exchange(self, order: OrderRequest) -> ExecutionResult:
        """Place the order on the exchange (blocking)."""
        # Format quantity with correct precision based on step size
        step = self._get_step_size(order.symbol)
        precision = max(0, round(-math.log10(step))) if step > 0 else 3
        quantity_str = f"{order.quantity:.{precision}f}"

        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "quantity": quantity_str,
        }

        if order.reduce_only:
            params["reduceOnly"] = "true"

        # Use time-synced timestamp
        if self._time_sync:
            params["timestamp"] = self._time_sync.synced_timestamp()
        else:
            params["timestamp"] = int(time.time() * 1000)

        params["recvWindow"] = 10000  # increased from 5000 for high-latency servers (rtt~248ms)

        # Sign and send
        client = self._client
        params["signature"] = client._sign(params)
        url = f"{client._base_url}/fapi/v1/order"
        response = client._session.request(
            "POST", url, params=params, timeout=client._timeout_s
        )

        # Parse Binance error code BEFORE raise_for_status so we log the real reason.
        # raise_for_status() only gives "400 Bad Request" without the Binance code
        # (e.g. -1021 timestamp, -1003 rate limit, -2010 balance, -1100 signature).
        if not response.ok:
            try:
                err_body = response.json()
                binance_code = err_body.get("code", 0)
                binance_msg = err_body.get("msg", "unknown")
                raise RuntimeError(
                    f"binance_{response.status_code} code={binance_code} msg={binance_msg}"
                )
            except (ValueError, KeyError):
                response.raise_for_status()

        data = response.json()

        # Parse fill
        filled_qty = float(data.get("executedQty", 0.0))
        avg_price = float(data.get("avgPrice", 0.0))
        order_id = str(data.get("orderId", ""))
        status = data.get("status", "UNKNOWN")

        # MARKET orders with status=NEW mean the fill is still processing.
        # Poll up to 3 times (300ms each) until FILLED or CANCELED.
        if order.order_type == "MARKET" and status == "NEW" and order_id:
            for _ in range(3):
                time.sleep(0.3)
                try:
                    check_params = {
                        "symbol": order.symbol,
                        "orderId": order_id,
                    }
                    check_params["timestamp"] = (
                        self._time_sync.synced_timestamp()
                        if self._time_sync
                        else int(time.time() * 1000)
                    )
                    check_params["recvWindow"] = 10000
                    check_params["signature"] = client._sign(check_params)
                    check_url = f"{client._base_url}/fapi/v1/order"
                    check_resp = client._session.get(
                        check_url, params=check_params, timeout=client._timeout_s
                    )
                    check_resp.raise_for_status()
                    check_data = check_resp.json()
                    filled_qty = float(check_data.get("executedQty", filled_qty))
                    avg_price = float(check_data.get("avgPrice", avg_price))
                    status = check_data.get("status", status)
                    if status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                        break
                except Exception:
                    break  # keep whatever we have

        if avg_price == 0 and filled_qty > 0:
            avg_price = float(data.get("price", order.price))

        error_msg = ""
        if filled_qty == 0:
            error_msg = f"zero_fill: status={status} orderId={order_id}"

        return ExecutionResult(
            success=filled_qty > 0,
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=0.0,  # Binance includes fee in separate endpoint
            error=error_msg,
        )

    async def _attempt_limit_entry(self, order: OrderRequest) -> Optional[ExecutionResult]:
        """
        Attempt a LIMIT POST_ONLY entry to capture maker rebate.

        Places a LIMIT order slightly inside the best bid/ask:
        - BUY: price = best_bid + offset_bps (join the bid, slightly aggressive)
        - SELL: price = best_ask - offset_bps (join the ask, slightly aggressive)

        Post-Only (timeInForce=GTX on Binance) guarantees maker execution:
        if the order would take liquidity, Binance rejects it instead of filling
        as taker. This prevents accidental taker fees on a LIMIT order.

        Returns ExecutionResult if filled, None if not filled (caller falls back to MARKET).
        """
        cfg = self._config
        t_start = time.time()

        try:
            # Fetch best bid/ask for limit price computation
            book = await asyncio.to_thread(
                self._client.get_order_book_depth, order.symbol, 5
            )

            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids or not asks:
                return None  # no book data, fall back to MARKET

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            if best_bid <= 0 or best_ask <= 0:
                return None

            # Compute limit price with offset
            offset_mult = cfg.limit_offset_bps / 10000.0
            if order.side == "BUY":
                # Place slightly above best bid (more aggressive, higher fill chance)
                limit_price = best_bid * (1.0 + offset_mult)
                # But never above best ask (that would be a taker)
                limit_price = min(limit_price, best_ask - 0.01)
            else:
                # Place slightly below best ask
                limit_price = best_ask * (1.0 - offset_mult)
                # But never below best bid
                limit_price = max(limit_price, best_bid + 0.01)

            if limit_price <= 0:
                return None

            # Place LIMIT POST_ONLY (GTX = Good Till Crossing)
            result = await asyncio.to_thread(
                self._place_limit_post_only, order, limit_price
            )

            if result is None:
                return None

            latency_ms = (time.time() - t_start) * 1000.0
            result.latency_ms = latency_ms

            if result.success:
                # Compute slippage vs original signal price
                if order.price > 0:
                    slippage_pct = abs(result.avg_price - order.price) / order.price * 100.0
                    result.slippage_bps = slippage_pct * 100.0

                logger.info(
                    "POST_ONLY LIMIT filled: %s %s qty=%.6f limit=%.4f avg=%.4f (saved taker fee)",
                    order.side, order.symbol, result.filled_qty, limit_price, result.avg_price,
                )
                return result

            # LIMIT was rejected or expired — check if it's still open
            if result.order_id and result.error and "POST_ONLY_REJECT" not in result.error:
                # Order exists but not filled yet — wait for fill
                filled = await self._wait_for_limit_fill(order.symbol, result.order_id)
                if filled and filled.success:
                    filled.latency_ms = (time.time() - t_start) * 1000.0
                    return filled
                # Timed out — cancel and fall back to MARKET
                await asyncio.to_thread(
                    self._client.cancel_order, order.symbol, result.order_id
                )

            return None  # fall back to MARKET

        except Exception as exc:
            logger.debug(
                "LIMIT entry attempt failed for %s, falling back to MARKET: %s",
                order.symbol, exc,
            )
            return None

    def _place_limit_post_only(
        self, order: OrderRequest, limit_price: float
    ) -> Optional[ExecutionResult]:
        """Place a LIMIT POST_ONLY order on the exchange (blocking)."""
        step = self._get_step_size(order.symbol)
        precision = max(0, round(-math.log10(step))) if step > 0 else 3
        quantity_str = f"{order.quantity:.{precision}f}"

        # Format price to tick size precision
        tick_size = self._symbol_info.get(order.symbol, {}).get("tick_size", 0.01)
        if tick_size <= 0:
            tick_size = 0.01
        price_precision = max(0, round(-math.log10(tick_size)))
        price_str = f"{limit_price:.{price_precision}f}"

        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": "LIMIT",
            "quantity": quantity_str,
            "price": price_str,
            "timeInForce": "GTX",  # Post-Only: reject if would be taker
        }

        if self._time_sync:
            params["timestamp"] = self._time_sync.synced_timestamp()
        else:
            params["timestamp"] = int(time.time() * 1000)

        params["recvWindow"] = 10000

        client = self._client
        params["signature"] = client._sign(params)
        url = f"{client._base_url}/fapi/v1/order"

        try:
            response = client._session.request(
                "POST", url, params=params, timeout=client._timeout_s
            )

            if not response.ok:
                try:
                    err_body = response.json()
                    binance_code = err_body.get("code", 0)
                    binance_msg = err_body.get("msg", "unknown")
                    # -5022 = "Order would immediately trigger" (GTX rejected)
                    if binance_code == -5022 or "would immediately" in str(binance_msg).lower():
                        return ExecutionResult(
                            success=False,
                            symbol=order.symbol,
                            side=order.side,
                            error="POST_ONLY_REJECT: would take liquidity",
                        )
                    return None  # other error, fall back to MARKET
                except (ValueError, KeyError):
                    return None

            data = response.json()
            filled_qty = float(data.get("executedQty", 0.0))
            avg_price = float(data.get("avgPrice", 0.0))
            order_id = str(data.get("orderId", ""))
            status = data.get("status", "UNKNOWN")

            if avg_price == 0 and filled_qty > 0:
                avg_price = limit_price

            return ExecutionResult(
                success=filled_qty > 0 and status == "FILLED",
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                filled_qty=filled_qty,
                avg_price=avg_price,
                error="" if filled_qty > 0 else f"limit_unfilled:{status}",
            )
        except Exception:
            return None

    async def _wait_for_limit_fill(
        self, symbol: str, order_id: str
    ) -> Optional[ExecutionResult]:
        """Poll for LIMIT order fill within timeout."""
        cfg = self._config
        deadline = time.time() + cfg.limit_fill_timeout_s
        client = self._client

        while time.time() < deadline:
            await asyncio.sleep(0.5)
            try:
                check_params: Dict[str, Any] = {
                    "symbol": symbol,
                    "orderId": order_id,
                }
                check_params["timestamp"] = (
                    self._time_sync.synced_timestamp()
                    if self._time_sync
                    else int(time.time() * 1000)
                )
                check_params["recvWindow"] = 10000
                check_params["signature"] = client._sign(check_params)
                check_url = f"{client._base_url}/fapi/v1/order"
                check_resp = client._session.get(
                    check_url, params=check_params, timeout=client._timeout_s
                )
                check_resp.raise_for_status()
                data = check_resp.json()

                status = data.get("status", "UNKNOWN")
                if status == "FILLED":
                    return ExecutionResult(
                        success=True,
                        order_id=order_id,
                        symbol=symbol,
                        side=data.get("side", ""),
                        filled_qty=float(data.get("executedQty", 0)),
                        avg_price=float(data.get("avgPrice", 0)),
                    )
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    return None
            except Exception:
                break

        return None

    def _get_step_size(self, symbol: str) -> float:
        """Get quantity step size for a symbol."""
        if symbol in self._symbol_info:
            return self._symbol_info[symbol].get("step_size", self._config.default_step_size)
        return self._config.default_step_size

    @staticmethod
    def _round_step(quantity: float, step: float) -> float:
        """Round quantity down to nearest step size."""
        if step <= 0:
            return quantity
        # Round quantity to the step's precision without float drift
        return math.floor(quantity / step) * step

    def load_symbol_info(self, step_sizes: Dict[str, float]) -> None:
        """Load symbol step sizes fetched from exchange info."""
        for symbol, step in step_sizes.items():
            if symbol not in self._symbol_info:
                self._symbol_info[symbol] = {}
            self._symbol_info[symbol]["step_size"] = step
        logger.info("loaded step sizes for %d symbols", len(step_sizes))

    def load_tick_sizes(self, tick_sizes: Dict[str, float]) -> None:
        """Load price tick sizes for LIMIT order formatting."""
        for symbol, tick in tick_sizes.items():
            if symbol not in self._symbol_info:
                self._symbol_info[symbol] = {}
            self._symbol_info[symbol]["tick_size"] = tick
        logger.info("loaded tick sizes for %d symbols", len(tick_sizes))

    def get_metrics(self) -> Dict[str, Any]:
        """Return execution metrics."""
        return {
            "total_orders": self._total_orders,
            "total_fills": self._total_fills,
            "total_rejects": self._total_rejects,
            "fill_rate_pct": (
                self._total_fills / self._total_orders * 100.0
                if self._total_orders > 0
                else 0.0
            ),
        }
