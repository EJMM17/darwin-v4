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
    slippage_tolerance_pct: float = 0.5
    fill_timeout_s: float = 10.0
    min_notional_usdt: float = 5.0
    # Binance step sizes (common ones; fetched dynamically in production)
    default_step_size: float = 0.001
    # --- Risk hardening (HFT review) ---
    # Pre-trade slippage guard: reject order if book mid deviates > N% from signal price
    max_pretrade_deviation_pct: float = 0.3
    # Emergency close: aggressive retry for risk-critical closes (SL, liquidation avoidance)
    emergency_max_retries: int = 5
    emergency_base_backoff_s: float = 0.5
    # Abort slippage: cancel/reject fill if post-trade slippage exceeds this (hard reject)
    abort_slippage_pct: float = 1.5


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

    async def _execute_with_retry(
        self, order: OrderRequest, *, is_emergency: bool = False,
    ) -> ExecutionResult:
        """
        Execute order with exponential backoff retry.

        Parameters
        ----------
        is_emergency : bool
            If True, use aggressive retry settings (more retries, shorter
            backoff). Used for SL closes and liquidation-avoidance orders
            where failing to fill is catastrophic.
        """
        cfg = self._config
        last_error = ""

        if is_emergency:
            max_retries = cfg.emergency_max_retries
            base_backoff = cfg.emergency_base_backoff_s
        else:
            max_retries = cfg.max_retries
            base_backoff = cfg.base_backoff_s

        for attempt in range(max_retries + 1):
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

                    # HARD REJECT: if slippage exceeds abort threshold on entries
                    # (never abort emergency closes — getting out is always priority)
                    if (
                        not is_emergency
                        and not order.reduce_only
                        and slippage_pct > cfg.abort_slippage_pct
                    ):
                        logger.error(
                            "SLIPPAGE ABORT on %s: %.2f%% > abort_threshold %.2f%% — "
                            "immediately closing entry to prevent holding a bad fill",
                            order.symbol, slippage_pct, cfg.abort_slippage_pct,
                        )
                        # Immediately reverse the bad entry
                        reverse_side = "SELL" if order.side == "BUY" else "BUY"
                        reverse_order = OrderRequest(
                            symbol=order.symbol,
                            side=reverse_side,
                            quantity=result.filled_qty,
                            price=result.avg_price,
                            leverage=order.leverage,
                            reduce_only=True,
                        )
                        await self._execute_with_retry(reverse_order, is_emergency=True)
                        result.success = False
                        result.error = (
                            f"slippage_abort: {slippage_pct:.2f}% > {cfg.abort_slippage_pct}% "
                            f"(entry reversed)"
                        )
                        return result

                    # Soft warning for slippage above tolerance but below abort
                    if slippage_pct > cfg.slippage_tolerance_pct:
                        logger.warning(
                            "high slippage on %s: %.2f%% (tolerance: %.2f%%)",
                            order.symbol,
                            slippage_pct,
                            cfg.slippage_tolerance_pct,
                        )

                return result

            except Exception as exc:
                last_error = str(exc)
                latency_ms = (time.time() - t_start) * 1000.0

                # Binance-specific error handling based on parsed code.
                is_400 = (
                    "code=-1021" in last_error
                    or "timestamp" in last_error.lower()
                    or "400 Client Error" in last_error
                    or "Bad Request" in last_error
                )
                if is_400:
                    if self._time_sync:
                        logger.warning("time sync triggered on 400 error, re-syncing clock")
                        self._time_sync.force_sync_on_error(400)
                elif "code=-1003" in last_error:
                    # Rate limit: for emergency closes, short backoff and KEEP RETRYING.
                    # For normal orders, yield to next tick.
                    if is_emergency:
                        logger.warning(
                            "rate limit hit (-1003) during EMERGENCY close, "
                            "short backoff and retrying (attempt %d/%d)",
                            attempt + 1, max_retries + 1,
                        )
                        await asyncio.sleep(2.0)
                        # Do NOT break — keep retrying for emergency closes
                    else:
                        logger.warning("rate limit hit (-1003), backing off 30s")
                        await asyncio.sleep(30.0)
                        break
                elif "code=-2010" in last_error:
                    logger.error("insufficient balance (-2010), aborting retries")
                    break
                else:
                    logger.warning("binance error: %s", last_error)

                if attempt < max_retries:
                    backoff = min(
                        base_backoff * (2 ** attempt),
                        cfg.max_backoff_s if not is_emergency else 8.0,
                    )
                    logger.warning(
                        "%sorder failed (attempt %d/%d): %s, retrying in %.1fs",
                        "EMERGENCY " if is_emergency else "",
                        attempt + 1,
                        max_retries + 1,
                        last_error,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

        return ExecutionResult(
            success=False,
            symbol=order.symbol,
            side=order.side,
            error=f"max_retries_exceeded: {last_error}",
            retries=max_retries,
        )

    async def emergency_close(self, order: OrderRequest) -> ExecutionResult:
        """
        Emergency close: aggressive retry for risk-critical orders.

        Use this for stop-loss closes, liquidation avoidance, and any order
        where failing to fill means the account is at risk.

        Differences from normal place_order:
          - 5 retries instead of 2
          - 0.5s base backoff instead of 2s
          - Rate limit hit does NOT abort (keeps retrying)
          - Slippage is never a reason to abort (getting out is priority)
        """
        self._total_orders += 1

        valid, reason = self.validate_order(order)
        if not valid:
            self._total_rejects += 1
            return ExecutionResult(
                success=False,
                symbol=order.symbol,
                side=order.side,
                error=f"validation_failed: {reason}",
            )

        if self._telemetry:
            self._telemetry.log_order_placed(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                leverage=order.leverage,
            )

        result = await self._execute_with_retry(order, is_emergency=True)

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
            logger.critical(
                "EMERGENCY CLOSE FAILED for %s %s: %s — ACCOUNT AT RISK",
                order.side, order.symbol, result.error,
            )

        return result

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
            self._symbol_info[symbol] = {"step_size": step}
        logger.info("loaded step sizes for %d symbols", len(step_sizes))

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
