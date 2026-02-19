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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("darwin.v5.execution")


@dataclass
class ExecutionConfig:
    """Configuration for the execution engine."""
    leverage: int = 5
    max_retries: int = 3
    base_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    slippage_tolerance_pct: float = 0.5
    fill_timeout_s: float = 10.0
    min_notional_usdt: float = 5.0
    # Binance step sizes (common ones; fetched dynamically in production)
    default_step_size: float = 0.001


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
        cfg = self._config
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

    async def _execute_with_retry(self, order: OrderRequest) -> ExecutionResult:
        """Execute order with exponential backoff retry."""
        cfg = self._config
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

                    # Check slippage tolerance
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

                # Check for timestamp error (400)
                if "400" in last_error or "timestamp" in last_error.lower():
                    if self._time_sync:
                        logger.info("re-syncing time after 400 error")
                        self._time_sync.force_sync_on_error(400)

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
        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "quantity": str(order.quantity),
        }

        if order.reduce_only:
            params["reduceOnly"] = "true"

        # Use time-synced timestamp
        if self._time_sync:
            params["timestamp"] = self._time_sync.synced_timestamp()
        else:
            params["timestamp"] = int(time.time() * 1000)

        params["recvWindow"] = 5000

        # Sign and send
        client = self._client
        params["signature"] = client._sign(params)
        url = f"{client._base_url}/fapi/v1/order"
        response = client._session.request(
            "POST", url, params=params, timeout=client._timeout_s
        )
        response.raise_for_status()
        data = response.json()

        # Parse fill
        filled_qty = float(data.get("executedQty", 0.0))
        avg_price = float(data.get("avgPrice", 0.0))
        if avg_price == 0 and filled_qty > 0:
            avg_price = float(data.get("price", order.price))

        return ExecutionResult(
            success=filled_qty > 0,
            order_id=str(data.get("orderId", "")),
            symbol=order.symbol,
            side=order.side,
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=0.0,  # Binance includes fee in separate endpoint
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
        precision = max(0, -int(math.floor(math.log10(step))))
        return math.floor(quantity / step) * step

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
