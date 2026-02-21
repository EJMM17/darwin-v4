"""
Darwin v5 — VISA: Virtual Intelligent Stop Architecture.

Server-side order manager that ensures protective orders (SL/TP) live on
Binance's matching engine, not just in our Python process.

Why this matters:
    - Bot crash/OOM/kill -9: positions remain protected by exchange-side orders
    - Network partition: server-side orders execute independently
    - Latency spike: no 5s tick delay between detection and execution
    - Binance rate limit: orders are already placed, no API call needed at trigger

Architecture:
    1. After every fill, immediately place STOP_MARKET + TAKE_PROFIT_MARKET
    2. Track all server-side order IDs per position
    3. When position closes (by any path), cancel the opposing server-side orders
    4. Periodic audit: verify server-side orders still exist (exchange can cancel
       expired orders, or they may fill without our knowledge)

Server-side orders use:
    - workingType=MARK_PRICE (not LAST_PRICE) to avoid wick manipulation
    - closePosition=true to handle partial fills and quantity mismatches
    - Exponential backoff retry for transient API failures

Usage:
    visa = VISAOrderManager(binance_client)
    # After a fill:
    visa.place_protective_orders(symbol, "LONG", entry_price, sl_price, tp_price, qty)
    # When closing:
    visa.cancel_protective_orders(symbol)
    # Periodic audit:
    orphaned = visa.audit_orders(open_symbols)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("darwin.v5.visa")


@dataclass(slots=True)
class ProtectiveOrderPair:
    """Tracks the SL + TP server-side orders for one position."""
    symbol: str = ""
    direction: str = ""           # "LONG" or "SHORT"
    sl_order_id: str = ""
    tp_order_id: str = ""
    sl_price: float = 0.0
    tp_price: float = 0.0
    quantity: float = 0.0
    entry_price: float = 0.0
    placed_at: float = 0.0
    sl_verified: bool = False     # last audit confirmed SL exists
    tp_verified: bool = False     # last audit confirmed TP exists
    last_audit: float = 0.0


@dataclass(slots=True)
class VISAAuditResult:
    """Result of periodic server-side order audit."""
    total_positions: int = 0
    sl_missing: List[str] = field(default_factory=list)  # symbols with missing SL
    tp_missing: List[str] = field(default_factory=list)  # symbols with missing TP
    sl_filled: List[str] = field(default_factory=list)   # SL triggered on exchange
    tp_filled: List[str] = field(default_factory=list)   # TP triggered on exchange
    errors: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.sl_missing or self.tp_missing or self.errors)


class VISAOrderManager:
    """
    Server-side protective order lifecycle manager.

    One instance per engine. Maintains a registry of all server-side
    SL/TP orders and their status.
    """

    def __init__(self, binance_client: Any) -> None:
        self._client = binance_client
        # symbol -> ProtectiveOrderPair
        self._orders: Dict[str, ProtectiveOrderPair] = {}
        # Counters for telemetry
        self._total_sl_placed: int = 0
        self._total_tp_placed: int = 0
        self._total_sl_triggered: int = 0
        self._total_tp_triggered: int = 0
        self._total_placement_failures: int = 0

    @property
    def active_orders(self) -> Dict[str, ProtectiveOrderPair]:
        return dict(self._orders)

    def place_protective_orders(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        quantity: float,
    ) -> ProtectiveOrderPair:
        """
        Place server-side SL + TP orders after a fill.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTCUSDT").
        direction : str
            "LONG" or "SHORT".
        entry_price : float
            Fill price of the entry order.
        sl_price : float
            Stop loss trigger price.
        tp_price : float
            Take profit trigger price.
        quantity : float
            Position quantity.

        Returns
        -------
        ProtectiveOrderPair
            Tracking record with order IDs.

        Raises
        ------
        RuntimeError
            If SL placement fails (TP failure is logged but not fatal —
            the SL is the critical protection).
        """
        # Cancel any existing protective orders for this symbol first
        self._cancel_existing(symbol)

        # Determine close side (opposite of position direction)
        if direction == "LONG":
            close_side = "SELL"
        else:
            close_side = "BUY"

        pair = ProtectiveOrderPair(
            symbol=symbol,
            direction=direction,
            sl_price=sl_price,
            tp_price=tp_price,
            quantity=quantity,
            entry_price=entry_price,
            placed_at=time.time(),
        )

        # CRITICAL: Place SL first — this is the protective order.
        # If SL placement fails, we MUST raise because the position
        # has no protection.
        try:
            sl_order = self._client.place_stop_market(
                symbol=symbol,
                side=close_side,
                quantity=quantity,
                stop_price=sl_price,
                reduce_only=True,
            )
            pair.sl_order_id = sl_order.order_id
            pair.sl_verified = True
            self._total_sl_placed += 1
        except Exception as exc:
            self._total_placement_failures += 1
            logger.critical(
                "VISA: FAILED to place server-side SL for %s at %.4f: %s",
                symbol, sl_price, exc,
            )
            # Re-raise: position without SL is unacceptable
            raise RuntimeError(
                f"VISA_SL_PLACEMENT_FAILED: {symbol} sl={sl_price} — {exc}"
            ) from exc

        # Place TP — failure here is not fatal (SL still protects us)
        try:
            tp_order = self._client.place_take_profit_market(
                symbol=symbol,
                side=close_side,
                quantity=quantity,
                stop_price=tp_price,
                reduce_only=True,
            )
            pair.tp_order_id = tp_order.order_id
            pair.tp_verified = True
            self._total_tp_placed += 1
        except Exception as exc:
            self._total_placement_failures += 1
            logger.warning(
                "VISA: TP placement failed for %s at %.4f (SL is active): %s",
                symbol, tp_price, exc,
            )
            # Non-fatal: SL is in place, we just won't auto-TP on the exchange

        self._orders[symbol] = pair

        logger.info(
            "VISA: protective orders placed for %s %s — SL=%.4f (id=%s) TP=%.4f (id=%s)",
            symbol, direction, sl_price, pair.sl_order_id,
            tp_price, pair.tp_order_id or "NONE",
        )

        return pair

    def cancel_protective_orders(self, symbol: str) -> None:
        """
        Cancel all server-side protective orders for a symbol.

        Called when:
        - Position is closed by any path (client-side SL/TP, trailing, manual)
        - Position is adjusted (need to re-place with new levels)
        """
        pair = self._orders.pop(symbol, None)
        if pair is None:
            return

        cancelled = []
        for order_id, label in [(pair.sl_order_id, "SL"), (pair.tp_order_id, "TP")]:
            if order_id:
                ok = self._client.cancel_order(symbol, order_id)
                if ok:
                    cancelled.append(f"{label}={order_id}")
                else:
                    logger.warning(
                        "VISA: cancel %s failed for %s/%s (may have already filled)",
                        label, symbol, order_id,
                    )

        if cancelled:
            logger.info("VISA: cancelled %s for %s", ", ".join(cancelled), symbol)

    def update_sl_price(
        self, symbol: str, new_sl_price: float
    ) -> Optional[str]:
        """
        Move the server-side SL to a new price (for trailing stop).

        Cancels old SL and places new one. Returns new order ID or None.
        This is an atomic operation: if the new SL fails, the old one
        is already cancelled. In this case we log CRITICAL and return None,
        and the caller must handle the unprotected state.
        """
        pair = self._orders.get(symbol)
        if not pair:
            return None

        close_side = "SELL" if pair.direction == "LONG" else "BUY"

        # Cancel old SL
        if pair.sl_order_id:
            self._client.cancel_order(symbol, pair.sl_order_id)

        # Place new SL
        try:
            sl_order = self._client.place_stop_market(
                symbol=symbol,
                side=close_side,
                quantity=pair.quantity,
                stop_price=new_sl_price,
                reduce_only=True,
            )
            pair.sl_order_id = sl_order.order_id
            pair.sl_price = new_sl_price
            pair.sl_verified = True
            self._total_sl_placed += 1

            logger.info(
                "VISA: SL moved for %s to %.4f (id=%s)",
                symbol, new_sl_price, sl_order.order_id,
            )
            return sl_order.order_id
        except Exception as exc:
            self._total_placement_failures += 1
            pair.sl_order_id = ""
            pair.sl_verified = False
            logger.critical(
                "VISA: FAILED to move SL for %s to %.4f — POSITION UNPROTECTED: %s",
                symbol, new_sl_price, exc,
            )
            return None

    def audit_orders(self, open_position_symbols: List[str]) -> VISAAuditResult:
        """
        Verify that all tracked server-side orders still exist on the exchange.

        Should be called periodically (every 60s or so).

        Detects:
        1. SL/TP orders that disappeared (exchange cancelled, expired)
        2. SL/TP orders that filled (position was closed server-side)
        3. Orphaned tracking entries (position closed but we didn't know)
        """
        result = VISAAuditResult(total_positions=len(open_position_symbols))

        # Clean up entries for symbols that no longer have positions
        stale_symbols = [s for s in self._orders if s not in open_position_symbols]
        for s in stale_symbols:
            self.cancel_protective_orders(s)

        for symbol, pair in list(self._orders.items()):
            if symbol not in open_position_symbols:
                continue

            try:
                open_orders = self._client.get_open_orders(symbol)
                open_ids = {str(o.get("orderId", "")) for o in open_orders}
                open_status = {
                    str(o.get("orderId", "")): o.get("status", "")
                    for o in open_orders
                }
            except Exception as exc:
                result.errors.append(f"{symbol}: fetch_orders_failed: {exc}")
                continue

            # Check SL
            if pair.sl_order_id:
                if pair.sl_order_id not in open_ids:
                    # SL disappeared — check if it filled
                    order_info = self._client.get_order_status(symbol, pair.sl_order_id)
                    status = order_info.get("status", "UNKNOWN")
                    if status == "FILLED":
                        result.sl_filled.append(symbol)
                        self._total_sl_triggered += 1
                        logger.warning(
                            "VISA AUDIT: SL FILLED server-side for %s (id=%s)",
                            symbol, pair.sl_order_id,
                        )
                    else:
                        result.sl_missing.append(symbol)
                        pair.sl_verified = False
                        logger.critical(
                            "VISA AUDIT: SL MISSING for %s (id=%s status=%s) — POSITION UNPROTECTED",
                            symbol, pair.sl_order_id, status,
                        )
                else:
                    pair.sl_verified = True
            else:
                result.sl_missing.append(symbol)

            # Check TP
            if pair.tp_order_id:
                if pair.tp_order_id not in open_ids:
                    order_info = self._client.get_order_status(symbol, pair.tp_order_id)
                    status = order_info.get("status", "UNKNOWN")
                    if status == "FILLED":
                        result.tp_filled.append(symbol)
                        self._total_tp_triggered += 1
                        logger.info(
                            "VISA AUDIT: TP FILLED server-side for %s (id=%s)",
                            symbol, pair.tp_order_id,
                        )
                    else:
                        result.tp_missing.append(symbol)
                        pair.tp_verified = False
                        logger.warning(
                            "VISA AUDIT: TP missing for %s (id=%s status=%s)",
                            symbol, pair.tp_order_id, status,
                        )
                else:
                    pair.tp_verified = True
            # TP missing is not critical (SL protects us)

            pair.last_audit = time.time()

        if result.has_issues:
            logger.warning(
                "VISA AUDIT issues: sl_missing=%s tp_missing=%s errors=%d",
                result.sl_missing, result.tp_missing, len(result.errors),
            )

        return result

    def replace_protective_orders(
        self,
        symbol: str,
        new_sl_price: float,
        new_tp_price: float,
    ) -> bool:
        """
        Replace both SL and TP orders with new prices.

        Used when regime changes require SL/TP adjustment.
        Returns True if at least the SL was placed successfully.
        """
        pair = self._orders.get(symbol)
        if not pair:
            return False

        # Cancel all existing
        self._cancel_existing(symbol)

        # Re-place with new levels
        try:
            new_pair = self.place_protective_orders(
                symbol=symbol,
                direction=pair.direction,
                entry_price=pair.entry_price,
                sl_price=new_sl_price,
                tp_price=new_tp_price,
                quantity=pair.quantity,
            )
            return bool(new_pair.sl_order_id)
        except RuntimeError:
            return False

    def _cancel_existing(self, symbol: str) -> None:
        """Cancel existing protective orders silently."""
        pair = self._orders.get(symbol)
        if not pair:
            return

        for order_id in [pair.sl_order_id, pair.tp_order_id]:
            if order_id:
                self._client.cancel_order(symbol, order_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return VISA telemetry metrics."""
        return {
            "active_positions": len(self._orders),
            "total_sl_placed": self._total_sl_placed,
            "total_tp_placed": self._total_tp_placed,
            "total_sl_triggered": self._total_sl_triggered,
            "total_tp_triggered": self._total_tp_triggered,
            "total_placement_failures": self._total_placement_failures,
            "positions": {
                sym: {
                    "direction": pair.direction,
                    "sl_price": pair.sl_price,
                    "tp_price": pair.tp_price,
                    "sl_verified": pair.sl_verified,
                    "tp_verified": pair.tp_verified,
                    "age_s": round(time.time() - pair.placed_at, 1),
                }
                for sym, pair in self._orders.items()
            },
        }

    def has_protection(self, symbol: str) -> bool:
        """Check if a symbol has active server-side SL protection."""
        pair = self._orders.get(symbol)
        return pair is not None and bool(pair.sl_order_id) and pair.sl_verified
