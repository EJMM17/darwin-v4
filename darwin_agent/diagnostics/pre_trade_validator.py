"""
Darwin v4 — Pre-Trade Validation Layer.

Before placing an order, re-validates:
  1. Fetch latest futures account balance
  2. Compute equity = wallet_balance + unrealized_pnl
  3. Recalculate position size dynamically
  4. Ensure leverage=5 for the symbol
  5. Ensure final_notional >= min_notional

If any check fails, logs the reason and returns a failed result.
The caller aborts the trade cleanly.

Does NOT modify any trading logic, risk engine math, or entry logic.
This is a pure pre-flight check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from darwin_agent.diagnostics.rejection_reason import RejectionReason

logger = logging.getLogger("darwin.diagnostics.pretrade")


@dataclass(slots=True)
class PreTradeResult:
    """Result of pre-trade validation."""
    valid: bool = True
    rejection_reason: RejectionReason = RejectionReason.NONE
    wallet_balance: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    recalculated_size_usdt: float = 0.0
    recalculated_quantity: float = 0.0
    leverage_confirmed: bool = False
    final_notional: float = 0.0
    min_notional: float = 5.0
    detail: str = ""


class PreTradeValidator:
    """
    Stateless pre-trade validation. Requires a balance_fetcher and
    a leverage_setter — both are callables to keep this decoupled
    from any specific exchange client.

    Usage:
        validator = PreTradeValidator(
            balance_fetcher=binance_client.get_futures_balance,
            leverage_setter=binance_client.set_leverage,
        )
        result = validator.validate(
            symbol="BTCUSDT",
            side="BUY",
            price=45000.0,
            risk_pct=2.0,
            leverage=5,
            min_notional=5.0,
        )
        if not result.valid:
            # abort trade, log result.rejection_reason
    """

    __slots__ = ("_balance_fetcher", "_leverage_setter", "_target_leverage")

    def __init__(
        self,
        balance_fetcher=None,
        leverage_setter=None,
        target_leverage: int = 5,
    ) -> None:
        self._balance_fetcher = balance_fetcher
        self._leverage_setter = leverage_setter
        self._target_leverage = target_leverage

    def validate(
        self,
        symbol: str,
        price: float,
        risk_pct: float,
        leverage: int = 5,
        min_notional: float = 5.0,
    ) -> PreTradeResult:
        """
        Synchronous pre-trade validation.

        Steps:
          1. Re-fetch account balance (wallet_balance + unrealized_pnl)
          2. Compute equity
          3. Recalculate position size: equity * (risk_pct / 100)
          4. Verify leverage is set to target
          5. Verify final_notional >= min_notional

        Returns PreTradeResult with valid=False and rejection_reason
        if any step fails.
        """
        result = PreTradeResult(min_notional=min_notional)

        # Step 1: Fetch balance
        if self._balance_fetcher is not None:
            try:
                balance_data = self._balance_fetcher()
                if isinstance(balance_data, dict):
                    result.wallet_balance = float(balance_data.get("wallet_balance", 0.0))
                    result.unrealized_pnl = float(balance_data.get("unrealized_pnl", 0.0))
                elif isinstance(balance_data, (int, float)):
                    result.wallet_balance = float(balance_data)
                    result.unrealized_pnl = 0.0
                else:
                    result.wallet_balance = 0.0
                    result.unrealized_pnl = 0.0
            except Exception as exc:
                logger.warning("pre-trade balance fetch failed: %s", exc)
                result.valid = False
                result.rejection_reason = RejectionReason.BALANCE_FETCH_FAILED
                result.detail = f"balance_fetch_error: {exc}"
                return result
        else:
            # No fetcher configured — skip live balance check
            result.wallet_balance = 0.0
            result.unrealized_pnl = 0.0

        # Step 2: Compute equity
        result.equity = result.wallet_balance + result.unrealized_pnl

        # Step 3: Recalculate size
        if result.equity > 0:
            result.recalculated_size_usdt = result.equity * (risk_pct / 100.0)
        else:
            result.recalculated_size_usdt = 0.0

        # Compute quantity
        if price > 0:
            result.recalculated_quantity = result.recalculated_size_usdt / price
        else:
            result.recalculated_quantity = 0.0

        # Step 4: Verify leverage
        if self._leverage_setter is not None:
            try:
                lev_ok = self._leverage_setter(symbol, self._target_leverage)
                result.leverage_confirmed = bool(lev_ok)
                if not result.leverage_confirmed:
                    result.valid = False
                    result.rejection_reason = RejectionReason.LEVERAGE_MISMATCH
                    result.detail = (
                        f"leverage_set returned False for {symbol} "
                        f"target={self._target_leverage}"
                    )
                    return result
            except Exception as exc:
                logger.warning("pre-trade leverage set failed for %s: %s", symbol, exc)
                result.valid = False
                result.rejection_reason = RejectionReason.LEVERAGE_SET_FAILED
                result.detail = f"leverage_set_error: {exc}"
                return result
        else:
            result.leverage_confirmed = True  # no setter, assume OK

        # Step 5: Notional check
        result.final_notional = result.recalculated_size_usdt * leverage
        if result.final_notional < min_notional:
            result.valid = False
            result.rejection_reason = RejectionReason.MIN_NOTIONAL_FAIL
            result.detail = (
                f"final_notional={result.final_notional:.4f} < "
                f"min_notional={min_notional:.4f}"
            )
            return result

        # All checks passed
        result.valid = True
        result.rejection_reason = RejectionReason.NONE
        return result
