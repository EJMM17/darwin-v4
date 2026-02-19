from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger("darwin.telegram")


class TelegramNotifier:
    """
    Telegram notifier. Works in no-op mode when bot_token or chat_id are empty.
    This allows running Darwin without Telegram configured.
    """

    def __init__(self, bot_token: str, chat_id: str, timeout_s: float = 10.0) -> None:
        self._enabled = bool(bot_token and chat_id)
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout_s = timeout_s
        self._session = requests.Session()
        if not self._enabled:
            logger.info("Telegram not configured — notifications disabled")

    def close(self) -> None:
        self._session.close()

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            response = self._session.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=self._timeout_s,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("telegram send failed: %s", exc)

    def notify_engine_connected(self, startup: dict[str, Any]) -> None:
        equity = startup.get("equity", 0.0)
        leverage = startup.get("leverage", 5)
        symbols = startup.get("symbols", [])
        dry_run = startup.get("dry_run", False)
        mode = "DRY-RUN" if dry_run else "LIVE"
        self.send(
            "\n".join([
                f"🟢 Darwin v5 Connected [{mode}]",
                f"Equity: ${equity:.2f}",
                f"Leverage: {leverage}x",
                f"Symbols: {', '.join(symbols)}",
            ])
        )

    def notify_engine_started(self) -> None:
        self.send("🟢 Darwin Engine started")

    def notify_engine_stopped(self) -> None:
        self.send("🔴 Darwin Engine stopped")

    def notify_error(self, message: str) -> None:
        self.send(f"❌ ERROR: {message}")

    def notify_trade_opened(
        self, symbol: str, side: str, price: float,
        quantity: float, risk_percent: float, equity: float
    ) -> None:
        self.send("\n".join([
            "🟢 TRADE OPENED",
            f"Symbol: {symbol}",
            f"Side: {side}",
            f"Entry: {price}",
            f"Size: {quantity}",
            f"Risk %: {risk_percent}",
            f"Equity: ${equity:.2f}",
        ]))

    def notify_trade_closed(
        self, symbol: str, price: float, pnl: float, equity: float
    ) -> None:
        emoji = "+" if pnl >= 0 else ""
        self.send("\n".join([
            "🔴 TRADE CLOSED",
            f"Symbol: {symbol}",
            f"Exit: {price}",
            f"PnL: {emoji}${pnl:.2f}",
            f"New Equity: ${equity:.2f}",
        ]))

    def notify_signal_generated(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        threshold: float,
        regime: str,
        factors: dict[str, float] | None = None,
    ) -> None:
        factor_str = ", ".join(
            f"{k}={v:.3f}" for k, v in (factors or {}).items()
        )
        self.send("\n".join([
            "📊 SIGNAL GENERATED",
            f"Symbol: {symbol}",
            f"Direction: {direction}",
            f"Confidence: {confidence:.3f} (threshold: {threshold:.3f})",
            f"Regime: {regime}",
            f"Factors: {factor_str}",
        ]))

    def notify_signal_rejected(
        self, symbol: str, reason: str, confidence: float = 0.0
    ) -> None:
        self.send("\n".join([
            "⚠️ SIGNAL REJECTED",
            f"Symbol: {symbol}",
            f"Reason: {reason}",
            f"Confidence: {confidence:.3f}",
        ]))

    def notify_order_placed(
        self, symbol: str, side: str, quantity: float,
        price: float, leverage: int = 5
    ) -> None:
        self.send("\n".join([
            "📤 ORDER PLACED",
            f"Symbol: {symbol}",
            f"Side: {side}",
            f"Quantity: {quantity}",
            f"Price: {price}",
            f"Leverage: {leverage}x",
        ]))

    def notify_order_filled(
        self, symbol: str, side: str, filled_qty: float,
        avg_price: float, fee: float, slippage_bps: float = 0.0
    ) -> None:
        self.send("\n".join([
            "✅ ORDER FILLED",
            f"Symbol: {symbol}",
            f"Side: {side}",
            f"Filled: {filled_qty}",
            f"Avg Price: {avg_price}",
            f"Fee: {fee:.4f}",
            f"Slippage: {slippage_bps:.1f} bps",
        ]))

    def notify_position_closed(
        self, symbol: str, pnl: float, pnl_pct: float,
        reason: str, equity: float
    ) -> None:
        emoji = "+" if pnl >= 0 else ""
        self.send("\n".join([
            "🔴 POSITION CLOSED",
            f"Symbol: {symbol}",
            f"PnL: {emoji}${pnl:.2f} ({emoji}{pnl_pct:.1f}%)",
            f"Reason: {reason}",
            f"Equity: ${equity:.2f}",
        ]))

    def notify_heartbeat(
        self, equity: float, positions: int,
        regime: str, drawdown_pct: float
    ) -> None:
        self.send("\n".join([
            "💓 HEARTBEAT",
            f"Equity: ${equity:.2f}",
            f"Positions: {positions}",
            f"Regime: {regime}",
            f"Drawdown: {drawdown_pct:.1f}%",
        ]))
