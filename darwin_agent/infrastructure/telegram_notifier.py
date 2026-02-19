from __future__ import annotations

from typing import Any

import requests


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout_s: float = 10.0) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def send(self, text: str) -> None:
        response = self._session.post(
            self._url,
            json={"chat_id": self._chat_id, "text": text},
            timeout=self._timeout_s,
        )
        response.raise_for_status()

    def notify_engine_started(self) -> None:
        self.send("🟢 Darwin Engine started")

    def notify_engine_stopped(self) -> None:
        self.send("🔴 Darwin Engine stopped")

    def notify_trade_opened(self, symbol: str, side: str, price: float, quantity: float, risk_percent: float, equity: float) -> None:
        self.send(
            "\n".join(
                [
                    "🟢 TRADE OPENED",
                    f"Symbol: {symbol}",
                    f"Side: {side}",
                    f"Entry: {price}",
                    f"Size: {quantity}",
                    "Leverage: 5x",
                    f"Risk %: {risk_percent}",
                    f"Equity: {equity}",
                ]
            )
        )

    def notify_trade_closed(self, symbol: str, price: float, pnl: float, equity: float) -> None:
        self.send(
            "\n".join(
                [
                    "🔴 TRADE CLOSED",
                    f"Symbol: {symbol}",
                    f"Exit: {price}",
                    f"PnL: {pnl}",
                    f"New Equity: {equity}",
                ]
            )
        )

    def notify_error(self, error_message: str) -> None:
        self.send("\n".join(["⚠️ ERROR", error_message]))

    def notify_reconnect_attempt(self, retry: int, max_retries: int) -> None:
        self.send(f"reconnect_attempt {retry}/{max_retries}")

    def notify_stop_loss_triggered(self, symbol: str, price: float) -> None:
        self.send(f"stop_loss_triggered {symbol} @ {price}")

    def notify_take_profit_triggered(self, symbol: str, price: float) -> None:
        self.send(f"take_profit_triggered {symbol} @ {price}")

    def notify_engine_connected(self, startup_payload: dict[str, Any]) -> None:
        wallet = startup_payload.get("wallet_balance")
        positions = len(startup_payload.get("open_positions", []))
        self.send(f"Engine connected. Wallet: {wallet}, Open positions: {positions}")

    # ── v5 signal notifications ───────────────────────────

    def notify_signal_generated(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        threshold: float,
        regime: str,
        factors: dict[str, float],
    ) -> None:
        factor_str = ", ".join(f"{k}={v:.2f}" for k, v in factors.items())
        self.send(
            "\n".join([
                "SIGNAL GENERATED",
                f"Symbol: {symbol}",
                f"Direction: {direction}",
                f"Confidence: {confidence:.3f} (threshold: {threshold:.3f})",
                f"Regime: {regime}",
                f"Factors: {factor_str}",
            ])
        )

    def notify_signal_rejected(
        self,
        symbol: str,
        reason: str,
        confidence: float = 0.0,
    ) -> None:
        self.send(
            "\n".join([
                "SIGNAL REJECTED",
                f"Symbol: {symbol}",
                f"Reason: {reason}",
                f"Confidence: {confidence:.3f}",
            ])
        )

    def notify_order_placed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: int = 5,
    ) -> None:
        self.send(
            "\n".join([
                "ORDER PLACED",
                f"Symbol: {symbol}",
                f"Side: {side}",
                f"Quantity: {quantity}",
                f"Price: {price}",
                f"Leverage: {leverage}x",
            ])
        )

    def notify_order_filled(
        self,
        symbol: str,
        side: str,
        filled_qty: float,
        avg_price: float,
        fee: float,
        slippage_bps: float = 0.0,
    ) -> None:
        self.send(
            "\n".join([
                "ORDER FILLED",
                f"Symbol: {symbol}",
                f"Side: {side}",
                f"Filled: {filled_qty}",
                f"Avg Price: {avg_price}",
                f"Fee: {fee:.4f}",
                f"Slippage: {slippage_bps:.1f} bps",
            ])
        )

    def notify_position_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
        equity: float,
    ) -> None:
        emoji = "+" if pnl >= 0 else ""
        self.send(
            "\n".join([
                "POSITION CLOSED",
                f"Symbol: {symbol}",
                f"PnL: {emoji}${pnl:.2f} ({emoji}{pnl_pct:.1f}%)",
                f"Reason: {reason}",
                f"Equity: ${equity:.2f}",
            ])
        )

    def notify_heartbeat(
        self,
        equity: float,
        positions: int,
        regime: str,
        drawdown_pct: float,
    ) -> None:
        self.send(
            "\n".join([
                "HEARTBEAT",
                f"Equity: ${equity:.2f}",
                f"Positions: {positions}",
                f"Regime: {regime}",
                f"Drawdown: {drawdown_pct:.1f}%",
            ])
        )
