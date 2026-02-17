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
