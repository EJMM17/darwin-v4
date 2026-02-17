from __future__ import annotations

import requests


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout_s: float = 10.0) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout_s = timeout_s

    def send(self, text: str) -> None:
        response = requests.post(
            self._url,
            json={"chat_id": self._chat_id, "text": text},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
