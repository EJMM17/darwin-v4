"""Non-blocking Telegram alerts for dashboard runtime events."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import requests

_ALERT_TIMEOUT_SECONDS = 5


def _send(payload: Dict[str, Any]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = {
        "chat_id": chat_id,
        "text": payload.get("text", "Darwin alert"),
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=body, timeout=_ALERT_TIMEOUT_SECONDS)
    except Exception:
        # Never raise from alert transport
        return


def send_telegram_alert(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Fire-and-forget Telegram alert sender."""
    text = message
    if details:
        suffix = ", ".join(f"{k}={v}" for k, v in sorted(details.items()))
        text = f"{message}\n{suffix}"

    t = threading.Thread(target=_send, args=({"text": text},), daemon=True, name="darwin-alert")
    t.start()
