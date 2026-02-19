from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path


RUNTIME_CREDENTIALS_PATH = Path("/app/data/runtime_credentials.json")


class CredentialsError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeCredentials:
    binance_api_key: str
    binance_api_secret: str
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, str]) -> "RuntimeCredentials":
        binance_api_key = (payload.get("binance_api_key") or "").strip()
        binance_api_secret = (payload.get("binance_api_secret") or "").strip()

        missing = []
        if not binance_api_key:
            missing.append("binance_api_key")
        if not binance_api_secret:
            missing.append("binance_api_secret")
        if missing:
            raise CredentialsError(f"missing required credentials: {', '.join(missing)}")

        return cls(
            binance_api_key=binance_api_key,
            binance_api_secret=binance_api_secret,
            telegram_bot_token=(payload.get("telegram_bot_token") or "").strip(),
            telegram_chat_id=(payload.get("telegram_chat_id") or "").strip(),
        )


def _load_from_env() -> RuntimeCredentials | None:
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    return RuntimeCredentials(
        binance_api_key=api_key,
        binance_api_secret=api_secret,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )


def _load_from_file(path: Path) -> RuntimeCredentials | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise CredentialsError("credential file content must be a JSON object")
    return RuntimeCredentials.from_mapping(payload)


def load_runtime_credentials(
    logger: logging.Logger,
    path: Path = RUNTIME_CREDENTIALS_PATH,
) -> RuntimeCredentials:
    try:
        env_creds = _load_from_env()
        if env_creds:
            logger.info("runtime credentials loaded from environment")
            return env_creds

        file_creds = _load_from_file(path)
        if file_creds:
            logger.info("runtime credentials loaded from %s", path)
            return file_creds

    except CredentialsError:
        raise
    except Exception as exc:
        raise CredentialsError(f"failed to load credentials: {exc}") from exc

    raise CredentialsError(
        "no credentials available: set BINANCE_API_KEY and BINANCE_API_SECRET "
        f"environment variables or provide {path}"
    )
