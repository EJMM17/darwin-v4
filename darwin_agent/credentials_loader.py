from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path


RUNTIME_CREDENTIALS_PATH = Path("/app/data/runtime_credentials.json")


class CredentialsError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeCredentials:
    binance_api_key: str
    binance_api_secret: str
    telegram_bot_token: str
    telegram_chat_id: str

    @classmethod
    def from_mapping(cls, payload: dict[str, str]) -> "RuntimeCredentials":
        creds = cls(
            binance_api_key=(payload.get("binance_api_key") or "").strip(),
            binance_api_secret=(payload.get("binance_api_secret") or "").strip(),
            telegram_bot_token=(payload.get("telegram_bot_token") or "").strip(),
            telegram_chat_id=(payload.get("telegram_chat_id") or "").strip(),
        )
        missing = [
            name
            for name, value in {
                "binance_api_key": creds.binance_api_key,
                "binance_api_secret": creds.binance_api_secret,
                "telegram_bot_token": creds.telegram_bot_token,
                "telegram_chat_id": creds.telegram_chat_id,
            }.items()
            if not value
        ]
        if missing:
            raise CredentialsError(f"missing required credentials: {', '.join(missing)}")
        return creds


def _load_from_env() -> RuntimeCredentials | None:
    values = {
        "binance_api_key": os.getenv("BINANCE_API_KEY", ""),
        "binance_api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }
    if not any(values.values()):
        return None
    return RuntimeCredentials.from_mapping(values)


def _load_from_file(path: Path) -> RuntimeCredentials | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise CredentialsError("credential file content must be a JSON object")
    return RuntimeCredentials.from_mapping(payload)


def _prompt_credentials() -> RuntimeCredentials:
    values = {
        "binance_api_key": input("binance_api_key: "),
        "binance_api_secret": getpass("binance_api_secret: "),
        "telegram_bot_token": getpass("telegram_bot_token: "),
        "telegram_chat_id": input("telegram_chat_id: "),
    }
    return RuntimeCredentials.from_mapping(values)


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

        if sys.stdin.isatty():
            logger.info("runtime credentials not found in env/file; prompting interactive input")
            creds = _prompt_credentials()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "binance_api_key": creds.binance_api_key,
                        "binance_api_secret": creds.binance_api_secret,
                        "telegram_bot_token": creds.telegram_bot_token,
                        "telegram_chat_id": creds.telegram_chat_id,
                    },
                    indent=2,
                )
            )
            return creds
    except CredentialsError:
        raise
    except Exception as exc:
        raise CredentialsError(f"failed to load credentials: {exc}") from exc

    raise CredentialsError(
        "no credentials available: set BINANCE_API_KEY/BINANCE_API_SECRET/TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
        f"or provide {path}"
    )
