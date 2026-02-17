from __future__ import annotations

import argparse
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from getpass import getpass
from pathlib import Path
from typing import Any

from darwin_agent.core.engine import DarwinCoreEngine, EngineConfig, EquitySnapshot
from darwin_agent.infrastructure.binance_client import BinanceCredentials, BinanceFuturesClient
from darwin_agent.infrastructure.telegram_notifier import TelegramNotifier
from darwin_agent.runtime.runtime_service import RuntimeService


LOG_PATH = Path("/app/logs/darwin_engine.log")
CREDENTIALS_PATH = Path("/app/data/runtime_credentials.json")
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("darwin")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=5)
        rotating.setFormatter(formatter)
        logger.addHandler(rotating)
    except Exception as exc:
        logger.warning("failed to attach file log handler: %s", exc)

    return logger


def _prompt_credentials() -> dict[str, str]:
    print("Enter runtime credentials for Darwin Engine:")
    return {
        "binance_api_key": input("binance_api_key: ").strip(),
        "binance_api_secret": getpass("binance_api_secret: ").strip(),
        "telegram_bot_token": getpass("telegram_bot_token: ").strip(),
        "telegram_chat_id": input("telegram_chat_id: ").strip(),
    }


def _load_or_prompt_credentials() -> dict[str, str]:
    if CREDENTIALS_PATH.exists():
        return json.loads(CREDENTIALS_PATH.read_text())
    credentials = _prompt_credentials()
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(json.dumps(credentials, indent=2))
    return credentials


def _create_runtime() -> tuple[RuntimeService, BinanceFuturesClient, TelegramNotifier, DarwinCoreEngine]:
    creds = _load_or_prompt_credentials()
    binance = BinanceFuturesClient(
        BinanceCredentials(
            api_key=creds["binance_api_key"],
            api_secret=creds["binance_api_secret"],
        )
    )
    telegram = TelegramNotifier(
        bot_token=creds["telegram_bot_token"],
        chat_id=creds["telegram_chat_id"],
    )
    engine = DarwinCoreEngine(EngineConfig(risk_percent=1.0, leverage=5))
    runtime = RuntimeService(
        engine=engine,
        binance_client=binance,
        telegram_notifier=telegram,
        symbols=DEFAULT_SYMBOLS,
        max_retries=5,
        safe_shutdown_flag=True,
    )
    return runtime, binance, telegram, engine


def startup_validation(
    binance: BinanceFuturesClient,
    telegram: TelegramNotifier,
    leverage: int,
    symbols: list[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    result = binance.validate_startup(symbols=symbols, leverage=leverage)
    telegram.send("Darwin Engine successfully connected.")
    logger.info("validation success")
    return result


def run_test_mode(logger: logging.Logger) -> int:
    runtime, binance, telegram, engine = _create_runtime()
    _ = runtime
    try:
        result = startup_validation(binance, telegram, leverage=5, symbols=DEFAULT_SYMBOLS, logger=logger)
        wallet = float(result["wallet_balance"])
        upnl = float(binance.get_unrealized_pnl())
        equity = EquitySnapshot(wallet_balance=wallet, unrealized_pnl=upnl).equity
        print(f"equity: {equity:.8f}")
        print("leverage confirmation:", result["leverage_result"])
        print(f"risk percent: {engine.evaluate(EquitySnapshot(wallet, upnl), []).get('risk_percent')}")
        print("test mode completed: no trades opened")
        return 0
    except Exception as exc:
        logger.error("startup/test validation failed: %s", exc, exc_info=True)
        try:
            telegram.send(f"Darwin Engine startup failure: {exc}")
        except Exception:
            pass
        return 1


def run_live(logger: logging.Logger) -> int:
    runtime, binance, telegram, _ = _create_runtime()
    try:
        startup_validation(binance, telegram, leverage=5, symbols=DEFAULT_SYMBOLS, logger=logger)
    except Exception as exc:
        logger.error("startup validation failed: %s", exc, exc_info=True)
        try:
            telegram.send(f"Darwin Engine startup failure: {exc}")
        except Exception:
            pass
        return 1

    runtime.start()
    logger.info("runtime started")
    try:
        while True:
            if not runtime.status().running:
                return 1
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt received, stopping runtime")
        runtime.stop()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Darwin v4 headless runtime")
    parser.add_argument("--test", action="store_true", help="Run startup validations and exit")
    args = parser.parse_args()

    logger = setup_logging()
    code = run_test_mode(logger) if args.test else run_live(logger)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
