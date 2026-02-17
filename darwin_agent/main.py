from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from darwin_agent.core.engine import DarwinCoreEngine, EngineConfig
from darwin_agent.credentials_loader import CredentialsError, load_runtime_credentials
from darwin_agent.infrastructure.binance_client import BinanceCredentials, BinanceFuturesClient
from darwin_agent.infrastructure.telegram_notifier import TelegramNotifier
from darwin_agent.runtime.runtime_service import RuntimeService


LOG_PATH = Path("/app/logs/darwin_engine.log")
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


def _create_runtime(logger: logging.Logger) -> tuple[RuntimeService, BinanceFuturesClient, TelegramNotifier, DarwinCoreEngine]:
    creds = load_runtime_credentials(logger)
    binance = BinanceFuturesClient(
        BinanceCredentials(
            api_key=creds.binance_api_key,
            api_secret=creds.binance_api_secret,
        )
    )
    telegram = TelegramNotifier(
        bot_token=creds.telegram_bot_token,
        chat_id=creds.telegram_chat_id,
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
    logger.info("startup validation begin")
    result = binance.validate_startup(symbols=symbols, leverage=leverage)
    telegram.notify_engine_connected(result)
    logger.info("startup validation success", extra={"event": "startup_validated", "symbols": len(symbols)})
    return result


async def run_test_mode(logger: logging.Logger) -> int:
    try:
        runtime, binance, telegram, _ = _create_runtime(logger)
    except CredentialsError as exc:
        logger.error("credentials error: %s", exc)
        return 1

    _ = runtime
    try:
        result = await asyncio.to_thread(startup_validation, binance, telegram, 5, DEFAULT_SYMBOLS, logger)
        wallet = float(result["wallet_balance"])
        upnl = float(binance.get_unrealized_pnl())
        equity = wallet + upnl
        print(f"equity: {equity:.8f}")
        print("leverage confirmation:", result["leverage_result"])
        print("test mode completed: no trades opened")
        return 0
    except Exception as exc:
        logger.error("startup/test validation failed: %s", exc, exc_info=True)
        try:
            telegram.notify_error(f"startup failure: {exc}")
        except Exception:
            pass
        return 1
    finally:
        binance.close()
        telegram.close()


async def run_live(logger: logging.Logger) -> int:
    try:
        runtime, binance, telegram, _ = _create_runtime(logger)
    except CredentialsError as exc:
        logger.error("credentials error: %s", exc)
        return 1

    stop_event = asyncio.Event()

    def _request_shutdown(signum: int, _frame: Any) -> None:
        logger.info("signal received: %s", signum)
        runtime.stop()
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    try:
        await asyncio.to_thread(startup_validation, binance, telegram, 5, DEFAULT_SYMBOLS, logger)
    except Exception as exc:
        logger.error("startup validation failed: %s", exc, exc_info=True)
        try:
            telegram.notify_error(f"startup failure: {exc}")
        except Exception:
            pass
        binance.close()
        telegram.close()
        return 1

    runtime_task = asyncio.create_task(runtime.run_forever())
    wait_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait({runtime_task, wait_task}, return_when=asyncio.FIRST_COMPLETED)
        if runtime_task in done:
            return runtime_task.result()
        runtime.stop()
        runtime_task.cancel()
        try:
            await runtime_task
        except asyncio.CancelledError:
            pass
        return 0
    finally:
        wait_task.cancel()
        binance.close()
        telegram.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Darwin v4 headless runtime")
    parser.add_argument("--test", action="store_true", help="Run startup validations and exit")
    args = parser.parse_args()

    logger = setup_logging()
    code = asyncio.run(run_test_mode(logger) if args.test else run_live(logger))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
