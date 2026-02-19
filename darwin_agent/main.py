from __future__ import annotations

import argparse
import asyncio
import logging
import os
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


def _create_clients(logger: logging.Logger) -> tuple[BinanceFuturesClient, TelegramNotifier]:
    """Create exchange and notification clients from credentials."""
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
    return binance, telegram


def _create_runtime(logger: logging.Logger) -> tuple[RuntimeService, BinanceFuturesClient, TelegramNotifier, DarwinCoreEngine]:
    """Create v4 runtime (preserved for backward compatibility)."""
    binance, telegram = _create_clients(logger)
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
    if not symbols:
        raise RuntimeError("No trading symbols configured. DEFAULT_SYMBOLS is empty.")

    logger.info("Initializing Binance Futures client...")
    try:
        binance.ping_futures()
    except Exception as exc:
        raise RuntimeError(f"Binance connection failed: {exc}") from exc
    logger.info("Binance connection OK")

    try:
        wallet = float(binance.get_wallet_balance())
    except Exception as exc:
        raise RuntimeError(f"Wallet balance fetch failed: {exc}") from exc
    if wallet <= 0:
        raise RuntimeError(f"Wallet balance must be > 0, got {wallet}")
    logger.info("Wallet balance: %.8f USDT", wallet)

    positions = binance.get_open_positions()
    logger.info("Positions synced: %d", len(positions))

    leverage_result: dict[str, bool] = {}
    for symbol in symbols:
        logger.info("Setting leverage 5x on %s", symbol)
        try:
            leverage_result[symbol] = bool(binance.set_leverage(symbol, leverage))
        except Exception as exc:
            raise RuntimeError(f"Leverage set failed for {symbol}: {exc}") from exc
        if not leverage_result[symbol]:
            raise RuntimeError(f"Leverage set failed for {symbol}: exchange did not confirm {leverage}x")

    result = {
        "wallet_balance": wallet,
        "open_positions": positions,
        "leverage_result": leverage_result,
        "symbols": list(symbols),
    }
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

    leverage = 5
    symbols = list(getattr(runtime, "_symbols", DEFAULT_SYMBOLS))
    if not symbols:
        logger.error("fatal startup validation failed: no symbols configured")
        binance.close()
        telegram.close()
        raise RuntimeError("No trading symbols configured. DEFAULT_SYMBOLS is empty.")

    try:
        startup = await asyncio.to_thread(startup_validation, binance, telegram, leverage, symbols, logger)
    except Exception as exc:
        logger.error("fatal startup validation failed: %s", exc, exc_info=True)
        try:
            telegram.notify_error(f"startup failure: {exc}")
        except Exception:
            pass
        binance.close()
        telegram.close()
        return 1

    if getattr(runtime, "_binance", None) is None:
        binance.close()
        telegram.close()
        raise RuntimeError("fatal startup validation failed: runtime has no exchange client")
    if float(startup["wallet_balance"]) <= 0:
        binance.close()
        telegram.close()
        raise RuntimeError("fatal startup validation failed: wallet balance must be > 0")
    if not all(bool(ok) for ok in startup["leverage_result"].values()):
        binance.close()
        telegram.close()
        raise RuntimeError("fatal startup validation failed: leverage enforcement did not succeed")

    runtime._startup_state = startup
    runtime._leverage = 5
    runtime._symbols = symbols

    logger.info("=== DARWIN LIVE STARTUP SUMMARY ===")
    logger.info("Wallet balance: %.8f USDT", startup["wallet_balance"])
    logger.info("Open positions: %d", len(startup["open_positions"]))
    logger.info("Symbols: %s", startup["symbols"])
    logger.info("Leverage: 5x (forced)")
    logger.info("====================================")

    runtime_task = asyncio.create_task(runtime.run_forever())
    logger.info("Runtime loop started")
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


# ═════════════════════════════════════════════════════════════
# v5 Engine Entry Points
# ═════════════════════════════════════════════════════════════

async def run_v5(logger: logging.Logger, dry_run: bool = False) -> int:
    """Run the Darwin v5 engine."""
    from darwin_agent.v5.engine import DarwinV5Engine, V5EngineConfig

    try:
        binance, telegram = _create_clients(logger)
    except CredentialsError as exc:
        logger.error("credentials error: %s", exc)
        return 1

    symbols = os.getenv("DARWIN_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    symbols = [s.strip() for s in symbols if s.strip()]

    config = V5EngineConfig(
        symbols=symbols,
        leverage=int(os.getenv("DARWIN_LEVERAGE", "5")),
        base_risk_pct=float(os.getenv("DARWIN_RISK_PERCENT", "1.0")),
        confidence_threshold=float(os.getenv("DARWIN_CONFIDENCE_THRESHOLD", "0.60")),
        dry_run=dry_run,
    )

    v5_engine = DarwinV5Engine(binance, telegram, config)

    stop_event = asyncio.Event()

    def _request_shutdown(signum: int, _frame: Any) -> None:
        logger.info("signal received: %s", signum)
        v5_engine.stop()
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    try:
        startup = await v5_engine.initialize()
        logger.info("=== DARWIN v5 STARTUP SUMMARY ===")
        logger.info("Equity: $%.4f", startup["equity"])
        logger.info("Positions: %d", startup["open_positions"])
        logger.info("Symbols: %s", startup["symbols"])
        logger.info("Leverage: %dx", config.leverage)
        logger.info("Dry run: %s", dry_run)
        logger.info("=================================")
    except Exception as exc:
        logger.error("v5 initialization failed: %s", exc, exc_info=True)
        try:
            telegram.notify_error(f"v5 startup failure: {exc}")
        except Exception:
            pass
        binance.close()
        telegram.close()
        return 1

    engine_task = asyncio.create_task(v5_engine.run_forever())
    wait_task = asyncio.create_task(stop_event.wait())

    try:
        done, _ = await asyncio.wait({engine_task, wait_task}, return_when=asyncio.FIRST_COMPLETED)
        if engine_task in done:
            return engine_task.result()
        v5_engine.stop()
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
        return 0
    finally:
        wait_task.cancel()
        binance.close()
        telegram.close()


async def run_v5_dry_run(logger: logging.Logger) -> int:
    """Run v5 engine in dry-run mode (signals only, no orders)."""
    return await run_v5(logger, dry_run=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Darwin v5 headless runtime")
    parser.add_argument("--test", action="store_true", help="Run startup validations and exit")
    parser.add_argument("--v5", action="store_true", help="Run the v5 engine (default)")
    parser.add_argument("--v4", action="store_true", help="Run the legacy v4 engine")
    parser.add_argument("--dry-run", action="store_true", help="v5 dry-run: signals only, no orders")
    args = parser.parse_args()

    logger = setup_logging()

    if args.test:
        code = asyncio.run(run_test_mode(logger))
    elif args.v4:
        code = asyncio.run(run_live(logger))
    elif args.dry_run:
        code = asyncio.run(run_v5_dry_run(logger))
    else:
        # Default: v5 engine
        code = asyncio.run(run_v5(logger))

    raise SystemExit(code)


if __name__ == "__main__":
    main()
