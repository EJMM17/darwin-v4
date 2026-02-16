import asyncio

from darwin_agent.config import DarwinConfig, ExchangeConfig
from darwin_agent.interfaces.enums import ExchangeID
from darwin_agent.interfaces.types import ExchangeStatus
from darwin_agent.exchanges.router import ExchangeRouter
from darwin_agent.main import build_exchange_router, validate_exchange_symbol_sync


class _Adapter:
    def __init__(self, eid):
        self.eid = eid
        self.closed = False

    async def get_status(self):
        return ExchangeStatus(exchange_id=self.eid, connected=True, latency_ms=1.0)

    async def close(self):
        self.closed = True


def test_router_backward_compatible_hooks_work():
    bybit = _Adapter(ExchangeID.BYBIT)
    router = ExchangeRouter(adapters={ExchangeID.BYBIT: bybit}, primary=ExchangeID.BYBIT)

    asyncio.run(router.connect_all())
    statuses = asyncio.run(router.get_all_statuses())
    assert ExchangeID.BYBIT in statuses
    assert statuses[ExchangeID.BYBIT].connected is True

    asyncio.run(router.disconnect_all())
    assert bybit.closed is True


def test_build_exchange_router_supports_binance_config():
    config = DarwinConfig(
        exchanges=[
            ExchangeConfig(
                exchange_id="binance",
                api_key="k",
                api_secret="s",
                enabled=True,
                testnet=True,
            )
        ]
    )

    router = build_exchange_router(config)
    assert router is not None
    assert ExchangeID.BINANCE in router._adapters


class _BinanceValidatorAdapter:
    def __init__(self, missing=None):
        self._missing = missing or []

    async def validate_symbols(self, symbols):
        return list(self._missing)


def test_validate_exchange_symbol_sync_passes_when_all_symbols_valid():
    config = DarwinConfig(
        mode="live",
        exchanges=[
            ExchangeConfig(
                exchange_id="binance",
                enabled=True,
                symbols=["BTCUSDT", "ETHUSDT"],
            )
        ],
    )
    router = ExchangeRouter(
        adapters={ExchangeID.BINANCE: _BinanceValidatorAdapter(missing=[])},
        primary=ExchangeID.BINANCE,
    )
    asyncio.run(validate_exchange_symbol_sync(router, config))


def test_validate_exchange_symbol_sync_raises_in_live_for_missing_symbols():
    config = DarwinConfig(
        mode="live",
        exchanges=[
            ExchangeConfig(
                exchange_id="binance",
                enabled=True,
                symbols=["BTCUSDT", "BADPAIR"],
            )
        ],
    )
    router = ExchangeRouter(
        adapters={ExchangeID.BINANCE: _BinanceValidatorAdapter(missing=["BADPAIR"])},
        primary=ExchangeID.BINANCE,
    )

    try:
        asyncio.run(validate_exchange_symbol_sync(router, config))
    except RuntimeError as exc:
        assert "BADPAIR" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for invalid live Binance symbols")
