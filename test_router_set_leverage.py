import asyncio

from darwin_agent.exchanges.router import ExchangeRouter
from darwin_agent.interfaces.enums import ExchangeID


class _AdapterTrue:
    async def set_leverage(self, symbol, leverage):
        return True


class _AdapterFalse:
    async def set_leverage(self, symbol, leverage):
        return False


def test_set_leverage_reflects_adapter_result_true():
    router = ExchangeRouter(
        adapters={ExchangeID.BYBIT: _AdapterTrue()},
        primary=ExchangeID.BYBIT,
    )
    assert asyncio.run(router.set_leverage("BTCUSDT", 10)) is True


def test_set_leverage_reflects_adapter_result_false():
    router = ExchangeRouter(
        adapters={ExchangeID.BYBIT: _AdapterFalse()},
        primary=ExchangeID.BYBIT,
    )
    assert asyncio.run(router.set_leverage("BTCUSDT", 10)) is False
