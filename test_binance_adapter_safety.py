import asyncio

from darwin_agent.exchanges.binance import BinanceAdapter
from darwin_agent.interfaces.enums import OrderSide, OrderType
from darwin_agent.interfaces.types import OrderRequest, Position


class _FakeBinance(BinanceAdapter):
    __slots__ = ("responses", "positions", "last_request")

    def __init__(self):
        super().__init__("k", "s")
        self.responses = []
        self.positions = []
        self.last_request = None

    async def _signed(self, method: str, path: str, params=None):
        self.last_request = {"method": method, "path": path, "params": params or {}}
        if self.responses:
            return self.responses.pop(0)
        return {}

    async def get_positions(self):
        return list(self.positions)


def test_place_order_passes_reduce_only_flag():
    adapter = _FakeBinance()
    adapter.responses = [{"orderId": "1", "executedQty": "0.01", "avgPrice": "50000"}]

    req = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.01,
        reduce_only=True,
    )
    res = asyncio.run(adapter.place_order(req))

    assert res.success is True
    assert adapter.last_request["params"].get("reduceOnly") == "true"


def test_close_position_fails_if_exchange_returns_error_payload():
    adapter = _FakeBinance()
    adapter.positions = [
        Position(symbol="BTCUSDT", side=OrderSide.BUY, size=0.2, entry_price=50000)
    ]
    adapter.responses = [{"code": -2011, "msg": "Order would immediately trigger"}]

    res = asyncio.run(adapter.close_position("BTCUSDT", OrderSide.BUY))
    assert res.success is False
    assert "trigger" in res.error.lower()


def test_close_position_targets_requested_side_in_hedge_mode():
    adapter = _FakeBinance()
    adapter.positions = [
        Position(symbol="BTCUSDT", side=OrderSide.SELL, size=0.5, entry_price=50010),
        Position(symbol="BTCUSDT", side=OrderSide.BUY, size=0.1, entry_price=50000),
    ]
    adapter.responses = [{"orderId": "99"}]

    res = asyncio.run(adapter.close_position("BTCUSDT", OrderSide.BUY))
    assert res.success is True
    # Closing BUY should send SELL with BUY position size (0.1), not SELL size (0.5)
    assert adapter.last_request["params"]["side"] == "SELL"
    assert adapter.last_request["params"]["quantity"] == "0.1"


def test_set_leverage_returns_false_on_binance_error_code():
    adapter = _FakeBinance()
    adapter.responses = [{"code": -1021, "msg": "Timestamp outside recvWindow"}]

    ok = asyncio.run(adapter.set_leverage("BTCUSDT", 20))
    assert ok is False
