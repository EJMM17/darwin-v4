from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests


@dataclass(slots=True)
class BinanceCredentials:
    api_key: str
    api_secret: str


class BinanceFuturesClient:
    def __init__(self, credentials: BinanceCredentials, timeout_s: float = 10.0) -> None:
        self._credentials = credentials
        self._timeout_s = timeout_s
        self._base_url = "https://fapi.binance.com"
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": credentials.api_key})

    def _sign(self, params: dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        params["signature"] = self._sign(params)
        url = f"{self._base_url}{path}"
        response = self._session.request(method, url, params=params, timeout=self._timeout_s)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._session.close()

    def ping_futures(self) -> None:
        response = self._session.get(f"{self._base_url}/fapi/v1/time", timeout=self._timeout_s)
        response.raise_for_status()

    def get_wallet_balance(self) -> float:
        payload = self._signed_request("GET", "/fapi/v2/account")
        return float(payload.get("totalWalletBalance", 0.0))

    def get_open_positions(self) -> list[dict[str, Any]]:
        payload = self._signed_request("GET", "/fapi/v2/positionRisk")
        if not isinstance(payload, list):
            return []
        return [p for p in payload if float(p.get("positionAmt", 0.0)) != 0.0]

    def get_unrealized_pnl(self) -> float:
        payload = self._signed_request("GET", "/fapi/v2/account")
        return float(payload.get("totalUnrealizedProfit", 0.0))

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        payload = self._signed_request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
        )
        return int(payload.get("leverage", 0)) == leverage

    def validate_startup(self, symbols: list[str], leverage: int) -> dict[str, Any]:
        self.ping_futures()
        wallet = self.get_wallet_balance()
        _ = self.get_unrealized_pnl()
        positions = self.get_open_positions()
        leverage_result: dict[str, bool] = {}
        for symbol in symbols:
            leverage_result[symbol] = self.set_leverage(symbol, leverage)
        return {
            "wallet_balance": wallet,
            "open_positions": positions,
            "leverage_result": leverage_result,
        }
