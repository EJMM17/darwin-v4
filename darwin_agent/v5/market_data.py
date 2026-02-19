"""
Darwin v5 — Market Data Layer.

Fetches and caches OHLCV data from Binance, computes technical
indicators used by all downstream layers (regime detection, signals,
position sizing).

Usage:
    md = MarketDataLayer(binance_client)
    md.fetch_candles("BTCUSDT", "15m", limit=200)
    features = md.compute_features("BTCUSDT")
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("darwin.v5.market_data")


@dataclass(slots=True)
class OHLCV:
    """Single candlestick bar."""
    timestamp: int  # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class FeatureSet:
    """Computed features for a symbol at current time."""
    symbol: str
    # Price data
    close: float = 0.0
    # Trend indicators
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    # Volatility
    atr_14: float = 0.0
    realized_vol_20: float = 0.0
    realized_vol_60: float = 0.0
    # Momentum
    roc_20: float = 0.0
    roc_50: float = 0.0
    roc_100: float = 0.0
    # Mean reversion
    z_score_20: float = 0.0
    z_score_50: float = 0.0
    distance_from_mean_20: float = 0.0
    # Trend strength
    adx_14: float = 0.0
    # Returns
    returns: List[float] = field(default_factory=list)
    # Raw data for downstream use
    closes: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)


class MarketDataLayer:
    """
    Fetches OHLCV from Binance and computes technical features.

    Maintains an in-memory cache of candle data per symbol/interval
    to avoid redundant API calls within the same tick.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client for fetching klines.
    cache_ttl_s : float
        How long cached candle data remains valid.
    """

    def __init__(
        self,
        binance_client: Any,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._client = binance_client
        self._cache_ttl_s = cache_ttl_s
        # Cache: (symbol, interval) -> (timestamp, candles)
        self._cache: Dict[Tuple[str, str], Tuple[float, List[OHLCV]]] = {}

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles from Binance.

        Uses cache if data is fresh enough.
        """
        cache_key = (symbol, interval)
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            ts, candles = self._cache[cache_key]
            if now - ts < self._cache_ttl_s:
                return candles

        # Fetch from exchange
        try:
            session = self._client._session
            base_url = self._client._base_url
            timeout = self._client._timeout_s

            response = session.get(
                f"{base_url}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=timeout,
            )
            response.raise_for_status()
            raw = response.json()

            candles = []
            for k in raw:
                candles.append(OHLCV(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                ))

            self._cache[cache_key] = (now, candles)
            return candles

        except Exception as exc:
            logger.warning("failed to fetch candles for %s: %s", symbol, exc)
            # Return cached data if available (stale is better than nothing)
            if cache_key in self._cache:
                return self._cache[cache_key][1]
            return []

    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch the current funding rate for a symbol."""
        try:
            session = self._client._session
            base_url = self._client._base_url
            timeout = self._client._timeout_s

            response = session.get(
                f"{base_url}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if data:
                return float(data[-1].get("fundingRate", 0.0))
            return 0.0
        except Exception as exc:
            logger.warning("failed to fetch funding rate for %s: %s", symbol, exc)
            return 0.0

    def compute_features(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
    ) -> FeatureSet:
        """
        Fetch candles and compute full feature set for a symbol.

        Returns a FeatureSet with all technical indicators computed.
        """
        candles = self.fetch_candles(symbol, interval, limit)
        if len(candles) < 50:
            logger.warning("insufficient candles for %s: %d", symbol, len(candles))
            return FeatureSet(symbol=symbol)

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        # Compute indicators
        ema_20 = _ema(closes, 20)
        ema_50 = _ema(closes, 50)
        ema_200 = _ema(closes, min(200, len(closes)))
        atr_14 = _atr(highs, lows, closes, 14)
        adx_14 = _adx(highs, lows, closes, 14)

        # Returns
        returns = _returns(closes)

        # Realized volatility
        rv_20 = _realized_vol(returns, 20)
        rv_60 = _realized_vol(returns, 60)

        # Rate of change
        roc_20 = _roc(closes, 20)
        roc_50 = _roc(closes, 50)
        roc_100 = _roc(closes, min(100, len(closes) - 1))

        # Z-scores
        z_20 = _z_score(closes, 20)
        z_50 = _z_score(closes, 50)

        # Distance from mean
        dist_20 = _distance_from_mean(closes, 20)

        return FeatureSet(
            symbol=symbol,
            close=closes[-1],
            ema_20=ema_20[-1],
            ema_50=ema_50[-1],
            ema_200=ema_200[-1],
            atr_14=atr_14[-1],
            realized_vol_20=rv_20,
            realized_vol_60=rv_60,
            roc_20=roc_20,
            roc_50=roc_50,
            roc_100=roc_100,
            z_score_20=z_20,
            z_score_50=z_50,
            distance_from_mean_20=dist_20,
            adx_14=adx_14[-1] if adx_14 else 0.0,
            returns=returns,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


# ── Technical Indicator Functions ─────────────────────────

def _ema(data: List[float], period: int) -> List[float]:
    """Exponential moving average."""
    n = len(data)
    if n == 0 or period < 1:
        return [0.0] * n
    k = 2.0 / (period + 1)
    out = [data[0]]
    for i in range(1, n):
        out.append(data[i] * k + out[-1] * (1.0 - k))
    return out


def _sma(data: List[float], period: int) -> List[float]:
    """Simple moving average."""
    n = len(data)
    out = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        window = data[start:i + 1]
        out[i] = sum(window) / len(window)
    return out


def _atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[float]:
    """Average True Range."""
    n = len(closes)
    if n == 0:
        return []
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))
    # Wilder's smoothing
    atr = [0.0] * n
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[float]:
    """Average Directional Index (ADX)."""
    n = len(closes)
    if n < period * 2:
        return [0.0] * n

    # True Range
    tr = [highs[0] - lows[0]]
    plus_dm = [0.0]
    minus_dm = [0.0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    # Smoothed TR, +DM, -DM (Wilder's)
    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = [0.0] * n
    minus_di = [0.0] * n
    dx = [0.0] * n

    for i in range(period - 1, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = (smoothed_plus[i] / smoothed_tr[i]) * 100.0
            minus_di[i] = (smoothed_minus[i] / smoothed_tr[i]) * 100.0
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100.0

    # ADX = smoothed DX
    adx = [0.0] * n
    start = 2 * period - 1
    if start < n:
        adx[start] = sum(dx[period - 1:start + 1]) / period
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def _wilder_smooth(data: List[float], period: int) -> List[float]:
    """Wilder's smoothing method."""
    n = len(data)
    out = [0.0] * n
    if n >= period:
        out[period - 1] = sum(data[:period])
        for i in range(period, n):
            out[i] = out[i - 1] - (out[i - 1] / period) + data[i]
    return out


def _returns(closes: List[float]) -> List[float]:
    """Compute log returns."""
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            returns.append(math.log(closes[i] / closes[i - 1]))
        else:
            returns.append(0.0)
    return returns


def _realized_vol(returns: List[float], period: int) -> float:
    """Realized volatility over last N returns."""
    if len(returns) < period:
        period = len(returns)
    if period < 2:
        return 0.0
    window = returns[-period:]
    mu = sum(window) / len(window)
    var = sum((r - mu) ** 2 for r in window) / (len(window) - 1)
    return math.sqrt(var)


def _roc(closes: List[float], period: int) -> float:
    """Rate of change over period."""
    if len(closes) <= period or closes[-period - 1] == 0:
        return 0.0
    return (closes[-1] - closes[-period - 1]) / closes[-period - 1]


def _z_score(closes: List[float], period: int) -> float:
    """Z-score of current price vs rolling mean/std."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mu = sum(window) / len(window)
    var = sum((c - mu) ** 2 for c in window) / len(window)
    std = math.sqrt(var)
    if std < 1e-10:
        return 0.0
    return (closes[-1] - mu) / std


def _distance_from_mean(closes: List[float], period: int) -> float:
    """Distance of current price from rolling mean, as percentage."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mu = sum(window) / len(window)
    if mu == 0:
        return 0.0
    return (closes[-1] - mu) / mu
