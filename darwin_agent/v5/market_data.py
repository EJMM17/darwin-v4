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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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
    z_score_ou: float = 0.0        # z-score computed with OU-calibrated window
    distance_from_mean_20: float = 0.0
    ou_half_life: float = 20.0     # Ornstein-Uhlenbeck half-life in bars
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
        # Thread safety: compute_features is called via asyncio.to_thread
        # from multiple coroutines.
        self._lock = threading.Lock()
        # Per-thread HTTP sessions to avoid requests.Session thread-safety issues.
        # requests.Session is NOT thread-safe: concurrent requests corrupt the
        # internal connection pool (urllib3 PoolManager). The old code grabbed
        # a reference to self._client._session under lock but then used it
        # OUTSIDE the lock, creating a race condition.
        # Fix: each thread gets its own session via threading.local().
        self._thread_local = threading.local()
        # Store API key for creating per-thread sessions
        self._api_key = binance_client._session.headers.get("X-MBX-APIKEY", "")
        self._base_url = binance_client._base_url
        self._timeout_s = binance_client._timeout_s

    def _get_session(self) -> "requests.Session":
        """Get or create a thread-local requests.Session."""
        import requests as _requests
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = _requests.Session()
            session.headers.update({"X-MBX-APIKEY": self._api_key})
            self._thread_local.session = session
        return session

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles from Binance.

        Uses cache if data is fresh enough. Lock is only held for cache
        reads/writes — released during the HTTP call to avoid serializing
        all concurrent symbol fetches behind one slow request.
        """
        cache_key = (symbol, interval)
        now = time.time()

        # 1. Check cache under lock
        with self._lock:
            if cache_key in self._cache:
                ts, candles = self._cache[cache_key]
                if now - ts < self._cache_ttl_s:
                    return candles

        # 2. Fetch from exchange WITHOUT holding the lock.
        # Uses thread-local session (each thread has its own connection pool).
        try:
            session = self._get_session()

            response = session.get(
                f"{self._base_url}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=self._timeout_s,
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

            # 3. Write to cache under lock
            with self._lock:
                self._cache[cache_key] = (now, candles)
            return candles

        except Exception as exc:
            logger.warning("failed to fetch candles for %s: %s", symbol, exc)
            # Return stale cached data if available (stale > nothing)
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key][1]
            return []

    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch the current funding rate for a symbol."""
        try:
            session = self._get_session()

            response = session.get(
                f"{self._base_url}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
                timeout=self._timeout_s,
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

        # Z-scores (fixed windows)
        z_20 = _z_score(closes, 20)
        z_50 = _z_score(closes, 50)

        # Distance from mean
        dist_20 = _distance_from_mean(closes, 20)

        # OU half-life calibration (Kakushadze & Serur §9 Ornstein-Uhlenbeck)
        # half_life = ln(2) / kappa, where kappa is fitted from price residuals.
        # This gives the dynamic window that best captures mean-reversion in the
        # current market conditions, instead of a fixed 20 or 50-bar window.
        ou_hl = _ou_half_life(closes, max_window=100)
        ou_window = max(5, min(100, int(round(ou_hl))))
        z_ou = _z_score(closes, ou_window)

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
            z_score_ou=z_ou,
            distance_from_mean_20=dist_20,
            ou_half_life=ou_hl,
            adx_14=adx_14[-1] if adx_14 else 0.0,
            returns=returns,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
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
    """
    Normalized rate of change over period.

    Kakushadze & Serur §18.2 eq.524: R̂(t) = [R(t) - R̄(t,T)] / σ(t,T)

    Normalizes the current bar return by the mean and std of bar returns
    over the lookback window, making momentum scale-invariant across
    volatility regimes. A 2%% move in a low-vol regime and a 2%% move in
    a high-vol regime now carry different weights, preventing false signals
    when the market transitions between regimes.
    """
    if len(closes) <= period or closes[-period - 1] == 0:
        return 0.0

    # Compute rolling single-bar returns over the window for normalization
    window = closes[-(period + 1):]  # period+1 prices -> period returns
    if len(window) < 3:
        # fallback: raw return
        return (closes[-1] - closes[-period - 1]) / closes[-period - 1]

    bar_returns = [
        (window[i] - window[i - 1]) / window[i - 1]
        for i in range(1, len(window))
        if window[i - 1] != 0
    ]
    if len(bar_returns) < 2:
        return (closes[-1] - closes[-period - 1]) / closes[-period - 1]

    mu = sum(bar_returns) / len(bar_returns)
    variance = sum((r - mu) ** 2 for r in bar_returns) / len(bar_returns)
    sigma = variance ** 0.5

    if sigma < 1e-10:
        return 0.0  # flat market, no meaningful signal

    # Normalize the most recent bar return vs the period distribution
    current_bar = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0.0
    return (current_bar - mu) / sigma


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

def _ou_half_life(closes: List[float], max_window: int = 100) -> float:
    """
    Estimate the Ornstein-Uhlenbeck half-life of mean reversion.

    Kakushadze & Serur §9: dX(t) = κ[a - X(t)]dt + σdW(t)
    The half-life = ln(2) / κ tells us how long it takes for a deviation
    from the mean to decay by half. This is the optimal lookback window
    for mean-reversion strategies.

    Method: OLS regression of ΔX(t) on X(t-1) (Dickey-Fuller style)
        ΔX(t) = α + β·X(t-1) + ε
        κ ≈ -β  (if β < 0, the process is mean-reverting)
        half_life = ln(2) / κ

    Returns half_life in bars. Falls back to 20 if regression fails or
    if the series is not mean-reverting (β >= 0).
    """
    import math as _math
    n = min(len(closes), max_window)
    if n < 20:
        return 20.0

    window = closes[-n:]

    # ΔX(t) = X(t) - X(t-1), X_lag = X(t-1)
    x_lag = window[:-1]
    delta_x = [window[i] - window[i - 1] for i in range(1, len(window))]

    if len(x_lag) < 10:
        return 20.0

    # OLS: delta_x = alpha + beta * x_lag
    beta, _ = _ols_simple(x_lag, delta_x)

    # beta should be negative for mean-reversion; kappa = -beta
    if beta >= 0:
        # Not mean-reverting in this window; return sensible default
        return 20.0

    kappa = -beta
    if kappa < 1e-6:
        return 100.0  # extremely slow reversion

    half_life = _math.log(2.0) / kappa

    # Clamp to [3, max_window] bars
    return max(3.0, min(float(max_window), half_life))


def _ols_simple(x: List[float], y: List[float]) -> tuple:
    """OLS regression y = alpha + beta * x. Returns (beta, alpha)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
    var = sum((xi - x_mean) ** 2 for xi in x) / n
    if var < 1e-20:
        return 0.0, y_mean
    beta = cov / var
    alpha = y_mean - beta * x_mean
    return beta, alpha


def compute_range_context(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    window: int = 48,
    atr_pct: float = 0.0,
) -> dict:
    """
    Detecta el rango de mercado (soporte/resistencia) para estrategias
    de mean-reversion en mercados laterales.

    Calcula:
      - range_high: resistencia (max de highs en la ventana)
      - range_low:  soporte (min de lows en la ventana)
      - range_pct:  amplitud del rango como % del precio
      - price_position: 0.0 = en soporte, 1.0 = en resistencia
      - near_support: True si precio está en el 20% inferior del rango
      - near_resistance: True si precio está en el 20% superior del rango
      - is_ranging: True si el rango es comprimido relative to the asset's vol

    Uso: Solo abrir longs cerca de soporte, cortos cerca de resistencia.

    The is_ranging threshold adapts to the asset's volatility:
      - BTC with ATR 0.8%: threshold = 5% (default)
      - PIPPIN with ATR 10%: threshold = max(5%, 3 * 10%) = 30%
    This prevents memecoins from never being classified as ranging.
    """
    n = min(len(closes), min(len(highs), len(lows)))
    if n < 10:
        return {
            "range_high": closes[-1] if closes else 0,
            "range_low": closes[-1] if closes else 0,
            "range_pct": 0.0,
            "price_position": 0.5,
            "near_support": False,
            "near_resistance": False,
            "is_ranging": False,
        }

    w = min(window, n)
    h_window = highs[-w:]
    l_window = lows[-w:]
    c_window = closes[-w:]

    range_high = max(h_window)
    range_low  = min(l_window)
    current    = closes[-1]

    if range_low <= 0 or range_high <= range_low:
        return {
            "range_high": range_high,
            "range_low": range_low,
            "range_pct": 0.0,
            "price_position": 0.5,
            "near_support": False,
            "near_resistance": False,
            "is_ranging": False,
        }

    range_pct = (range_high - range_low) / range_low

    # Posición del precio dentro del rango (0=soporte, 1=resistencia)
    price_position = (current - range_low) / (range_high - range_low)

    # Zona exterior = 20% desde los extremos
    EDGE_ZONE = 0.20
    near_support    = price_position <= EDGE_ZONE
    near_resistance = price_position >= (1.0 - EDGE_ZONE)

    # Mercado en rango si la amplitud es pequeña relative to the asset's volatility.
    # For BTC (ATR ~0.8%): threshold = 5% (48h range < 5% = ranging)
    # For memecoins (ATR ~10%): threshold = 30% (48h range < 30% = ranging for this asset)
    # This allows memecoins to be classified as ranging during consolidation phases.
    ranging_threshold = max(0.05, 3.0 * atr_pct) if atr_pct > 0 else 0.05
    is_ranging = range_pct < ranging_threshold

    return {
        "range_high": range_high,
        "range_low": range_low,
        "range_pct": range_pct,
        "price_position": price_position,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "is_ranging": is_ranging,
    }
