"""
Darwin v5 — Order Flow Intelligence.

Detects whale manipulation and market microstructure signals BEFORE
price moves, giving the bot a 5-30 second edge over candle-based signals.

Three layers of protection:

1. Order Book Imbalance (OBI)
   Measures bid vs ask pressure at the top of the book.
   Whales accumulate orders BEFORE moving price → visible as imbalance spike.
   Formula: OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)
   Range: -1.0 (pure sell pressure) to +1.0 (pure buy pressure)

2. Trade Flow Delta (TFD)
   Classifies recent trades as buyer-initiated vs seller-initiated.
   Aggressive buying (trades hitting the ask) = bullish flow.
   Aggressive selling (trades hitting the bid) = bearish flow.
   Whale stop-hunt: sudden delta reversal after a price spike.

3. Fake Breakout Detector
   Identifies when a price spike is likely a manipulation trap:
   - Price spikes hard in one direction (spike_ratio > threshold)
   - BUT volume on the spike is LOW relative to average
   - AND order book imbalance is OPPOSITE to spike direction
   → High probability fake breakout: do not enter in spike direction

Usage:
    ofi = OrderFlowIntelligence(binance_client)
    ctx = ofi.analyze("SOLUSDT", current_price=83.32, atr=0.45)
    if ctx.is_fake_breakout:
        # skip or trade opposite direction
    signal_modifier = ctx.combined_score  # -1 to +1
"""
from __future__ import annotations

import logging

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("darwin.v5.order_flow")


@dataclass
class OrderFlowContext:
    """
    Microstructure analysis output for one symbol.

    All scores range from -1.0 (strong bearish) to +1.0 (strong bullish).
    """
    symbol: str = ""

    # Order book imbalance: bid pressure vs ask pressure
    obi: float = 0.0          # -1 to +1
    obi_depth: int = 0        # levels used
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    # Trade flow delta: aggressive buyers vs sellers
    tfd: float = 0.0          # -1 to +1
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0

    # Fake breakout detection
    is_fake_breakout: bool = False
    fake_breakout_direction: str = ""   # "UP" or "DOWN" — the TRAP direction
    spike_ratio: float = 0.0           # how big the spike was vs ATR

    # Combined signal: OBI + TFD weighted
    combined_score: float = 0.0

    # Meta
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obi": round(self.obi, 4),
            "tfd": round(self.tfd, 4),
            "combined": round(self.combined_score, 4),
            "is_fake_breakout": self.is_fake_breakout,
            "fake_direction": self.fake_breakout_direction,
            "spike_ratio": round(self.spike_ratio, 3),
            "bid_qty": round(self.bid_qty, 2),
            "ask_qty": round(self.ask_qty, 2),
            "buy_vol": round(self.buy_volume, 2),
            "sell_vol": round(self.sell_volume, 2),
        }


@dataclass
class OrderFlowConfig:
    """Tunable parameters for order flow analysis."""
    # Order book
    book_depth: int = 10          # how many levels to sum (top 10 bids/asks)
    obi_weight: float = 0.55      # weight of OBI in combined score

    # Trade flow
    trade_lookback: int = 50      # last N trades to analyze
    tfd_weight: float = 0.45      # weight of TFD in combined score

    # Fake breakout detection
    spike_atr_multiplier: float = 0.8   # spike > 0.8x ATR = significant spike
    fake_breakout_volume_ratio: float = 0.7  # spike volume < 0.7x avg = low conviction
    fake_breakout_obi_threshold: float = 0.2  # OBI opposing spike by > 0.2 = fake

    # Caching
    cache_ttl_s: float = 4.0     # reuse result within same tick (5s tick interval)


class OrderFlowIntelligence:
    """
    Real-time order flow analysis using Binance REST API.

    Called once per tick per symbol, results cached for the tick duration
    to avoid redundant API calls.
    """

    def __init__(
        self,
        binance_client: Any,
        config: OrderFlowConfig | None = None,
    ) -> None:
        self._client = binance_client
        self._config = config or OrderFlowConfig()
        # Cache: symbol -> (timestamp, OrderFlowContext)
        self._cache: Dict[str, Tuple[float, OrderFlowContext]] = {}
        # Rolling OBI history per symbol for normalization
        self._obi_history: Dict[str, List[float]] = {}
        self._tfd_history: Dict[str, List[float]] = {}
        self._max_history = 100

    def analyze(
        self,
        symbol: str,
        current_price: float = 0.0,
        atr: float = 0.0,
        prev_price: float = 0.0,
    ) -> OrderFlowContext:
        """
        Fetch and analyze order book + trade flow for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "SOLUSDT".
        current_price : float
            Current mark price (used for spike detection).
        atr : float
            Current ATR-14 (used to normalize spike size).
        prev_price : float
            Previous tick price (used for spike detection).

        Returns
        -------
        OrderFlowContext
            Full microstructure analysis.
        """
        cfg = self._config
        now = time.time()

        # Check cache
        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < cfg.cache_ttl_s:
                return cached

        t_start = time.time()
        ctx = OrderFlowContext(symbol=symbol)

        try:
            # Fetch order book and trades in parallel (both are cheap REST calls)
            obi, bid_qty, ask_qty, depth = self._fetch_order_book_imbalance(symbol)
            tfd, buy_vol, sell_vol, n_trades = self._fetch_trade_flow_delta(symbol)

            ctx.obi = obi
            ctx.obi_depth = depth
            ctx.bid_qty = bid_qty
            ctx.ask_qty = ask_qty
            ctx.tfd = tfd
            ctx.buy_volume = buy_vol
            ctx.sell_volume = sell_vol
            ctx.trade_count = n_trades

            # Update rolling histories for normalization
            self._update_history(symbol, obi, tfd)

            # Normalize OBI and TFD against their own histories
            obi_norm = self._normalize(obi, self._obi_history.get(symbol, [obi]))
            tfd_norm = self._normalize(tfd, self._tfd_history.get(symbol, [tfd]))

            # Combined score: weighted average of normalized signals
            ctx.combined_score = (
                cfg.obi_weight * obi_norm + cfg.tfd_weight * tfd_norm
            )
            ctx.combined_score = max(-1.0, min(1.0, ctx.combined_score))

            # Fake breakout detection
            if current_price > 0 and prev_price > 0 and atr > 0:
                ctx.spike_ratio, ctx.is_fake_breakout, ctx.fake_breakout_direction = (
                    self._detect_fake_breakout(
                        current_price, prev_price, atr, obi,
                        buy_vol, sell_vol, cfg,
                    )
                )

        except Exception as exc:
            ctx.error = str(exc)
            logger.warning("order flow analysis failed for %s: %s", symbol, exc)

        ctx.latency_ms = (time.time() - t_start) * 1000.0
        self._cache[symbol] = (now, ctx)
        return ctx

    def _fetch_order_book_imbalance(
        self, symbol: str
    ) -> Tuple[float, float, float, int]:
        """
        Fetch top-N order book levels and compute imbalance.

        Returns (obi, total_bid_qty, total_ask_qty, depth_levels)
        """
        cfg = self._config
        session = self._client._session
        base_url = self._client._base_url
        timeout = self._client._timeout_s

        response = session.get(
            f"{base_url}/fapi/v1/depth",
            params={"symbol": symbol, "limit": cfg.book_depth},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        bid_qty = sum(float(b[1]) for b in bids)
        ask_qty = sum(float(a[1]) for a in asks)
        total = bid_qty + ask_qty

        if total < 1e-10:
            return 0.0, 0.0, 0.0, 0

        obi = (bid_qty - ask_qty) / total
        return obi, bid_qty, ask_qty, min(len(bids), len(asks))

    def _fetch_trade_flow_delta(
        self, symbol: str
    ) -> Tuple[float, float, float, int]:
        """
        Fetch recent trades and compute buyer vs seller flow delta.

        Binance marks each trade: isBuyerMaker=True means the BUYER was
        the market maker → the trade was SELLER-initiated (aggressive sell).
        isBuyerMaker=False → BUYER-initiated (aggressive buy).

        Returns (tfd, buy_volume, sell_volume, n_trades)
        """
        cfg = self._config
        session = self._client._session
        base_url = self._client._base_url
        timeout = self._client._timeout_s

        response = session.get(
            f"{base_url}/fapi/v1/aggTrades",
            params={"symbol": symbol, "limit": cfg.trade_lookback},
            timeout=timeout,
        )
        response.raise_for_status()
        trades = response.json()

        buy_vol = 0.0
        sell_vol = 0.0

        for t in trades:
            qty = float(t.get("q", 0))
            is_buyer_maker = t.get("m", False)
            if is_buyer_maker:
                # Seller initiated (aggressive sell)
                sell_vol += qty
            else:
                # Buyer initiated (aggressive buy)
                buy_vol += qty

        total_vol = buy_vol + sell_vol
        if total_vol < 1e-10:
            return 0.0, 0.0, 0.0, 0

        tfd = (buy_vol - sell_vol) / total_vol
        return tfd, buy_vol, sell_vol, len(trades)

    def _detect_fake_breakout(
        self,
        current_price: float,
        prev_price: float,
        atr: float,
        obi: float,
        buy_vol: float,
        sell_vol: float,
        cfg: OrderFlowConfig,
    ) -> Tuple[float, bool, str]:
        """
        Detect if the recent price move is a manipulation trap.

        A fake breakout (stop hunt) has 3 characteristics:
        1. Price moved significantly (> spike_atr_multiplier * ATR)
        2. Volume on the spike was LOW (< fake_breakout_volume_ratio * avg)
        3. Order book imbalance is OPPOSITE to the spike direction

        Returns (spike_ratio, is_fake, direction_of_trap)
        """
        price_move = current_price - prev_price
        spike_ratio = abs(price_move) / atr if atr > 0 else 0.0

        # No significant spike
        if spike_ratio < cfg.spike_atr_multiplier:
            return spike_ratio, False, ""

        spike_direction = "UP" if price_move > 0 else "DOWN"

        # Check if OBI opposes the spike direction
        # Spike UP but order book shows sell pressure → fake up spike
        # Spike DOWN but order book shows buy pressure → fake down spike
        obi_opposes = (
            (spike_direction == "UP" and obi < -cfg.fake_breakout_obi_threshold) or
            (spike_direction == "DOWN" and obi > cfg.fake_breakout_obi_threshold)
        )

        # Check if trade volume was low conviction
        total_vol = buy_vol + sell_vol
        # Low conviction: if spike is UP, buy volume should dominate
        # If buy_vol / total < 0.55 during an UP spike → sellers absorbing = fake
        if total_vol > 0 and spike_direction == "UP":
            buy_ratio = buy_vol / total_vol
            low_conviction = buy_ratio < cfg.fake_breakout_volume_ratio
        elif total_vol > 0 and spike_direction == "DOWN":
            sell_ratio = sell_vol / total_vol
            low_conviction = sell_ratio < cfg.fake_breakout_volume_ratio
        else:
            low_conviction = False

        # Fake breakout requires at least 2 of 3 conditions
        # (OBI opposing is the strongest signal, so it alone counts double)
        conditions_met = int(obi_opposes) * 2 + int(low_conviction)
        is_fake = conditions_met >= 2

        return spike_ratio, is_fake, spike_direction if is_fake else ""

    def _update_history(self, symbol: str, obi: float, tfd: float) -> None:
        """Update rolling history for normalization."""
        if symbol not in self._obi_history:
            self._obi_history[symbol] = []
            self._tfd_history[symbol] = []

        self._obi_history[symbol].append(obi)
        self._tfd_history[symbol].append(tfd)

        if len(self._obi_history[symbol]) > self._max_history:
            self._obi_history[symbol] = self._obi_history[symbol][-self._max_history:]
            self._tfd_history[symbol] = self._tfd_history[symbol][-self._max_history:]

    @staticmethod
    def _normalize(value: float, history: List[float]) -> float:
        """
        Normalize value to [-1, 1] using rolling min-max.
        Falls back to raw value if history is too short.
        """
        if len(history) < 5:
            return max(-1.0, min(1.0, value))

        lo = min(history)
        hi = max(history)
        span = hi - lo

        if span < 1e-10:
            return 0.0

        # Map [lo, hi] → [-1, +1]
        normalized = 2.0 * (value - lo) / span - 1.0
        return max(-1.0, min(1.0, normalized))
