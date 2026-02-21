"""
Darwin v5 — Backtester.

Replays the full signal pipeline on historical Binance kline data to
validate the strategy's edge before risking real capital.

Simulates:
  - Market data layer (from historical candles)
  - Regime detection
  - Multi-factor signal generation
  - Position sizing (with all scaling factors)
  - SL/TP/trailing stop execution
  - Realistic fees (taker 0.04%) and slippage (1-3 bps)
  - Circuit breakers and kill switches

Usage:
    python -m darwin_agent.v5.backtester \
        --symbols BTCUSDT ETHUSDT SOLUSDT \
        --start 2024-01-01 --end 2025-01-01 \
        --equity 80.0 --leverage 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# Local imports — reuse the real pipeline modules
from darwin_agent.v5.market_data import (
    OHLCV,
    FeatureSet,
    MarketDataLayer,
    _ema,
    _atr,
    _adx,
    _returns,
    _realized_vol,
    _roc,
    _z_score,
    _distance_from_mean,
    _ou_half_life,
)
from darwin_agent.v5.regime_detector import RegimeDetector, RegimeState
from darwin_agent.v5.signal_generator import SignalGenerator, TradeSignal
from darwin_agent.v5.position_sizer import PositionSizer, SizerConfig, SizeResult
from darwin_agent.v5.performance_analytics import PerformanceAnalytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("darwin.v5.backtest")


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    interval: str = "15m"
    initial_equity: float = 80.0
    leverage: int = 10
    # Fees
    taker_fee_pct: float = 0.04       # 0.04% per side (Binance futures taker)
    slippage_bps: float = 2.0         # 2 bps average slippage
    # Risk
    base_risk_pct: float = 1.5
    confidence_threshold: float = 0.55
    confidence_threshold_trending: float = 0.55
    confidence_threshold_ranging: float = 0.70
    # SL/TP
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 3.0
    atr_sl_multiplier: float = 2.0
    sl_pct_trending: float = 1.5
    tp_pct_trending: float = 3.0
    sl_pct_ranging: float = 0.7
    tp_pct_ranging: float = 1.2
    # Trailing
    trailing_stop_enabled: bool = True
    trailing_activation_pct: float = 1.5
    trailing_distance_pct: float = 0.8
    # Kill switches
    max_daily_loss_pct: float = 5.0
    max_drawdown_kill_pct: float = 20.0
    max_loss_streak: int = 8                # allow more losses before halting
    # Cooldowns
    post_loss_cooldown_bars: int = 2
    ranging_trade_cooldown_bars: int = 20   # ~5h at 15m bars
    min_atr_pct_to_trade: float = 0.15


# ── Synthetic Data Generation ────────────────────────────────────────────

import random

def generate_synthetic_candles(
    symbol: str,
    start_date: str,
    end_date: str,
    interval_minutes: int = 15,
    seed: int = 42,
) -> List[OHLCV]:
    """
    Generate realistic synthetic crypto OHLCV data using Geometric Brownian
    Motion with regime switching (trending/ranging/volatile).

    This produces data with statistical properties matching real crypto:
    - Fat tails (kurtosis > 3)
    - Volatility clustering
    - Regime changes (trending -> ranging -> volatile)
    - Realistic volume patterns (higher at support/resistance)
    """
    rng = random.Random(seed if symbol == "BTCUSDT" else hash(symbol) + seed)

    # Starting prices per symbol
    start_prices = {
        "BTCUSDT": 42000.0,
        "ETHUSDT": 2300.0,
        "SOLUSDT": 100.0,
        "BNBUSDT": 300.0,
    }
    price = start_prices.get(symbol, 1000.0)

    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    interval_ms = interval_minutes * 60 * 1000

    candles: List[OHLCV] = []
    current_ts = start_ts

    # Regime state machine
    regime = "trending_up"  # trending_up, trending_down, ranging, volatile
    regime_duration = 0
    regime_target = rng.randint(200, 600)  # bars per regime

    # Volatility clustering (GARCH-like)
    base_vol = 0.002  # 0.2% per 15min bar
    current_vol = base_vol

    while current_ts < end_ts:
        # Regime transitions
        regime_duration += 1
        if regime_duration >= regime_target:
            regime_duration = 0
            regime_target = rng.randint(200, 600)
            old_regime = regime
            regime = rng.choice(["trending_up", "trending_down", "ranging", "volatile"])
            # Avoid immediate same-direction trend reversal
            if old_regime == "trending_up" and regime == "trending_up":
                regime = "ranging"
            elif old_regime == "trending_down" and regime == "trending_down":
                regime = "ranging"

        # Drift and volatility based on regime
        if regime == "trending_up":
            drift = 0.0003  # ~0.03% per bar = ~2.9% per day
            vol_mult = 1.0
        elif regime == "trending_down":
            drift = -0.0003
            vol_mult = 1.2  # slightly higher vol in downtrends
        elif regime == "ranging":
            drift = 0.0
            vol_mult = 0.6  # lower vol in range
        else:  # volatile
            drift = 0.0001 * (1 if rng.random() > 0.5 else -1)
            vol_mult = 2.5  # high vol spikes

        # GARCH-like volatility clustering
        vol_shock = rng.gauss(0, base_vol * 0.3)
        current_vol = max(
            base_vol * 0.3,
            0.9 * current_vol + 0.1 * abs(vol_shock) + base_vol * 0.05
        )
        effective_vol = current_vol * vol_mult

        # Generate return with fat tails (mix of normal + jump)
        normal_return = rng.gauss(drift, effective_vol)
        # 3% chance of a jump (fat tails)
        if rng.random() < 0.03:
            jump = rng.gauss(0, effective_vol * 3.0)
            normal_return += jump

        # Apply return
        new_price = price * math.exp(normal_return)

        # Generate realistic OHLC within the bar
        intrabar_vol = effective_vol * 0.5
        high_ext = abs(rng.gauss(0, intrabar_vol))
        low_ext = abs(rng.gauss(0, intrabar_vol))

        open_price = price
        close_price = new_price
        high_price = max(open_price, close_price) * (1.0 + high_ext)
        low_price = min(open_price, close_price) * (1.0 - low_ext)

        # Volume: higher during volatile regimes and at price extremes
        base_volume = 1000.0 * (price / 42000.0)  # scale with price
        vol_scale = vol_mult * (1.0 + abs(normal_return) * 50.0)
        volume = base_volume * vol_scale * rng.uniform(0.5, 2.0)

        candles.append(OHLCV(
            timestamp=current_ts,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 2),
        ))

        price = new_price
        current_ts += interval_ms

    logger.info(
        "generated %d synthetic candles for %s (%.0f -> %.0f)",
        len(candles), symbol, candles[0].close if candles else 0, candles[-1].close if candles else 0,
    )
    return candles


# ── Historical Data Fetching ─────────────────────────────────────────────

def fetch_historical_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> List[OHLCV]:
    """
    Fetch historical klines from Binance public API.
    Paginates automatically (max 1500 per request).
    """
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_candles: List[OHLCV] = []
    current_ts = start_ts

    logger.info("fetching %s %s from %s to %s ...", symbol, interval, start_date, end_date)

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": 1500,
        }
        for attempt in range(4):
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                raw = resp.json()
                break
            except Exception as exc:
                if attempt == 3:
                    logger.error("failed to fetch %s after 4 retries: %s", symbol, exc)
                    return all_candles
                wait = 2 ** (attempt + 1)
                logger.warning("retry %d for %s: %s (waiting %ds)", attempt + 1, symbol, exc, wait)
                time.sleep(wait)

        if not raw:
            break

        for k in raw:
            all_candles.append(OHLCV(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))

        # Move past the last candle
        current_ts = int(raw[-1][0]) + 1

        # Rate limit protection
        time.sleep(0.2)

    logger.info("fetched %d candles for %s", len(all_candles), symbol)
    return all_candles


def compute_features_from_candles(
    symbol: str,
    candles: List[OHLCV],
    window_end: int,
    lookback: int = 200,
) -> FeatureSet:
    """Compute features from a slice of historical candles."""
    start = max(0, window_end - lookback)
    window = candles[start:window_end]

    if len(window) < 50:
        return FeatureSet(symbol=symbol)

    closes = [c.close for c in window]
    highs = [c.high for c in window]
    lows = [c.low for c in window]
    volumes = [c.volume for c in window]

    ema_20 = _ema(closes, 20)
    ema_50 = _ema(closes, 50)
    ema_200 = _ema(closes, min(200, len(closes)))
    atr_14 = _atr(highs, lows, closes, 14)
    adx_14 = _adx(highs, lows, closes, 14)
    returns = _returns(closes)
    rv_20 = _realized_vol(returns, 20)
    rv_60 = _realized_vol(returns, 60)
    roc_20 = _roc(closes, 20)
    roc_50 = _roc(closes, 50)
    roc_100 = _roc(closes, min(100, len(closes) - 1))
    z_20 = _z_score(closes, 20)
    z_50 = _z_score(closes, 50)
    dist_20 = _distance_from_mean(closes, 20)
    ou_hl = _ou_half_life(closes, max_window=100)
    ou_window = max(5, min(100, int(round(ou_hl))))
    z_ou = _z_score(closes, ou_window)

    return FeatureSet(
        symbol=symbol,
        close=closes[-1],
        ema_20=ema_20[-1],
        ema_50=ema_50[-1],
        ema_200=ema_200[-1],
        atr_14=atr_14[-1] if atr_14 else 0.0,
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


# ── Position Tracking ────────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    """A simulated open position."""
    symbol: str
    side: str          # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    entry_bar: int
    trailing_peak_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class ClosedTrade:
    """A completed trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_pct: float
    fees_usdt: float
    entry_bar: int
    exit_bar: int
    exit_reason: str


# ── Backtester Engine ────────────────────────────────────────────────────

class Backtester:
    """
    Event-driven backtester that replays the Darwin v5 pipeline
    on historical data.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config
        self.equity = config.initial_equity
        self.peak_equity = config.initial_equity
        self.daily_start_equity = config.initial_equity

        # Pipeline modules (same as live)
        self.regime_detector = RegimeDetector()
        self.signal_generator = SignalGenerator()
        self.position_sizer = PositionSizer(SizerConfig(
            base_risk_pct=config.base_risk_pct,
            leverage=config.leverage,
        ))
        self.analytics = PerformanceAnalytics(initial_equity=config.initial_equity)

        # State
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[int, float]] = []  # (timestamp, equity)
        self.total_fees: float = 0.0
        self.current_day: int = -1
        self.trading_halted: bool = False
        self.halt_reason: str = ""
        self.cooldown_until: Dict[str, int] = {}   # symbol -> bar index
        self.last_trade_bar: Dict[str, int] = {}    # symbol -> bar index

        # Step sizes for common symbols
        self._step_sizes = {
            "BTCUSDT": 0.001,
            "ETHUSDT": 0.01,
            "SOLUSDT": 0.1,
            "BNBUSDT": 0.01,
            "XRPUSDT": 0.1,
            "DOGEUSDT": 1.0,
            "ADAUSDT": 1.0,
            "AVAXUSDT": 0.1,
        }

    def run(
        self,
        candles_by_symbol: Dict[str, List[OHLCV]],
    ) -> Dict[str, Any]:
        """
        Run the backtest over all historical candles.

        Returns a summary dict with performance metrics.
        """
        # Find the symbol with fewest candles to determine loop range
        btc_candles = candles_by_symbol.get("BTCUSDT", [])
        if not btc_candles:
            logger.error("BTCUSDT candles required for backtest")
            return {}

        n_bars = min(len(c) for c in candles_by_symbol.values())
        warmup = 200  # need 200 bars for indicators

        if n_bars <= warmup:
            logger.error("insufficient candles: %d (need > %d)", n_bars, warmup)
            return {}

        logger.info(
            "starting backtest: %d symbols, %d bars, $%.2f equity, %dx leverage",
            len(self.cfg.symbols), n_bars - warmup, self.cfg.initial_equity, self.cfg.leverage,
        )

        for bar_idx in range(warmup, n_bars):
            self._process_bar(bar_idx, candles_by_symbol)

        return self._compute_results()

    def _process_bar(
        self,
        bar_idx: int,
        candles_by_symbol: Dict[str, List[OHLCV]],
    ) -> None:
        """Process a single bar across all symbols."""
        btc_candles = candles_by_symbol["BTCUSDT"]
        current_ts = btc_candles[bar_idx].timestamp

        # Daily reset check
        bar_day = datetime.datetime.utcfromtimestamp(current_ts / 1000).toordinal()
        if self.current_day != bar_day:
            self.current_day = bar_day
            unrealized = self._compute_unrealized_pnl(candles_by_symbol, bar_idx)
            self.daily_start_equity = self.equity + unrealized
            # Reset daily halts
            if self.trading_halted and "daily_loss" in self.halt_reason:
                self.trading_halted = False
                self.halt_reason = ""

        # Compute unrealized PnL for equity tracking
        unrealized_pnl = self._compute_unrealized_pnl(candles_by_symbol, bar_idx)
        current_equity = self.equity + unrealized_pnl

        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown_pct = 0.0
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100.0

        # Record equity curve
        self.equity_curve.append((current_ts, current_equity))
        self.analytics.update_equity(current_equity)

        # Kill switch checks
        self._check_kill_switches(current_equity, drawdown_pct)
        if self.trading_halted:
            # Still manage open positions for SL/TP even when halted
            self._manage_positions(bar_idx, candles_by_symbol, current_equity, drawdown_pct)
            return

        # Manage open positions first (SL/TP/trailing)
        self._manage_positions(bar_idx, candles_by_symbol, current_equity, drawdown_pct)

        # Compute BTC features (needed for all symbols)
        btc_features = compute_features_from_candles("BTCUSDT", btc_candles, bar_idx)

        # Process each symbol
        for symbol in self.cfg.symbols:
            self._process_symbol(
                symbol, bar_idx, candles_by_symbol, btc_features,
                current_equity, drawdown_pct,
            )

    def _compute_unrealized_pnl(
        self,
        candles_by_symbol: Dict[str, List[OHLCV]],
        bar_idx: int,
    ) -> float:
        """Compute total unrealized PnL of open positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in candles_by_symbol and bar_idx < len(candles_by_symbol[symbol]):
                current_price = candles_by_symbol[symbol][bar_idx].close
                if pos.side == "LONG":
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                total += pnl_pct * pos.entry_price * pos.quantity
        return total

    def _check_kill_switches(self, equity: float, drawdown_pct: float) -> None:
        """Check daily loss and max drawdown limits."""
        if self.trading_halted:
            return

        # Daily loss
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100.0
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                self.trading_halted = True
                self.halt_reason = f"daily_loss_limit: -{daily_loss_pct:.2f}%"
                logger.warning("HALT: %s", self.halt_reason)

        # Max drawdown
        if drawdown_pct >= self.cfg.max_drawdown_kill_pct:
            self.trading_halted = True
            self.halt_reason = f"max_drawdown: -{drawdown_pct:.2f}%"
            logger.warning("HALT: %s", self.halt_reason)

        # Circuit breaker (loss streak)
        daily_pnl = equity - self.daily_start_equity
        breached, reason = self.analytics.circuit_breaker_check(
            daily_pnl_usdt=daily_pnl,
            daily_start_equity=self.daily_start_equity,
            max_loss_streak=self.cfg.max_loss_streak,
        )
        if breached:
            self.trading_halted = True
            self.halt_reason = f"circuit_breaker: {reason}"
            logger.warning("HALT: %s", self.halt_reason)

    def _manage_positions(
        self,
        bar_idx: int,
        candles_by_symbol: Dict[str, List[OHLCV]],
        equity: float,
        drawdown_pct: float,
    ) -> None:
        """Check all open positions for SL/TP/trailing stop."""
        to_close: List[Tuple[str, str]] = []  # (symbol, reason)

        for symbol, pos in self.positions.items():
            if symbol not in candles_by_symbol or bar_idx >= len(candles_by_symbol[symbol]):
                continue

            candle = candles_by_symbol[symbol][bar_idx]
            current_price = candle.close

            # Use high/low of the bar for realistic SL/TP fill simulation
            bar_high = candle.high
            bar_low = candle.low

            # PnL %
            if pos.side == "LONG":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                worst_price = bar_low  # worst case for longs
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                worst_price = bar_high  # worst case for shorts

            # Compute SL/TP based on regime
            features = compute_features_from_candles(
                symbol, candles_by_symbol[symbol], bar_idx
            )
            atr_pct = 0.0
            if features.atr_14 > 0 and features.close > 0:
                atr_pct = features.atr_14 / features.close

            regime = self.regime_detector.detect(features)
            is_ranging = regime.regime.value in ("range_bound", "low_vol")

            if is_ranging:
                sl_floor = self.cfg.sl_pct_ranging / 100.0
                tp_pct = self.cfg.tp_pct_ranging / 100.0
            else:
                sl_floor = self.cfg.sl_pct_trending / 100.0
                tp_pct = self.cfg.tp_pct_trending / 100.0

            sl_pct = max(sl_floor, self.cfg.atr_sl_multiplier * atr_pct)

            # Check worst-case price within the bar for SL
            if pos.side == "LONG":
                worst_pnl = (bar_low - pos.entry_price) / pos.entry_price
            else:
                worst_pnl = (pos.entry_price - bar_high) / pos.entry_price

            # Trailing stop
            trail_sl = None
            if self.cfg.trailing_stop_enabled and pnl_pct > 0:
                pos.trailing_peak_pnl = max(pos.trailing_peak_pnl, pnl_pct)
                if pos.trailing_peak_pnl >= self.cfg.trailing_activation_pct / 100.0:
                    trail_sl = pos.trailing_peak_pnl - self.cfg.trailing_distance_pct / 100.0

            # Determine close reason
            close_reason = ""
            if worst_pnl <= -sl_pct:
                close_reason = f"STOP_LOSS ({worst_pnl*100:.2f}%)"
            elif trail_sl is not None and pnl_pct <= trail_sl:
                close_reason = f"TRAILING_STOP ({pnl_pct*100:.2f}%)"
            elif pnl_pct >= tp_pct:
                close_reason = f"TAKE_PROFIT ({pnl_pct*100:.2f}%)"

            if close_reason:
                to_close.append((symbol, close_reason))

        # Execute closes
        for symbol, reason in to_close:
            self._close_position(symbol, bar_idx, candles_by_symbol, reason)

    def _close_position(
        self,
        symbol: str,
        bar_idx: int,
        candles_by_symbol: Dict[str, List[OHLCV]],
        reason: str,
    ) -> None:
        """Close a position and record the trade."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        candle = candles_by_symbol[symbol][bar_idx]

        # Simulate fill price with slippage
        slippage_mult = self.cfg.slippage_bps / 10000.0
        if pos.side == "LONG":
            # Closing a long = selling — slippage worsens the price
            exit_price = candle.close * (1.0 - slippage_mult)
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            exit_price = candle.close * (1.0 + slippage_mult)
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        # For stop losses, simulate fill at the SL price (worst case)
        if "STOP_LOSS" in reason:
            # SL was hit within the bar — fill is at SL price, not close
            features = compute_features_from_candles(
                symbol, candles_by_symbol[symbol], bar_idx
            )
            atr_pct = 0.0
            if features.atr_14 > 0 and features.close > 0:
                atr_pct = features.atr_14 / features.close
            regime = self.regime_detector.detect(features)
            is_ranging = regime.regime.value in ("range_bound", "low_vol")
            sl_floor = (self.cfg.sl_pct_ranging if is_ranging else self.cfg.sl_pct_trending) / 100.0
            sl_pct = max(sl_floor, self.cfg.atr_sl_multiplier * atr_pct)

            if pos.side == "LONG":
                exit_price = pos.entry_price * (1.0 - sl_pct)
            else:
                exit_price = pos.entry_price * (1.0 + sl_pct)
            pnl_pct = -sl_pct  # capped at SL level

        gross_pnl = pnl_pct * pos.entry_price * pos.quantity

        # Fees: entry + exit
        entry_fee = pos.entry_price * pos.quantity * (self.cfg.taker_fee_pct / 100.0)
        exit_fee = exit_price * pos.quantity * (self.cfg.taker_fee_pct / 100.0)
        total_fees = entry_fee + exit_fee
        self.total_fees += total_fees

        net_pnl = gross_pnl - total_fees

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl_usdt=net_pnl,
            pnl_pct=pnl_pct * 100.0,
            fees_usdt=total_fees,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
            exit_reason=reason,
        )
        self.trades.append(trade)

        self.equity += net_pnl
        self.position_sizer.record_trade_pnl(net_pnl)
        self.analytics.record_trade(pnl_usdt=net_pnl, is_win=(net_pnl > 0))

        # Cooldown after stop loss
        if "STOP_LOSS" in reason:
            self.cooldown_until[symbol] = bar_idx + self.cfg.post_loss_cooldown_bars

        logger.debug(
            "CLOSE %s %s pnl=$%.4f (%.2f%%) fees=$%.4f reason=%s",
            pos.side, symbol, net_pnl, pnl_pct * 100, total_fees, reason,
        )

    def _process_symbol(
        self,
        symbol: str,
        bar_idx: int,
        candles_by_symbol: Dict[str, List[OHLCV]],
        btc_features: FeatureSet,
        equity: float,
        drawdown_pct: float,
    ) -> None:
        """Process one symbol: compute features -> signal -> sizing -> entry."""
        if symbol in self.positions:
            return  # already positioned

        # Cooldown check
        if bar_idx < self.cooldown_until.get(symbol, 0):
            return

        candles = candles_by_symbol[symbol]
        if bar_idx >= len(candles):
            return

        # Compute features
        if symbol == "BTCUSDT":
            features = btc_features
        else:
            features = compute_features_from_candles(symbol, candles, bar_idx)

        if features.close <= 0:
            return

        # Detect regime
        regime = self.regime_detector.detect(features, btc_features)
        is_ranging = regime.regime.value in ("range_bound", "low_vol")

        # ATR filter
        if features.atr_14 > 0 and features.close > 0:
            atr_pct = features.atr_14 / features.close * 100.0
            if atr_pct < self.cfg.min_atr_pct_to_trade:
                return

        # Generate signal (no order flow in backtest)
        signal = self.signal_generator.generate(
            features, regime, btc_features, funding_rate=0.0,
        )

        if not signal.passed or not signal.direction:
            return

        # Regime-adaptive confidence threshold
        effective_threshold = (
            self.cfg.confidence_threshold_ranging if is_ranging
            else self.cfg.confidence_threshold_trending
        )
        if signal.confidence < effective_threshold:
            return

        # Ranging trade cooldown
        if is_ranging:
            last_bar = self.last_trade_bar.get(symbol, 0)
            if bar_idx - last_bar < self.cfg.ranging_trade_cooldown_bars:
                return

        # Position sizing
        current_exposure = sum(
            p.entry_price * p.quantity for p in self.positions.values()
        )
        effective_daily_pnl = equity - self.daily_start_equity

        size_result = self.position_sizer.compute(
            equity=equity,
            price=features.close,
            realized_vol=features.realized_vol_20,
            drawdown_pct=drawdown_pct,
            regime_multiplier=regime.risk_multiplier,
            signal_confidence=signal.confidence,
            current_exposure=current_exposure,
            step_size=self._step_sizes.get(symbol, 0.0),
            daily_pnl=effective_daily_pnl,
            daily_start_equity=self.daily_start_equity,
        )

        if not size_result.approved:
            return

        # Simulate entry with slippage
        slippage_mult = self.cfg.slippage_bps / 10000.0
        if signal.direction == "LONG":
            entry_price = features.close * (1.0 + slippage_mult)
        else:
            entry_price = features.close * (1.0 - slippage_mult)

        # Open position
        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            side=signal.direction,
            entry_price=entry_price,
            quantity=size_result.quantity,
            entry_bar=bar_idx,
        )
        self.last_trade_bar[symbol] = bar_idx

        logger.debug(
            "OPEN %s %s qty=%.6f price=%.2f conf=%.3f regime=%s",
            signal.direction, symbol, size_result.quantity, entry_price,
            signal.confidence, regime.regime.value,
        )

    def _compute_results(self) -> Dict[str, Any]:
        """Compute final backtest results."""
        if not self.trades:
            logger.warning("no trades executed in backtest")
            return {"error": "no trades"}

        snap = self.analytics.snapshot()

        wins = [t for t in self.trades if t.pnl_usdt > 0]
        losses = [t for t in self.trades if t.pnl_usdt <= 0]

        total_pnl = sum(t.pnl_usdt for t in self.trades)
        gross_profit = sum(t.pnl_usdt for t in wins)
        gross_loss = sum(abs(t.pnl_usdt) for t in losses)
        win_rate = len(wins) / len(self.trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = sum(t.pnl_usdt for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(abs(t.pnl_usdt) for t in losses) / len(losses) if losses else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Compute return-based Sharpe from equity curve
        if len(self.equity_curve) > 10:
            equities = [e[1] for e in self.equity_curve]
            daily_returns = []
            # Sample at daily intervals (96 bars per day for 15m)
            bars_per_day = 96
            for i in range(bars_per_day, len(equities), bars_per_day):
                prev = equities[i - bars_per_day]
                if prev > 0:
                    daily_returns.append((equities[i] - prev) / prev)

            if len(daily_returns) >= 5:
                mu = sum(daily_returns) / len(daily_returns)
                var = sum((r - mu) ** 2 for r in daily_returns) / len(daily_returns)
                std = var ** 0.5
                sharpe = (mu / std) * (252 ** 0.5) if std > 1e-12 else 0.0

                neg_sq = [r ** 2 for r in daily_returns if r < 0]
                downside_dev = (sum(neg_sq) / len(daily_returns)) ** 0.5 if neg_sq else 0.0
                sortino = (mu / downside_dev) * (252 ** 0.5) if downside_dev > 1e-12 else 0.0
            else:
                sharpe = 0.0
                sortino = 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max drawdown from equity curve
        max_dd = 0.0
        peak = self.cfg.initial_equity
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        total_return_pct = (self.equity - self.cfg.initial_equity) / self.cfg.initial_equity * 100.0
        calmar = (total_return_pct / 100.0) / max_dd if max_dd > 0 else 0.0

        # Per-symbol breakdown
        symbol_stats: Dict[str, Dict] = {}
        for symbol in self.cfg.symbols:
            sym_trades = [t for t in self.trades if t.symbol == symbol]
            if sym_trades:
                sym_wins = [t for t in sym_trades if t.pnl_usdt > 0]
                symbol_stats[symbol] = {
                    "trades": len(sym_trades),
                    "win_rate": len(sym_wins) / len(sym_trades) * 100.0,
                    "pnl": sum(t.pnl_usdt for t in sym_trades),
                    "fees": sum(t.fees_usdt for t in sym_trades),
                }

        results = {
            "initial_equity": self.cfg.initial_equity,
            "final_equity": round(self.equity, 4),
            "total_return_pct": round(total_return_pct, 2),
            "total_pnl": round(total_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "fees_pct_of_gross": round(
                self.total_fees / (gross_profit + gross_loss) * 100.0
                if (gross_profit + gross_loss) > 0 else 0.0, 2
            ),
            "n_trades": len(self.trades),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "expectancy": round(expectancy, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "leverage": self.cfg.leverage,
            "symbol_breakdown": symbol_stats,
            "equity_curve_length": len(self.equity_curve),
        }

        return results


# ── CLI Entry Point ──────────────────────────────────────────────────────

def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print backtest results."""
    if "error" in results:
        print(f"\nBACKTEST FAILED: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("  DARWIN v5 BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Equity:     ${results['initial_equity']:.2f}")
    print(f"  Final Equity:       ${results['final_equity']:.2f}")
    print(f"  Total Return:       {results['total_return_pct']:+.2f}%")
    print(f"  Total PnL:          ${results['total_pnl']:+.4f}")
    print(f"  Total Fees:         ${results['total_fees']:.4f} ({results['fees_pct_of_gross']:.1f}% of volume)")
    print(f"  Leverage:           {results['leverage']}x")
    print("-" * 60)
    print(f"  Trades:             {results['n_trades']}")
    print(f"  Win Rate:           {results['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:      {results['profit_factor']:.3f}")
    print(f"  Avg Win:            ${results['avg_win']:.4f}")
    print(f"  Avg Loss:           ${results['avg_loss']:.4f}")
    print(f"  Expectancy:         ${results['expectancy']:+.4f} per trade")
    print("-" * 60)
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:      {results['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio:       {results['calmar_ratio']:.3f}")
    print(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    print("-" * 60)

    # Grading
    sharpe = results["sharpe_ratio"]
    if sharpe >= 2.0:
        grade = "A (Hedge Fund Grade)"
    elif sharpe >= 1.5:
        grade = "B (Strong)"
    elif sharpe >= 1.0:
        grade = "C (Acceptable)"
    elif sharpe >= 0.5:
        grade = "D (Weak)"
    else:
        grade = "F (No Edge)"

    scalable = results["max_drawdown_pct"] < 20.0 and results["profit_factor"] > 1.5
    print(f"  Grade:              {grade}")
    print(f"  Scalable:           {'YES' if scalable else 'NO'}")

    # Symbol breakdown
    if results.get("symbol_breakdown"):
        print("-" * 60)
        print("  Per-Symbol:")
        for sym, stats in results["symbol_breakdown"].items():
            print(
                f"    {sym:12s} | {stats['trades']:3d} trades | "
                f"WR {stats['win_rate']:.0f}% | "
                f"PnL ${stats['pnl']:+.4f} | "
                f"Fees ${stats['fees']:.4f}"
            )

    print("=" * 60)


def run_monte_carlo_backtest(
    n_seeds: int = 20,
    config: BacktestConfig | None = None,
) -> Dict[str, Any]:
    """Run the backtest across multiple synthetic seeds for statistical significance."""
    cfg = config or BacktestConfig()
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    int_min = interval_minutes.get(cfg.interval, 15)

    all_results = []
    for seed in range(1, n_seeds + 1):
        candles_by_symbol: Dict[str, List[OHLCV]] = {}
        for symbol in cfg.symbols:
            candles_by_symbol[symbol] = generate_synthetic_candles(
                symbol, cfg.start_date, cfg.end_date,
                interval_minutes=int_min, seed=seed * 100,
            )

        bt = Backtester(cfg)
        result = bt.run(candles_by_symbol)
        if "error" not in result:
            all_results.append(result)
            logger.info(
                "seed %d: return=%.2f%% trades=%d WR=%.0f%% sharpe=%.3f",
                seed, result["total_return_pct"], result["n_trades"],
                result["win_rate_pct"], result["sharpe_ratio"],
            )

    if not all_results:
        return {"error": "no valid results"}

    # Aggregate
    returns = [r["total_return_pct"] for r in all_results]
    sharpes = [r["sharpe_ratio"] for r in all_results]
    win_rates = [r["win_rate_pct"] for r in all_results]
    trade_counts = [r["n_trades"] for r in all_results]
    max_dds = [r["max_drawdown_pct"] for r in all_results]

    avg_return = sum(returns) / len(returns)
    avg_sharpe = sum(sharpes) / len(sharpes)
    avg_wr = sum(win_rates) / len(win_rates)
    avg_trades = sum(trade_counts) / len(trade_counts)
    avg_dd = sum(max_dds) / len(max_dds)
    pct_profitable = sum(1 for r in returns if r > 0) / len(returns) * 100.0

    print("\n" + "=" * 60)
    print(f"  MONTE CARLO BACKTEST ({n_seeds} seeds)")
    print("=" * 60)
    print(f"  Avg Return:         {avg_return:+.2f}%")
    print(f"  Avg Sharpe:         {avg_sharpe:.3f}")
    print(f"  Avg Win Rate:       {avg_wr:.1f}%")
    print(f"  Avg Trades:         {avg_trades:.0f}")
    print(f"  Avg Max DD:         {avg_dd:.2f}%")
    print(f"  % Seeds Profitable: {pct_profitable:.0f}%")
    print(f"  Best Return:        {max(returns):+.2f}%")
    print(f"  Worst Return:       {min(returns):+.2f}%")
    print(f"  Best Sharpe:        {max(sharpes):.3f}")
    print(f"  Worst Sharpe:       {min(sharpes):.3f}")
    print("=" * 60)

    return {
        "n_seeds": n_seeds,
        "avg_return_pct": round(avg_return, 2),
        "avg_sharpe": round(avg_sharpe, 3),
        "avg_win_rate": round(avg_wr, 1),
        "avg_trades": round(avg_trades, 0),
        "avg_max_dd": round(avg_dd, 2),
        "pct_profitable": round(pct_profitable, 0),
        "best_return": round(max(returns), 2),
        "worst_return": round(min(returns), 2),
        "all_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Darwin v5 Backtester")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="15m", help="Candle interval")
    parser.add_argument("--equity", type=float, default=80.0, help="Initial equity in USDT")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage multiplier")
    parser.add_argument("--save", default="", help="Save results to JSON file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Binance API")
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N",
                        help="Run N seeds of Monte Carlo synthetic backtest")
    args = parser.parse_args()

    config = BacktestConfig(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        initial_equity=args.equity,
        leverage=args.leverage,
    )

    # Monte Carlo mode
    if args.monte_carlo > 0:
        mc_results = run_monte_carlo_backtest(n_seeds=args.monte_carlo, config=config)
        if args.save:
            with open(args.save, "w") as f:
                json.dump(mc_results, f, indent=2, default=str)
            print(f"\nResults saved to {args.save}")
        return mc_results

    # Fetch historical data (or generate synthetic)
    candles_by_symbol: Dict[str, List[OHLCV]] = {}
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    int_min = interval_minutes.get(config.interval, 15)

    for symbol in config.symbols:
        if args.synthetic:
            candles = generate_synthetic_candles(
                symbol, config.start_date, config.end_date,
                interval_minutes=int_min,
            )
        else:
            candles = fetch_historical_klines(
                symbol, config.interval, config.start_date, config.end_date,
            )
        if not candles:
            logger.error("no candles for %s, aborting", symbol)
            sys.exit(1)
        candles_by_symbol[symbol] = candles

    # Run backtest
    backtester = Backtester(config)
    results = backtester.run(candles_by_symbol)

    # Display results
    print_results(results)

    # Save if requested
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")

    return results


if __name__ == "__main__":
    main()
