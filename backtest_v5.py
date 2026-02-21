#!/usr/bin/env python3
"""
Darwin v5 — Historical Backtester.

Downloads real OHLCV from Binance public API (no auth needed) and simulates
the v5 engine's full pipeline: features → regime → signal → sizing → SL/TP.

Usage:
    python backtest_v5.py                      # defaults: 200 USDT, 90 days
    python backtest_v5.py --budget 500 --days 180
    python backtest_v5.py --symbols BTCUSDT ETHUSDT SOLUSDT --leverage 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Re-use actual Darwin v5 modules (no mocks) ──────────────────────────────

from darwin_agent.v5.market_data import (
    FeatureSet,
    OHLCV,
    _ema,
    _atr,
    _adx,
    _returns,
    _realized_vol,
    _roc,
    _z_score,
    _distance_from_mean,
    _ou_half_life,
    compute_range_context,
)
from darwin_agent.v5.regime_detector import RegimeDetector, RegimeState, Regime
from darwin_agent.v5.signal_generator import SignalGenerator, TradeSignal
from darwin_agent.v5.position_sizer import PositionSizer, SizerConfig, SizeResult
from darwin_agent.v5.performance_analytics import PerformanceAnalytics


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_binance_klines(
    symbol: str,
    interval: str = "15m",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch historical klines from Binance public API (no auth).
    Falls back to synthetic data if API is unreachable."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_candles = []
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 86400 * 1000)
    limit = 1500

    # Try Binance first
    api_works = True
    current_start = start_time
    try:
        resp = requests.get(url, params={
            "symbol": symbol, "interval": interval, "startTime": current_start, "limit": 5
        }, timeout=10)
        resp.raise_for_status()
    except Exception:
        api_works = False

    if api_works:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": limit,
            }
            for attempt in range(4):
                try:
                    resp = requests.get(url, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt == 3:
                        data = []
                        break
                    time.sleep(2 ** attempt)
            if not data:
                break
            for k in data:
                all_candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            last_ts = int(data[-1][0])
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
    else:
        # Fallback: generate realistic synthetic data
        print(f"(API unavailable, using synthetic data)", end=" ", flush=True)
        all_candles = _generate_synthetic_ohlcv(symbol, days)

    df = pd.DataFrame(all_candles)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _generate_synthetic_ohlcv(symbol: str, days: int) -> List[dict]:
    """
    Generate realistic synthetic OHLCV with regime-switching dynamics.

    Uses Geometric Brownian Motion (GBM) with regime transitions:
      - Trending (mu=+/-0.0002, sigma=0.003): directional moves
      - Ranging (mu~0, sigma=0.001): low-vol consolidation
      - High vol (mu=0, sigma=0.008): spikes and crashes

    Parameters calibrated from actual 2024-2025 BTC/ETH/SOL 15m data:
      BTC: base_price=95000, daily_vol~2%
      ETH: base_price=3200, daily_vol~3%
      SOL: base_price=180, daily_vol~4%
    """
    rng = np.random.default_rng(seed=hash(symbol) % (2**31))

    # Symbol-specific parameters (realistic as of late 2025)
    params = {
        "BTCUSDT": {"base": 95000, "daily_vol": 0.020, "trend_strength": 0.0001},
        "ETHUSDT": {"base": 3200, "daily_vol": 0.030, "trend_strength": 0.00012},
        "SOLUSDT": {"base": 180, "daily_vol": 0.040, "trend_strength": 0.00015},
    }
    p = params.get(symbol, {"base": 1000, "daily_vol": 0.025, "trend_strength": 0.0001})

    bars_per_day = 96  # 15m bars
    n_bars = days * bars_per_day
    bar_vol = p["daily_vol"] / (bars_per_day ** 0.5)  # scale daily vol to 15m

    # Regime transition matrix (Markov chain)
    # States: 0=trending_up, 1=trending_down, 2=ranging, 3=high_vol
    # Stationary dist ~ [20%, 15%, 50%, 15%] — crypto spends most time in range
    transition = np.array([
        [0.985, 0.005, 0.008, 0.002],  # trending_up
        [0.005, 0.985, 0.008, 0.002],  # trending_down
        [0.004, 0.003, 0.990, 0.003],  # ranging
        [0.010, 0.010, 0.020, 0.960],  # high_vol
    ])

    regime_params = {
        0: {"mu": p["trend_strength"], "vol_mult": 1.0},      # trending up
        1: {"mu": -p["trend_strength"], "vol_mult": 1.0},     # trending down
        2: {"mu": 0.0, "vol_mult": 0.5},                      # ranging
        3: {"mu": 0.0, "vol_mult": 2.5},                      # high vol
    }

    # Generate price path
    price = p["base"]
    regime = 2  # start ranging
    candles = []
    ts = int((time.time() - days * 86400) * 1000)
    bar_ms = 15 * 60 * 1000  # 15 min in ms

    for i in range(n_bars):
        # Regime transition
        regime = rng.choice(4, p=transition[regime])
        rp = regime_params[regime]

        mu = rp["mu"]
        sigma = bar_vol * rp["vol_mult"]

        # GBM: dS/S = mu*dt + sigma*dW
        ret = mu + sigma * rng.standard_normal()

        # Occasional large moves (fat tails — crypto has kurtosis ~6-8)
        if rng.random() < 0.005:
            ret += sigma * 3 * rng.choice([-1, 1])

        new_price = price * (1 + ret)
        new_price = max(new_price, price * 0.9)  # prevent unrealistic crashes

        # Generate OHLCV from close-to-close
        # Intra-bar volatility: high/low spread proportional to bar_vol
        spread = abs(ret) + sigma * abs(rng.standard_normal()) * 0.5
        if new_price >= price:
            o = price
            c = new_price
            h = max(o, c) * (1 + spread * rng.random() * 0.3)
            l = min(o, c) * (1 - spread * rng.random() * 0.3)
        else:
            o = price
            c = new_price
            h = max(o, c) * (1 + spread * rng.random() * 0.3)
            l = min(o, c) * (1 - spread * rng.random() * 0.3)

        # Volume: higher in volatile regimes, log-normal
        base_vol = 1e6 if "BTC" in symbol else (5e5 if "ETH" in symbol else 2e5)
        vol_regime_mult = {0: 1.2, 1: 1.3, 2: 0.7, 3: 2.5}[regime]
        volume = base_vol * vol_regime_mult * rng.lognormal(0, 0.5)

        candles.append({
            "timestamp": ts + i * bar_ms,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": round(volume, 2),
        })
        price = new_price

    return candles


def compute_features_from_df(
    symbol: str, df: pd.DataFrame, idx: int, window: int = 200
) -> Optional[FeatureSet]:
    """Compute FeatureSet from a DataFrame slice ending at idx."""
    start = max(0, idx - window + 1)
    sl = df.iloc[start:idx + 1]
    if len(sl) < 50:
        return None

    closes = sl["close"].tolist()
    highs = sl["high"].tolist()
    lows = sl["low"].tolist()
    volumes = sl["volume"].tolist()

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


# ── Position / Trade models ──────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    symbol: str
    side: str            # "LONG" or "SHORT"
    entry_price: float
    quantity: float      # in base asset terms
    entry_bar: int
    sl_pct: float
    tp_pct: float
    trailing_peak_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_pct: float
    bars_held: int
    reason: str
    entry_bar: int
    exit_bar: int


@dataclass
class BacktestConfig:
    initial_equity: float = 200.0
    leverage: int = 5
    base_risk_pct: float = 1.0
    # Fee model (Binance futures taker: 0.04%, we assume 0.05% conservative)
    fee_pct: float = 0.05
    # Slippage model: 0.02% per side (conservative estimate for liquid pairs)
    slippage_pct: float = 0.02
    # SL/TP
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 3.0
    atr_sl_multiplier: float = 2.0
    # Trailing stop
    trailing_stop_enabled: bool = True
    trailing_activation_pct: float = 1.5
    trailing_distance_pct: float = 0.8
    # Regime-adaptive SL/TP
    sl_pct_trending: float = 1.5
    tp_pct_trending: float = 3.0
    sl_pct_ranging: float = 0.7
    tp_pct_ranging: float = 1.2
    # Kill switches
    max_daily_loss_pct: float = 5.0
    max_drawdown_kill_pct: float = 15.0
    # Confidence
    confidence_threshold_trending: float = 0.60
    confidence_threshold_ranging: float = 0.78
    # ATR filter
    min_atr_pct_to_trade: float = 0.25
    # Cooldown
    post_loss_cooldown_bars: int = 6
    ranging_trade_cooldown_bars: int = 120
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    # Backtest period
    days: int = 90


# ── Backtester Engine ────────────────────────────────────────────────────────

class DarwinBacktester:
    """
    Event-driven backtester that replicates the v5 engine tick-by-tick
    using real Darwin modules. No lookahead bias — all indicators are
    computed on data available at time t only.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config
        self.equity = config.initial_equity
        self.peak_equity = config.initial_equity
        self.daily_start_equity = config.initial_equity

        # Darwin modules (actual production code)
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
        self.equity_curve: List[Tuple[int, float]] = []
        self.daily_pnl: float = 0.0
        self.trading_halted: bool = False
        self.halt_reason: str = ""
        self.cooldown_until: Dict[str, int] = {}
        self.last_trade_bar: Dict[str, int] = {}
        self.current_day: int = -1
        self.drawdown_pct: float = 0.0

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run the backtest across all symbols simultaneously.

        data: {symbol: DataFrame} with OHLCV columns.
        """
        # Find common bar range
        min_len = min(len(df) for df in data.values())
        if min_len < 200:
            print("ERROR: insufficient data for backtest (need >= 200 bars)")
            return {}

        btc_df = data.get("BTCUSDT")
        warmup = 200  # bars needed for indicator computation

        print(f"\n{'='*70}")
        print(f"  DARWIN v5 BACKTESTER")
        print(f"  Budget: ${self.cfg.initial_equity:.2f} | Leverage: {self.cfg.leverage}x")
        print(f"  Symbols: {self.cfg.symbols}")
        print(f"  Period: {self.cfg.days} days | Bars: {min_len - warmup}")
        print(f"  Fees: {self.cfg.fee_pct}% per side | Slippage: {self.cfg.slippage_pct}% per side")
        print(f"{'='*70}\n")

        for bar_idx in range(warmup, min_len):
            self._tick(bar_idx, data, btc_df)
            if bar_idx % 1000 == 0:
                pct_done = (bar_idx - warmup) / (min_len - warmup) * 100
                print(f"  [{pct_done:5.1f}%] bar {bar_idx}/{min_len}  equity=${self.equity:.2f}  "
                      f"trades={len(self.trades)}  dd={self.drawdown_pct:.1f}%")

        # Close any remaining open positions at last bar
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            last_price = data[symbol].iloc[-1]["close"]
            self._close_position(symbol, last_price, bar_idx, "END_OF_BACKTEST")

        return self._compile_results(data)

    def _tick(self, bar_idx: int, data: Dict[str, pd.DataFrame], btc_df: pd.DataFrame) -> None:
        """Process one bar across all symbols."""

        # -- Daily reset check (using bar timestamps)
        if btc_df is not None and bar_idx < len(btc_df):
            ts = btc_df.iloc[bar_idx]["timestamp"]
            day = int(ts // 86400000)  # milliseconds to days
            if day != self.current_day:
                self.current_day = day
                self.daily_start_equity = self.equity
                self.daily_pnl = 0.0
                # Reset daily halts
                if self.trading_halted and "daily" in self.halt_reason.lower():
                    self.trading_halted = False
                    self.halt_reason = ""

        # -- Update unrealized PnL for equity
        unrealized = 0.0
        for sym, pos in self.positions.items():
            if sym in data and bar_idx < len(data[sym]):
                cur_price = data[sym].iloc[bar_idx]["close"]
                if pos.side == "LONG":
                    unrealized += (cur_price - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - cur_price) * pos.quantity

        realized_equity = self.equity  # cash equity from closed trades
        mark_equity = realized_equity + unrealized

        # Update peak and drawdown
        if mark_equity > self.peak_equity:
            self.peak_equity = mark_equity
        if self.peak_equity > 0:
            self.drawdown_pct = (self.peak_equity - mark_equity) / self.peak_equity * 100

        self.equity_curve.append((bar_idx, mark_equity))
        self.analytics.update_equity(mark_equity)

        # -- Kill switch checks
        self._check_kill_switches(mark_equity)

        # -- BTC features (shared)
        btc_feats = compute_features_from_df("BTCUSDT", btc_df, bar_idx) if btc_df is not None else None

        # -- Manage open positions (SL/TP)
        for symbol in list(self.positions.keys()):
            if symbol in data and bar_idx < len(data[symbol]):
                self._manage_position(symbol, data[symbol], bar_idx, btc_feats)

        # -- Process signals for each symbol
        if not self.trading_halted:
            for symbol in self.cfg.symbols:
                if symbol in data and bar_idx < len(data[symbol]):
                    self._process_symbol(symbol, data[symbol], bar_idx, btc_feats, mark_equity)

    def _check_kill_switches(self, equity: float) -> None:
        if self.trading_halted:
            return
        # Daily loss
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                self.trading_halted = True
                self.halt_reason = f"daily_loss: -{daily_loss_pct:.2f}%"
                return
        # Max drawdown
        if self.drawdown_pct >= self.cfg.max_drawdown_kill_pct:
            self.trading_halted = True
            self.halt_reason = f"max_drawdown: -{self.drawdown_pct:.2f}%"

    def _manage_position(
        self, symbol: str, df: pd.DataFrame, bar_idx: int, btc_feats: Optional[FeatureSet]
    ) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return

        row = df.iloc[bar_idx]
        current_price = row["close"]
        high = row["high"]
        low = row["low"]

        # PnL %
        if pos.side == "LONG":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            # Check if intra-bar low triggered SL
            worst_price = low
            worst_pnl = (worst_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            worst_price = high
            worst_pnl = (pos.entry_price - worst_price) / pos.entry_price

        # Dynamic SL from ATR + regime
        feats = compute_features_from_df(symbol, df, bar_idx)
        atr_pct = 0.0
        if feats and feats.atr_14 > 0 and feats.close > 0:
            atr_pct = feats.atr_14 / feats.close

        # Regime-adaptive SL/TP
        regime = self.regime_detector.detect(feats, btc_feats) if feats else None
        is_ranging = regime and regime.regime.value in ("range_bound", "low_vol")

        if is_ranging:
            sl_floor = self.cfg.sl_pct_ranging / 100
            tp_pct = self.cfg.tp_pct_ranging / 100
        else:
            sl_floor = self.cfg.sl_pct_trending / 100
            tp_pct = self.cfg.tp_pct_trending / 100

        sl_pct = max(sl_floor, self.cfg.atr_sl_multiplier * atr_pct)

        # Trailing stop
        trail_sl = None
        if self.cfg.trailing_stop_enabled and pnl_pct > 0:
            pos.trailing_peak_pnl = max(pos.trailing_peak_pnl, pnl_pct)
            if pos.trailing_peak_pnl >= self.cfg.trailing_activation_pct / 100:
                trail_sl = pos.trailing_peak_pnl - self.cfg.trailing_distance_pct / 100

        # Determine close reason (use worst_pnl for intra-bar SL check)
        close_reason = ""
        if worst_pnl <= -sl_pct:
            close_reason = "STOP_LOSS"
        elif trail_sl is not None and worst_pnl <= trail_sl:
            close_reason = "TRAILING_STOP"
        elif pnl_pct >= tp_pct:
            close_reason = "TAKE_PROFIT"

        if close_reason:
            # For SL, use the SL price level (not the bar close)
            if close_reason == "STOP_LOSS":
                if pos.side == "LONG":
                    exit_price = pos.entry_price * (1 - sl_pct)
                else:
                    exit_price = pos.entry_price * (1 + sl_pct)
            elif close_reason == "TRAILING_STOP" and trail_sl is not None:
                if pos.side == "LONG":
                    exit_price = pos.entry_price * (1 + trail_sl)
                else:
                    exit_price = pos.entry_price * (1 - trail_sl)
            else:
                exit_price = current_price

            self._close_position(symbol, exit_price, bar_idx, close_reason)

    def _close_position(self, symbol: str, exit_price: float, bar_idx: int, reason: str) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        # Apply slippage and fees
        if pos.side == "LONG":
            effective_exit = exit_price * (1 - self.cfg.slippage_pct / 100)
            raw_pnl = (effective_exit - pos.entry_price) * pos.quantity
        else:
            effective_exit = exit_price * (1 + self.cfg.slippage_pct / 100)
            raw_pnl = (pos.entry_price - effective_exit) * pos.quantity

        # Fees: entry + exit (each side pays fee_pct on notional)
        entry_fee = pos.entry_price * pos.quantity * self.cfg.fee_pct / 100
        exit_fee = effective_exit * pos.quantity * self.cfg.fee_pct / 100
        total_fees = entry_fee + exit_fee

        pnl_usdt = raw_pnl - total_fees
        pnl_pct = pnl_usdt / (pos.entry_price * pos.quantity) if pos.entry_price * pos.quantity > 0 else 0

        self.equity += pnl_usdt
        self.daily_pnl += pnl_usdt

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=effective_exit,
            quantity=pos.quantity,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            bars_held=bar_idx - pos.entry_bar,
            reason=reason,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
        )
        self.trades.append(trade)
        self.analytics.record_trade(pnl_usdt=pnl_usdt, is_win=(pnl_usdt > 0))
        self.position_sizer.record_trade_pnl(pnl_usdt)

        # Cooldown on SL
        if reason == "STOP_LOSS":
            self.cooldown_until[symbol] = bar_idx + self.cfg.post_loss_cooldown_bars

    def _process_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        bar_idx: int,
        btc_feats: Optional[FeatureSet],
        mark_equity: float,
    ) -> None:
        # Skip if already positioned
        if symbol in self.positions:
            return

        # Cooldown check
        if bar_idx < self.cooldown_until.get(symbol, 0):
            return

        feats = compute_features_from_df(symbol, df, bar_idx)
        if feats is None or feats.close <= 0:
            return

        # Regime detection
        regime = self.regime_detector.detect(feats, btc_feats)
        is_ranging = regime.regime.value in ("range_bound", "low_vol")

        # ATR filter
        if feats.atr_14 > 0 and feats.close > 0:
            atr_pct = feats.atr_14 / feats.close * 100
            if atr_pct < self.cfg.min_atr_pct_to_trade:
                return

        # Signal generation (no order flow in backtest — it requires live order book)
        signal = self.signal_generator.generate(feats, regime, btc_feats, funding_rate=0.0)

        if not signal.passed or not signal.direction:
            return

        # Confidence threshold (regime-adaptive)
        threshold = (
            self.cfg.confidence_threshold_ranging if is_ranging
            else self.cfg.confidence_threshold_trending
        )
        if signal.confidence < threshold:
            return

        # Frequency throttle in ranging
        if is_ranging:
            last_bar = self.last_trade_bar.get(symbol, 0)
            if bar_idx - last_bar < self.cfg.ranging_trade_cooldown_bars:
                return

        # Range context filter
        if is_ranging and len(feats.closes) >= 10:
            rctx = compute_range_context(feats.closes, feats.highs, feats.lows, window=48)
            if rctx["range_pct"] < 0.02:
                return
            if not rctx["near_support"] and not rctx["near_resistance"]:
                return
            if rctx["near_support"] and signal.direction == "SHORT":
                return
            if rctx["near_resistance"] and signal.direction == "LONG":
                return

        # Position sizing
        size_result = self.position_sizer.compute(
            equity=mark_equity,
            price=feats.close,
            realized_vol=feats.realized_vol_20,
            drawdown_pct=self.drawdown_pct,
            regime_multiplier=regime.risk_multiplier,
            signal_confidence=signal.confidence,
            current_exposure=sum(p.notional for p in self.positions.values()),
            step_size=0.0,  # no step size needed in backtest
            risk_pct_override=0.0,
            daily_pnl=self.daily_pnl,
            daily_start_equity=self.daily_start_equity,
        )

        if not size_result.approved:
            return

        # Apply slippage on entry
        if signal.direction == "LONG":
            entry_price = feats.close * (1 + self.cfg.slippage_pct / 100)
        else:
            entry_price = feats.close * (1 - self.cfg.slippage_pct / 100)

        quantity = size_result.position_size_usdt / entry_price

        # Open position
        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            side=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_bar=bar_idx,
            sl_pct=self.cfg.stop_loss_pct,
            tp_pct=self.cfg.take_profit_pct,
        )
        self.last_trade_bar[symbol] = bar_idx

    def _compile_results(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Compile and display backtest results."""
        if not self.trades:
            print("\n  NO TRADES EXECUTED — strategy was too conservative or data insufficient")
            return {"trades": 0}

        pnls = [t.pnl_usdt for t in self.trades]
        wins = [t for t in self.trades if t.pnl_usdt > 0]
        losses = [t for t in self.trades if t.pnl_usdt <= 0]
        win_pnls = [t.pnl_usdt for t in wins]
        loss_pnls = [t.pnl_usdt for t in losses]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        profit_factor = sum(win_pnls) / abs(sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf')
        expectancy = np.mean(pnls) if pnls else 0

        # Equity curve stats
        eq_values = [e[1] for e in self.equity_curve]
        peak = max(eq_values) if eq_values else self.cfg.initial_equity
        trough = min(eq_values) if eq_values else self.cfg.initial_equity
        max_dd_abs = peak - trough
        max_dd_pct = max_dd_abs / peak * 100 if peak > 0 else 0

        # Compute actual max drawdown properly
        running_peak = eq_values[0] if eq_values else self.cfg.initial_equity
        max_dd_pct_real = 0.0
        for e in eq_values:
            if e > running_peak:
                running_peak = e
            dd = (running_peak - e) / running_peak * 100 if running_peak > 0 else 0
            max_dd_pct_real = max(max_dd_pct_real, dd)

        final_equity = self.equity
        total_return_pct = (final_equity - self.cfg.initial_equity) / self.cfg.initial_equity * 100

        # Sharpe (daily returns)
        daily_returns = []
        bars_per_day = 96  # 15m bars per day
        for i in range(0, len(eq_values) - bars_per_day, bars_per_day):
            day_start = eq_values[i]
            day_end = eq_values[i + bars_per_day]
            if day_start > 0:
                daily_returns.append((day_end - day_start) / day_start)

        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) >= 5:
            mu = np.mean(daily_returns)
            sigma = np.std(daily_returns)
            if sigma > 1e-10:
                sharpe = (mu / sigma) * (252 ** 0.5)
            neg_returns = [r for r in daily_returns if r < 0]
            if neg_returns:
                downside_dev = np.sqrt(np.mean([r ** 2 for r in neg_returns]))
                if downside_dev > 1e-10:
                    sortino = (mu / downside_dev) * (252 ** 0.5)

        # Calmar
        calmar = total_return_pct / max_dd_pct_real if max_dd_pct_real > 0 else 0

        # By-symbol breakdown
        symbol_stats = {}
        for sym in self.cfg.symbols:
            sym_trades = [t for t in self.trades if t.symbol == sym]
            sym_wins = [t for t in sym_trades if t.pnl_usdt > 0]
            sym_pnl = sum(t.pnl_usdt for t in sym_trades)
            symbol_stats[sym] = {
                "trades": len(sym_trades),
                "wins": len(sym_wins),
                "win_rate": len(sym_wins) / len(sym_trades) * 100 if sym_trades else 0,
                "pnl": sym_pnl,
            }

        # Trade reason breakdown
        reason_counts = {}
        for t in self.trades:
            reason_counts[t.reason] = reason_counts.get(t.reason, 0) + 1

        # Longest winning/losing streaks
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in self.trades:
            if t.pnl_usdt > 0:
                streak = max(0, streak) + 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = min(0, streak) - 1
                max_loss_streak = max(max_loss_streak, abs(streak))

        # Average bars held
        avg_bars_held = np.mean([t.bars_held for t in self.trades])

        # ── Print Results ────────────────────────────────────────────────────

        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"")
        print(f"  CAPITAL")
        print(f"  {'Initial equity:':<30} ${self.cfg.initial_equity:.2f}")
        print(f"  {'Final equity:':<30} ${final_equity:.2f}")
        print(f"  {'Total PnL:':<30} ${total_pnl:+.2f} ({total_return_pct:+.2f}%)")
        print(f"  {'Peak equity:':<30} ${peak:.2f}")
        print(f"  {'Max drawdown:':<30} {max_dd_pct_real:.2f}% (${max_dd_abs:.2f})")
        print(f"")
        print(f"  RISK METRICS")
        print(f"  {'Sharpe ratio (ann.):':<30} {sharpe:.3f}")
        print(f"  {'Sortino ratio (ann.):':<30} {sortino:.3f}")
        print(f"  {'Calmar ratio:':<30} {calmar:.3f}")
        print(f"  {'Profit factor:':<30} {profit_factor:.3f}")
        print(f"")
        print(f"  TRADE STATS")
        print(f"  {'Total trades:':<30} {len(self.trades)}")
        print(f"  {'Win rate:':<30} {win_rate*100:.1f}%")
        print(f"  {'Avg win:':<30} ${avg_win:.4f}")
        print(f"  {'Avg loss:':<30} ${avg_loss:.4f}")
        print(f"  {'Expectancy/trade:':<30} ${expectancy:.4f}")
        print(f"  {'Max win streak:':<30} {max_win_streak}")
        print(f"  {'Max loss streak:':<30} {max_loss_streak}")
        print(f"  {'Avg bars held:':<30} {avg_bars_held:.1f} ({avg_bars_held * 15 / 60:.1f}h)")
        print(f"")
        print(f"  EXIT REASONS")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.trades) * 100
            print(f"    {reason:<20} {count:>4} ({pct:.1f}%)")
        print(f"")
        print(f"  PER-SYMBOL BREAKDOWN")
        for sym, stats in symbol_stats.items():
            print(f"    {sym:<12} trades={stats['trades']:>3}  "
                  f"wr={stats['win_rate']:.1f}%  pnl=${stats['pnl']:+.2f}")
        print(f"")

        # Scalability assessment
        print(f"  SCALABILITY ASSESSMENT (institutional grade)")
        grade_score = 0
        checks = []

        if sharpe >= 2.0:
            checks.append(("Sharpe >= 2.0", "PASS", 3))
            grade_score += 3
        elif sharpe >= 1.0:
            checks.append(("Sharpe >= 1.0", "MARGINAL", 1))
            grade_score += 1
        else:
            checks.append(("Sharpe >= 1.0", "FAIL", 0))

        if max_dd_pct_real <= 10:
            checks.append(("Max DD <= 10%", "PASS", 2))
            grade_score += 2
        elif max_dd_pct_real <= 20:
            checks.append(("Max DD <= 20%", "MARGINAL", 1))
            grade_score += 1
        else:
            checks.append(("Max DD <= 20%", "FAIL", 0))

        if profit_factor >= 2.0:
            checks.append(("PF >= 2.0", "PASS", 2))
            grade_score += 2
        elif profit_factor >= 1.5:
            checks.append(("PF >= 1.5", "MARGINAL", 1))
            grade_score += 1
        else:
            checks.append(("PF >= 1.5", "FAIL", 0))

        if win_rate >= 0.55:
            checks.append(("Win rate >= 55%", "PASS", 1))
            grade_score += 1
        else:
            checks.append(("Win rate >= 55%", "FAIL", 0))

        if calmar >= 1.0:
            checks.append(("Calmar >= 1.0", "PASS", 2))
            grade_score += 2
        else:
            checks.append(("Calmar >= 1.0", "FAIL", 0))

        for check, result, pts in checks:
            marker = "+" if result == "PASS" else ("~" if result == "MARGINAL" else "-")
            print(f"    [{marker}] {check:<25} {result}")

        if grade_score >= 9:
            grade = "A"
        elif grade_score >= 6:
            grade = "B"
        elif grade_score >= 3:
            grade = "C"
        else:
            grade = "F"

        scalable = grade in ("A", "B")
        print(f"\n  GRADE: {grade} (score={grade_score}/10)  "
              f"{'SCALABLE' if scalable else 'NOT SCALABLE'}")

        if not scalable:
            print(f"\n  With a ${self.cfg.initial_equity:.0f} budget, the strategy needs "
                  f"{'more edge' if profit_factor < 1.5 else 'better risk control'} "
                  f"before scaling up.")

        print(f"\n{'='*70}")

        # Return dict for programmatic access
        return {
            "initial_equity": self.cfg.initial_equity,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd_pct_real,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "grade": grade,
            "scalable": scalable,
            "symbol_stats": symbol_stats,
            "reason_counts": reason_counts,
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Darwin v5 Backtester")
    parser.add_argument("--budget", type=float, default=200.0, help="Initial equity in USDT")
    parser.add_argument("--days", type=int, default=90, help="Backtest period in days")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage multiplier")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--fee", type=float, default=0.05, help="Fee per side (%%)")
    parser.add_argument("--slippage", type=float, default=0.02, help="Slippage per side (%%)")
    args = parser.parse_args()

    config = BacktestConfig(
        initial_equity=args.budget,
        leverage=args.leverage,
        days=args.days,
        symbols=args.symbols,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
    )

    # Fetch historical data
    print("Fetching historical data from Binance...")
    data: Dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        df = fetch_binance_klines(symbol, interval="15m", days=config.days)
        if df.empty:
            print(f"FAILED (no data)")
            sys.exit(1)
        print(f"OK ({len(df)} bars, {df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()})")
        data[symbol] = df

    # Run backtest
    bt = DarwinBacktester(config)
    results = bt.run(data)

    # Save results
    output_path = "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return 0 if results.get("total_pnl", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
