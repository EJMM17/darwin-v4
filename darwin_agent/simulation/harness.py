"""
Darwin v4 — Simulation Harness.

Offline evolutionary simulation with ZERO exchange calls.
Fully deterministic via seed. Integrates every production
component (EvolutionEngine, PortfolioRiskEngine, Diagnostics)
in an isolated sandbox.

SIMULATION MODES:

  1. HISTORICAL REPLAY — Feed recorded candle data. Agents trade
     against actual market history. Deterministic replay of any
     period to measure strategy robustness.

  2. MONTE CARLO — Synthetic price paths via geometric Brownian
     motion with configurable drift, volatility, and fat tails.
     Run N paths to build confidence intervals on strategy fitness.

  3. SHOCK INJECTION — Layer market shocks onto any base scenario:
     flash crashes, volatility spikes, liquidity gaps, momentum
     reversals. Tests drawdown recovery and circuit breaker behavior.

ARCHITECTURE:

  MarketScenario (ABC)        → generates price series per symbol
    ├─ HistoricalScenario     → replays from recorded data
    ├─ MonteCarloScenario     → synthetic GBM paths
    └─ ShockScenario          → wraps any base scenario + injects shocks

  SimulatedAgent              → lightweight agent that trades against
                                 a price series using DNA genes. No exchange
                                 calls. Tracks PnL, exposure, trade history.

  SimulationHarness           → orchestrator: seeds → spawn agents → run
                                 generations → evaluate → evolve → collect
                                 metrics → export results.

  ResultsExporter             → CSV + JSON export of per-generation metrics,
                                 agent rankings, gene distributions, and
                                 diagnostic reports.

USAGE:

    from darwin_agent.simulation.harness import SimulationHarness, SimConfig
    from darwin_agent.simulation.scenarios import MonteCarloScenario

    scenario = MonteCarloScenario(symbols=["BTCUSDT", "ETHUSDT"])
    harness = SimulationHarness(SimConfig(
        scenario=scenario,
        generations=50,
        pool_size=8,
        starting_capital=50.0,
        seed=42,
    ))
    results = await harness.run()
    results.export_csv("sim_results.csv")
    results.export_json("sim_results.json")
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import random
import statistics
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin_agent.interfaces.enums import PortfolioRiskState
from darwin_agent.interfaces.types import (
    AgentEvalData, AgentMetrics, DNAData,
    GenerationSnapshot,
)
from darwin_agent.evolution.engine import EvolutionEngine, DEFAULT_GENE_RANGES
from darwin_agent.evolution.diagnostics import EvolutionDiagnostics
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits

logger = logging.getLogger("darwin.simulation")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ═════════════════════════════════════════════════════════════
# Market scenarios — price generation
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Tick:
    """Single price point for one symbol at one time step."""
    symbol: str
    step: int
    price: float
    volume: float = 1.0
    # Technical indicators (populated by HistoricalMarketScenario)
    ema_fast: float = 0.0      # EMA(12)
    ema_mid: float = 0.0       # EMA(50)
    ema_slow: float = 0.0      # EMA(48)
    rsi: float = 50.0          # RSI(14), 0-100
    atr: float = 0.0           # ATR(14)
    rolling_vol: float = 0.0   # Rolling std of returns(20)
    trend_strength: float = 0.0  # |ema_fast - ema_slow| / price
    macro_trend: float = 0.0    # (close - EMA200) / EMA200


class MarketScenario(ABC):
    """
    Abstract base. Generates price ticks for N symbols over T steps.
    Subclasses implement the actual price generation logic.
    Every scenario is seeded for deterministic replay.
    """

    @abstractmethod
    def generate(
        self, n_steps: int, rng: random.Random,
    ) -> Dict[str, List[Tick]]:
        """
        Return {symbol: [Tick, ...]} with n_steps ticks per symbol.
        Must be deterministic given the same rng state.
        """
        ...

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Return list of symbols this scenario produces."""
        ...


class HistoricalScenario(MarketScenario):
    """
    Replay from pre-recorded price data.

    Accepts {symbol: [price_0, price_1, ...]} and replays verbatim.
    If n_steps exceeds data length, wraps around (cyclic replay).
    """

    __slots__ = ("_data", "_symbols")

    def __init__(self, data: Dict[str, List[float]]) -> None:
        if not data:
            raise ValueError("Historical data cannot be empty")
        self._data = data
        self._symbols = sorted(data.keys())

    def get_symbols(self) -> List[str]:
        return list(self._symbols)

    def generate(
        self, n_steps: int, rng: random.Random,
    ) -> Dict[str, List[Tick]]:
        result: Dict[str, List[Tick]] = {}
        for sym in self._symbols:
            prices = self._data[sym]
            ticks = []
            for step in range(n_steps):
                idx = step % len(prices)
                ticks.append(Tick(
                    symbol=sym, step=step, price=prices[idx],
                    volume=1.0 + rng.random() * 0.5,
                ))
            result[sym] = ticks
        return result


class HistoricalMarketScenario(MarketScenario):
    """
    CSV-backed historical market scenario.

    Loads OHLCV data from CSV files, slices by timeframe, and replays
    close prices step-by-step. Fully deterministic — no Monte Carlo
    randomness. Volume comes from actual data.

    CSV format expected:
        timestamp,open,high,low,close,volume
    or:
        date,open,high,low,close,volume

    Usage:
        scenario = HistoricalMarketScenario(
            csv_paths={"BTCUSDT": "btc_1h.csv", "ETHUSDT": "eth_1h.csv"},
            timeframe="1h",
        )
    """

    @dataclass(frozen=True, slots=True)
    class OHLCVBar:
        timestamp: float   # unix epoch
        open: float
        high: float
        low: float
        close: float
        volume: float

    __slots__ = (
        "_symbols", "_bars", "_generation_size", "_total_steps",
    )

    def __init__(
        self,
        csv_paths: Dict[str, str] | None = None,
        csv_data: Dict[str, List["HistoricalMarketScenario.OHLCVBar"]] | None = None,
        timeframe: str = "1h",
        generation_size: int | None = None,
    ) -> None:
        """
        Args:
            csv_paths:  {symbol: "/path/to/file.csv"} — loaded from disk.
            csv_data:   {symbol: [OHLCVBar, ...]} — pre-loaded bars.
                        Exactly one of csv_paths or csv_data must be given.
            timeframe:  Resampling target. One of:
                        "1m","5m","15m","30m","1h","4h","1d".
                        If source data is already the desired resolution,
                        no resampling occurs.
            generation_size: Ticks per generation (overrides SimConfig's
                        trades_per_generation when set).
        """
        if csv_paths and csv_data:
            raise ValueError("Provide csv_paths OR csv_data, not both")
        if not csv_paths and not csv_data:
            raise ValueError("Provide csv_paths or csv_data")

        self._generation_size = generation_size

        if csv_paths:
            bars = {}
            for sym, path in csv_paths.items():
                bars[sym] = self._load_csv(path)
            csv_data = bars

        # Resample if needed
        tf_seconds = self._timeframe_to_seconds(timeframe)
        resampled: Dict[str, List[HistoricalMarketScenario.OHLCVBar]] = {}
        for sym, bar_list in csv_data.items():
            if not bar_list:
                raise ValueError(f"No data for {sym}")
            resampled[sym] = self._resample(bar_list, tf_seconds)

        self._bars = resampled
        self._symbols = sorted(resampled.keys())

        # Align all symbols to same length (shortest)
        min_len = min(len(v) for v in self._bars.values())
        for sym in self._symbols:
            self._bars[sym] = self._bars[sym][:min_len]
        self._total_steps = min_len

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        if tf not in mapping:
            raise ValueError(
                f"Unknown timeframe '{tf}'. Use: {list(mapping.keys())}"
            )
        return mapping[tf]

    @staticmethod
    def _load_csv(path: str) -> List["HistoricalMarketScenario.OHLCVBar"]:
        """Load OHLCV from CSV. Auto-detects header and timestamp format."""
        import csv as _csv
        bars: List[HistoricalMarketScenario.OHLCVBar] = []
        with open(path, "r") as f:
            reader = _csv.reader(f)
            header = next(reader)
            # Normalize header
            hdr = [h.strip().lower() for h in header]
            # Find column indices
            ts_col = None
            for candidate in ("timestamp", "date", "time", "datetime"):
                if candidate in hdr:
                    ts_col = hdr.index(candidate)
                    break
            if ts_col is None:
                ts_col = 0  # assume first column
            o_col = hdr.index("open") if "open" in hdr else 1
            h_col = hdr.index("high") if "high" in hdr else 2
            l_col = hdr.index("low") if "low" in hdr else 3
            c_col = hdr.index("close") if "close" in hdr else 4
            v_col = hdr.index("volume") if "volume" in hdr else 5

            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    ts_raw = row[ts_col].strip()
                    # Try numeric timestamp first
                    try:
                        ts = float(ts_raw)
                        # If it looks like milliseconds, convert
                        if ts > 1e12:
                            ts /= 1000.0
                    except ValueError:
                        # Try ISO date parsing
                        from datetime import datetime as _dt
                        for fmt in (
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d",
                            "%m/%d/%Y %H:%M",
                        ):
                            try:
                                ts = _dt.strptime(ts_raw, fmt).timestamp()
                                break
                            except ValueError:
                                continue
                        else:
                            continue  # skip unparseable rows

                    bars.append(HistoricalMarketScenario.OHLCVBar(
                        timestamp=ts,
                        open=float(row[o_col]),
                        high=float(row[h_col]),
                        low=float(row[l_col]),
                        close=float(row[c_col]),
                        volume=float(row[v_col]),
                    ))
                except (ValueError, IndexError):
                    continue
        # Sort by timestamp
        bars.sort(key=lambda b: b.timestamp)
        return bars

    @staticmethod
    def _resample(
        bars: List["HistoricalMarketScenario.OHLCVBar"],
        tf_seconds: int,
    ) -> List["HistoricalMarketScenario.OHLCVBar"]:
        """Resample bars to target timeframe. If already correct, passthrough."""
        if len(bars) < 2:
            return list(bars)
        # Detect source resolution from median gap
        gaps = [bars[i+1].timestamp - bars[i].timestamp
                for i in range(min(20, len(bars)-1))]
        gaps.sort()
        src_tf = gaps[len(gaps)//2]
        # If source matches target (within 20% tolerance), passthrough
        if abs(src_tf - tf_seconds) / max(tf_seconds, 1) < 0.2:
            return list(bars)
        if src_tf > tf_seconds:
            # Cannot upsample — return as-is
            return list(bars)
        # Downsample: bucket bars into tf_seconds windows
        result: List[HistoricalMarketScenario.OHLCVBar] = []
        bucket_start = bars[0].timestamp
        bucket: List[HistoricalMarketScenario.OHLCVBar] = []
        for bar in bars:
            if bar.timestamp >= bucket_start + tf_seconds and bucket:
                result.append(HistoricalMarketScenario.OHLCVBar(
                    timestamp=bucket[0].timestamp,
                    open=bucket[0].open,
                    high=max(b.high for b in bucket),
                    low=min(b.low for b in bucket),
                    close=bucket[-1].close,
                    volume=sum(b.volume for b in bucket),
                ))
                bucket_start = bar.timestamp
                bucket = [bar]
            else:
                bucket.append(bar)
        if bucket:
            result.append(HistoricalMarketScenario.OHLCVBar(
                timestamp=bucket[0].timestamp,
                open=bucket[0].open,
                high=max(b.high for b in bucket),
                low=min(b.low for b in bucket),
                close=bucket[-1].close,
                volume=sum(b.volume for b in bucket),
            ))
        return result

    @property
    def total_bars(self) -> int:
        return self._total_steps

    def get_symbols(self) -> List[str]:
        return list(self._symbols)

    def generate(
        self, n_steps: int, rng: random.Random,
    ) -> Dict[str, List[Tick]]:
        """
        Deterministic replay with technical indicators.

        Computes EMA(12), EMA(48), RSI(14), ATR(14), rolling_vol(20),
        trend_strength from the full bar history, then slices the
        requested window. Indicators are pre-computed over the entire
        dataset so warmup artifacts don't bleed into generation windows.
        """
        offset = rng.randint(0, max(self._total_steps - 1, 0))

        result: Dict[str, List[Tick]] = {}
        for sym in self._symbols:
            bars = self._bars[sym]

            # Pre-compute indicators over full bar history
            closes = [b.close for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            n_bars = len(closes)

            ema_f = self._compute_ema(closes, 12)
            ema_s = self._compute_ema(closes, 48)
            ema_m = self._compute_ema(closes, 50)
            rsi_arr = self._compute_rsi(closes, 14)
            atr_arr = self._compute_atr(highs, lows, closes, 14)
            rvol = self._compute_rolling_vol(closes, 20)
            trend = [
                abs(ema_f[i] - ema_s[i]) / closes[i] if closes[i] > 0 else 0.0
                for i in range(n_bars)
            ]
            # Macro regime: (close - EMA200) / EMA200
            ema_200 = self._compute_ema(closes, 200)
            macro = [
                (closes[i] - ema_200[i]) / ema_200[i] if ema_200[i] > 0 else 0.0
                for i in range(n_bars)
            ]

            ticks = []
            for step in range(n_steps):
                idx = (offset + step) % n_bars
                bar = bars[idx]
                ticks.append(Tick(
                    symbol=sym, step=step,
                    price=bar.close, volume=bar.volume,
                    ema_fast=ema_f[idx], ema_mid=ema_m[idx],
                    ema_slow=ema_s[idx],
                    rsi=rsi_arr[idx], atr=atr_arr[idx],
                    rolling_vol=rvol[idx], trend_strength=trend[idx],
                    macro_trend=macro[idx],
                ))
            result[sym] = ticks
        return result

    @staticmethod
    def _compute_ema(prices: List[float], period: int) -> List[float]:
        """Exponential moving average. Warmup uses SMA seed."""
        if not prices:
            return []
        result = [0.0] * len(prices)
        k = 2.0 / (period + 1)
        # Seed with SMA of first `period` bars
        if len(prices) >= period:
            result[period - 1] = sum(prices[:period]) / period
            for i in range(period, len(prices)):
                result[i] = prices[i] * k + result[i - 1] * (1 - k)
            # Backfill warmup with SMA
            for i in range(period - 1):
                result[i] = sum(prices[:i+1]) / (i + 1)
        else:
            for i in range(len(prices)):
                result[i] = sum(prices[:i+1]) / (i + 1)
        return result

    @staticmethod
    def _compute_rsi(prices: List[float], period: int) -> List[float]:
        """Wilder's RSI."""
        n = len(prices)
        result = [50.0] * n
        if n < period + 1:
            return result
        gains = [0.0] * n
        losses = [0.0] * n
        for i in range(1, n):
            delta = prices[i] - prices[i - 1]
            if delta > 0:
                gains[i] = delta
            else:
                losses[i] = -delta
        avg_gain = sum(gains[1:period+1]) / period
        avg_loss = sum(losses[1:period+1]) / period
        for i in range(period, n):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - 100.0 / (1.0 + rs)
        return result

    @staticmethod
    def _compute_atr(
        highs: List[float], lows: List[float],
        closes: List[float], period: int,
    ) -> List[float]:
        """Average True Range."""
        n = len(closes)
        result = [0.0] * n
        if n < 2:
            return result
        tr = [highs[0] - lows[0]]
        for i in range(1, n):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        if n >= period:
            result[period - 1] = sum(tr[:period]) / period
            for i in range(period, n):
                result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
            for i in range(period - 1):
                result[i] = sum(tr[:i+1]) / (i + 1)
        else:
            for i in range(n):
                result[i] = sum(tr[:i+1]) / (i + 1)
        return result

    @staticmethod
    def _compute_rolling_vol(
        prices: List[float], period: int,
    ) -> List[float]:
        """Rolling standard deviation of log returns."""
        n = len(prices)
        result = [0.0] * n
        if n < 2:
            return result
        log_ret = [0.0]
        for i in range(1, n):
            if prices[i] > 0 and prices[i - 1] > 0:
                log_ret.append(math.log(prices[i] / prices[i - 1]))
            else:
                log_ret.append(0.0)
        for i in range(period, n):
            window = log_ret[i - period + 1:i + 1]
            mean = sum(window) / period
            var = sum((x - mean) ** 2 for x in window) / period
            result[i] = math.sqrt(var)
        return result


class MonteCarloScenario(MarketScenario):
    """
    Geometric Brownian Motion with configurable parameters.

    Each symbol gets independent paths with optional fat tails
    (Student-t innovations) and mean reversion overlay.

    Parameters per symbol (or global defaults):
        base_price:   Starting price
        annual_drift: Expected annual return (0.05 = 5%)
        annual_vol:   Annual volatility (0.80 = 80%, typical crypto)
        fat_tail_df:  Degrees of freedom for t-distribution (inf = normal)
        mean_rev_k:   Mean reversion strength (0 = pure GBM)
    """

    @dataclass
    class SymbolParams:
        base_price: float = 50000.0
        annual_drift: float = 0.0
        annual_vol: float = 0.80
        fat_tail_df: float = 5.0     # Student-t DoF; >=30 ≈ normal
        mean_rev_k: float = 0.01     # 0 = no mean reversion

    __slots__ = ("_params",)

    def __init__(
        self,
        symbols: List[str] | None = None,
        params: Dict[str, "MonteCarloScenario.SymbolParams"] | None = None,
    ) -> None:
        default_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._params: Dict[str, MonteCarloScenario.SymbolParams] = {}
        defaults = {
            "BTCUSDT": self.SymbolParams(base_price=50000, annual_vol=0.75),
            "ETHUSDT": self.SymbolParams(base_price=3000, annual_vol=0.85),
            "SOLUSDT": self.SymbolParams(base_price=100, annual_vol=1.0),
        }
        for sym in default_symbols:
            if params and sym in params:
                self._params[sym] = params[sym]
            elif sym in defaults:
                self._params[sym] = defaults[sym]
            else:
                self._params[sym] = self.SymbolParams(base_price=100)

    def get_symbols(self) -> List[str]:
        return sorted(self._params.keys())

    def generate(
        self, n_steps: int, rng: random.Random,
    ) -> Dict[str, List[Tick]]:
        result: Dict[str, List[Tick]] = {}
        # Assume each step = 1 hour → 8760 steps/year
        dt = 1.0 / 8760.0

        for sym, sp in self._params.items():
            prices = []
            price = sp.base_price
            mu_dt = sp.annual_drift * dt
            sig_sqrt_dt = sp.annual_vol * math.sqrt(dt)

            for step in range(n_steps):
                # Innovation: Student-t for fat tails
                z = self._student_t(rng, sp.fat_tail_df)

                # Mean reversion overlay
                log_ratio = math.log(price / sp.base_price) if price > 0 else 0
                mr_pull = -sp.mean_rev_k * log_ratio * dt

                # GBM step
                ret = mu_dt + mr_pull + sig_sqrt_dt * z
                price = price * math.exp(ret)
                price = max(price, sp.base_price * 0.001)  # floor at 0.1%

                prices.append(Tick(
                    symbol=sym, step=step, price=round(price, 4),
                    volume=1.0 + abs(z) * 0.5,
                ))
            result[sym] = prices
        return result

    @staticmethod
    def _student_t(rng: random.Random, df: float) -> float:
        """Sample from Student-t distribution using standard trick."""
        if df >= 100:
            return rng.gauss(0, 1)
        # t = Z / sqrt(V/df), V ~ chi2(df)
        z = rng.gauss(0, 1)
        v = sum(rng.gauss(0, 1) ** 2 for _ in range(max(1, int(df))))
        return z / math.sqrt(v / df) if v > 0 else z


@dataclass(slots=True)
class ShockEvent:
    """
    A market shock injected at a specific step.

    Types:
      flash_crash:  Instant price drop of `magnitude` percent
      volatility:   Multiply volatility by `magnitude` for `duration` steps
      gap:          Price jumps by `magnitude` percent (can be positive)
      regime_shift: Drift changes by `magnitude` for `duration` steps
    """
    step: int
    shock_type: str          # "flash_crash", "volatility", "gap", "regime_shift"
    magnitude: float         # e.g. -0.20 = 20% crash, 3.0 = 3x vol
    duration: int = 1        # steps the shock persists
    symbols: List[str] = field(default_factory=list)  # empty = all symbols


class ShockScenario(MarketScenario):
    """
    Wraps any base MarketScenario and injects shock events.
    The base scenario generates prices first, then shocks modify them.
    """

    __slots__ = ("_base", "_shocks")

    def __init__(
        self,
        base: MarketScenario,
        shocks: List[ShockEvent],
    ) -> None:
        self._base = base
        self._shocks = sorted(shocks, key=lambda s: s.step)

    def get_symbols(self) -> List[str]:
        return self._base.get_symbols()

    def generate(
        self, n_steps: int, rng: random.Random,
    ) -> Dict[str, List[Tick]]:
        base_data = self._base.generate(n_steps, rng)

        for shock in self._shocks:
            targets = shock.symbols or list(base_data.keys())
            for sym in targets:
                if sym not in base_data:
                    continue
                ticks = base_data[sym]
                self._apply_shock(ticks, shock, rng)

        return base_data

    def _apply_shock(
        self, ticks: List[Tick], shock: ShockEvent, rng: random.Random,
    ) -> None:
        n = len(ticks)
        start = min(shock.step, n - 1)
        end = min(start + shock.duration, n)

        if shock.shock_type == "flash_crash":
            # Instant drop then partial recovery
            if start < n:
                pre = ticks[start].price
                crash_price = pre * (1.0 + shock.magnitude)  # magnitude is negative
                ticks[start] = Tick(
                    symbol=ticks[start].symbol, step=start,
                    price=max(crash_price, 0.01), volume=ticks[start].volume * 5,
                )
                # Gradual recovery over duration
                for i in range(start + 1, end):
                    recovery_pct = (i - start) / max(shock.duration, 1)
                    recovered = crash_price + (pre - crash_price) * recovery_pct * 0.7
                    ticks[i] = Tick(
                        symbol=ticks[i].symbol, step=i,
                        price=max(recovered, 0.01),
                        volume=ticks[i].volume * (3 - 2 * recovery_pct),
                    )

        elif shock.shock_type == "volatility":
            # Amplify price moves
            for i in range(start, end):
                if i == 0:
                    continue
                prev_price = ticks[i - 1].price
                current_ret = (ticks[i].price - prev_price) / max(prev_price, 0.01)
                amplified_ret = current_ret * shock.magnitude
                new_price = prev_price * (1.0 + amplified_ret)
                ticks[i] = Tick(
                    symbol=ticks[i].symbol, step=i,
                    price=max(new_price, 0.01),
                    volume=ticks[i].volume * abs(shock.magnitude),
                )

        elif shock.shock_type == "gap":
            # Instant price gap
            if start < n:
                gap_factor = 1.0 + shock.magnitude
                for i in range(start, n):
                    ticks[i] = Tick(
                        symbol=ticks[i].symbol, step=i,
                        price=max(ticks[i].price * gap_factor, 0.01),
                        volume=ticks[i].volume,
                    )

        elif shock.shock_type == "regime_shift":
            # Drift change over duration
            for i in range(start, end):
                drift_per_step = shock.magnitude / max(shock.duration, 1)
                factor = 1.0 + drift_per_step
                ticks[i] = Tick(
                    symbol=ticks[i].symbol, step=i,
                    price=max(ticks[i].price * factor, 0.01),
                    volume=ticks[i].volume,
                )


# ═════════════════════════════════════════════════════════════
# Simulated agent — trades against price series using DNA
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SimTrade:
    """Record of one simulated trade."""
    symbol: str
    entry_price: float
    exit_price: float
    entry_step: int
    exit_step: int
    size: float
    pnl: float
    pnl_pct: float
    side: str  # "long" or "short"


class SimulatedAgent:
    """
    Lightweight agent that trades price ticks using DNA genes.
    No exchange calls. Pure math.

    Strategy logic (derived from DNA):
      - Confidence threshold → how often it enters trades
      - Risk % → position size
      - Stop loss / take profit → exit rules
      - Strategy weights → directional bias
      - Leverage aggression → notional multiplier
    """

    __slots__ = (
        "agent_id", "dna", "_capital", "_initial_capital",
        "_wins", "_losses", "_peak", "_max_dd",
        "_closed_trades", "_pnl_series", "_trend_series", "_exposure",
        "_position", "_rng", "_sharpe_returns",
        "_cooldown_until", "_exposure_mult", "_fee_rate", "_n_bars",
        "_bars_gated",
    )

    def __init__(
        self, agent_id: str, dna: DNAData,
        capital: float, rng: random.Random,
        exposure_mult: float = 1.0,
        fee_rate: float = 0.0,
    ) -> None:
        self.agent_id = agent_id
        self.dna = dna
        self._capital = capital
        self._initial_capital = capital
        self._wins = 0
        self._losses = 0
        self._peak = capital
        self._max_dd = 0.0
        self._closed_trades: List[SimTrade] = []
        self._pnl_series: List[float] = []
        self._trend_series: List[float] = []
        self._exposure: Dict[str, float] = {}
        self._position: Optional[Dict] = None  # active position
        self._rng = rng
        self._sharpe_returns: List[float] = []
        self._cooldown_until = 0
        self._exposure_mult = exposure_mult
        self._fee_rate = fee_rate
        self._n_bars = 0
        self._bars_gated = 0

    def step(self, ticks: Dict[str, Tick], step: int) -> None:
        """Process one time step across all symbols."""
        self._n_bars += 1
        genes = self.dna.genes

        # Check existing position
        if self._position:
            self._check_exit(ticks, step, genes)

        # Try new entry if no position and not on cooldown
        if not self._position and step >= self._cooldown_until:
            self._check_entry(ticks, step, genes)

    def _check_entry(
        self, ticks: Dict[str, Tick], step: int, genes: Dict[str, float],
    ) -> None:
        """Decide whether to enter a trade."""
        conf_threshold = genes.get("confidence_threshold", 0.6)
        risk_pct = genes.get("risk_pct", 2.0) / 100.0
        leverage = genes.get("leverage_aggression", 0.5) * 20  # up to 10x

        # DDT 2.0: Smooth Drawdown Dynamic Throttle
        # exposure_mult = max(0.2, 1 - 1.5 * rolling_dd)
        # Scales risk_pct + leverage. No halt. No gene modification.
        if self._exposure_mult < 1.0:
            risk_pct *= self._exposure_mult
            leverage *= self._exposure_mult

        # Strategy weight preferences (determine which "style" dominates)
        momentum_w = genes.get("momentum_weight", 0.5)
        mean_rev_w = genes.get("mean_rev_weight", 0.5)
        scalping_w = genes.get("scalping_weight", 0.5)
        breakout_w = genes.get("breakout_weight", 0.5)

        # ── Structural genes ─────────────────────────────
        regime_bias = genes.get("regime_bias", 0.0)      # -1..+1
        vol_thresh = genes.get("volatility_threshold", 0.3)  # 0..1

        # Volatility gate: estimate local vol from recent price memory
        # Use the agent-local RNG to produce a consistent vol proxy
        # (in live trading this comes from real price data)
        vol_proxy = abs(self._rng.gauss(0, 0.5))  # ~half-normal, E≈0.4
        if vol_proxy < vol_thresh:
            return  # market too quiet for this agent's taste

        # regime_bias shifts strategy weights:
        #   +1 → amplify momentum/breakout, suppress mean_rev
        #   -1 → amplify mean_rev, suppress momentum/breakout
        # The shift is additive and bounded by [0, max_original_weight * 2]
        bias = regime_bias  # already in [-1, 1]
        momentum_w = max(0.0, momentum_w + bias * 0.3)
        breakout_w = max(0.0, breakout_w + bias * 0.2)
        mean_rev_w = max(0.0, mean_rev_w - bias * 0.3)

        # Pick symbol with strongest pseudo-signal
        best_sym = None
        best_signal = 0.0

        # Technical indicator gene weights
        ema_w = genes.get("ema_weight", 0.0)
        rsi_w = genes.get("rsi_weight", 0.0)
        atr_w = genes.get("atr_weight", 0.0)
        trend_w = genes.get("trend_strength_weight", 0.0)
        vol_w = genes.get("volatility_weight", 0.0)
        macro_w = genes.get("macro_regime_weight", 0.0)
        # regime_trend_gate / regime_vol_gate genes preserved in genome for
        # future soft regime modulation layer. Bypassed in current F4.4 mode.

        for sym, tick in ticks.items():
            # ── Legacy pseudo-signals (always active) ────
            mom_sig = self._rng.gauss(0, 1.0)        # trend-following
            mr_sig = -mom_sig * 0.6 + self._rng.gauss(0, 0.3)  # mean-revert
            scalp_sig = self._rng.gauss(0, 0.5)      # noise scalp
            brk_sig = mom_sig * 1.5 if abs(mom_sig) > 1.0 else 0  # breakout

            raw = (momentum_w * mom_sig +
                   mean_rev_w * mr_sig +
                   scalping_w * scalp_sig +
                   breakout_w * brk_sig)

            # ── Technical indicator signals (when available) ──
            # EMA crossover: fast > slow → bullish (+1), fast < slow → bearish (-1)
            if tick.ema_fast > 0 and tick.ema_slow > 0:
                ema_diff = (tick.ema_fast - tick.ema_slow) / tick.ema_slow
                ema_sig = math.tanh(ema_diff * 50)  # scale to [-1, 1]
                raw += ema_w * ema_sig

            # RSI: >70 → overbought (short signal), <30 → oversold (long)
            if tick.rsi != 50.0:  # non-default
                rsi_sig = -(tick.rsi - 50.0) / 50.0  # [-1, 1], negative when overbought
                raw += rsi_w * rsi_sig

            # ATR: high ATR → volatile → scale signal magnitude
            if tick.atr > 0 and tick.price > 0:
                atr_pct = tick.atr / tick.price
                atr_sig = math.tanh(atr_pct * 30) * (1 if raw > 0 else -1)
                raw += atr_w * atr_sig

            # Trend strength: strong trend → amplify directional bias
            if tick.trend_strength > 0:
                trend_sig = math.tanh(tick.trend_strength * 100)
                ema_dir = 1.0 if tick.ema_fast >= tick.ema_slow else -1.0
                raw += trend_w * trend_sig * ema_dir

            # Rolling volatility: high vol → amplify signal, low → dampen
            if tick.rolling_vol > 0:
                vol_sig = math.tanh(tick.rolling_vol * 50)
                raw *= (1.0 + vol_w * vol_sig)

            # Macro regime: EMA200 trend filter
            # macro_trend > 0 → bullish macro, < 0 → bearish macro
            # macro_regime_weight in [-1, +1]: positive = trend-follow, negative = contrarian
            if tick.macro_trend != 0.0:
                macro_sig = math.tanh(tick.macro_trend * 5)
                raw += macro_w * macro_sig

            # ── Soft Regime Modulation (bypassed for pure F4.4) ──
            # regime genes remain in genome for future activation
            # regime_factor = 1.0 (no modulation)
            # Normalize to approximate [-1, 1] via tanh
            signal = math.tanh(raw)

            if abs(signal) > abs(best_signal):
                best_signal = signal
                best_sym = sym

        if best_sym is None or abs(best_signal) < conf_threshold:
            return

        # Size position
        tick = ticks[best_sym]
        notional = self._capital * risk_pct * max(leverage, 1.0)
        size = notional / tick.price if tick.price > 0 else 0
        if size <= 0 or notional < 1.0:
            return

        side = "long" if best_signal > 0 else "short"
        self._position = {
            "symbol": best_sym,
            "entry_price": tick.price,
            "entry_step": step,
            "size": size,
            "side": side,
            "notional": notional,
        }
        self._exposure = {best_sym: notional / max(self._capital, 0.01)}

    def _check_exit(
        self, ticks: Dict[str, Tick], step: int, genes: Dict[str, float],
    ) -> None:
        """Check stop loss, take profit, trailing stop."""
        pos = self._position
        sym = pos["symbol"]
        if sym not in ticks:
            return

        price = ticks[sym].price
        entry = pos["entry_price"]
        side = pos["side"]

        if side == "long":
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        sl = genes.get("stop_loss_pct", 1.5) / 100.0
        tp = genes.get("take_profit_pct", 3.0) / 100.0

        # timeframe_bias → max hold duration
        # [0, 0.33) = 1m scalper: 20 steps
        # [0.33, 0.66) = 5m swing: 60 steps
        # [0.66, 1.0] = 15m position: 150 steps
        tf = genes.get("timeframe_bias", 0.5)
        if tf < 0.33:
            max_hold = 20
        elif tf < 0.66:
            max_hold = 60
        else:
            max_hold = 150

        should_exit = False
        if pnl_pct <= -sl:
            should_exit = True
        elif pnl_pct >= tp:
            should_exit = True
        elif step - pos["entry_step"] > max_hold:
            should_exit = True

        if should_exit:
            pnl = pos["size"] * entry * pnl_pct
            # Deduct round-trip fees (entry + exit) from PnL
            if self._fee_rate > 0:
                notional = pos["size"] * entry
                pnl -= notional * self._fee_rate * 2  # entry + exit
            self._capital += pnl
            self._pnl_series.append(pnl)
            self._trend_series.append(ticks[sym].trend_strength)
            self._sharpe_returns.append(pnl_pct)

            if pnl >= 0:
                self._wins += 1
            else:
                self._losses += 1

            if self._capital > self._peak:
                self._peak = self._capital
            dd = (self._peak - self._capital) / max(self._peak, 0.01) * 100
            if dd > self._max_dd:
                self._max_dd = dd

            self._closed_trades.append(SimTrade(
                symbol=sym, entry_price=entry, exit_price=price,
                entry_step=pos["entry_step"], exit_step=step,
                size=pos["size"], pnl=round(pnl, 4),
                pnl_pct=round(pnl_pct * 100, 2), side=side,
            ))

            self._position = None
            self._exposure = {}
            # Cooldown
            cd = int(genes.get("cooldown_minutes", 15))
            self._cooldown_until = step + max(1, cd // 5)

    def get_eval_data(self) -> AgentEvalData:
        """Build AgentEvalData for EvolutionEngine."""
        total = self._wins + self._losses
        dd = (self._peak - self._capital) / max(self._peak, 0.01) * 100
        realized = sum(self._pnl_series)

        sharpe = 0.0
        if len(self._sharpe_returns) >= 2:
            mu = statistics.mean(self._sharpe_returns)
            sd = statistics.stdev(self._sharpe_returns)
            if sd > 0:
                # Annualize by trades/year proxy.
                # 4H bars: ~2190/year; normalize by sqrt(trade_count/year)
                # Use sqrt(252) as conservative proxy (matches industry standard
                # for daily-equivalent risk-adjusted returns at this scale).
                n_trades = max(len(self._sharpe_returns), 1)
                annualization = math.sqrt(min(n_trades, 252))
                sharpe = (mu / sd) * annualization

        # v4.2: compute gross profit, gross loss, total notional
        gross_profit = 0.0
        gross_loss = 0.0
        total_notional = 0.0
        for t in self._closed_trades:
            notional = t.size * t.entry_price
            total_notional += notional
            if t.pnl >= 0:
                gross_profit += t.pnl
            else:
                gross_loss += abs(t.pnl)

        return AgentEvalData(
            metrics=AgentMetrics(
                agent_id=self.agent_id,
                generation=self.dna.generation,
                phase="dead",
                capital=round(self._capital, 2),
                realized_pnl=round(realized, 2),
                total_trades=total,
                winning_trades=self._wins,
                losing_trades=self._losses,
                sharpe_ratio=round(sharpe, 4),
                max_drawdown_pct=round(dd, 2),
                fitness=0.0,  # will be overwritten by engine
            ),
            initial_capital=self._initial_capital,
            pnl_series=list(self._pnl_series),
            exposure=dict(self._exposure),
            dna=self.dna,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_notional=total_notional,
            trend_series=list(self._trend_series),
            n_bars=self._n_bars,
            bars_gated=self._bars_gated,
        )


# ═════════════════════════════════════════════════════════════
# Per-generation results
# ═════════════════════════════════════════════════════════════

@dataclass
class GenerationResult:
    """Complete results for one simulated generation."""
    generation: int = 0
    snapshot: GenerationSnapshot = field(default_factory=GenerationSnapshot)
    diagnostics_dict: Dict[str, Any] = field(default_factory=dict)
    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_state: str = ""
    pool_capital: float = 0.0
    pool_pnl: float = 0.0
    gene_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════
# Simulation results + export
# ═════════════════════════════════════════════════════════════

@dataclass
class SimulationResults:
    """Complete simulation output across all generations."""
    seed: int = 0
    generations_run: int = 0
    total_elapsed_ms: float = 0.0
    starting_capital: float = 0.0
    final_capital: float = 0.0
    best_fitness_ever: float = 0.0
    best_agent_id: str = ""
    scenario_type: str = ""
    generation_results: List[GenerationResult] = field(default_factory=list)
    hall_of_fame: List[Dict[str, Any]] = field(default_factory=list)
    final_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def export_csv(self, path: str) -> str:
        """Export per-generation summary to CSV. Returns path written."""
        rows = []
        for gr in self.generation_results:
            s = gr.snapshot
            diag = gr.diagnostics_dict
            rows.append({
                "generation": gr.generation,
                "population": s.population_size,
                "best_fitness": round(s.best_fitness, 4),
                "avg_fitness": round(s.avg_fitness, 4),
                "worst_fitness": round(s.worst_fitness, 4),
                "total_trades": s.total_trades,
                "total_pnl": round(s.total_pnl, 2),
                "pool_capital": round(gr.pool_capital, 2),
                "pool_win_rate": round(s.pool_win_rate * 100, 1),
                "pool_sharpe": round(s.pool_sharpe, 4),
                "pool_max_drawdown": round(s.pool_max_drawdown, 2),
                "survivors": s.survivors,
                "eliminated": s.eliminated,
                "portfolio_state": gr.portfolio_state,
                "diversity_score": round(
                    diag.get("diversity", {}).get("overall_diversity_score", 0), 4),
                "health_score": round(diag.get("health_score", 0), 4),
                "capital_gini": round(
                    diag.get("concentration", {}).get("capital_gini", 0), 4),
                "stagnating": diag.get("fitness", {}).get("stagnating", False),
                "n_alerts": len(diag.get("alerts", [])),
            })

        p = Path(path)
        if rows:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info("CSV exported: %s (%d rows)", p, len(rows))
        return str(p)

    def export_json(self, path: str) -> str:
        """Export full results to JSON."""
        data = {
            "meta": {
                "seed": self.seed,
                "generations_run": self.generations_run,
                "elapsed_ms": round(self.total_elapsed_ms, 1),
                "starting_capital": self.starting_capital,
                "final_capital": round(self.final_capital, 2),
                "best_fitness": round(self.best_fitness_ever, 4),
                "best_agent_id": self.best_agent_id,
                "scenario_type": self.scenario_type,
            },
            "hall_of_fame": self.hall_of_fame,
            "final_diagnostics": self.final_diagnostics,
            "generations": [
                {
                    "generation": gr.generation,
                    "pool_capital": round(gr.pool_capital, 2),
                    "pool_pnl": round(gr.pool_pnl, 2),
                    "portfolio_state": gr.portfolio_state,
                    "snapshot": {
                        "best_fitness": round(gr.snapshot.best_fitness, 4),
                        "avg_fitness": round(gr.snapshot.avg_fitness, 4),
                        "worst_fitness": round(gr.snapshot.worst_fitness, 4),
                        "total_trades": gr.snapshot.total_trades,
                        "total_pnl": round(gr.snapshot.total_pnl, 2),
                        "survivors": gr.snapshot.survivors,
                        "eliminated": gr.snapshot.eliminated,
                        "agent_rankings": gr.snapshot.agent_rankings,
                    },
                    "diagnostics": gr.diagnostics_dict,
                    "gene_stats": gr.gene_stats,
                }
                for gr in self.generation_results
            ],
        }

        p = Path(path)
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("JSON exported: %s", p)
        return str(p)

    def to_csv_string(self) -> str:
        """Return CSV as string (for in-memory use)."""
        buf = io.StringIO()
        rows = []
        for gr in self.generation_results:
            s = gr.snapshot
            rows.append({
                "generation": gr.generation,
                "best_fitness": round(s.best_fitness, 4),
                "avg_fitness": round(s.avg_fitness, 4),
                "total_pnl": round(s.total_pnl, 2),
                "pool_capital": round(gr.pool_capital, 2),
            })
        if rows:
            writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return buf.getvalue()


# ═════════════════════════════════════════════════════════════
# Simulation configuration
# ═════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """All simulation parameters."""
    scenario: MarketScenario = field(
        default_factory=lambda: MonteCarloScenario())
    generations: int = 20
    pool_size: int = 5
    trades_per_generation: int = 200   # ticks per agent per generation
    starting_capital: float = 50.0
    seed: int = 42
    # Evolution
    survival_rate: float = 0.5
    elitism_count: int = 1
    mutation_rate: float = 0.15
    # Risk
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    # Capital floor: if pool capital drops below this fraction of
    # starting_capital, trigger defensive reallocation.
    # 0.45 = trigger when capital < 45% of start.
    # 0.0 = disabled.
    capital_floor_pct: float = 0.45
    # Elite archive for cross-run genotype seeding (optional)
    elite_archive: Any = None  # EliteArchive instance or None
    # Drawdown Dynamic Throttle (DDT)
    enable_ddt: bool = False
    # Trading costs
    maker_fee: float = 0.0002   # 0.02% maker
    taker_fee: float = 0.0005   # 0.05% taker
    slippage: float = 0.0003    # 0.03% per trade
    # Cluster-aware initialization (optional)
    cluster_type: str | None = None  # AssetProfiler archetype


# ═════════════════════════════════════════════════════════════
# Simulation Harness — main orchestrator
# ═════════════════════════════════════════════════════════════

class SimulationHarness:
    """
    Orchestrates complete offline evolutionary simulations.

    Flow per generation:
      1. Generate/advance price data from scenario
      2. Each agent trades the price series using its DNA
      3. Collect AgentEvalData from all agents
      4. Update PortfolioRiskEngine with pool equity
      5. EvolutionEngine.evaluate_generation() scores + ranks
      6. Diagnostics.diagnose() produces health report
      7. Select survivors, breed next generation
      8. Record metrics
      9. Repeat
    """

    __slots__ = (
        "_cfg", "_rng", "_evolution", "_risk_engine",
        "_diagnostics", "_results",
    )

    def __init__(self, config: SimConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)
        self._evolution = EvolutionEngine(
            survival_rate=config.survival_rate,
            elitism_count=config.elitism_count,
            base_mutation_rate=config.mutation_rate,
        )
        self._risk_engine = PortfolioRiskEngine(limits=config.risk_limits)
        self._diagnostics = EvolutionDiagnostics()
        self._results: List[GenerationResult] = []

    async def run(self) -> SimulationResults:
        """
        Execute the full simulation. Returns structured results.
        Deterministic: same seed → same results.
        """
        import time
        t0 = time.monotonic()

        cfg = self._cfg
        scenario = cfg.scenario
        symbols = scenario.get_symbols()
        scenario_type = type(scenario).__name__

        logger.info(
            "SIM START: seed=%d gens=%d pool=%d capital=$%.2f scenario=%s symbols=%s",
            cfg.seed, cfg.generations, cfg.pool_size,
            cfg.starting_capital, scenario_type, symbols,
        )

        # DETERMINISM: Lock all RNGs and reset ID counter
        from darwin_agent.determinism import lock_determinism, reset_id_counter
        lock_determinism(cfg.seed)
        reset_id_counter(cfg.seed)

        # DETERMINISM: Seed the global random module so that
        # EvolutionEngine (which uses random.random/gauss/sample)
        # produces identical results across runs with same seed.
        random.seed(cfg.seed)

        # Initialize risk engine with starting equity
        total_capital = cfg.starting_capital
        self._risk_engine.update_equity(total_capital)

        # Seed generation 0: archive elites + random DNA
        dna_pool = self._evolution.seed_initial_population(
            pool_size=cfg.pool_size,
            archive=cfg.elite_archive,
            rng=random.Random(self._rng.randint(0, 2**32)),
            cluster_type=cfg.cluster_type,
        )

        best_fitness_ever = 0.0
        best_agent_id = ""
        alloc_mult = 1.0  # reduced to 0.5 during capital floor breach

        # DDT: Drawdown Dynamic Throttle — tracks peak equity
        ddt_peak = cfg.starting_capital
        ddt_mult = 1.0

        for gen in range(cfg.generations):
            gen_rng = random.Random(self._rng.randint(0, 2**32))

            # 1. Generate prices for this generation
            price_data = scenario.generate(cfg.trades_per_generation, gen_rng)

            # 1b. DDT 2.0: Smooth Drawdown Dynamic Throttle
            #     exposure = max(0.2, 1 - 1.5 * rolling_dd)
            if cfg.enable_ddt:
                if total_capital > ddt_peak:
                    ddt_peak = total_capital
                rolling_dd = (ddt_peak - total_capital) / ddt_peak if ddt_peak > 0 else 0.0
                ddt_mult = max(0.2, 1.0 - 1.5 * rolling_dd)
            else:
                ddt_mult = 1.0

            # 2. Spawn simulated agents
            #    DDT 2.0 throttles agent-level risk only (leverage + risk_pct).
            #    Capital allocation stays full — no death spiral.
            per_agent_cap = (total_capital * alloc_mult) / max(cfg.pool_size, 1)
            _fee_rate = cfg.taker_fee + cfg.slippage  # combined cost per side
            agents: List[SimulatedAgent] = []
            for i, dna in enumerate(dna_pool[:cfg.pool_size]):
                agent = SimulatedAgent(
                    agent_id=f"sim-g{gen}-{i}",
                    dna=dna,
                    capital=per_agent_cap,
                    rng=random.Random(gen_rng.randint(0, 2**32)),
                    exposure_mult=ddt_mult,
                    fee_rate=_fee_rate,
                )
                agents.append(agent)

            # 3. Run all agents through price series
            for step in range(cfg.trades_per_generation):
                step_ticks = {
                    sym: ticks[step] for sym, ticks in price_data.items()
                }
                for agent in agents:
                    agent.step(step_ticks, step)

            # 4. Collect eval data
            eval_data = [a.get_eval_data() for a in agents]

            # 5. Update risk engine with pool equity
            pool_capital = sum(ed.metrics.capital for ed in eval_data)
            pool_pnl = sum(ed.metrics.realized_pnl for ed in eval_data)
            self._risk_engine.update_equity(pool_capital)
            portfolio_snap = self._risk_engine.get_portfolio_state()

            # 5b. Capital floor check
            #     If pool capital < floor_pct × starting_capital:
            #       - Halve allocation multiplier (reduces next-gen position sizes)
            #       - Reinitialize bottom 40% of agents with fresh DNA
            #       - Force risk state to CRITICAL
            #     If capital recovers above floor, restore full allocation.
            # Capital floor check
            floor_abs = cfg.capital_floor_pct * cfg.starting_capital
            if cfg.capital_floor_pct > 0 and pool_capital < floor_abs:
                alloc_mult = 0.5
                logger.info(
                    "  CAPITAL FLOOR BREACH gen %d: $%.2f < $%.2f (%.0f%% floor) "
                    "→ halve allocation, reinit bottom 40%%",
                    gen, pool_capital, floor_abs,
                    cfg.capital_floor_pct * 100,
                )

                # Force risk state to critical
                portfolio_snap.risk_state = PortfolioRiskState.CRITICAL

                # Reinitialize bottom 40% of dna_pool with fresh random DNA
                sorted_evals = sorted(
                    eval_data, key=lambda e: e.metrics.fitness)
                n_reinit = max(1, int(len(sorted_evals) * 0.4))
                reinit_ids = sorted(
                    e.metrics.agent_id for e in sorted_evals[:n_reinit]
                )

                # Replace their DNA — deterministic via gen_rng
                reinit_rng = random.Random(gen_rng.randint(0, 2**32))
                for i, ed in enumerate(eval_data):
                    if ed.metrics.agent_id in reinit_ids and i < len(dna_pool):
                        old_state = random.getstate()
                        random.seed(reinit_rng.randint(0, 2**32))
                        dna_pool[i] = self._evolution.create_random_dna()
                        random.setstate(old_state)
            else:
                alloc_mult = 1.0

            # 6. Evaluate generation (scores + ranks)
            snapshot = await self._evolution.evaluate_generation(
                eval_data, portfolio_snapshot=portfolio_snap)

            # 7. Diagnostics
            diag_report = self._diagnostics.diagnose(gen, eval_data, dna_pool)

            # 8. Track best ever
            if snapshot.best_fitness > best_fitness_ever:
                best_fitness_ever = snapshot.best_fitness
                best_agent_id = snapshot.best_agent_id

            # 9. Compute gene stats for this generation
            gene_stats = self._compute_gene_stats(dna_pool)

            # 10. Build per-agent result dicts
            agent_results = []
            for ed in eval_data:
                m = ed.metrics
                agent_results.append({
                    "agent_id": m.agent_id,
                    "fitness": round(m.fitness, 4),
                    "capital": round(m.capital, 2),
                    "pnl": round(m.realized_pnl, 2),
                    "trades": m.total_trades,
                    "wins": m.winning_trades,
                    "losses": m.losing_trades,
                    "win_rate": round(m.win_rate * 100, 1),
                    "sharpe": round(m.sharpe_ratio, 4),
                    "max_dd": round(m.max_drawdown_pct, 2),
                })

            # 11. Store generation result
            self._results.append(GenerationResult(
                generation=gen,
                snapshot=snapshot,
                diagnostics_dict=diag_report.to_dict(),
                agent_results=agent_results,
                portfolio_state=portfolio_snap.risk_state.value,
                pool_capital=round(pool_capital, 2),
                pool_pnl=round(pool_pnl, 2),
                gene_stats=gene_stats,
            ))

            # 12. Evolution: select survivors → breed next gen
            survivor_ids = self._evolution.select_survivors(
                [ed.metrics for ed in eval_data])
            survivor_dna = [
                ed.dna for ed in eval_data
                if ed.metrics.agent_id in survivor_ids and ed.dna
            ]
            # Attach fitness to survivor DNA for breeding priority
            for ed in eval_data:
                if ed.dna and ed.metrics.agent_id in survivor_ids:
                    ed.dna.fitness = ed.metrics.fitness

            dna_pool = self._evolution.create_next_generation(
                survivor_dna, cfg.pool_size)

            # Update total capital for next gen
            total_capital = max(pool_capital, 1.0)

            # Log progress
            if gen % 5 == 0 or gen == cfg.generations - 1:
                logger.info(
                    "  gen %d/%d │ best=%.4f avg=%.4f │ "
                    "capital=$%.2f │ pnl=$%.2f │ state=%s │ alerts=%d",
                    gen, cfg.generations - 1,
                    snapshot.best_fitness, snapshot.avg_fitness,
                    pool_capital, pool_pnl,
                    portfolio_snap.risk_state.value,
                    len(diag_report.alerts),
                )

        elapsed = (time.monotonic() - t0) * 1000

        # Build final results
        hof = self._evolution.get_hall_of_fame()
        final_diag = self._diagnostics.diagnose(
            cfg.generations - 1, eval_data, dna_pool).to_dict()

        results = SimulationResults(
            seed=cfg.seed,
            generations_run=cfg.generations,
            total_elapsed_ms=elapsed,
            starting_capital=cfg.starting_capital,
            final_capital=total_capital,
            best_fitness_ever=best_fitness_ever,
            best_agent_id=best_agent_id,
            scenario_type=scenario_type,
            generation_results=self._results,
            hall_of_fame=[d.to_dict() for d in hof],
            final_diagnostics=final_diag,
        )

        logger.info(
            "SIM COMPLETE: %d gens in %.0fms │ "
            "$%.2f → $%.2f │ best_fitness=%.4f",
            cfg.generations, elapsed,
            cfg.starting_capital, total_capital,
            best_fitness_ever,
        )

        # Deposit top genotypes into archive if provided
        if cfg.elite_archive is not None:
            capital_return = ((total_capital - cfg.starting_capital) /
                              cfg.starting_capital * 100
                              if cfg.starting_capital > 0 else 0.0)
            deposit_entries = [
                {
                    "genes": dict(d.genes),
                    "fitness": d.fitness,
                    "return_pct": capital_return,
                    "generation": d.generation,
                    "dna_id": d.dna_id,
                }
                for d in hof if d.genes
            ]
            cfg.elite_archive.deposit(deposit_entries, seed=cfg.seed)

        return results

    def _compute_gene_stats(
        self, dna_pool: List[DNAData],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/std/min/max per gene across pool."""
        if not dna_pool:
            return {}
        all_genes = sorted(DEFAULT_GENE_RANGES.keys())
        stats: Dict[str, Dict[str, float]] = {}
        for gene in all_genes:
            values = [d.genes.get(gene, 0.0) for d in dna_pool]
            stats[gene] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
        return stats
