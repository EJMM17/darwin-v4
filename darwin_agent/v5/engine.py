"""
Darwin v5 — Engine Orchestrator.

Integrates all v5 layers into a unified trading engine:
    1. Market Data Layer       → fetch OHLCV, compute features
    2. Feature Engineering     → technical indicators
    3. Regime Detection        → classify market state
    4. Multi-Factor Signals    → generate trade signals
    5. Risk & Position Sizing  → dynamic sizing with vol/dd scaling
    6. Execution Engine        → validated order placement with retry
    7. Telemetry & Logging     → structured observability

Usage:
    engine = DarwinV5Engine(binance_client, telegram_notifier, config)
    await engine.initialize()
    await engine.run_forever()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from darwin_agent.v5.time_sync import TimeSyncManager
from darwin_agent.v5.health_monitor import HealthMonitor
from darwin_agent.v5.telemetry import TelemetryReporter
from darwin_agent.v5.market_data import MarketDataLayer
from darwin_agent.v5.regime_detector import RegimeDetector
from darwin_agent.v5.signal_generator import SignalGenerator
from darwin_agent.v5.position_sizer import PositionSizer, SizerConfig
from darwin_agent.v5.execution_engine import (
    ExecutionEngine,
    ExecutionConfig,
    OrderRequest as V5OrderRequest,
)
from darwin_agent.v5.monte_carlo import MonteCarloValidator
from darwin_agent.v5.portfolio_constructor import PortfolioConstructor

logger = logging.getLogger("darwin.v5.engine")


@dataclass
class V5EngineConfig:
    """Top-level configuration for the v5 engine."""
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    # Timing
    tick_interval_s: float = 5.0
    heartbeat_interval_ticks: int = 12  # log heartbeat every 12 ticks (60s at 5s interval)
    monte_carlo_interval_ticks: int = 360  # run MC every 30 min
    # Risk
    leverage: int = 5
    base_risk_pct: float = 1.0
    confidence_threshold: float = 0.60
    # Mode
    dry_run: bool = False
    # Max retries before fatal
    max_consecutive_errors: int = 10


@dataclass(slots=True)
class V5EngineState:
    """Runtime state of the v5 engine."""
    equity: float = 0.0
    wallet_balance: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    current_regime: str = "range_bound"
    tick_count: int = 0
    trade_pnls: List[float] = field(default_factory=list)
    signal_confidences: List[float] = field(default_factory=list)
    daily_start_equity: float = 0.0
    daily_pnl: float = 0.0
    last_heartbeat_tick: int = 0


class DarwinV5Engine:
    """
    Institutional-grade trading engine orchestrator.

    Manages the full pipeline from market data to execution,
    with comprehensive observability and risk management.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client.
    telegram_notifier : TelegramNotifier
        Notification service.
    config : V5EngineConfig, optional
        Engine configuration.
    """

    def __init__(
        self,
        binance_client: Any,
        telegram_notifier: Any,
        config: V5EngineConfig | None = None,
    ) -> None:
        self._config = config or V5EngineConfig()
        self._binance = binance_client
        self._telegram = telegram_notifier

        # Infrastructure modules
        self._time_sync = TimeSyncManager(binance_client)
        self._health_monitor = HealthMonitor(binance_client, telegram_notifier)
        self._telemetry = TelemetryReporter(telegram_notifier)

        # Trading pipeline modules
        self._market_data = MarketDataLayer(binance_client)
        self._regime_detector = RegimeDetector()
        self._signal_generator = SignalGenerator()
        self._position_sizer = PositionSizer(SizerConfig(
            base_risk_pct=self._config.base_risk_pct,
            leverage=self._config.leverage,
        ))
        self._execution_engine = ExecutionEngine(
            binance_client,
            self._time_sync,
            self._telemetry,
            ExecutionConfig(leverage=self._config.leverage),
        )
        self._monte_carlo = MonteCarloValidator()
        self._portfolio = PortfolioConstructor(self._config.symbols)

        # State
        self._state = V5EngineState()
        self._stop_event = asyncio.Event()
        self._running = False
        self._consecutive_errors = 0

    @property
    def state(self) -> V5EngineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the engine: sync time, validate connectivity,
        set leverage, fetch initial state.

        Returns startup summary dict.
        """
        logger.info("=== Darwin v5 Engine Initialization ===")

        # 1. Time sync
        logger.info("syncing time with exchange...")
        self._time_sync.sync()
        logger.info("time offset: %dms", self._time_sync.offset_ms)

        # 2. Health check
        logger.info("checking exchange connectivity...")
        connected = await self._health_monitor.wait_for_connection(timeout_s=30.0)
        if not connected:
            raise RuntimeError("Failed to connect to exchange")
        logger.info("exchange connected")

        # 3. Fetch wallet state
        wallet = await asyncio.to_thread(self._binance.get_wallet_balance)
        upnl = await asyncio.to_thread(self._binance.get_unrealized_pnl)
        positions = await asyncio.to_thread(self._binance.get_open_positions)

        self._state.wallet_balance = float(wallet)
        self._state.unrealized_pnl = float(upnl)
        self._state.equity = self._state.wallet_balance + self._state.unrealized_pnl
        self._state.peak_equity = self._state.equity
        self._state.daily_start_equity = self._state.equity
        self._state.open_positions = list(positions)

        if self._state.equity <= 0:
            raise RuntimeError(f"Equity must be > 0, got {self._state.equity}")

        # 4. Set leverage on all symbols
        leverage_results = {}
        for symbol in self._config.symbols:
            try:
                ok = await asyncio.to_thread(
                    self._binance.set_leverage, symbol, self._config.leverage
                )
                leverage_results[symbol] = bool(ok)
            except Exception as exc:
                raise RuntimeError(f"Failed to set leverage on {symbol}: {exc}") from exc

        # 5. Load symbol step sizes from Binance exchange info
        try:
            step_sizes = await asyncio.to_thread(
                self._binance.get_symbol_step_sizes, self._config.symbols
            )
            self._execution_engine.load_symbol_info(step_sizes)
            logger.info("symbol step sizes loaded: %s", step_sizes)
        except Exception as exc:
            logger.warning("could not load symbol step sizes, using defaults: %s", exc)

        # 6. Initialize position sizer daily tracking
        self._position_sizer.reset_daily(self._state.equity)

        startup = {
            "wallet_balance": self._state.wallet_balance,
            "unrealized_pnl": self._state.unrealized_pnl,
            "equity": self._state.equity,
            "open_positions": len(positions),
            "leverage_results": leverage_results,
            "time_offset_ms": self._time_sync.offset_ms,
            "symbols": self._config.symbols,
            "dry_run": self._config.dry_run,
        }

        logger.info("=== v5 Initialization Complete ===")
        logger.info("Equity: $%.4f", self._state.equity)
        logger.info("Positions: %d", len(positions))
        logger.info("Symbols: %s", self._config.symbols)
        logger.info("Leverage: %dx", self._config.leverage)
        logger.info("Dry run: %s", self._config.dry_run)

        self._telegram.notify_engine_connected(startup)
        return startup

    def stop(self) -> None:
        """Request engine shutdown."""
        self._stop_event.set()
        self._running = False

    async def run_forever(self) -> int:
        """
        Main engine loop. Returns exit code (0=clean, 1=error).
        """
        self._stop_event.clear()
        self._running = True
        self._telegram.notify_engine_started()

        # Start health monitor in background
        health_task = asyncio.create_task(self._health_monitor.run())

        logger.info(
            "v5 engine started (interval=%.0fs, dry_run=%s)",
            self._config.tick_interval_s,
            self._config.dry_run,
        )

        try:
            while not self._stop_event.is_set():
                try:
                    await self._tick()
                    self._consecutive_errors = 0
                    await asyncio.sleep(self._config.tick_interval_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._consecutive_errors += 1
                    logger.error(
                        "tick error (%d/%d): %s",
                        self._consecutive_errors,
                        self._config.max_consecutive_errors,
                        exc,
                        exc_info=True,
                    )
                    self._telemetry.log_risk_event("tick_error", {
                        "error": str(exc),
                        "consecutive": self._consecutive_errors,
                    })

                    if self._consecutive_errors >= self._config.max_consecutive_errors:
                        logger.error("max consecutive errors reached, shutting down")
                        self._telegram.notify_error(
                            f"v5 engine fatal: {self._consecutive_errors} consecutive errors"
                        )
                        return 1

                    # Exponential backoff
                    backoff = min(2 ** self._consecutive_errors, 60)
                    await asyncio.sleep(backoff)

            return 0
        finally:
            self._running = False
            self._health_monitor.stop()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
            self._telegram.notify_engine_stopped()

    async def _tick(self) -> None:
        """Execute one tick of the engine pipeline."""
        self._state.tick_count += 1
        tick = self._state.tick_count

        # 1. Fetch current portfolio state
        wallet = await asyncio.to_thread(self._binance.get_wallet_balance)
        upnl = await asyncio.to_thread(self._binance.get_unrealized_pnl)
        positions = await asyncio.to_thread(self._binance.get_open_positions)

        self._state.wallet_balance = float(wallet)
        self._state.unrealized_pnl = float(upnl)
        self._state.equity = self._state.wallet_balance + self._state.unrealized_pnl
        self._state.open_positions = list(positions)

        # Update peak equity
        if self._state.equity > self._state.peak_equity:
            self._state.peak_equity = self._state.equity

        # Drawdown
        if self._state.peak_equity > 0:
            self._state.drawdown_pct = (
                (self._state.peak_equity - self._state.equity)
                / self._state.peak_equity * 100.0
            )

        # 2. Compute features for all symbols + BTC reference
        btc_features = await asyncio.to_thread(
            self._market_data.compute_features, "BTCUSDT"
        )

        for symbol in self._config.symbols:
            await self._process_symbol(symbol, btc_features)

        # 3. Heartbeat logging
        if tick - self._state.last_heartbeat_tick >= self._config.heartbeat_interval_ticks:
            self._state.last_heartbeat_tick = tick
            self._telemetry.log_heartbeat(
                equity=self._state.equity,
                wallet_balance=self._state.wallet_balance,
                unrealized_pnl=self._state.unrealized_pnl,
                open_positions=len(positions),
                regime=self._state.current_regime,
                drawdown_pct=self._state.drawdown_pct,
            )

        # 4. Periodic Monte Carlo validation
        if (
            tick % self._config.monte_carlo_interval_ticks == 0
            and len(self._state.trade_pnls) >= 10
        ):
            mc_result = self._monte_carlo.validate(self._state.trade_pnls)
            self._telemetry.log_monte_carlo_result(
                n_simulations=mc_result.n_simulations,
                mean_pnl=mc_result.mean_random_pnl,
                p5_pnl=mc_result.p5_pnl,
                p95_pnl=mc_result.p95_pnl,
                edge_ratio=mc_result.edge_ratio,
                max_drawdown_p95=mc_result.max_drawdown_p95,
            )

        logger.info(
            "tick %d: equity=$%.4f dd=%.1f%% positions=%d regime=%s",
            tick,
            self._state.equity,
            self._state.drawdown_pct,
            len(positions),
            self._state.current_regime,
        )

    async def _process_symbol(self, symbol: str, btc_features: Any) -> None:
        """Process one symbol through the full pipeline."""
        # 1. Compute features
        if symbol == "BTCUSDT":
            features = btc_features
        else:
            features = await asyncio.to_thread(
                self._market_data.compute_features, symbol
            )

        if features.close <= 0:
            return  # no data

        # 2. Detect regime
        regime = self._regime_detector.detect(features, btc_features)
        self._state.current_regime = regime.regime.value

        # 3. Fetch funding rate
        funding_rate = await asyncio.to_thread(
            self._market_data.fetch_funding_rate, symbol
        )

        # 4. Generate signal
        signal = self._signal_generator.generate(
            features, regime, btc_features, funding_rate
        )

        # Log signal
        self._telemetry.log_signal_generated(
            symbol=symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            threshold=signal.threshold,
            regime=regime.regime.value,
            factors=signal.factors,
            position_size_usdt=0.0,
        )

        # Update portfolio constructor
        self._portfolio.update_signal(symbol, signal)

        # 5. Check if signal passes
        if not signal.passed:
            if signal.rejection_reason:
                self._telemetry.log_signal_rejected(
                    symbol=symbol,
                    reason=signal.rejection_reason,
                    confidence=signal.confidence,
                    regime=regime.regime.value,
                )
            return

        if not signal.direction:
            return

        # 6. Check if we already have a position in this symbol
        for pos in self._state.open_positions:
            if pos.get("symbol") == symbol:
                # Already positioned, skip
                return

        # 7. Position sizing
        current_exposure = sum(
            abs(float(p.get("notional", 0)))
            for p in self._state.open_positions
        )

        size_result = self._position_sizer.compute(
            equity=self._state.equity,
            price=features.close,
            realized_vol=features.realized_vol_20,
            drawdown_pct=self._state.drawdown_pct,
            regime_multiplier=regime.risk_multiplier,
            signal_confidence=signal.confidence,
            current_exposure=current_exposure,
        )

        if not size_result.approved:
            self._telemetry.log_signal_rejected(
                symbol=symbol,
                reason=f"sizing_rejected: {size_result.rejection_reason}",
                confidence=signal.confidence,
                regime=regime.regime.value,
                details=size_result.to_dict(),
            )
            return

        # 8. Build order
        side = "BUY" if signal.direction == "LONG" else "SELL"
        order = V5OrderRequest(
            symbol=symbol,
            side=side,
            quantity=size_result.quantity,
            price=features.close,
            leverage=self._config.leverage,
        )

        # 9. Execute
        result = await self._execution_engine.place_order(
            order, dry_run=self._config.dry_run
        )

        if result.success:
            logger.info(
                "order filled: %s %s qty=%.6f price=%.2f slippage=%.1fbps",
                side, symbol, result.filled_qty, result.avg_price, result.slippage_bps,
            )
        else:
            logger.warning("order failed: %s %s error=%s", side, symbol, result.error)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return comprehensive diagnostics for all modules."""
        return {
            "time_sync": self._time_sync.get_diagnostics(),
            "health": self._health_monitor.get_diagnostics(),
            "execution": self._execution_engine.get_metrics(),
            "state": {
                "equity": self._state.equity,
                "wallet_balance": self._state.wallet_balance,
                "unrealized_pnl": self._state.unrealized_pnl,
                "drawdown_pct": self._state.drawdown_pct,
                "open_positions": len(self._state.open_positions),
                "tick_count": self._state.tick_count,
                "regime": self._state.current_regime,
                "total_trades": len(self._state.trade_pnls),
                "dry_run": self._config.dry_run,
            },
            "portfolio_weights": self._portfolio.get_weights(),
            "telemetry_events": self._telemetry.event_count,
        }
