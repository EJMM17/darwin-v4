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
import datetime
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
from darwin_agent.v5.order_flow import OrderFlowIntelligence
from darwin_agent.v5.performance_analytics import PerformanceAnalytics
from darwin_agent.v5.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig

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
    # Risk per trade — SL dinámico basado en ATR
    stop_loss_pct: float = 1.5      # SL mínimo (% del precio), floor de seguridad
    take_profit_pct: float = 3.0    # TP fijo (% del precio), ratio 2:1
    atr_sl_multiplier: float = 2.0  # SL = max(stop_loss_pct, atr_sl_multiplier * ATR%)
    # Trailing stop: protege ganancias parciales
    trailing_stop_enabled: bool = True
    trailing_activation_pct: float = 1.5  # activa trailing cuando PnL > 1.5%
    trailing_distance_pct: float = 0.8    # trailing a 0.8% debajo del máximo
    # Cooldown post-pérdida: evita re-entrar inmediatamente después de un stop
    post_loss_cooldown_ticks: int = 6     # esperar 6 ticks (~30s) tras un SL
    # Kill switches institucionales
    max_daily_loss_pct: float = 5.0       # detener trading si perdemos >5% del equity en un día
    max_drawdown_kill_pct: float = 15.0   # halt total si drawdown supera 15% desde el pico

    # ── Regime-adaptive thresholds ────────────────────────────────────────────
    # En TRENDING: configuración normal de momentum.
    # En RANGE_BOUND / LOW_VOL: más exigente, SL/TP ajustados, throttle de freq.

    # Confianza mínima por régimen
    confidence_threshold_trending: float = 0.60   # umbral en tendencia
    confidence_threshold_ranging:  float = 0.78   # más exigente en lateral

    # SL/TP por régimen
    sl_pct_trending:  float = 1.5    # SL en tendencia (ratio 2:1)
    tp_pct_trending:  float = 3.0    # TP en tendencia
    sl_pct_ranging:   float = 0.7    # SL ajustado en lateral (ratio 1.5:1)
    tp_pct_ranging:   float = 1.2    # TP pequeño pero alcanzable en lateral

    # Frequency throttle: mínimo de ticks entre trades en lateral
    # 120 ticks x 5s = 10 minutos entre trades por simbolo
    ranging_trade_cooldown_ticks: int = 120

    # ATR minimo para abrir trade (filtro de actividad)
    # Si ATR < umbral, las fees erosionan el profit
    min_atr_pct_to_trade: float = 0.25

    # Leverage multiplier en lateral (reservado para uso futuro)
    leverage_multiplier_ranging: float = 0.5

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
    current_day: int = -1  # track UTC date for daily reset
    # Cooldown tracking: symbol -> tick number when cooldown expires
    cooldown_until: dict = field(default_factory=dict)
    # Trailing stop tracking: symbol -> peak_pnl_pct seen so far
    trailing_peaks: dict = field(default_factory=dict)
    # Frequency throttle en lateral: symbol -> tick del último trade abierto
    last_trade_tick: dict = field(default_factory=dict)
    # Kill switch state
    trading_halted: bool = False
    halt_reason: str = ""


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
        self._order_flow = OrderFlowIntelligence(binance_client)
        self._analytics = PerformanceAnalytics()
        self._portfolio_risk = PortfolioRiskManager(PortfolioRiskConfig())
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
        self._symbol_step_sizes: Dict[str, float] = {}  # populated on startup
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

        # 3. Fetch wallet state (single API call)
        wallet, upnl = await asyncio.to_thread(self._binance.get_account_snapshot)
        positions = await asyncio.to_thread(self._binance.get_open_positions)

        self._state.wallet_balance = float(wallet)
        self._state.unrealized_pnl = float(upnl)
        self._state.equity = self._state.wallet_balance + self._state.unrealized_pnl
        self._state.peak_equity = self._state.equity
        # Send daily performance report before resetting (except on first run)
        if self._state.daily_start_equity > 0:
            perf = self._analytics.get_report()
            if perf and perf.total_trades > 0:
                try:
                    self._telegram.send_message(perf.to_telegram())
                    logger.info("Daily performance report sent to Telegram")
                except Exception as _e:
                    logger.warning("Failed to send daily report: %s", _e)

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
            self._symbol_step_sizes = step_sizes  # also keep locally for position sizing
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

        # 1. Fetch current portfolio state (single API call for wallet+upnl)
        wallet, upnl = await asyncio.to_thread(self._binance.get_account_snapshot)
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

        # 1b. Daily reset check (UTC midnight)
        today = datetime.datetime.now(datetime.timezone.utc).toordinal()
        if self._state.current_day != today:
            self._state.current_day = today
            self._position_sizer.reset_daily(self._state.equity)
            # Reset kill switch daily loss trigger (max drawdown halt persists)
            if self._state.trading_halted and "daily_loss_limit" in self._state.halt_reason:
                self._state.trading_halted = False
                self._state.halt_reason = ""
                logger.info("KILL SWITCH reset: new trading day started")
            # Export equity curve CSV at day rollover (institutional audit trail)
            self._export_equity_csv()
            logger.info("daily reset: new equity baseline $%.4f", self._state.equity)

        # 2. Manage open positions (SL/TP check) BEFORE new signals
        await self._manage_open_positions(positions)

        # 3. Compute features for all symbols + BTC reference
        btc_features = await asyncio.to_thread(
            self._market_data.compute_features, "BTCUSDT"
        )

        # 3b. Kill switch: verificar límites de pérdida antes de abrir nuevas posiciones
        await self._check_kill_switches()
        if self._state.trading_halted:
            logger.warning(
                "TRADING HALTED: %s | equity=$%.4f | daily_pnl=%.2f%%",
                self._state.halt_reason,
                self._state.equity,
                self._state.daily_pnl / max(self._state.daily_start_equity, 1.0) * 100,
            )
        else:
            for symbol in self._config.symbols:
                await self._process_symbol(symbol, btc_features)

        # 4. Heartbeat logging
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
            # Feed equity snapshot to performance analytics (institutional metrics)
            self._analytics.update_equity(self._state.equity)
            # Log live metrics every 5 minutes (every 5 heartbeats at 12 ticks/heartbeat)
            if tick % (self._config.heartbeat_interval_ticks * 5) == 0:
                snap = self._analytics.snapshot()
                if snap and snap.n_trades > 0:
                    logger.info(self._analytics.summary_line())
                    # Circuit breaker check (institutional risk controls)
                    breached, reason = self._analytics.circuit_breaker_check()
                    if breached:
                        logger.warning("CIRCUIT BREAKER: %s", reason)
            # Record equity for real-time Sharpe/Sortino/Calmar computation
            self._analytics.record_equity(self._state.equity)
            # Update portfolio risk daily equity (circuit breaker + Kelly)
            self._portfolio_risk.update_daily_equity(self._state.equity)
            # Log performance metrics every heartbeat (60s)
            perf = self._analytics.get_report()
            if perf:
                logger.info(perf.to_log())

        # 5. Periodic Monte Carlo validation
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




    def _export_equity_csv(self) -> None:
        """
        Export equity curve and trade history to CSV for institutional audit trail.

        Files written to /data/analytics/ (Docker volume):
          equity_curve_YYYY-MM-DD.csv   — timestamp, equity per heartbeat
          trade_history_YYYY-MM-DD.csv  — all closed trades with PnL

        These CSVs allow:
          - Independent auditor verification of performance claims
          - Backtesting ground truth comparison
          - Investor reporting automation
          - Regulatory compliance documentation
        """
        import csv
        import os
        import datetime as _dt

        report = self._analytics.get_report()
        if not report:
            return

        today_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = "/data/analytics"
        os.makedirs(out_dir, exist_ok=True)

        # Equity curve
        eq_path = f"{out_dir}/equity_curve_{today_str}.csv"
        try:
            snap = self._analytics.snapshot()
            with open(eq_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc", "equity_usdt", "drawdown_pct"])
                # Use snapshot summary for the CSV
                writer.writerow([today_str, f"{snap.equity_current:.4f}", f"{snap.cur_drawdown_pct:.4f}"])
            logger.info("equity snapshot exported: %s", eq_path)
        except Exception as exc:
            logger.warning("equity CSV export failed: %s", exc)

        # Trade history
        trades_path = f"{out_dir}/trade_history_{today_str}.csv"
        try:
            with open(trades_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pnl_usdt"])
                for pnl in self._analytics._trade_pnls:
                    writer.writerow([f"{pnl:.4f}"])
            logger.info("trade history exported: %s (%d trades)", trades_path, len(self._analytics._trade_pnls))
        except Exception as exc:
            logger.warning("trade history CSV export failed: %s", exc)

        # Summary metrics to log
        if snap and snap.n_trades >= 5:
            logger.info("DAILY REPORT | %s", self._analytics.summary_line())

    async def _check_kill_switches(self) -> None:
        """
        Institutional kill switches: halt trading if risk limits are breached.

        Two independent triggers (AIMA crypto hedge fund standards 2025):

        1. Daily Loss Limit: if we lose > max_daily_loss_pct of the day's
           opening equity, halt trading for the rest of the calendar day.
           Prevents runaway losses during adverse market conditions.
           Resets automatically on next daily_reset.

        2. Max Drawdown Halt: if peak-to-trough drawdown exceeds
           max_drawdown_kill_pct, halt trading entirely and require manual
           restart. This protects capital at the portfolio level and
           prevents the bot from trading through a strategy failure.
        """
        if self._state.trading_halted:
            return  # already halted

        cfg = self._config
        equity = self._state.equity
        daily_start = self._state.daily_start_equity

        # ── Trigger 1: Daily loss limit ──────────────────────────────────
        if daily_start > 0:
            daily_loss_pct = (daily_start - equity) / daily_start * 100.0
            if daily_loss_pct >= cfg.max_daily_loss_pct:
                self._state.trading_halted = True
                self._state.halt_reason = (
                    f"daily_loss_limit_breached: -{daily_loss_pct:.2f}% "
                    f"(limit={cfg.max_daily_loss_pct}%)"
                )
                self._telegram.send_message(
                    f"🚨 KILL SWITCH ACTIVADO — Daily Loss Limit\n"
                    f"Pérdida del día: -{daily_loss_pct:.2f}%\n"
                    f"Límite configurado: {cfg.max_daily_loss_pct}%\n"
                    f"Trading suspendido hasta el próximo día UTC.\n"
                    f"Equity actual: ${equity:.2f}"
                )
                logger.critical(
                    "KILL SWITCH: daily loss limit breached %.2f%% (limit=%.1f%%)",
                    daily_loss_pct, cfg.max_daily_loss_pct,
                )
                return

        # ── Trigger 2: Max drawdown from peak ────────────────────────────
        if self._state.peak_equity > 0:
            drawdown_pct = (
                (self._state.peak_equity - equity) / self._state.peak_equity * 100.0
            )
            if drawdown_pct >= cfg.max_drawdown_kill_pct:
                self._state.trading_halted = True
                self._state.halt_reason = (
                    f"max_drawdown_breached: -{drawdown_pct:.2f}% from peak "
                    f"(limit={cfg.max_drawdown_kill_pct}%)"
                )
                self._telegram.send_message(
                    f"🚨 KILL SWITCH ACTIVADO — Max Drawdown\n"
                    f"Drawdown desde el pico: -{drawdown_pct:.2f}%\n"
                    f"Límite configurado: {cfg.max_drawdown_kill_pct}%\n"
                    f"Trading suspendido. Requiere reinicio manual.\n"
                    f"Peak equity: ${self._state.peak_equity:.2f}\n"
                    f"Equity actual: ${equity:.2f}"
                )
                logger.critical(
                    "KILL SWITCH: max drawdown breached %.2f%% from peak $%.2f (limit=%.1f%%)",
                    drawdown_pct, self._state.peak_equity, cfg.max_drawdown_kill_pct,
                )

    async def _manage_open_positions(self, positions: list) -> None:
        """
        Check all open positions for stop loss / take profit.

        Three risk management layers:
        1. Dynamic SL: max(fixed_sl_pct, atr_multiplier * ATR%) — adapts to volatility
        2. Trailing stop: once profit > activation threshold, trail at distance below peak
        3. Post-loss cooldown: after a SL, suppress new entries for N ticks
        """
        if not positions:
            return

        cfg = self._config
        tick = self._state.tick_count

        for pos in positions:
            symbol = pos.get("symbol", "")
            amt = float(pos.get("positionAmt", 0.0))
            if amt == 0.0:
                continue

            entry_price = float(pos.get("entryPrice", 0.0))
            if entry_price <= 0:
                continue

            # Get current price (lightweight single-field call)
            try:
                current_price = await asyncio.to_thread(
                    self._binance.get_current_price, symbol
                )
            except Exception:
                continue

            if current_price <= 0:
                continue

            # Direction and raw PnL %
            is_long = amt > 0
            pnl_pct = (
                (current_price - entry_price) / entry_price
                if is_long
                else (entry_price - current_price) / entry_price
            )

            # ── Layer 1: Dynamic SL/TP (ATR-based + regime-adaptive) ───────
            # Fetch ATR for this symbol from cached features (non-blocking)
            atr_pct = 0.0
            pos_regime = "trending"
            try:
                feats = self._market_data.compute_features(symbol)
                if feats.atr_14 > 0 and feats.close > 0:
                    atr_pct = feats.atr_14 / feats.close
            except Exception:
                pass  # fallback to fixed SL

            # Detectar régimen de la posición para SL/TP adaptativos
            # (usamos el régimen actual del engine)
            current_regime_val = self._state.current_regime
            is_ranging_pos = current_regime_val in ("range_bound", "low_vol")

            if is_ranging_pos:
                # En lateral: SL/TP más ajustados para capturar moves pequeños
                # Ratio 1.5:1 (vs 2:1 en tendencia) — adecuado para rango
                sl_floor = cfg.sl_pct_ranging / 100.0
                tp_pct   = cfg.tp_pct_ranging / 100.0
            else:
                sl_floor = cfg.sl_pct_trending / 100.0
                tp_pct   = cfg.tp_pct_trending / 100.0

            # Effective SL = max(config floor por régimen, ATR multiple)
            sl_pct = max(sl_floor, cfg.atr_sl_multiplier * atr_pct)

            # ── Layer 2: Trailing stop ─────────────────────────────────────
            trail_sl = None
            if cfg.trailing_stop_enabled and pnl_pct > 0:
                # Update peak profit seen for this position
                peak_key = symbol
                prev_peak = self._state.trailing_peaks.get(peak_key, 0.0)
                new_peak = max(prev_peak, pnl_pct)
                self._state.trailing_peaks[peak_key] = new_peak

                # Once profit crosses activation threshold, set trailing SL
                if new_peak >= cfg.trailing_activation_pct / 100.0:
                    trail_sl = new_peak - cfg.trailing_distance_pct / 100.0
                    # Trailing SL is only triggered when PnL falls below it
                    # (trail_sl could be negative if we gave back a lot)

            # ── Determine close reason ─────────────────────────────────────
            close_reason = ""
            if pnl_pct <= -sl_pct:
                close_reason = f"STOP_LOSS ({pnl_pct*100:.2f}% | sl={sl_pct*100:.2f}% atr={atr_pct*100:.3f}%)"
            elif trail_sl is not None and pnl_pct <= trail_sl:
                close_reason = f"TRAILING_STOP ({pnl_pct*100:.2f}% | trail={trail_sl*100:.2f}% peak={self._state.trailing_peaks.get(symbol, 0)*100:.2f}%)"
            elif pnl_pct >= tp_pct:
                close_reason = f"TAKE_PROFIT ({pnl_pct*100:.2f}%)"

            if not close_reason:
                continue

            # Clean up trailing peak for this symbol
            self._state.trailing_peaks.pop(symbol, None)

            # Close position: opposite side, reduceOnly
            close_side = "SELL" if is_long else "BUY"
            close_qty = abs(amt)

            logger.info(
                "closing %s %s qty=%.6f reason=%s entry=%.2f current=%.2f",
                close_side, symbol, close_qty, close_reason, entry_price, current_price,
            )

            close_order = V5OrderRequest(
                symbol=symbol,
                side=close_side,
                quantity=close_qty,
                price=current_price,
                leverage=cfg.leverage,
                reduce_only=True,
            )

            result = await self._execution_engine.place_order(
                close_order, dry_run=cfg.dry_run
            )

            if result.success:
                pnl_usdt = pnl_pct * entry_price * close_qty
                self._position_sizer.record_trade_pnl(pnl_usdt)
                self._state.trade_pnls.append(pnl_usdt)
                # Feed trade to institutional performance analytics
                self._analytics.record_trade(pnl_usdt=pnl_usdt, is_win=(pnl_usdt > 0))
                self._portfolio_risk.record_trade(pnl_pct=pnl_pct)

                # ── Layer 3: Cooldown post-pérdida ────────────────────────
                # If this was a stop loss (not TP or trailing), enforce cooldown
                if "STOP_LOSS" in close_reason:
                    cooldown_expiry = tick + cfg.post_loss_cooldown_ticks
                    self._state.cooldown_until[symbol] = cooldown_expiry
                    logger.info(
                        "%s cooldown activo hasta tick %d (%d ticks)",
                        symbol, cooldown_expiry, cfg.post_loss_cooldown_ticks,
                    )

                self._telegram.notify_position_closed(
                    symbol=symbol,
                    pnl=pnl_usdt,
                    pnl_pct=pnl_pct * 100.0,
                    reason=close_reason,
                    equity=self._state.equity,
                )
                logger.info(
                    "position closed: %s pnl=%.4f (%+.2f%%) reason=%s",
                    symbol, pnl_usdt, pnl_pct * 100.0, close_reason,
                )
            else:
                logger.error(
                    "FAILED to close position %s: %s", symbol, result.error
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

        # 4. Order Flow Intelligence — fetch microstructure data in parallel
        # Provides real-time OBI + trade flow delta + fake breakout detection.
        # This runs as a separate thread to keep latency low.
        prev_close = features.closes[-2] if len(features.closes) >= 2 else 0.0
        order_flow_ctx = await asyncio.to_thread(
            self._order_flow.analyze,
            symbol,
            features.close,
            features.atr_14,
            prev_close,
        )

        if order_flow_ctx.error:
            logger.debug("%s order flow error (continuing): %s", symbol, order_flow_ctx.error)

        # 5. Generate signal (with order flow context)
        signal = self._signal_generator.generate(
            features, regime, btc_features, funding_rate, order_flow_ctx
        )

        # Log signal with order flow context
        if order_flow_ctx and not order_flow_ctx.error:
            logger.debug(
                "%s ofl: obi=%.2f tfd=%.2f combined=%.2f fake=%s",
                symbol,
                order_flow_ctx.obi,
                order_flow_ctx.tfd,
                order_flow_ctx.combined_score,
                order_flow_ctx.is_fake_breakout,
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

        # 6. Check if we already have a position in this symbol, or in cooldown
        for pos in self._state.open_positions:
            if pos.get("symbol") == symbol:
                # Already positioned, skip
                return

        # Cooldown check: don't re-enter too soon after a stop loss
        cooldown_until = self._state.cooldown_until.get(symbol, 0)
        if self._state.tick_count < cooldown_until:
            remaining = cooldown_until - self._state.tick_count
            logger.debug("%s in cooldown — %d ticks remaining", symbol, remaining)
            return

        # 6b. Regime-adaptive guards — crítico para mercados laterales
        cfg = self._config
        is_ranging = regime.regime.value in ("range_bound", "low_vol")

        # GAP 6: ATR mínimo filter — no operar si el mercado está demasiado quieto
        # Si ATR < min_atr_pct, las fees (0.1% round-trip) se comen el profit
        if features.atr_14 > 0 and features.close > 0:
            atr_pct = features.atr_14 / features.close * 100.0
            min_atr = cfg.min_atr_pct_to_trade
            if atr_pct < min_atr:
                logger.debug(
                    "%s ATR_filter: atr=%.3f%% < min=%.3f%% (mercado demasiado quieto)",
                    symbol, atr_pct, min_atr,
                )
                return

        # GAP 1: Confidence threshold más alto en mercado lateral
        effective_threshold = (
            cfg.confidence_threshold_ranging if is_ranging
            else cfg.confidence_threshold_trending
        )
        if signal.confidence < effective_threshold:
            logger.debug(
                "%s confidence_filter: %.3f < %.3f (regime=%s)",
                symbol, signal.confidence, effective_threshold, regime.regime.value,
            )
            return

        # GAP 2: Frequency throttle en lateral
        # Solo 1 trade por símbolo cada N ticks para no desperdiciar fees en ruido
        if is_ranging:
            last_tick = self._state.last_trade_tick.get(symbol, 0)
            ticks_since_last = self._state.tick_count - last_tick
            if ticks_since_last < cfg.ranging_trade_cooldown_ticks:
                logger.debug(
                    "%s ranging_throttle: %d/%d ticks",
                    symbol, ticks_since_last, cfg.ranging_trade_cooldown_ticks,
                )
                return

        # GAP 4: Range context — solo operar en extremos del rango en lateral
        if is_ranging and len(features.closes) >= 10:
            from darwin_agent.v5.market_data import compute_range_context
            rctx = compute_range_context(
                features.closes, features.highs, features.lows, window=48
            )
            if rctx["range_pct"] < 0.02:
                # Rango demasiado estrecho (< 2%) — no hay espacio para el trade
                logger.debug(
                    "%s range_filter: range_pct=%.2f%% muy comprimido",
                    symbol, rctx["range_pct"] * 100,
                )
                return

            # Si el precio no está cerca de los extremos del rango → skip
            # (no entramos en el medio del rango)
            if not rctx["near_support"] and not rctx["near_resistance"]:
                logger.debug(
                    "%s range_filter: price_pos=%.2f (no en extremo del rango)",
                    symbol, rctx["price_position"],
                )
                return

            # Filtro de dirección: solo long en soporte, solo short en resistencia
            if rctx["near_support"] and signal.direction == "SHORT":
                logger.debug(
                    "%s range_dir_filter: en soporte pero señal SHORT — inverting to LONG",
                    symbol,
                )
                # En lateral: cerca del soporte = comprar, no vender
                # Rechazar señal corta en soporte (probable rebote al alza)
                return
            if rctx["near_resistance"] and signal.direction == "LONG":
                logger.debug(
                    "%s range_dir_filter: en resistencia pero señal LONG — corto más probable",
                    symbol,
                )
                return

        # 7. Position sizing — with Half-Kelly + Portfolio Risk checks

        # Update price history for correlation filter
        self._portfolio_risk.update_price(symbol, features.close)

        # Estimate proposed notional for heat check (before exact sizing)
        equity = self._state.equity
        base_risk_pct = self._config.base_risk_pct / 100.0
        estimated_notional = equity * base_risk_pct * self._config.leverage

        # Portfolio risk check: heat, correlation, circuit breaker
        risk_check = self._portfolio_risk.check_new_position(
            symbol=symbol,
            equity=equity,
            price=features.close,
            proposed_notional=estimated_notional,
            open_positions=self._state.open_positions,
        )

        if not risk_check.approved:
            logger.debug(
                "portfolio_risk REJECTED %s: %s",
                symbol, risk_check.rejection_reason,
            )
            return

        # Use Kelly risk % if available, otherwise fallback to config
        kelly_risk_pct = risk_check.kelly_risk_pct
        effective_risk_pct = kelly_risk_pct * risk_check.sizing_multiplier

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
            step_size=self._symbol_step_sizes.get(symbol, 0.0),
            # Pass Kelly-derived risk % override (replaces base_risk_pct for this trade)
            risk_pct_override=effective_risk_pct,
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

        # Log portfolio heat and Kelly info
        logger.debug(
            "%s approved: kelly_risk=%.2f%% cb_level=%d heat=%.1f%% mult=%.2f",
            symbol, kelly_risk_pct, risk_check.circuit_level,
            risk_check.portfolio_heat_pct, risk_check.sizing_multiplier,
        )

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
