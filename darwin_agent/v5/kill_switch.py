"""
Darwin v5 — Enhanced Kill Switch.

Institutional-grade emergency shutdown system that monitors:

1. LATENCY MONITOR
   Measures round-trip time to Binance on every API call.
   If RTT exceeds threshold for N consecutive ticks, halt trading.
   High latency = stale prices = blind trading = guaranteed losses.
   Threshold: 2000ms sustained (Binance orders become unreliable above 1.5s).

2. FLASH CRASH DETECTOR
   Monitors price change per tick across all symbols.
   If any symbol drops/spikes > flash_crash_pct in one tick (5s window),
   trigger emergency mode:
     - Cancel ALL open orders on ALL symbols
     - Close ALL open positions at market
     - Halt trading for cooldown period
   This protects against 2021-style BTC flash crashes (-50% in minutes)
   and memecoin rug-pulls (LUNA, FTT, etc).

3. CONNECTIVITY WATCHDOG
   If we fail to reach Binance API for N consecutive ticks,
   we CANNOT close positions client-side. In this case:
     - Rely entirely on server-side VISA orders
     - Alert via Telegram (if reachable)
     - Halt all new entries until connectivity restored

4. MARGIN SAFETY
   Monitors margin ratio from Binance account data.
   If margin_ratio > 80% → reduce exposure
   If margin_ratio > 90% → emergency close all

All actions are logged and sent via Telegram for manual override capability.

Usage:
    ks = EnhancedKillSwitch(binance_client, telegram, config)
    ks.check_latency(rtt_ms=248)
    ks.check_flash_crash("WIFUSDT", prev_price=0.50, cur_price=0.35)
    if ks.should_halt:
        await ks.emergency_shutdown()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger("darwin.v5.kill_switch")


class KillSwitchTrigger(Enum):
    """What triggered the kill switch."""
    NONE = auto()
    HIGH_LATENCY = auto()
    FLASH_CRASH = auto()
    CONNECTIVITY_LOST = auto()
    MARGIN_CRITICAL = auto()
    MANUAL = auto()


@dataclass
class KillSwitchConfig:
    """Tunable parameters for kill switch."""
    # Latency monitor
    latency_warning_ms: float = 1000.0   # warn above 1s
    latency_critical_ms: float = 2000.0  # halt above 2s
    latency_consecutive_threshold: int = 3  # N consecutive high-latency ticks
    latency_window_size: int = 20        # rolling window for avg latency

    # Flash crash detection
    flash_crash_pct: float = 8.0         # >8% move in one tick = flash crash
    flash_crash_cooldown_s: float = 300.0  # 5 min cooldown after flash crash

    # Connectivity watchdog
    max_consecutive_api_failures: int = 5  # 5 failures = connectivity lost
    connectivity_recovery_threshold: int = 3  # 3 successes = recovered

    # Margin safety
    margin_reduce_threshold: float = 80.0  # reduce exposure above 80%
    margin_emergency_threshold: float = 90.0  # emergency close above 90%

    # Emergency shutdown behavior
    cancel_all_orders_on_trigger: bool = True
    close_all_positions_on_trigger: bool = True
    notify_telegram: bool = True


@dataclass
class KillSwitchState:
    """Current state of all kill switch monitors."""
    # Latency
    last_rtt_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    consecutive_high_latency: int = 0
    latency_halted: bool = False

    # Flash crash
    flash_crash_detected: bool = False
    flash_crash_symbol: str = ""
    flash_crash_pct: float = 0.0
    flash_crash_cooldown_until: float = 0.0

    # Connectivity
    consecutive_api_failures: int = 0
    consecutive_api_successes: int = 0
    connectivity_lost: bool = False
    last_successful_api_ts: float = 0.0

    # Margin
    current_margin_ratio: float = 0.0
    margin_critical: bool = False

    # Overall
    trigger: KillSwitchTrigger = KillSwitchTrigger.NONE
    halted: bool = False
    halt_reason: str = ""
    halt_timestamp: float = 0.0
    emergency_actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "halted": self.halted,
            "trigger": self.trigger.name,
            "halt_reason": self.halt_reason,
            "last_rtt_ms": round(self.last_rtt_ms, 1),
            "avg_rtt_ms": round(self.avg_rtt_ms, 1),
            "consecutive_high_latency": self.consecutive_high_latency,
            "flash_crash_detected": self.flash_crash_detected,
            "flash_crash_symbol": self.flash_crash_symbol,
            "connectivity_lost": self.connectivity_lost,
            "margin_ratio": round(self.current_margin_ratio, 2),
            "emergency_actions": self.emergency_actions_taken,
        }


class EnhancedKillSwitch:
    """
    Multi-factor emergency shutdown system.

    Monitors latency, flash crashes, connectivity, and margin in real-time.
    When triggered, cancels all orders and closes all positions.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client for emergency actions.
    telegram_notifier : TelegramNotifier
        For alerting (may be None in tests).
    config : KillSwitchConfig, optional
        Tunable thresholds.
    """

    def __init__(
        self,
        binance_client: Any,
        telegram_notifier: Any = None,
        config: KillSwitchConfig | None = None,
    ) -> None:
        self._client = binance_client
        self._telegram = telegram_notifier
        self._config = config or KillSwitchConfig()
        self._state = KillSwitchState()
        self._rtt_window: deque = deque(maxlen=self._config.latency_window_size)
        # Price tracking for flash crash detection: symbol → last_price
        self._last_prices: Dict[str, float] = {}

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def should_halt(self) -> bool:
        """Check if trading should be halted."""
        s = self._state
        if s.halted:
            return True
        if s.latency_halted:
            return True
        if s.flash_crash_detected and time.time() < s.flash_crash_cooldown_until:
            return True
        if s.connectivity_lost:
            return True
        if s.margin_critical:
            return True
        return False

    # ── Latency Monitor ────────────────────────────────────────────

    def record_latency(self, rtt_ms: float) -> None:
        """
        Record an API round-trip time measurement.

        Called after every API call to Binance. Tracks rolling average
        and detects sustained high latency.
        """
        cfg = self._config
        self._state.last_rtt_ms = rtt_ms
        self._rtt_window.append(rtt_ms)

        if len(self._rtt_window) > 0:
            self._state.avg_rtt_ms = sum(self._rtt_window) / len(self._rtt_window)

        if rtt_ms > cfg.latency_critical_ms:
            self._state.consecutive_high_latency += 1
            if self._state.consecutive_high_latency >= cfg.latency_consecutive_threshold:
                if not self._state.latency_halted:
                    self._state.latency_halted = True
                    self._trigger_halt(
                        KillSwitchTrigger.HIGH_LATENCY,
                        f"sustained high latency: {rtt_ms:.0f}ms "
                        f"({self._state.consecutive_high_latency} consecutive > "
                        f"{cfg.latency_critical_ms}ms)",
                    )
        elif rtt_ms < cfg.latency_warning_ms:
            # Recovery: latency back to normal
            if self._state.consecutive_high_latency > 0:
                self._state.consecutive_high_latency = max(
                    0, self._state.consecutive_high_latency - 1
                )
            if self._state.latency_halted and self._state.consecutive_high_latency == 0:
                self._state.latency_halted = False
                logger.info("KILL_SWITCH: latency recovered (rtt=%.0fms)", rtt_ms)

    # ── Flash Crash Detector ───────────────────────────────────────

    def check_price_change(self, symbol: str, current_price: float) -> bool:
        """
        Check if price moved dangerously in one tick.

        Returns True if flash crash detected.
        """
        if current_price <= 0:
            return False

        cfg = self._config
        prev = self._last_prices.get(symbol, 0.0)
        self._last_prices[symbol] = current_price

        if prev <= 0:
            return False  # first tick, no comparison

        pct_change = abs(current_price - prev) / prev * 100.0

        if pct_change >= cfg.flash_crash_pct:
            direction = "CRASH" if current_price < prev else "SPIKE"
            self._state.flash_crash_detected = True
            self._state.flash_crash_symbol = symbol
            self._state.flash_crash_pct = pct_change
            self._state.flash_crash_cooldown_until = time.time() + cfg.flash_crash_cooldown_s

            self._trigger_halt(
                KillSwitchTrigger.FLASH_CRASH,
                f"FLASH {direction}: {symbol} moved {pct_change:.1f}% in one tick "
                f"({prev:.6f} → {current_price:.6f})",
            )
            return True

        # Check cooldown expiry
        if (
            self._state.flash_crash_detected
            and time.time() >= self._state.flash_crash_cooldown_until
        ):
            self._state.flash_crash_detected = False
            self._state.flash_crash_symbol = ""
            logger.info("KILL_SWITCH: flash crash cooldown expired, trading resumed")

        return False

    # ── Connectivity Watchdog ──────────────────────────────────────

    def record_api_success(self) -> None:
        """Record a successful API call."""
        self._state.consecutive_api_failures = 0
        self._state.consecutive_api_successes += 1
        self._state.last_successful_api_ts = time.time()

        cfg = self._config
        if (
            self._state.connectivity_lost
            and self._state.consecutive_api_successes >= cfg.connectivity_recovery_threshold
        ):
            self._state.connectivity_lost = False
            logger.info("KILL_SWITCH: connectivity restored after %d successes",
                        self._state.consecutive_api_successes)

    def record_api_failure(self) -> None:
        """Record a failed API call."""
        self._state.consecutive_api_failures += 1
        self._state.consecutive_api_successes = 0

        cfg = self._config
        if self._state.consecutive_api_failures >= cfg.max_consecutive_api_failures:
            if not self._state.connectivity_lost:
                self._state.connectivity_lost = True
                self._trigger_halt(
                    KillSwitchTrigger.CONNECTIVITY_LOST,
                    f"connectivity lost: {self._state.consecutive_api_failures} "
                    f"consecutive API failures",
                )

    # ── Margin Safety ──────────────────────────────────────────────

    def check_margin_ratio(self, margin_ratio_pct: float) -> bool:
        """
        Check if margin ratio is dangerous.

        Returns True if emergency action needed.
        """
        cfg = self._config
        self._state.current_margin_ratio = margin_ratio_pct

        if margin_ratio_pct >= cfg.margin_emergency_threshold:
            self._state.margin_critical = True
            self._trigger_halt(
                KillSwitchTrigger.MARGIN_CRITICAL,
                f"MARGIN CRITICAL: {margin_ratio_pct:.1f}% "
                f"(threshold={cfg.margin_emergency_threshold}%)",
            )
            return True

        if margin_ratio_pct >= cfg.margin_reduce_threshold:
            logger.warning(
                "KILL_SWITCH: margin warning %.1f%% (reduce threshold=%.1f%%)",
                margin_ratio_pct, cfg.margin_reduce_threshold,
            )

        # Recovery
        if self._state.margin_critical and margin_ratio_pct < cfg.margin_reduce_threshold:
            self._state.margin_critical = False
            logger.info("KILL_SWITCH: margin ratio recovered to %.1f%%", margin_ratio_pct)

        return False

    # ── Emergency Shutdown ─────────────────────────────────────────

    async def emergency_shutdown(
        self,
        symbols: List[str],
        positions: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Execute emergency shutdown: cancel all orders, close all positions.

        Returns list of actions taken.

        This is the nuclear option. It:
        1. Cancels ALL open orders on ALL symbols (prevents new fills)
        2. Closes ALL open positions at MARKET (immediate exit)
        3. Sends Telegram alert
        """
        cfg = self._config
        actions = []

        # Step 1: Cancel all orders on all symbols
        if cfg.cancel_all_orders_on_trigger:
            for symbol in symbols:
                try:
                    await asyncio.to_thread(
                        self._client.cancel_all_orders, symbol
                    )
                    actions.append(f"cancelled_all_orders:{symbol}")
                    logger.warning("EMERGENCY: cancelled all orders for %s", symbol)
                except Exception as exc:
                    actions.append(f"cancel_failed:{symbol}:{exc}")
                    logger.error("EMERGENCY: cancel orders failed for %s: %s", symbol, exc)

        # Step 2: Close all open positions
        if cfg.close_all_positions_on_trigger:
            for pos in positions:
                symbol = pos.get("symbol", "")
                amt = float(pos.get("positionAmt", 0.0))
                if amt == 0.0 or not symbol:
                    continue

                close_side = "SELL" if amt > 0 else "BUY"
                close_qty = abs(amt)

                try:
                    await asyncio.to_thread(
                        self._client.emergency_close_position,
                        symbol, close_side, close_qty,
                    )
                    actions.append(f"emergency_close:{symbol}:{close_side}:{close_qty}")
                    logger.warning(
                        "EMERGENCY: closed %s %s qty=%.6f", symbol, close_side, close_qty
                    )
                except Exception as exc:
                    actions.append(f"close_failed:{symbol}:{exc}")
                    logger.critical(
                        "EMERGENCY: CLOSE FAILED for %s — server-side SL is last defense: %s",
                        symbol, exc,
                    )

        # Step 3: Notify
        if cfg.notify_telegram and self._telegram:
            try:
                msg = (
                    f"EMERGENCY SHUTDOWN\n"
                    f"Trigger: {self._state.trigger.name}\n"
                    f"Reason: {self._state.halt_reason}\n"
                    f"Actions: {len(actions)}\n"
                    + "\n".join(f"  - {a}" for a in actions[:10])
                )
                self._telegram.send_message(msg)
            except Exception:
                pass  # telegram failure during emergency is non-critical

        self._state.emergency_actions_taken = actions
        return actions

    def reset(self) -> None:
        """Manual reset after emergency. Requires operator intervention."""
        self._state = KillSwitchState()
        self._rtt_window.clear()
        logger.info("KILL_SWITCH: manual reset performed")

    # ── Internal ───────────────────────────────────────────────────

    def _trigger_halt(self, trigger: KillSwitchTrigger, reason: str) -> None:
        """Set halt state and log/notify."""
        self._state.halted = True
        self._state.trigger = trigger
        self._state.halt_reason = reason
        self._state.halt_timestamp = time.time()

        logger.critical("KILL_SWITCH TRIGGERED: [%s] %s", trigger.name, reason)

        if self._config.notify_telegram and self._telegram:
            try:
                self._telegram.send_message(
                    f"KILL SWITCH: {trigger.name}\n{reason}"
                )
            except Exception:
                pass  # can't fail here

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return kill switch diagnostics."""
        return self._state.to_dict()
