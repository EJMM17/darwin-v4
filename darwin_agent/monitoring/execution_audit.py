"""
Darwin v4 — Execution Audit Engine.

Pure observability layer. Does NOT modify strategy, genome, or portfolio logic.

Tracks:
    - Order intent vs actual execution (slippage, partial fills, rejects)
    - Fill quality metrics (slippage distribution, latency, fill rates)
    - Real-time risk drift (sim vs actual position, margin, funding, liquidation)
    - Circuit breaker execution validation (orphan position detection)
    - Prometheus-style counters for external monitoring

Usage:
    audit = ExecutionAudit(log_dir="logs/audit")

    # Before placing order:
    intent = audit.record_intent(symbol, side, qty, price, leverage, signal, bar)

    # After order fills:
    audit.record_fill(intent, filled_qty, avg_price, fee, order_id, latency_ms)

    # Each bar:
    audit.check_exposure_drift(symbol, sim_size, actual_size, mark, liq, margin, funding, bar)
    audit.check_risk_state(equity, peak, dd, rbe, gmrt, lev, cap, positions, bar)

    # On CB trigger:
    audit.validate_circuit_breaker(bar, eq_before, eq_after, loss, closed, actual_positions)

    # Periodic:
    metrics = audit.get_metrics()
    audit.print_summary()
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from darwin_agent.monitoring.audit_logger import AuditLogger


# ═══════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════

@dataclass
class OrderIntent:
    """Snapshot of what the system intended to execute."""
    intent_id: int
    timestamp_ns: int
    symbol: str
    side: str           # "BUY" or "SELL"
    intended_qty: float
    intended_price: float
    leverage: float
    signal_strength: float
    bar_index: int
    filled: bool = False


@dataclass
class FillRecord:
    """What actually happened on the exchange."""
    intent_id: int
    filled_qty: float
    avg_fill_price: float
    fee: float
    order_id: str
    latency_ms: float
    slippage_bps: float
    slippage_pct: float
    partial: bool
    rejected: bool


@dataclass
class DriftSnapshot:
    """Point-in-time exposure drift measurement."""
    symbol: str
    bar_index: int
    simulated_size: float
    actual_size: float
    drift_pct: float
    mark_price: float
    liquidation_price: float
    liq_distance_pct: float
    margin_ratio: float
    funding_rate: float


# ═══════════════════════════════════════════════════════
# ALERT THRESHOLDS
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class AlertThresholds:
    """Configurable alert thresholds. All defaults per spec."""
    max_slippage_pct: float = 0.5       # alert if slippage > 0.5%
    max_fill_latency_ms: float = 3000.0 # alert if latency > 3s
    max_reject_rate_pct: float = 2.0    # alert if reject rate > 2%
    max_leverage_overshoot: float = 0.0 # alert if real lev > configured cap
    min_liq_distance_pct: float = 15.0  # alert if liq price within 15%
    max_drift_pct: float = 10.0         # alert if exposure drift > 10%
    max_funding_rate: float = 0.001     # alert if funding > 0.1% per interval


# ═══════════════════════════════════════════════════════
# RUNNING STATISTICS (no numpy dependency)
# ═══════════════════════════════════════════════════════

class RunningStats:
    """Online mean/variance/percentile tracking without numpy."""

    __slots__ = ("_values", "_max_size")

    def __init__(self, max_size: int = 10000):
        self._values: Deque[float] = deque(maxlen=max_size)
        self._max_size = max_size

    def add(self, v: float) -> None:
        self._values.append(v)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        var = sum((v - m) ** 2 for v in self._values) / (len(self._values) - 1)
        return math.sqrt(var)

    def percentile(self, p: float) -> float:
        """Compute percentile (0-100). Linear interpolation."""
        if not self._values:
            return 0.0
        s = sorted(self._values)
        k = (p / 100.0) * (len(s) - 1)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        d = k - f
        return s[f] + d * (s[c] - s[f])

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def max_val(self) -> float:
        return max(self._values) if self._values else 0.0

    @property
    def min_val(self) -> float:
        return min(self._values) if self._values else 0.0


# ═══════════════════════════════════════════════════════
# EXECUTION AUDIT ENGINE
# ═══════════════════════════════════════════════════════

class ExecutionAudit:
    """
    Pure observability layer for live trading execution.

    Tracks order flow, fill quality, risk drift, and CB validation.
    Fires alerts when thresholds are breached.
    Exposes Prometheus-style metrics.

    Does NOT modify any trading logic.
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        thresholds: AlertThresholds = AlertThresholds(),
    ):
        self._logger = AuditLogger(log_dir=log_dir)
        self._thresholds = thresholds

        # Counters
        self._intent_counter: int = 0
        self._total_intents: int = 0
        self._total_fills: int = 0
        self._total_partial: int = 0
        self._total_rejected: int = 0
        self._total_alerts: int = 0
        self._cb_triggers: int = 0
        self._cb_orphans: int = 0

        # Running stats
        self._slippage_stats = RunningStats()
        self._latency_stats = RunningStats()
        self._drift_stats: Dict[str, RunningStats] = {}
        self._leverage_stats = RunningStats()

        # Recent records for inspection
        self._recent_intents: Deque[OrderIntent] = deque(maxlen=1000)
        self._recent_fills: Deque[FillRecord] = deque(maxlen=1000)
        self._recent_drifts: Deque[DriftSnapshot] = deque(maxlen=500)
        self._alerts: Deque[Dict] = deque(maxlen=500)

    # ─────────────────────────────────────────────────────
    # 1. Order Intent vs Execution Tracking
    # ─────────────────────────────────────────────────────

    def record_intent(
        self,
        symbol: str,
        side: str,
        intended_qty: float,
        intended_price: float,
        leverage: float,
        signal_strength: float,
        bar_index: int,
    ) -> OrderIntent:
        """
        Record what the system intends to execute.
        Call BEFORE placing the order on the exchange.
        Returns an OrderIntent to pass to record_fill().
        """
        self._intent_counter += 1
        self._total_intents += 1

        intent = OrderIntent(
            intent_id=self._intent_counter,
            timestamp_ns=time.time_ns(),
            symbol=symbol,
            side=side,
            intended_qty=intended_qty,
            intended_price=intended_price,
            leverage=leverage,
            signal_strength=signal_strength,
            bar_index=bar_index,
        )
        self._recent_intents.append(intent)

        self._logger.log_order_intent(
            symbol=symbol,
            side=side,
            intended_qty=intended_qty,
            intended_price=intended_price,
            leverage=leverage,
            signal_strength=signal_strength,
            bar_index=bar_index,
        )
        return intent

    def record_fill(
        self,
        intent: OrderIntent,
        filled_qty: float,
        avg_fill_price: float,
        fee: float,
        order_id: str,
        latency_ms: float,
    ) -> FillRecord:
        """
        Record what actually happened on the exchange.
        Call AFTER the order is filled (or rejected).
        """
        intent.filled = True
        rejected = filled_qty <= 0.0
        partial = (
            not rejected
            and filled_qty < intent.intended_qty * 0.999
        )

        slippage_pct = 0.0
        slippage_bps = 0.0
        if intent.intended_price > 0 and not rejected:
            slippage_pct = (
                abs(avg_fill_price - intent.intended_price)
                / intent.intended_price * 100.0
            )
            slippage_bps = slippage_pct * 100.0

        rec = FillRecord(
            intent_id=intent.intent_id,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            fee=fee,
            order_id=order_id,
            latency_ms=latency_ms,
            slippage_bps=slippage_bps,
            slippage_pct=slippage_pct,
            partial=partial,
            rejected=rejected,
        )

        self._recent_fills.append(rec)
        self._total_fills += 1
        if partial:
            self._total_partial += 1
        if rejected:
            self._total_rejected += 1

        # Update running stats
        if not rejected:
            self._slippage_stats.add(slippage_bps)
            self._latency_stats.add(latency_ms)

        # Log
        self._logger.log_fill(
            symbol=intent.symbol,
            side=intent.side,
            intended_qty=intent.intended_qty,
            filled_qty=filled_qty,
            intended_price=intent.intended_price,
            avg_fill_price=avg_fill_price,
            fee=fee,
            order_id=order_id,
            latency_ms=latency_ms,
            partial=partial,
            rejected=rejected,
        )

        # Check alerts
        self._check_fill_alerts(intent, rec)

        return rec

    def _check_fill_alerts(self, intent: OrderIntent, fill: FillRecord) -> None:
        th = self._thresholds

        if fill.slippage_pct > th.max_slippage_pct:
            self._fire_alert("HIGH_SLIPPAGE", {
                "symbol": intent.symbol,
                "slippage_pct": fill.slippage_pct,
                "slippage_bps": fill.slippage_bps,
                "threshold_pct": th.max_slippage_pct,
                "intended_price": intent.intended_price,
                "fill_price": fill.avg_fill_price,
                "order_id": fill.order_id,
            })

        if fill.latency_ms > th.max_fill_latency_ms:
            self._fire_alert("HIGH_LATENCY", {
                "symbol": intent.symbol,
                "latency_ms": fill.latency_ms,
                "threshold_ms": th.max_fill_latency_ms,
                "order_id": fill.order_id,
            })

        if fill.rejected:
            reject_rate = self.reject_rate_pct
            if reject_rate > th.max_reject_rate_pct:
                self._fire_alert("HIGH_REJECT_RATE", {
                    "reject_rate_pct": reject_rate,
                    "threshold_pct": th.max_reject_rate_pct,
                    "total_intents": self._total_intents,
                    "total_rejected": self._total_rejected,
                })

    # ─────────────────────────────────────────────────────
    # 3. Real-Time Risk Drift Monitoring
    # ─────────────────────────────────────────────────────

    def check_exposure_drift(
        self,
        symbol: str,
        simulated_size: float,
        actual_size: float,
        mark_price: float,
        liquidation_price: float,
        margin_ratio: float,
        funding_rate: float,
        bar_index: int,
    ) -> Optional[DriftSnapshot]:
        """
        Compare simulated exposure vs actual position.
        Call each bar for every active symbol.
        """
        drift_pct = 0.0
        if abs(simulated_size) > 0.0001:
            drift_pct = (actual_size - simulated_size) / abs(simulated_size) * 100.0
        elif abs(actual_size) > 0.0001:
            drift_pct = 100.0  # have position when sim says flat

        liq_distance_pct = 100.0
        if liquidation_price > 0 and mark_price > 0:
            liq_distance_pct = (
                abs(mark_price - liquidation_price) / mark_price * 100.0
            )

        snap = DriftSnapshot(
            symbol=symbol,
            bar_index=bar_index,
            simulated_size=simulated_size,
            actual_size=actual_size,
            drift_pct=drift_pct,
            mark_price=mark_price,
            liquidation_price=liquidation_price,
            liq_distance_pct=liq_distance_pct,
            margin_ratio=margin_ratio,
            funding_rate=funding_rate,
        )

        self._recent_drifts.append(snap)

        if symbol not in self._drift_stats:
            self._drift_stats[symbol] = RunningStats()
        self._drift_stats[symbol].add(abs(drift_pct))

        # Log
        self._logger.log_exposure_drift(
            symbol=symbol,
            simulated_size=simulated_size,
            actual_size=actual_size,
            drift_pct=drift_pct,
            mark_price=mark_price,
            liquidation_price=liquidation_price,
            margin_ratio=margin_ratio,
            funding_rate=funding_rate,
            bar_index=bar_index,
        )

        # Alerts
        th = self._thresholds
        if liq_distance_pct < th.min_liq_distance_pct:
            self._fire_alert("LIQUIDATION_PROXIMITY", {
                "symbol": symbol,
                "mark_price": mark_price,
                "liquidation_price": liquidation_price,
                "distance_pct": liq_distance_pct,
                "threshold_pct": th.min_liq_distance_pct,
                "bar": bar_index,
            })

        if abs(drift_pct) > th.max_drift_pct:
            self._fire_alert("EXPOSURE_DRIFT", {
                "symbol": symbol,
                "simulated": simulated_size,
                "actual": actual_size,
                "drift_pct": drift_pct,
                "threshold_pct": th.max_drift_pct,
                "bar": bar_index,
            })

        if abs(funding_rate) > th.max_funding_rate:
            self._fire_alert("HIGH_FUNDING_RATE", {
                "symbol": symbol,
                "funding_rate": funding_rate,
                "threshold": th.max_funding_rate,
                "bar": bar_index,
            })

        return snap

    def check_risk_state(
        self,
        equity: float,
        peak: float,
        drawdown_pct: float,
        rbe_mult: float,
        gmrt_mult: float,
        effective_leverage: float,
        configured_lev_cap: float,
        positions: Dict[str, float],
        bar_index: int,
    ) -> None:
        """
        Log current risk state and check for leverage overshoot.
        Call each bar.
        """
        self._leverage_stats.add(effective_leverage)

        self._logger.log_risk_state(
            equity=equity,
            peak=peak,
            drawdown_pct=drawdown_pct,
            rbe_mult=rbe_mult,
            gmrt_mult=gmrt_mult,
            effective_leverage=effective_leverage,
            configured_lev_cap=configured_lev_cap,
            positions=positions,
            bar_index=bar_index,
        )

        if effective_leverage > configured_lev_cap:
            self._fire_alert("LEVERAGE_OVERSHOOT", {
                "effective": effective_leverage,
                "cap": configured_lev_cap,
                "overshoot_pct": (effective_leverage - configured_lev_cap)
                    / configured_lev_cap * 100.0,
                "bar": bar_index,
            })

    # ─────────────────────────────────────────────────────
    # 4. Circuit Breaker Validation
    # ─────────────────────────────────────────────────────

    def validate_circuit_breaker(
        self,
        trigger_bar: int,
        equity_before: float,
        equity_after: float,
        loss_pct: float,
        positions_closed: int,
        actual_open_positions: List[str],
        cooldown_bars: int,
    ) -> bool:
        """
        Validate that CB actually closed all positions.
        Returns True if clean (no orphans), False if orphan positions found.
        """
        self._cb_triggers += 1
        orphan_count = len(actual_open_positions)

        if orphan_count > 0:
            self._cb_orphans += orphan_count

        self._logger.log_circuit_breaker(
            trigger_bar=trigger_bar,
            equity_before=equity_before,
            equity_after=equity_after,
            loss_pct=loss_pct,
            positions_closed=positions_closed,
            orphan_positions=orphan_count,
            cooldown_bars=cooldown_bars,
        )

        if orphan_count > 0:
            self._fire_alert("CB_ORPHAN_POSITIONS", {
                "trigger_bar": trigger_bar,
                "positions_closed": positions_closed,
                "orphan_symbols": actual_open_positions,
                "orphan_count": orphan_count,
            })
            return False

        return True

    # ─────────────────────────────────────────────────────
    # 6. Alerting
    # ─────────────────────────────────────────────────────

    def _fire_alert(self, alert_type: str, details: Dict) -> None:
        self._total_alerts += 1
        alert_record = {
            "alert_type": alert_type,
            "details": details,
            "alert_number": self._total_alerts,
        }
        self._alerts.append(alert_record)
        self._logger.log_alert(alert_type, details)

    # ─────────────────────────────────────────────────────
    # 2. Fill Quality Metrics (Prometheus-style)
    # ─────────────────────────────────────────────────────

    @property
    def reject_rate_pct(self) -> float:
        if self._total_intents == 0:
            return 0.0
        return self._total_rejected / self._total_intents * 100.0

    @property
    def partial_fill_rate_pct(self) -> float:
        if self._total_fills == 0:
            return 0.0
        return self._total_partial / self._total_fills * 100.0

    def get_metrics(self) -> Dict[str, float]:
        """
        Return Prometheus-style flat metrics dict.
        All metric names follow the convention:
            darwin_<subsystem>_<metric>_<unit>
        """
        m: Dict[str, float] = {}

        # ── Order flow ──
        m["darwin_orders_intents_total"] = float(self._total_intents)
        m["darwin_orders_fills_total"] = float(self._total_fills)
        m["darwin_orders_partial_total"] = float(self._total_partial)
        m["darwin_orders_rejected_total"] = float(self._total_rejected)
        m["darwin_orders_reject_rate_pct"] = self.reject_rate_pct
        m["darwin_orders_partial_rate_pct"] = self.partial_fill_rate_pct

        # ── Slippage ──
        m["darwin_slippage_mean_bps"] = self._slippage_stats.mean
        m["darwin_slippage_std_bps"] = self._slippage_stats.std
        m["darwin_slippage_p95_bps"] = self._slippage_stats.p95
        m["darwin_slippage_p99_bps"] = self._slippage_stats.p99
        m["darwin_slippage_max_bps"] = self._slippage_stats.max_val
        m["darwin_slippage_count"] = float(self._slippage_stats.count)

        # ── Latency ──
        m["darwin_latency_mean_ms"] = self._latency_stats.mean
        m["darwin_latency_std_ms"] = self._latency_stats.std
        m["darwin_latency_p95_ms"] = self._latency_stats.p95
        m["darwin_latency_p99_ms"] = self._latency_stats.p99
        m["darwin_latency_max_ms"] = self._latency_stats.max_val

        # ── Leverage ──
        m["darwin_leverage_mean"] = self._leverage_stats.mean
        m["darwin_leverage_max"] = self._leverage_stats.max_val
        m["darwin_leverage_p95"] = self._leverage_stats.p95

        # ── Drift per symbol ──
        for sym in sorted(self._drift_stats.keys()):
            ds = self._drift_stats[sym]
            m["darwin_drift_{}_mean_pct".format(sym.lower())] = ds.mean
            m["darwin_drift_{}_max_pct".format(sym.lower())] = ds.max_val
            m["darwin_drift_{}_p95_pct".format(sym.lower())] = ds.p95

        # ── Circuit breaker ──
        m["darwin_cb_triggers_total"] = float(self._cb_triggers)
        m["darwin_cb_orphans_total"] = float(self._cb_orphans)

        # ── Alerts ──
        m["darwin_alerts_total"] = float(self._total_alerts)

        return m

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        return list(self._alerts)[-n:]

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────

    def print_summary(self) -> str:
        """Print human-readable execution audit summary."""
        m = self.get_metrics()
        lines = []
        lines.append("=" * 65)
        lines.append("  EXECUTION AUDIT SUMMARY")
        lines.append("=" * 65)

        lines.append("\n  Order Flow:")
        lines.append("    Intents:     {}".format(int(m["darwin_orders_intents_total"])))
        lines.append("    Fills:       {}".format(int(m["darwin_orders_fills_total"])))
        lines.append("    Partial:     {} ({:.1f}%)".format(
            int(m["darwin_orders_partial_total"]),
            m["darwin_orders_partial_rate_pct"]))
        lines.append("    Rejected:    {} ({:.1f}%)".format(
            int(m["darwin_orders_rejected_total"]),
            m["darwin_orders_reject_rate_pct"]))

        lines.append("\n  Slippage (bps):")
        lines.append("    Mean: {:.1f}  Std: {:.1f}  P95: {:.1f}  P99: {:.1f}  Max: {:.1f}".format(
            m["darwin_slippage_mean_bps"], m["darwin_slippage_std_bps"],
            m["darwin_slippage_p95_bps"], m["darwin_slippage_p99_bps"],
            m["darwin_slippage_max_bps"]))

        lines.append("\n  Latency (ms):")
        lines.append("    Mean: {:.0f}  P95: {:.0f}  P99: {:.0f}  Max: {:.0f}".format(
            m["darwin_latency_mean_ms"], m["darwin_latency_p95_ms"],
            m["darwin_latency_p99_ms"], m["darwin_latency_max_ms"]))

        lines.append("\n  Leverage:")
        lines.append("    Mean: {:.2f}x  Max: {:.2f}x  P95: {:.2f}x".format(
            m["darwin_leverage_mean"], m["darwin_leverage_max"],
            m["darwin_leverage_p95"]))

        lines.append("\n  Circuit Breaker:")
        lines.append("    Triggers: {}  Orphans: {}".format(
            int(m["darwin_cb_triggers_total"]),
            int(m["darwin_cb_orphans_total"])))

        lines.append("\n  Alerts: {}".format(int(m["darwin_alerts_total"])))
        for a in list(self._alerts)[-5:]:
            lines.append("    {} {}".format(
                a["alert_type"],
                json.dumps(a["details"], separators=(",", ":"))[:80]
                if "details" in a else ""
            ))

        lines.append("=" * 65)
        out = "\n".join(lines)
        print(out)
        return out

    def close(self) -> None:
        self._logger.close()


# Avoid circular import — json only needed for summary display
import json
