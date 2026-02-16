"""
Tests for Execution Audit & Monitoring Layer.

Covers all 7 requirement areas:
    1. Order intent vs execution tracking
    2. Fill quality metrics
    3. Risk drift monitoring
    4. Circuit breaker validation
    5. Logging (structured JSON)
    6. Alerting hooks
    7. Prometheus-style metrics
"""

import json
import os
import tempfile
import pytest

from darwin_agent.monitoring.execution_audit import (
    ExecutionAudit,
    AlertThresholds,
    RunningStats,
)
from darwin_agent.monitoring.audit_logger import AuditLogger, LogLevel


# ═══ RunningStats ═══

class TestRunningStats:
    def test_empty(self):
        rs = RunningStats()
        assert rs.count == 0
        assert rs.mean == 0.0
        assert rs.std == 0.0
        assert rs.p95 == 0.0

    def test_single_value(self):
        rs = RunningStats()
        rs.add(5.0)
        assert rs.mean == 5.0
        assert rs.count == 1

    def test_known_distribution(self):
        rs = RunningStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.add(v)
        assert rs.mean == 3.0
        assert rs.count == 5
        assert abs(rs.percentile(50) - 3.0) < 0.01
        assert rs.min_val == 1.0
        assert rs.max_val == 5.0


# ═══ AuditLogger ═══

class TestAuditLogger:
    def test_creates_jsonl_file(self):
        with tempfile.TemporaryDirectory() as td:
            logger = AuditLogger(log_dir=td)
            logger.log("TEST_EVENT", LogLevel.INFO, {"key": "value"})
            logger.close()

            files = os.listdir(td)
            assert len(files) == 1
            assert files[0].startswith("audit_")
            assert files[0].endswith(".jsonl")

            with open(os.path.join(td, files[0])) as f:
                line = f.readline()
                record = json.loads(line)
                assert record["event"] == "TEST_EVENT"
                assert record["level"] == "INFO"
                assert record["data"]["key"] == "value"
                assert "ts" in record


# ═══ 1. Order Intent vs Execution ═══

class TestOrderIntentTracking:
    def _make_audit(self):
        td = tempfile.mkdtemp()
        return ExecutionAudit(log_dir=td), td

    def test_record_intent(self):
        audit, _ = self._make_audit()
        intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 65000.0, 3.0, 0.8, 100)
        assert intent.symbol == "BTCUSDT"
        assert intent.intended_qty == 0.1
        assert intent.intent_id == 1
        assert not intent.filled
        audit.close()

    def test_record_fill(self):
        audit, _ = self._make_audit()
        intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 65000.0, 3.0, 0.8, 100)
        fill = audit.record_fill(intent, 0.1, 65010.0, 0.05, "ord-1", 150.0)
        assert intent.filled
        assert fill.slippage_bps > 0
        assert not fill.partial
        assert not fill.rejected
        audit.close()

    def test_detect_partial_fill(self):
        audit, _ = self._make_audit()
        intent = audit.record_intent("BTCUSDT", "BUY", 1.0, 65000.0, 3.0, 0.8, 100)
        fill = audit.record_fill(intent, 0.7, 65000.0, 0.03, "ord-2", 200.0)
        assert fill.partial
        assert not fill.rejected
        assert audit._total_partial == 1
        audit.close()

    def test_detect_rejected_order(self):
        audit, _ = self._make_audit()
        intent = audit.record_intent("BTCUSDT", "BUY", 1.0, 65000.0, 3.0, 0.8, 100)
        fill = audit.record_fill(intent, 0.0, 0.0, 0.0, "", 50.0)
        assert fill.rejected
        assert not fill.partial
        assert audit._total_rejected == 1
        audit.close()

    def test_slippage_calculation(self):
        audit, _ = self._make_audit()
        intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, 100)
        fill = audit.record_fill(intent, 0.1, 50100.0, 0.05, "ord-3", 100.0)
        assert abs(fill.slippage_pct - 0.2) < 0.001  # 100/50000 = 0.2%
        assert abs(fill.slippage_bps - 20.0) < 0.1    # 0.2% = 20bps
        audit.close()


# ═══ 2. Fill Quality Metrics ═══

class TestFillQualityMetrics:
    def test_slippage_distribution(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        for i in range(100):
            intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, i)
            slip = 50000.0 + (i % 10) * 5  # 0-45 dollar slippage
            audit.record_fill(intent, 0.1, slip, 0.05, "ord-{}".format(i), 100.0 + i)

        m = audit.get_metrics()
        assert m["darwin_slippage_count"] == 100
        assert m["darwin_slippage_mean_bps"] > 0
        assert m["darwin_slippage_p95_bps"] >= m["darwin_slippage_mean_bps"]
        assert m["darwin_slippage_p99_bps"] >= m["darwin_slippage_p95_bps"]
        assert m["darwin_latency_mean_ms"] > 100
        audit.close()

    def test_reject_rate(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        for i in range(50):
            intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, i)
            if i < 2:
                audit.record_fill(intent, 0.0, 0.0, 0.0, "", 10.0)
            else:
                audit.record_fill(intent, 0.1, 50000.0, 0.05, "o-{}".format(i), 50.0)
        assert abs(audit.reject_rate_pct - 4.0) < 0.01  # 2/50 = 4%
        audit.close()


# ═══ 3. Risk Drift Monitoring ═══

class TestRiskDriftMonitoring:
    def test_exposure_drift_detection(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        snap = audit.check_exposure_drift(
            "BTCUSDT", 0.5, 0.55, 65000.0, 58000.0, 0.15, 0.0001, 100,
        )
        assert abs(snap.drift_pct - 10.0) < 0.01  # (0.55-0.5)/0.5 = 10%
        assert snap.liq_distance_pct > 0
        audit.close()

    def test_liquidation_proximity_alert(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        # Liq price within 10% of mark = should alert (threshold 15%)
        audit.check_exposure_drift(
            "BTCUSDT", 0.5, 0.5, 65000.0, 60000.0, 0.15, 0.0001, 100,
        )
        alerts = audit.get_recent_alerts()
        liq_alerts = [a for a in alerts if a["alert_type"] == "LIQUIDATION_PROXIMITY"]
        assert len(liq_alerts) == 1
        audit.close()

    def test_funding_rate_alert(self):
        audit = ExecutionAudit(
            log_dir=tempfile.mkdtemp(),
            thresholds=AlertThresholds(max_funding_rate=0.001),
        )
        audit.check_exposure_drift(
            "BTCUSDT", 0.5, 0.5, 65000.0, 50000.0, 0.15, 0.005, 100,
        )
        alerts = [a for a in audit.get_recent_alerts()
                  if a["alert_type"] == "HIGH_FUNDING_RATE"]
        assert len(alerts) == 1
        audit.close()


# ═══ 4. Circuit Breaker Validation ═══

class TestCircuitBreakerValidation:
    def test_clean_cb(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        clean = audit.validate_circuit_breaker(
            trigger_bar=100, equity_before=1000.0, equity_after=850.0,
            loss_pct=15.0, positions_closed=3, actual_open_positions=[],
            cooldown_bars=6,
        )
        assert clean
        assert audit._cb_triggers == 1
        assert audit._cb_orphans == 0
        audit.close()

    def test_orphan_detection(self):
        audit, _ = TestOrderIntentTracking._make_audit(None)
        clean = audit.validate_circuit_breaker(
            trigger_bar=100, equity_before=1000.0, equity_after=850.0,
            loss_pct=15.0, positions_closed=2,
            actual_open_positions=["SOLUSDT"],
            cooldown_bars=6,
        )
        assert not clean
        assert audit._cb_orphans == 1
        alerts = [a for a in audit.get_recent_alerts()
                  if a["alert_type"] == "CB_ORPHAN_POSITIONS"]
        assert len(alerts) == 1
        audit.close()


# ═══ 5. JSON Log Structure ═══

class TestLogStructure:
    def test_all_events_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            audit = ExecutionAudit(log_dir=td)

            intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 65000.0, 3.0, 0.8, 1)
            audit.record_fill(intent, 0.1, 65010.0, 0.05, "ord-1", 150.0)
            audit.check_risk_state(800.0, 800.0, 0.0, 1.0, 0.9, 3.5, 5.0,
                                   {"BTC": 0.1}, 1)
            audit.check_exposure_drift("BTC", 0.1, 0.1, 65000.0, 55000.0,
                                       0.10, 0.0001, 1)
            audit.validate_circuit_breaker(1, 800.0, 700.0, 12.5, 3, [], 6)
            audit.close()

            files = [f for f in os.listdir(td) if f.endswith(".jsonl")]
            assert len(files) == 1

            with open(os.path.join(td, files[0])) as f:
                for line_num, line in enumerate(f, 1):
                    record = json.loads(line)
                    assert "ts" in record
                    assert "event" in record
                    assert "level" in record


# ═══ 6. Alerting Hooks ═══

class TestAlertingHooks:
    def test_high_slippage_alert(self):
        audit = ExecutionAudit(
            log_dir=tempfile.mkdtemp(),
            thresholds=AlertThresholds(max_slippage_pct=0.5),
        )
        intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, 1)
        # 1% slippage = 500 dollar on 50k
        audit.record_fill(intent, 0.1, 50500.0, 0.05, "ord-1", 100.0)
        alerts = [a for a in audit.get_recent_alerts()
                  if a["alert_type"] == "HIGH_SLIPPAGE"]
        assert len(alerts) == 1
        audit.close()

    def test_high_latency_alert(self):
        audit = ExecutionAudit(
            log_dir=tempfile.mkdtemp(),
            thresholds=AlertThresholds(max_fill_latency_ms=3000.0),
        )
        intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, 1)
        audit.record_fill(intent, 0.1, 50000.0, 0.05, "ord-1", 5000.0)
        alerts = [a for a in audit.get_recent_alerts()
                  if a["alert_type"] == "HIGH_LATENCY"]
        assert len(alerts) == 1
        audit.close()

    def test_leverage_overshoot_alert(self):
        audit = ExecutionAudit(log_dir=tempfile.mkdtemp())
        audit.check_risk_state(
            equity=800.0, peak=800.0, drawdown_pct=0.0,
            rbe_mult=1.0, gmrt_mult=1.0,
            effective_leverage=5.5, configured_lev_cap=5.0,
            positions={"BTC": 4400.0}, bar_index=1,
        )
        alerts = [a for a in audit.get_recent_alerts()
                  if a["alert_type"] == "LEVERAGE_OVERSHOOT"]
        assert len(alerts) == 1
        audit.close()


# ═══ 7. Prometheus-style Metrics ═══

class TestPrometheusMetrics:
    def test_metric_names(self):
        audit = ExecutionAudit(log_dir=tempfile.mkdtemp())
        m = audit.get_metrics()
        required = [
            "darwin_orders_intents_total",
            "darwin_orders_fills_total",
            "darwin_orders_partial_total",
            "darwin_orders_rejected_total",
            "darwin_orders_reject_rate_pct",
            "darwin_slippage_mean_bps",
            "darwin_slippage_p95_bps",
            "darwin_slippage_p99_bps",
            "darwin_latency_mean_ms",
            "darwin_latency_p95_ms",
            "darwin_leverage_mean",
            "darwin_leverage_max",
            "darwin_cb_triggers_total",
            "darwin_cb_orphans_total",
            "darwin_alerts_total",
        ]
        for name in required:
            assert name in m, "Missing metric: {}".format(name)
            assert isinstance(m[name], float), "Metric {} not float".format(name)
        audit.close()

    def test_all_values_numeric(self):
        audit = ExecutionAudit(log_dir=tempfile.mkdtemp())
        # Generate some data
        for i in range(10):
            intent = audit.record_intent("BTCUSDT", "BUY", 0.1, 50000.0, 3.0, 0.8, i)
            audit.record_fill(intent, 0.1, 50010.0, 0.05, "o-{}".format(i), 100.0)
        audit.check_exposure_drift("BTC", 0.1, 0.11, 50000.0, 45000.0, 0.1, 0.0001, 10)

        m = audit.get_metrics()
        for k, v in m.items():
            assert isinstance(v, (int, float)), "{} is {}".format(k, type(v))
        audit.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
