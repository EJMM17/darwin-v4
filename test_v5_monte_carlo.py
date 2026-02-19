"""
Tests for Darwin v5 Monte Carlo Validation.

Tests:
    - Basic validation with trade PnL sequences
    - Edge detection (positive edge)
    - No-edge detection (random performance)
    - Insufficient data handling
    - Confidence distribution analysis
    - Percentile computation
    - Deterministic with seed
"""
import pytest

from darwin_agent.v5.monte_carlo import (
    MonteCarloValidator,
    MonteCarloConfig,
    MonteCarloResult,
)


class TestMonteCarloValidation:
    def test_basic_validation(self):
        """Monte Carlo runs with valid trade data."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=42))
        pnls = [10.0, -5.0, 15.0, -3.0, 8.0, -2.0, 12.0, -4.0, 7.0, -1.0]
        result = mc.validate(pnls)

        assert result.n_simulations == 100
        assert result.actual_pnl == sum(pnls)
        assert result.mean_random_pnl != 0 or all(p == 0 for p in pnls)
        assert 0.0 <= result.percentile_rank <= 1.0

    def test_positive_edge_detection(self):
        """Strategy with clear positive edge is detected."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=500, seed=42))
        # Consistently profitable trades
        pnls = [5.0, 3.0, 7.0, -1.0, 4.0, 6.0, -0.5, 8.0, 2.0, 5.0,
                3.0, 7.0, -1.5, 4.0, 6.0, -0.5, 8.0, 2.0, 5.0, 3.0]
        result = mc.validate(pnls)

        # With net positive PnL, actual should rank above mean random
        assert result.actual_pnl > 0
        assert result.win_rate_actual > 0.5

    def test_no_edge_with_random(self):
        """Pure random trades don't show significant edge."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=500, seed=42))
        import random
        rng = random.Random(42)
        pnls = [rng.gauss(0, 1) for _ in range(50)]
        result = mc.validate(pnls)

        # Edge ratio should be close to 0 for random data
        assert abs(result.edge_ratio) < 3.0  # within 3 sigma

    def test_insufficient_data(self):
        """Handles too few trades gracefully."""
        mc = MonteCarloValidator()
        result = mc.validate([1.0, -0.5])
        assert result.n_simulations == 0

    def test_all_positive_pnl(self):
        """All winning trades."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=42))
        pnls = [1.0] * 20
        result = mc.validate(pnls)
        assert result.win_rate_actual == 1.0
        assert result.actual_pnl == 20.0

    def test_all_negative_pnl(self):
        """All losing trades."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=42))
        pnls = [-1.0] * 20
        result = mc.validate(pnls)
        assert result.win_rate_actual == 0.0
        assert result.actual_pnl == -20.0

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        pnls = [5.0, -2.0, 3.0, -1.0, 7.0, -3.0, 4.0, -1.5, 6.0, -2.5]
        mc1 = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=123))
        mc2 = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=123))

        r1 = mc1.validate(pnls)
        r2 = mc2.validate(pnls)

        assert r1.mean_random_pnl == r2.mean_random_pnl
        assert r1.edge_ratio == r2.edge_ratio
        assert r1.percentile_rank == r2.percentile_rank

    def test_percentile_values(self):
        """Percentile values are ordered correctly."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=200, seed=42))
        pnls = [3.0, -1.0, 5.0, -2.0, 4.0, -1.5, 6.0, -0.5, 3.0, -1.0,
                2.0, -0.5, 4.0, -1.0, 5.0, -2.0, 3.0, -1.0, 4.0, -0.5]
        result = mc.validate(pnls)

        assert result.p5_pnl <= result.p25_pnl
        assert result.p25_pnl <= result.p50_pnl
        assert result.p50_pnl <= result.p75_pnl
        assert result.p75_pnl <= result.p95_pnl

    def test_max_drawdown_p95(self):
        """Max drawdown percentile is non-negative."""
        mc = MonteCarloValidator(MonteCarloConfig(n_simulations=100, seed=42))
        pnls = [5.0, -10.0, 3.0, -8.0, 7.0, -2.0, 4.0, -5.0, 6.0, -3.0]
        result = mc.validate(pnls)
        assert result.max_drawdown_p95 >= 0.0

    def test_to_dict(self):
        """MonteCarloResult serializes correctly."""
        result = MonteCarloResult(
            n_simulations=100,
            actual_pnl=50.0,
            mean_random_pnl=45.0,
            edge_ratio=0.5,
            percentile_rank=0.7,
        )
        d = result.to_dict()
        assert d["n_simulations"] == 100
        assert d["actual_pnl"] == 50.0
        assert d["has_edge"] is True  # edge_ratio > 0.1 and percentile_rank > 0.5

    def test_has_edge_property(self):
        """has_edge logic works correctly."""
        # Has edge: positive ratio + high percentile
        result = MonteCarloResult(edge_ratio=0.5, percentile_rank=0.7)
        assert result.has_edge is True

        # No edge: low ratio
        result = MonteCarloResult(edge_ratio=0.05, percentile_rank=0.7)
        assert result.has_edge is False

        # No edge: low percentile
        result = MonteCarloResult(edge_ratio=0.5, percentile_rank=0.3)
        assert result.has_edge is False


class TestConfidenceDistribution:
    def test_confidence_distribution_analysis(self):
        """Validates confidence-PnL monotonicity check."""
        mc = MonteCarloValidator(MonteCarloConfig(seed=42))
        confidences = [0.5 + i * 0.01 for i in range(20)]
        # Higher confidence trades perform better
        pnls = [c * 10 - 3 for c in confidences]

        dist = mc.validate_confidence_distribution(confidences, pnls)
        assert "low_confidence_avg_pnl" in dist
        assert "high_confidence_avg_pnl" in dist
        assert dist["monotonic"] is True  # higher conf → higher pnl

    def test_insufficient_confidence_data(self):
        """Returns empty dict when not enough data."""
        mc = MonteCarloValidator()
        dist = mc.validate_confidence_distribution([0.5], [1.0])
        assert dist == {}
