"""
Darwin v4 — Asset Structural Classification.

Standalone module. No dependency on evolution, fitness, or harness.

Computes per-symbol structural metrics from OHLCV data, normalizes
across the asset universe, and clusters symbols by market behavior.

Cluster archetypes (typical):
  0 = High-trend / High-vol  (momentum assets)
  1 = Low-trend / Low-vol    (range-bound / stable)
  2 = Mean-reverting          (RSI oscillators)
  3 = Vol-clustered           (regime-switching)
  4 = Mixed / Transitional

Usage:
    profiler = AssetProfiler(k=5)
    profiler.add_symbol("BTCUSDT", "/path/to/btc_1h.csv")
    profiler.add_symbol("ETHUSDT", "/path/to/eth_1h.csv")
    result = profiler.analyze()
    result.print_summary()
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Data structures ──────────────────────────────────────────

@dataclass
class SymbolProfile:
    """Raw + normalized metrics for one symbol."""
    symbol: str
    n_bars: int = 0
    # Raw metrics
    trend_persistence: float = 0.0     # lag-1 autocorrelation of returns
    volatility_ratio: float = 0.0      # mean(ATR14 / price)
    mean_reversion_score: float = 0.0  # mean |RSI - 50| / 50
    vol_clustering: float = 0.0        # std(rolling_std) / mean(rolling_std)
    avg_volume: float = 0.0            # mean daily volume
    # Normalized (z-score across universe)
    z_trend: float = 0.0
    z_vol: float = 0.0
    z_mr: float = 0.0
    z_vc: float = 0.0
    z_volume: float = 0.0
    # Cluster assignment
    cluster: int = -1

    def raw_vector(self) -> List[float]:
        return [self.trend_persistence, self.volatility_ratio,
                self.mean_reversion_score, self.vol_clustering,
                self.avg_volume]

    def z_vector(self) -> List[float]:
        return [self.z_trend, self.z_vol, self.z_mr, self.z_vc, self.z_volume]


@dataclass
class ClusterInfo:
    """Summary for one cluster."""
    cluster_id: int
    symbols: List[str] = field(default_factory=list)
    centroid: List[float] = field(default_factory=list)
    representative: str = ""   # symbol closest to centroid
    archetype: str = ""        # human-readable label


@dataclass
class ProfileResult:
    """Complete profiling output."""
    profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    clusters: Dict[int, ClusterInfo] = field(default_factory=dict)
    k: int = 5
    metric_names: Tuple[str, ...] = (
        "trend_persistence", "volatility_ratio",
        "mean_reversion_score", "vol_clustering", "avg_volume",
    )

    def print_summary(self) -> str:
        lines = []
        lines.append("=" * 90)
        lines.append("  ASSET STRUCTURAL CLASSIFICATION")
        lines.append("=" * 90)

        # Per-symbol raw metrics
        lines.append("")
        lines.append("  %-12s %8s %8s %8s %8s %12s" % (
            "Symbol", "Trend", "VolRatio", "MeanRev", "VolClust", "AvgVolume"))
        lines.append("  " + "-" * 62)
        for sym in sorted(self.profiles):
            p = self.profiles[sym]
            lines.append("  %-12s %+8.4f %8.4f %8.4f %8.4f %12.0f" % (
                sym, p.trend_persistence, p.volatility_ratio,
                p.mean_reversion_score, p.vol_clustering, p.avg_volume))

        # Z-score table
        lines.append("")
        lines.append("  Z-SCORES (normalized across universe)")
        lines.append("  %-12s %8s %8s %8s %8s %8s  %s" % (
            "Symbol", "zTrend", "zVol", "zMR", "zVC", "zVol$", "Cluster"))
        lines.append("  " + "-" * 68)
        for sym in sorted(self.profiles):
            p = self.profiles[sym]
            lines.append("  %-12s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f  C%d" % (
                sym, p.z_trend, p.z_vol, p.z_mr, p.z_vc, p.z_volume, p.cluster))

        # Cluster summary
        lines.append("")
        lines.append("  CLUSTERS")
        lines.append("  " + "-" * 68)
        for cid in sorted(self.clusters):
            ci = self.clusters[cid]
            syms = ", ".join(ci.symbols)
            lines.append("  C%d [%s]: %s" % (cid, ci.archetype, syms))
            lines.append("       Representative: %s" % ci.representative)
            if ci.centroid:
                cents = " ".join("%+.2f" % c for c in ci.centroid)
                lines.append("       Centroid: [%s]" % cents)

        lines.append("")
        lines.append("=" * 90)
        out = "\n".join(lines)
        print(out)
        return out


# ── Core profiler ────────────────────────────────────────────

class AssetProfiler:
    """
    Compute structural metrics from OHLCV CSVs and cluster assets.

    Parameters
    ----------
    k : int
        Number of KMeans clusters. Auto-reduced if fewer symbols.
    seed : int
        Random seed for deterministic clustering.
    """

    def __init__(self, k: int = 5, seed: int = 42) -> None:
        self._k = k
        self._seed = seed
        self._symbols: Dict[str, str] = {}  # symbol -> csv_path

    def add_symbol(self, symbol: str, csv_path: str) -> None:
        """Register a symbol with its OHLCV CSV path."""
        self._symbols[symbol] = csv_path

    def add_symbols(self, mapping: Dict[str, str]) -> None:
        """Register multiple symbols at once."""
        self._symbols.update(mapping)

    def analyze(self) -> ProfileResult:
        """Run full pipeline: compute → normalize → cluster → label."""
        if len(self._symbols) < 2:
            raise ValueError("Need at least 2 symbols for profiling")

        # 1. Compute raw metrics per symbol
        profiles = {}
        for sym, path in self._symbols.items():
            profiles[sym] = self._compute_metrics(sym, path)

        # 2. Z-score normalize
        self._normalize(profiles)

        # 3. Cluster
        k = min(self._k, len(profiles))
        clusters = self._kmeans(profiles, k)

        # 4. Label archetypes
        self._label_archetypes(clusters, profiles)

        return ProfileResult(profiles=profiles, clusters=clusters, k=k)

    # ── Metric computation ───────────────────────────────────

    @staticmethod
    def _load_ohlcv(path: str) -> Tuple[
        List[float], List[float], List[float], List[float], List[float]
    ]:
        """Load OHLCV from CSV. Returns (opens, highs, lows, closes, volumes)."""
        opens, highs, lows, closes, volumes = [], [], [], [], []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                opens.append(float(row["open"]))
                highs.append(float(row["high"]))
                lows.append(float(row["low"]))
                closes.append(float(row["close"]))
                volumes.append(float(row["volume"]))
        return opens, highs, lows, closes, volumes

    def _compute_metrics(self, symbol: str, csv_path: str) -> SymbolProfile:
        """Compute all 5 structural metrics from OHLCV data."""
        opens, highs, lows, closes, volumes = self._load_ohlcv(csv_path)
        n = len(closes)
        if n < 250:
            raise ValueError(f"{symbol}: need ≥250 bars, got {n}")

        profile = SymbolProfile(symbol=symbol, n_bars=n)

        # Returns series
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, n)]

        # 1. Trend persistence: lag-1 autocorrelation of returns
        profile.trend_persistence = self._autocorrelation(returns, lag=1)

        # 2. Volatility ratio: mean(ATR14 / price)
        atr = self._compute_atr(highs, lows, closes, period=14)
        ratios = [atr[i] / closes[i] for i in range(n) if closes[i] > 0 and atr[i] > 0]
        profile.volatility_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        # 3. Mean reversion score: mean(|RSI - 50|) / 50
        #    Higher = RSI spends more time at extremes (trending)
        #    Lower = RSI hovers near 50 (mean-reverting)
        rsi = self._compute_rsi(closes, period=14)
        rsi_devs = [abs(rsi[i] - 50.0) / 50.0 for i in range(n) if rsi[i] > 0]
        profile.mean_reversion_score = sum(rsi_devs) / len(rsi_devs) if rsi_devs else 0.0

        # 4. Volatility clustering: coefficient of variation of rolling std
        #    High = vol comes in bursts (regime-switching)
        #    Low = steady vol (predictable)
        window = 24  # 24h rolling
        if len(returns) >= window:
            rolling_stds = []
            for i in range(window, len(returns)):
                chunk = returns[i - window:i]
                mu = sum(chunk) / window
                var = sum((x - mu) ** 2 for x in chunk) / window
                rolling_stds.append(math.sqrt(var) if var > 0 else 0.0)
            if rolling_stds:
                rs_mean = sum(rolling_stds) / len(rolling_stds)
                rs_mu = rs_mean
                rs_var = sum((s - rs_mu) ** 2 for s in rolling_stds) / len(rolling_stds)
                rs_std = math.sqrt(rs_var) if rs_var > 0 else 0.0
                profile.vol_clustering = rs_std / rs_mean if rs_mean > 0 else 0.0

        # 5. Average volume (in quote currency units)
        profile.avg_volume = sum(
            closes[i] * volumes[i] for i in range(n)
        ) / n

        return profile

    # ── Normalization ────────────────────────────────────────

    @staticmethod
    def _normalize(profiles: Dict[str, SymbolProfile]) -> None:
        """Z-score normalize all metrics across the universe."""
        syms = list(profiles.keys())
        n = len(syms)
        if n < 2:
            return

        fields = [
            ("trend_persistence", "z_trend"),
            ("volatility_ratio", "z_vol"),
            ("mean_reversion_score", "z_mr"),
            ("vol_clustering", "z_vc"),
            ("avg_volume", "z_volume"),
        ]

        for raw_field, z_field in fields:
            vals = [getattr(profiles[s], raw_field) for s in syms]
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / n
            std = math.sqrt(var) if var > 0 else 1.0
            for s in syms:
                raw = getattr(profiles[s], raw_field)
                setattr(profiles[s], z_field, (raw - mu) / std)

    # ── KMeans clustering (pure Python, no sklearn) ──────────

    def _kmeans(
        self, profiles: Dict[str, SymbolProfile], k: int,
        max_iter: int = 100,
    ) -> Dict[int, ClusterInfo]:
        """KMeans on z-scored feature vectors. Deterministic via seed."""
        syms = sorted(profiles.keys())
        vectors = {s: profiles[s].z_vector() for s in syms}
        dim = 5
        rng = random.Random(self._seed)

        # Init centroids: k-means++ style
        centroids = [list(vectors[rng.choice(syms)])]
        for _ in range(1, k):
            dists = []
            for s in syms:
                min_d = min(self._dist(vectors[s], c) for c in centroids)
                dists.append(min_d)
            total = sum(dists)
            if total <= 0:
                centroids.append(list(vectors[rng.choice(syms)]))
                continue
            pick = rng.random() * total
            cum = 0.0
            for i, s in enumerate(syms):
                cum += dists[i]
                if cum >= pick:
                    centroids.append(list(vectors[s]))
                    break

        # Iterate
        assignments = {s: 0 for s in syms}
        for _ in range(max_iter):
            # Assign
            changed = False
            for s in syms:
                dists = [self._dist(vectors[s], c) for c in centroids]
                best = dists.index(min(dists))
                if assignments[s] != best:
                    assignments[s] = best
                    changed = True
            if not changed:
                break
            # Update centroids
            for ci in range(k):
                members = [s for s in syms if assignments[s] == ci]
                if members:
                    centroids[ci] = [
                        sum(vectors[s][d] for s in members) / len(members)
                        for d in range(dim)
                    ]

        # Build cluster info
        clusters: Dict[int, ClusterInfo] = {}
        for ci in range(k):
            members = [s for s in syms if assignments[s] == ci]
            if not members:
                continue
            centroid = centroids[ci]
            # Representative: closest to centroid
            rep = min(members, key=lambda s: self._dist(vectors[s], centroid))
            clusters[ci] = ClusterInfo(
                cluster_id=ci, symbols=members,
                centroid=[round(c, 4) for c in centroid],
                representative=rep,
            )

        # Set cluster assignment on profiles
        for s in syms:
            profiles[s].cluster = assignments[s]

        return clusters

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    # ── Archetype labeling ───────────────────────────────────

    @staticmethod
    def _label_archetypes(
        clusters: Dict[int, ClusterInfo],
        profiles: Dict[str, SymbolProfile],
    ) -> None:
        """Assign human-readable archetype labels based on centroid position."""
        # Metric indices: 0=trend, 1=vol, 2=mean_rev, 3=vol_clust, 4=volume
        for ci, info in clusters.items():
            c = info.centroid
            if len(c) < 5:
                info.archetype = "unknown"
                continue

            trend, vol, mr, vc, volume = c

            # Classification heuristics based on dominant z-scores
            if trend > 0.5 and vol > 0.5:
                info.archetype = "momentum-volatile"
            elif trend > 0.5 and vol <= 0.5:
                info.archetype = "trending-stable"
            elif trend < -0.5 and mr < -0.3:
                info.archetype = "mean-reverting"
            elif vc > 0.7:
                info.archetype = "regime-switching"
            elif vol < -0.5 and abs(trend) < 0.5:
                info.archetype = "range-bound"
            elif volume > 1.0:
                info.archetype = "high-liquidity"
            elif volume < -0.8:
                info.archetype = "low-liquidity"
            else:
                info.archetype = "mixed"

    # ── Technical indicator helpers (zero-dependency) ────────

    @staticmethod
    def _autocorrelation(series: List[float], lag: int = 1) -> float:
        """Lag-k autocorrelation. Returns [-1, +1]."""
        n = len(series)
        if n <= lag + 1:
            return 0.0
        mu = sum(series) / n
        var = sum((x - mu) ** 2 for x in series) / n
        if var < 1e-12:
            return 0.0
        cov = sum(
            (series[i] - mu) * (series[i - lag] - mu)
            for i in range(lag, n)
        ) / (n - lag)
        return cov / var

    @staticmethod
    def _compute_atr(
        highs: List[float], lows: List[float],
        closes: List[float], period: int = 14,
    ) -> List[float]:
        """Average True Range. Returns list same length as input."""
        n = len(closes)
        tr = [highs[0] - lows[0]]
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))
        atr = [0.0] * n
        if n >= period:
            atr[period - 1] = sum(tr[:period]) / period
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _compute_rsi(closes: List[float], period: int = 14) -> List[float]:
        """RSI(period). Returns list same length as input, 0 for warmup."""
        n = len(closes)
        rsi = [0.0] * n
        if n < period + 1:
            return rsi
        gains, losses = 0.0, 0.0
        for i in range(1, period + 1):
            d = closes[i] - closes[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[period] = 100.0
        for i in range(period + 1, n):
            d = closes[i] - closes[i - 1]
            g = d if d > 0 else 0.0
            l = -d if d < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + l) / period
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
            else:
                rsi[i] = 100.0
        return rsi
