"""
Darwin v4 — Determinism Hardening & Genome Freeze.

PHASE 1: Global determinism lock (already in determinism.py, extended here)
PHASE 2: GA reproducibility enforcement
PHASE 3: Genome freeze export/import
PHASE 4: Reproduction verification
PHASE 5: Docker lockdown config

Usage:
    from darwin_agent.freeze import (
        lock_all_determinism,
        freeze_run,
        load_frozen_run,
        verify_reproduction,
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from darwin_agent.determinism import (
    lock_determinism,
    reset_id_counter,
    genome_hash,
    equity_curve_hash,
)


# ═══════════════════════════════════════════════════════
# PHASE 1 — Global determinism lock (extended)
# ═══════════════════════════════════════════════════════

def lock_all_determinism(seed: int = 42) -> None:
    """
    Complete determinism lock. Call at process start.
    Extends lock_determinism with additional guards.
    """
    # Core lock (random, numpy, torch, PYTHONHASHSEED)
    lock_determinism(seed)
    reset_id_counter(seed)

    # Additional environment hardening
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Disable async parallelism in numpy/scipy
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"


# ═══════════════════════════════════════════════════════
# PHASE 3 — Genome Freeze Export
# ═══════════════════════════════════════════════════════

@dataclass
class FrozenGenome:
    symbol: str
    genes: Dict[str, float]
    fitness: float
    pf: float
    max_dd: float
    trades: int
    genome_sha256: str


@dataclass
class FrozenRun:
    """Complete snapshot of a canonical reference run."""
    timestamp: str
    seed: int
    config: Dict[str, Any]
    genomes: List[FrozenGenome]
    equity_curve: List[float]
    equity_curve_sha256: str
    pool_sha256: str
    final_equity: float
    cagr: float
    sharpe: float
    max_dd: float
    trades: int
    monthly_returns: List[float]


def freeze_run(
    seed: int,
    config: Dict[str, Any],
    active_genomes: Dict[str, Dict[str, float]],
    genome_meta: Dict[str, Dict[str, Any]],
    equity_curve: List[float],
    final_equity: float,
    cagr: float,
    sharpe: float,
    max_dd: float,
    trades: int,
    monthly_returns: List[float],
    output_dir: str = "artifacts",
) -> str:
    """
    Export a frozen canonical run to JSON.

    Returns path to the frozen run file.
    """
    frozen_genomes = []
    all_hashes = []

    for sym in sorted(active_genomes.keys()):
        genes = active_genomes[sym]
        meta = genome_meta.get(sym, {})
        gh = genome_hash(genes)
        all_hashes.append(gh)
        frozen_genomes.append(FrozenGenome(
            symbol=sym,
            genes=genes,
            fitness=meta.get("fitness", 0.0),
            pf=meta.get("pf", 0.0),
            max_dd=meta.get("max_dd", 0.0),
            trades=meta.get("trades", 0),
            genome_sha256=gh,
        ))

    # Pool hash = hash of sorted genome hashes
    pool_hash_data = "|".join(sorted(all_hashes))
    pool_sha = hashlib.sha256(pool_hash_data.encode()).hexdigest()

    eq_sha = equity_curve_hash(equity_curve)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run = FrozenRun(
        timestamp=ts,
        seed=seed,
        config=config,
        genomes=frozen_genomes,
        equity_curve=equity_curve,
        equity_curve_sha256=eq_sha,
        pool_sha256=pool_sha,
        final_equity=final_equity,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        trades=trades,
        monthly_returns=monthly_returns,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frozen_run_{ts}.json")

    # Custom serializer for dataclasses
    def serialize(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return str(obj)

    with open(path, "w") as f:
        json.dump(asdict(run), f, indent=2, default=serialize)

    return path


def load_frozen_run(path: str) -> FrozenRun:
    """Load a frozen run from JSON."""
    with open(path) as f:
        data = json.load(f)

    genomes = [FrozenGenome(**g) for g in data.pop("genomes")]
    return FrozenRun(genomes=genomes, **data)


def extract_genomes(frozen: FrozenRun) -> Dict[str, Dict[str, float]]:
    """Extract active genomes dict from a frozen run."""
    return {g.symbol: g.genes for g in frozen.genomes}


# ═══════════════════════════════════════════════════════
# PHASE 4 — Reproduction Verification
# ═══════════════════════════════════════════════════════

@dataclass
class ReproductionResult:
    """Result of a reproduction verification run."""
    equity_match: bool
    genome_match: bool
    trade_count_match: bool
    equity_sha256_expected: str
    equity_sha256_actual: str
    pool_sha256_expected: str
    pool_sha256_actual: str
    expected_equity: float
    actual_equity: float
    expected_trades: int
    actual_trades: int
    mismatches: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.equity_match and self.genome_match and self.trade_count_match


def verify_reproduction(
    frozen: FrozenRun,
    actual_equity_curve: List[float],
    actual_genomes: Dict[str, Dict[str, float]],
    actual_trades: int,
) -> ReproductionResult:
    """
    Compare a reproduction run against a frozen canonical run.
    """
    eq_sha_actual = equity_curve_hash(actual_equity_curve)
    eq_match = eq_sha_actual == frozen.equity_curve_sha256

    # Genome comparison
    actual_hashes = []
    for sym in sorted(actual_genomes.keys()):
        actual_hashes.append(genome_hash(actual_genomes[sym]))
    pool_hash_data = "|".join(sorted(actual_hashes))
    pool_sha_actual = hashlib.sha256(pool_hash_data.encode()).hexdigest()
    genome_match = pool_sha_actual == frozen.pool_sha256

    trade_match = actual_trades == frozen.trades

    mismatches = []
    if not eq_match:
        mismatches.append(
            f"Equity curve hash mismatch: "
            f"expected={frozen.equity_curve_sha256[:16]}... "
            f"actual={eq_sha_actual[:16]}..."
        )
        # Find first divergence point
        for i, (exp, act) in enumerate(zip(frozen.equity_curve, actual_equity_curve)):
            if abs(exp - act) > 1e-8:
                mismatches.append(
                    f"First divergence at bar {i}: "
                    f"expected={exp:.10f} actual={act:.10f} "
                    f"delta={act-exp:.2e}"
                )
                break

    if not genome_match:
        for fg in frozen.genomes:
            ag = actual_genomes.get(fg.symbol)
            if ag is None:
                mismatches.append(f"Missing genome for {fg.symbol}")
            else:
                ah = genome_hash(ag)
                if ah != fg.genome_sha256:
                    mismatches.append(
                        f"{fg.symbol} genome hash mismatch: "
                        f"expected={fg.genome_sha256[:16]}... "
                        f"actual={ah[:16]}..."
                    )
                    # Find differing genes
                    for k in sorted(set(fg.genes) | set(ag)):
                        ev = fg.genes.get(k, float('nan'))
                        av = ag.get(k, float('nan'))
                        if abs(ev - av) > 1e-12:
                            mismatches.append(
                                f"  gene {k}: expected={ev:.15e} actual={av:.15e}"
                            )

    if not trade_match:
        mismatches.append(
            f"Trade count mismatch: expected={frozen.trades} actual={actual_trades}"
        )

    actual_eq = actual_equity_curve[-1] if actual_equity_curve else 0.0

    return ReproductionResult(
        equity_match=eq_match,
        genome_match=genome_match,
        trade_count_match=trade_match,
        equity_sha256_expected=frozen.equity_curve_sha256,
        equity_sha256_actual=eq_sha_actual,
        pool_sha256_expected=frozen.pool_sha256,
        pool_sha256_actual=pool_sha_actual,
        expected_equity=frozen.final_equity,
        actual_equity=actual_eq,
        expected_trades=frozen.trades,
        actual_trades=actual_trades,
        mismatches=mismatches,
    )
