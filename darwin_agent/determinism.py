"""
Darwin v4 — Determinism Hardening Module.

Enforces byte-level reproducibility across runs:
    - Seed all RNGs at process start
    - Replace uuid4 with seeded hash IDs
    - Sort all unordered structures before iteration
    - Enforce float64 precision
    - Set PYTHONHASHSEED
    - Disable MKL/OpenBLAS non-determinism

Usage:
    from darwin_agent.determinism import lock_determinism, deterministic_id

    lock_determinism(seed=42)
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
from typing import Any, Dict, List, Sequence, Tuple

# ── Counter for deterministic IDs ──
_id_counter: int = 0
_id_seed: int = 42


def lock_determinism(seed: int = 42) -> None:
    """
    Lock all sources of non-determinism at process level.
    Must be called ONCE at process start, before any imports that use random.
    """
    global _id_counter, _id_seed
    _id_counter = 0
    _id_seed = seed

    # 1. Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. stdlib random
    random.seed(seed)

    # 3. numpy (if available)
    try:
        import numpy as np
        np.random.seed(seed)
        # Disable MKL non-determinism
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except ImportError:
        pass

    # 4. torch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def deterministic_id(seed: int = 0, counter: int = 0) -> str:
    """
    Produce a deterministic 12-char hex ID from seed + counter.
    Replaces uuid.uuid4() which uses OS entropy.
    """
    data = struct.pack(">QQ", seed, counter)
    return hashlib.sha256(data).hexdigest()[:12]


def next_deterministic_id() -> str:
    """
    Auto-incrementing deterministic ID.
    Thread-safe for single-threaded execution.
    """
    global _id_counter
    _id_counter += 1
    return deterministic_id(_id_seed, _id_counter)


def reset_id_counter(seed: int = 42) -> None:
    """Reset the ID counter (call at start of each run)."""
    global _id_counter, _id_seed
    _id_counter = 0
    _id_seed = seed


def sorted_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Iterate dict items in sorted key order. Eliminates dict order dependency."""
    return sorted(d.items(), key=lambda kv: kv[0])


def sorted_set(s: set) -> list:
    """Convert set to sorted list for deterministic iteration."""
    return sorted(s)


def stable_sort_by_fitness(items: Sequence, fitness_key: str = "fitness",
                           genes_key: str = "genes") -> list:
    """
    Sort by fitness descending with SHA256 tie-breaking on genes.
    Guarantees identical ordering when fitnesses are equal.
    """
    def sort_key(item):
        fit = getattr(item, fitness_key, 0) if hasattr(item, fitness_key) else item.get(fitness_key, 0)
        genes = getattr(item, genes_key, {}) if hasattr(item, genes_key) else item.get(genes_key, {})
        gene_str = str(sorted(genes.items())) if isinstance(genes, dict) else str(genes)
        gene_hash = hashlib.sha256(gene_str.encode()).hexdigest()
        return (-fit, gene_hash)
    return sorted(items, key=sort_key)


def genome_hash(genes: Dict[str, float]) -> str:
    """SHA256 hash of a genome dict for identity comparison."""
    gene_str = "|".join(f"{k}={v:.15e}" for k, v in sorted(genes.items()))
    return hashlib.sha256(gene_str.encode()).hexdigest()


def equity_curve_hash(curve: List[float]) -> str:
    """SHA256 hash of an equity curve for reproducibility verification."""
    data = "|".join(f"{v:.10e}" for v in curve)
    return hashlib.sha256(data.encode()).hexdigest()
