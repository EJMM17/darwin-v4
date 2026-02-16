"""
Darwin v4 — Capital allocator.

Layer 4 (capital). Depends only on interfaces.types + interfaces.enums.
Decides HOW MUCH capital each agent gets.  Pure computation.
The orchestrator calls ``allocate()`` and passes the result to agents.
"""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Dict, List

from darwin_agent.interfaces.enums import GrowthPhase
from darwin_agent.interfaces.types import AllocationSlice, PhaseParams

from darwin_agent.capital.phases import PhaseManager


class AllocationStrategy(Enum):
    EQUAL_WEIGHT        = auto()
    PERFORMANCE_WEIGHTED = auto()
    FITNESS_WEIGHTED    = auto()


class CapitalAllocator:
    """
    Splits total capital across a set of agent IDs according to a strategy.

    * ``EQUAL_WEIGHT`` — every agent gets ``total / n``.
    * ``PERFORMANCE_WEIGHTED`` — proportional to fitness score (min floor 10 %).
    * ``FITNESS_WEIGHTED`` — softmax on fitness (sharper concentration).

    Each returned :class:`AllocationSlice` also carries the
    :class:`PhaseParams` for that agent's capital size.
    """

    __slots__ = ("_strategy", "_phase_manager", "_min_floor_pct")

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.PERFORMANCE_WEIGHTED,
        phase_manager: PhaseManager | None = None,
        min_floor_pct: float = 10.0,
    ) -> None:
        self._strategy = strategy
        self._phase_manager = phase_manager or PhaseManager()
        self._min_floor_pct = min_floor_pct

    # ── Public API ───────────────────────────────────────────

    def allocate(
        self,
        agent_ids: List[str],
        total_capital: float,
        agent_fitness: Dict[str, float] | None = None,
    ) -> Dict[str, AllocationSlice]:
        """
        Return a mapping ``agent_id → AllocationSlice``.

        Parameters
        ----------
        agent_ids:
            IDs of alive agents.
        total_capital:
            Pool capital to distribute.
        agent_fitness:
            Optional fitness/performance scores (higher = better).
            Ignored for ``EQUAL_WEIGHT``.
        """
        if not agent_ids:
            return {}

        weights = self._compute_weights(agent_ids, agent_fitness or {})
        result: Dict[str, AllocationSlice] = {}

        for aid in agent_ids:
            amount = round(total_capital * weights[aid], 4)
            params = self._phase_manager.get_params(amount)
            result[aid] = AllocationSlice(
                agent_id=aid,
                amount=amount,
                phase_params=params,
            )

        return result

    def get_phase_params(self, capital: float) -> PhaseParams:
        """Convenience passthrough to the phase manager."""
        return self._phase_manager.get_params(capital)

    # ── Internals ────────────────────────────────────────────

    def _compute_weights(
        self,
        agent_ids: List[str],
        fitness: Dict[str, float],
    ) -> Dict[str, float]:
        n = len(agent_ids)

        if self._strategy == AllocationStrategy.EQUAL_WEIGHT or not fitness:
            equal = 1.0 / n
            return {aid: equal for aid in agent_ids}

        # Floor: every agent gets at least min_floor_pct of equal share
        floor = (self._min_floor_pct / 100.0) / n

        if self._strategy == AllocationStrategy.PERFORMANCE_WEIGHTED:
            return self._linear_weighted(agent_ids, fitness, floor)

        if self._strategy == AllocationStrategy.FITNESS_WEIGHTED:
            return self._softmax_weighted(agent_ids, fitness, floor)

        # Fallback
        equal = 1.0 / n
        return {aid: equal for aid in agent_ids}

    @staticmethod
    def _linear_weighted(
        agent_ids: List[str],
        fitness: Dict[str, float],
        floor: float,
    ) -> Dict[str, float]:
        """Proportional to fitness, with a minimum floor."""
        raw = {aid: max(fitness.get(aid, 0.0), 0.0) for aid in agent_ids}
        total = sum(raw.values())
        if total == 0:
            equal = 1.0 / len(agent_ids)
            return {aid: equal for aid in agent_ids}

        weights = {aid: (v / total) for aid, v in raw.items()}

        # Apply floor and re-normalise
        floored = {aid: max(w, floor) for aid, w in weights.items()}
        ftotal = sum(floored.values())
        return {aid: w / ftotal for aid, w in floored.items()}

    @staticmethod
    def _softmax_weighted(
        agent_ids: List[str],
        fitness: Dict[str, float],
        floor: float,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Softmax concentration on highest-fitness agents."""
        scores = [fitness.get(aid, 0.0) for aid in agent_ids]
        max_s = max(scores) if scores else 0.0
        # Numerically stable softmax
        exps = [math.exp((s - max_s) / max(temperature, 0.01)) for s in scores]
        total = sum(exps)
        if total == 0:
            equal = 1.0 / len(agent_ids)
            return {aid: equal for aid in agent_ids}

        weights = {aid: e / total for aid, e in zip(agent_ids, exps)}
        floored = {aid: max(w, floor) for aid, w in weights.items()}
        ftotal = sum(floored.values())
        return {aid: w / ftotal for aid, w in floored.items()}
