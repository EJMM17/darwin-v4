"""
Darwin v4 — Capital growth phase manager.

Layer 4 (capital). Depends only on interfaces.enums + interfaces.types.
Pure stateless computation: capital → PhaseParams.
"""
from __future__ import annotations

from darwin_agent.interfaces.enums import GrowthPhase
from darwin_agent.interfaces.types import PhaseParams


# ── Default phase boundaries (overridable via config) ────────

_DEFAULT_RULES = {
    GrowthPhase.BOOTSTRAP: {
        "max_capital": 100.0,
        "risk_pct": 2.0,
        "max_positions": 2,
        "leverage": 10,
    },
    GrowthPhase.SCALING: {
        "max_capital": 500.0,
        "risk_pct": 3.0,
        "max_positions": 3,
        "leverage": 15,
    },
    GrowthPhase.ACCELERATION: {
        "max_capital": 2_000.0,
        "risk_pct": 4.0,
        "max_positions": 4,
        "leverage": 20,
    },
    GrowthPhase.CONSOLIDATION: {
        "max_capital": float("inf"),
        "risk_pct": 2.5,
        "max_positions": 3,
        "leverage": 10,
    },
}


class PhaseManager:
    """
    Resolves the current :class:`GrowthPhase` and its risk parameters
    from a capital value.

    Stateless — safe to call from any module at any time.
    """

    __slots__ = ("_rules",)

    def __init__(self, overrides: dict | None = None) -> None:
        self._rules = dict(_DEFAULT_RULES)
        if overrides:
            for phase, params in overrides.items():
                if phase in self._rules:
                    self._rules[phase].update(params)

    def get_phase(self, capital: float) -> GrowthPhase:
        """Return the phase for the given *capital*."""
        for phase in (
            GrowthPhase.BOOTSTRAP,
            GrowthPhase.SCALING,
            GrowthPhase.ACCELERATION,
        ):
            if capital < self._rules[phase]["max_capital"]:
                return phase
        return GrowthPhase.CONSOLIDATION

    def get_params(self, capital: float) -> PhaseParams:
        """Return full :class:`PhaseParams` for the given *capital*."""
        phase = self.get_phase(capital)
        rule = self._rules[phase]
        return PhaseParams(
            phase=phase,
            risk_pct=rule["risk_pct"],
            max_positions=rule["max_positions"],
            leverage=rule["leverage"],
        )
