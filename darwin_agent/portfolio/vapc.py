"""
Darwin v4 — Volatility-Adjusted Position Cap (VAPC).

Limits maximum single-bar equity risk by dynamically capping leverage
based on current realized volatility. Acts as a pre-trade sizing filter
at the portfolio layer — does NOT modify genomes, GA, RQRE, or fitness.

Architecture:
    ATR(14) / price → bar_vol
    max_bar_risk / bar_vol → vol_adjusted_cap
    effective_leverage = min(dynamic_lev, flat_cap, vol_adjusted_cap)

Normal vol (bar_vol ≈ 0.02): cap ≈ 6.0 → 5x flat cap binds → no change
Crash vol  (bar_vol ≈ 0.10): cap ≈ 1.2 → leverage automatically reduced
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VAPCConfig:
    """Tunable parameters for the Volatility-Adjusted Position Cap."""
    max_bar_risk: float = 0.20    # max equity risk per bar (20%)
    min_leverage: float = 1.0     # floor — always allow at least 1x
    max_leverage: float = 5.0     # ceiling — never exceed flat cap
    fallback_leverage: float = 5.0  # used when ATR/price invalid


def compute_vapc_leverage(
    atr: float,
    price: float,
    config: VAPCConfig = VAPCConfig(),
) -> float:
    """
    Compute the volatility-adjusted leverage cap for a single symbol/bar.

    Parameters
    ----------
    atr : float
        ATR(14) value at current bar.
    price : float
        Current close price.
    config : VAPCConfig
        VAPC parameters.

    Returns
    -------
    float
        Clamped leverage cap ∈ [config.min_leverage, config.max_leverage].

    Edge cases:
        - atr <= 0 or price <= 0 → fallback_leverage
        - Extremely low vol → clamp to max_leverage
        - Extremely high vol → clamp to min_leverage
    """
    # Guard: invalid inputs → fallback to flat cap
    if atr <= 0.0 or price <= 0.0:
        return config.fallback_leverage

    bar_vol = atr / price

    # Guard: near-zero vol → would produce infinite cap
    if bar_vol < 1e-10:
        return config.max_leverage

    vol_adjusted_cap = config.max_bar_risk / bar_vol

    # Clamp to [min, max]
    return max(config.min_leverage, min(config.max_leverage, vol_adjusted_cap))


def compute_effective_leverage(
    dynamic_leverage: float,
    atr: float,
    price: float,
    config: VAPCConfig = VAPCConfig(),
) -> float:
    """
    Compute final effective leverage = min(dynamic, flat_cap, vapc_cap).

    This is the top-level function used in the portfolio sizing pipeline.
    Apply BEFORE PAE weight, GMRT multiplier, and RBE multiplier.

    Parameters
    ----------
    dynamic_leverage : float
        Leverage from the existing dynamic formula (dlev(gm, rm)).
    atr : float
        ATR(14) at current bar.
    price : float
        Current close price.
    config : VAPCConfig
        VAPC parameters.

    Returns
    -------
    float
        Effective leverage ∈ [config.min_leverage, config.max_leverage].
    """
    vapc_cap = compute_vapc_leverage(atr, price, config)
    return max(config.min_leverage, min(dynamic_leverage, vapc_cap))
