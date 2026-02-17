from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(slots=True)
class EngineConfig:
    risk_percent: float = 1.0
    leverage: int = 5
    max_drawdown_alert_pct: float = 10.0


@dataclass(slots=True)
class EquitySnapshot:
    wallet_balance: float
    unrealized_pnl: float

    @property
    def equity(self) -> float:
        return self.wallet_balance + self.unrealized_pnl


class DarwinCoreEngine:
    """Pure core logic: no exchange, Telegram, OS, or logging dependencies."""

    def __init__(
        self,
        config: EngineConfig,
        evolutionary_logic: Callable[[dict], dict] | None = None,
        rqre: Callable[[dict], dict] | None = None,
        gmrt: Callable[[dict], dict] | None = None,
        pae: Callable[[dict], dict] | None = None,
        risk_management: Callable[[dict], dict] | None = None,
    ) -> None:
        self._config = config
        self._evolutionary_logic = evolutionary_logic or (lambda ctx: ctx)
        self._rqre = rqre or (lambda ctx: ctx)
        self._gmrt = gmrt or (lambda ctx: ctx)
        self._pae = pae or (lambda ctx: ctx)
        self._risk_management = risk_management or (lambda ctx: ctx)
        self._peak_equity = 0.0

    def update_peak(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity

    def position_size(self, snapshot: EquitySnapshot) -> float:
        return snapshot.equity * (self._config.risk_percent / 100.0)

    def drawdown_pct(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, ((self._peak_equity - equity) / self._peak_equity) * 100.0)

    def evaluate(self, snapshot: EquitySnapshot, open_positions: Iterable[dict]) -> dict:
        equity = snapshot.equity
        self.update_peak(equity)
        ctx: dict = {
            "wallet_balance": snapshot.wallet_balance,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "equity": equity,
            "open_positions": list(open_positions),
            "leverage": self._config.leverage,
            "risk_percent": self._config.risk_percent,
            "position_size": self.position_size(snapshot),
            "drawdown_pct": self.drawdown_pct(equity),
            "drawdown_alert": self.drawdown_pct(equity) > self._config.max_drawdown_alert_pct,
        }
        ctx = self._evolutionary_logic(ctx)
        ctx = self._rqre(ctx)
        ctx = self._gmrt(ctx)
        ctx = self._pae(ctx)
        return self._risk_management(ctx)
