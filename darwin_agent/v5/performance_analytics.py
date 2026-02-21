"""
Performance Analytics — Institutional Grade
============================================
Calcula métricas estándar de hedge funds en tiempo real:
  - Sharpe Ratio       (retorno / volatilidad total)
  - Sortino Ratio      (retorno / volatilidad bajista)
  - Calmar Ratio       (retorno anualizado / max drawdown)
  - Win Rate / Profit Factor / Expectancy
  - Alpha vs BTC / Information Ratio
  - Half-Kelly position sizing dinámico
  - Circuit Breaker automático

Umbrales institucionales (AIMA/PwC 2024-2025):
  Sharpe  > 1.0 (mínimo), > 2.0 (hedge fund grade)
  Sortino > 2.0 | Calmar > 1.0 (elite > 2.0)
  Max DD  < 20% para capital escalable
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# Thresholds institucionales
SHARPE_MIN       = 1.0
SHARPE_HEDGE     = 2.0
SORTINO_MIN      = 2.0
CALMAR_MIN       = 1.0
CALMAR_ELITE     = 2.0
PROFIT_FACTOR_OK = 1.5
PROFIT_FACTOR_HF = 2.0
MAX_DD_SCALABLE  = 0.20

# Frecuencia de snapshots: ticks de 5s
TICKS_PER_YEAR = int(365 * 24 * 3600 / 5)   # ~6.3M
ANN_FACTOR     = TICKS_PER_YEAR ** 0.5


@dataclass
class PerformanceSnapshot:
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    total_pnl: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    alpha_pct: float = 0.0
    information_ratio: float = 0.0
    half_kelly_pct: float = 2.0
    kelly_edge: float = 0.0
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    grade: str = "N/A"
    scalable: bool = False
    computed_at: float = field(default_factory=time.time)

    def to_log(self) -> str:
        """One-line summary for periodic heartbeat logging."""
        return (
            f"PERF | trades={self.total_trades} WR={self.win_rate:.1%} "
            f"PF={self.profit_factor:.2f} PnL=${self.total_pnl:.2f} "
            f"Sharpe={self.sharpe_ratio:.2f} Sortino={self.sortino_ratio:.2f} "
            f"Calmar={self.calmar_ratio:.2f} DD={self.current_drawdown:.1%} "
            f"maxDD={self.max_drawdown:.1%} Kelly={self.half_kelly_pct:.1f}% "
            f"grade={self.grade}"
        )


class PerformanceAnalytics:
    """
    Institutional analytics module for Darwin v5.
    Calcula todo en O(1) usando suficiencias estadísticas.
    Historial máximo: 2000 ticks de equity, 500 trades.
    """

    MAX_HISTORY = 2000
    MAX_TRADES  = 500

    def __init__(self, initial_equity: float = 0.0) -> None:
        # Tick returns: solo para drawdown intraday real
        self._tick_returns: List[float] = []
        # DAILY returns: para Sharpe/Sortino comparables con industria (sqrt(252))
        self._daily_returns: List[float] = []
        self._daily_equity_start: float = initial_equity
        self._ticks_today: int = 0
        self._ticks_per_day: int = int(86400 / 5)  # 17280 ticks de 5s por día

        self._btc_returns:     List[float] = []
        self._btc_daily_start: float = 0.0
        self._trade_pnls:      List[float] = []

        self._peak_equity    = initial_equity
        self._current_equity = initial_equity
        self._initial_equity = initial_equity
        self._last_equity    = initial_equity
        self._last_btc_price = 0.0

        self._n_trades     = 0
        self._n_wins       = 0
        self._gross_profit = 0.0
        self._gross_loss   = 0.0
        self._sum_pnl      = 0.0
        self._max_drawdown = 0.0
        self._cur_drawdown = 0.0

        self._streak          = 0
        self._max_win_streak  = 0
        self._max_loss_streak = 0

    # ── Ingesta ───────────────────────────────────────────────────────────────

    def record_trade(self, pnl_usdt: float, is_win: bool) -> None:
        self._n_trades += 1
        self._sum_pnl  += pnl_usdt

        if is_win:
            self._n_wins       += 1
            self._gross_profit += pnl_usdt
            self._streak = max(0, self._streak) + 1
            self._max_win_streak = max(self._max_win_streak, self._streak)
        else:
            self._gross_loss += abs(pnl_usdt)
            self._streak = min(0, self._streak) - 1
            self._max_loss_streak = max(self._max_loss_streak, abs(self._streak))

        self._trade_pnls.append(pnl_usdt)
        if len(self._trade_pnls) > self.MAX_TRADES:
            self._trade_pnls.pop(0)

    def update_equity(self, equity: float, btc_price: float = 0.0) -> None:
        if equity <= 0:
            return

        # Drawdown (tick-by-tick, máxima precisión)
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            self._cur_drawdown = (self._peak_equity - equity) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, self._cur_drawdown)

        self._current_equity = equity

        # Tick return (para drawdown tracking)
        if self._last_equity > 0:
            r = (equity - self._last_equity) / self._last_equity
            self._tick_returns.append(r)
            if len(self._tick_returns) > self.MAX_HISTORY:
                self._tick_returns.pop(0)
        self._last_equity = equity

        # Acumular retorno diario (cada N ticks = 1 día)
        # Sharpe calculado con retornos diarios → comparable con industria (sqrt(252))
        self._ticks_today += 1
        if self._ticks_today >= self._ticks_per_day:
            if self._daily_equity_start > 0:
                daily_r = (equity - self._daily_equity_start) / self._daily_equity_start
                self._daily_returns.append(daily_r)
                if len(self._daily_returns) > self.MAX_HISTORY:
                    self._daily_returns.pop(0)
                # BTC daily return
                if self._btc_daily_start > 0 and btc_price > 0:
                    btc_dr = (btc_price - self._btc_daily_start) / self._btc_daily_start
                    self._btc_returns.append(btc_dr)
                    if len(self._btc_returns) > self.MAX_HISTORY:
                        self._btc_returns.pop(0)
                    self._btc_daily_start = btc_price
            self._daily_equity_start = equity
            self._ticks_today = 0

        # BTC price tracking
        if btc_price > 0:
            if self._btc_daily_start <= 0:
                self._btc_daily_start = btc_price
            self._last_btc_price = btc_price

    # ── Cálculos ──────────────────────────────────────────────────────────────

    def _mean_std(self, data: List[float]) -> Tuple[float, float]:
        n = len(data)
        if n < 2:
            return 0.0, 0.0
        mu = sum(data) / n
        var = sum((x - mu) ** 2 for x in data) / n
        return mu, var ** 0.5

    def _sharpe(self) -> float:
        """Sharpe anualizado usando retornos DIARIOS (sqrt(252) — estándar industria)."""
        rets = self._daily_returns
        if len(rets) < 5:
            return 0.0  # insuficiente historial diario
        mu, sigma = self._mean_std(rets)
        if sigma < 1e-12:
            return 0.0
        return (mu / sigma) * (252 ** 0.5)

    def _sortino(self) -> float:
        """Sortino anualizado sobre retornos diarios (solo downside deviation)."""
        rets = self._daily_returns
        if len(rets) < 5:
            return 0.0
        mu = sum(rets) / len(rets)
        neg_sq = [r ** 2 for r in rets if r < 0]
        if not neg_sq:
            return 99.0  # sin días negativos
        downside_dev = (sum(neg_sq) / len(rets)) ** 0.5
        if downside_dev < 1e-12:
            return 0.0
        return (mu / downside_dev) * (252 ** 0.5)

    def _calmar(self) -> float:
        if self._max_drawdown < 1e-6 or self._initial_equity <= 0:
            return 0.0
        total_return = (self._current_equity - self._initial_equity) / self._initial_equity
        return total_return / self._max_drawdown

    def _half_kelly(self) -> Tuple[float, float]:
        """Returns (half_kelly_pct, expectancy_per_trade)."""
        if self._n_trades < 10:
            return 2.0, 0.0

        wins   = [p for p in self._trade_pnls if p > 0]
        losses = [abs(p) for p in self._trade_pnls if p < 0]

        if not wins or not losses:
            return 2.0, 0.0

        p       = len(wins) / self._n_trades
        q       = 1.0 - p
        avg_win  = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss < 1e-8:
            return 2.0, 0.0

        b = avg_win / avg_loss
        kelly_f = (p * b - q) / b
        edge    = p * avg_win - q * avg_loss

        if kelly_f <= 0:
            return 0.5, edge  # sin edge: mínimo de seguridad

        # Half-Kelly en % del capital: Kelly da fracción óptima del capital por trade.
        # Multiplicamos por 100 para tener %, luego tomamos la mitad.
        half_k_pct = min(5.0, max(0.5, kelly_f * 50.0))
        return half_k_pct, edge

    def _alpha_ir(self) -> Tuple[float, float]:
        """Alpha e Information Ratio usando retornos diarios vs BTC benchmark."""
        n = min(len(self._daily_returns), len(self._btc_returns))
        if n < 5:
            return 0.0, 0.0
        excess = [
            self._daily_returns[-(n - i)] - self._btc_returns[-(n - i)]
            for i in range(n)
        ]
        mu, sigma = self._mean_std(excess)
        # Anualizar: alpha diario * 252 días * 100 para %
        alpha_annual_pct = mu * 252 * 100.0
        ir = (mu / sigma) * (252 ** 0.5) if sigma > 1e-12 else 0.0
        return alpha_annual_pct, ir

    def _grade(self, sharpe: float, calmar: float, dd: float, pf: float) -> Tuple[str, bool]:
        score = 0
        if sharpe >= SHARPE_HEDGE:     score += 3
        elif sharpe >= SHARPE_MIN:     score += 1
        if calmar >= CALMAR_ELITE:     score += 3
        elif calmar >= CALMAR_MIN:     score += 1
        if dd <= 0.10:                 score += 2
        elif dd <= MAX_DD_SCALABLE:    score += 1
        if pf >= PROFIT_FACTOR_HF:     score += 2
        elif pf >= PROFIT_FACTOR_OK:   score += 1

        if score >= 10:  return "A", True
        if score >= 7:   return "B", True
        if score >= 4:   return "C", False
        if score >= 2:   return "D", False
        return "F", False

    # ── API pública ───────────────────────────────────────────────────────────

    def snapshot(self) -> PerformanceSnapshot:
        n = self._n_trades
        if n == 0:
            return PerformanceSnapshot(
                equity=self._current_equity,
                peak_equity=self._peak_equity,
            )

        wins   = [p for p in self._trade_pnls if p > 0]
        losses = [abs(p) for p in self._trade_pnls if p < 0]

        wr    = self._n_wins / n
        pf    = self._gross_profit / self._gross_loss if self._gross_loss > 0 else 99.0
        avgw  = sum(wins) / len(wins) if wins else 0.0
        avgl  = sum(losses) / len(losses) if losses else 0.0
        exp_  = wr * avgw - (1 - wr) * avgl

        sharpe  = self._sharpe()
        sortino = self._sortino()
        calmar  = self._calmar()
        hk, edge = self._half_kelly()
        alpha, ir = self._alpha_ir()
        grade, scalable = self._grade(sharpe, calmar, self._max_drawdown, pf)

        return PerformanceSnapshot(
            total_trades=n,
            win_trades=self._n_wins,
            loss_trades=n - self._n_wins,
            total_pnl=round(self._sum_pnl, 4),
            equity=self._current_equity,
            peak_equity=self._peak_equity,
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            win_rate=round(wr, 4),
            profit_factor=round(min(pf, 99.0), 3),
            avg_win=round(avgw, 4),
            avg_loss=round(avgl, 4),
            expectancy=round(exp_, 4),
            max_drawdown=round(self._max_drawdown, 4),
            current_drawdown=round(self._cur_drawdown, 4),
            best_trade=round(max(self._trade_pnls), 4),
            worst_trade=round(min(self._trade_pnls), 4),
            alpha_pct=round(alpha, 2),
            information_ratio=round(ir, 3),
            half_kelly_pct=round(hk, 2),
            kelly_edge=round(edge, 4),
            current_streak=self._streak,
            max_win_streak=self._max_win_streak,
            max_loss_streak=self._max_loss_streak,
            grade=grade,
            scalable=scalable,
            computed_at=time.time(),
        )

    def summary_line(self) -> str:
        s = self.snapshot()
        return (
            f"[PERF] trades={s.total_trades} wr={s.win_rate*100:.1f}% "
            f"pf={s.profit_factor:.2f} exp=${s.expectancy:.4f} "
            f"sharpe={s.sharpe_ratio:.2f} sortino={s.sortino_ratio:.2f} "
            f"calmar={s.calmar_ratio:.2f} maxDD={s.max_drawdown*100:.1f}% "
            f"kelly={s.half_kelly_pct:.1f}% "
            f"alpha={s.alpha_pct:+.1f}% IR={s.information_ratio:.2f} "
            f"grade={s.grade} scalable={s.scalable}"
        )

    def circuit_breaker_check(
        self,
        daily_loss_limit_pct: float = 0.05,
        max_dd_limit_pct: float = 0.20,
        max_loss_streak: int = 5,
        daily_pnl_usdt: float = 0.0,
        daily_start_equity: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Evalúa si el trading debe pausarse.
        Returns: (should_halt, reason_string)
        """
        # 1. Daily loss limit
        if daily_start_equity > 0 and daily_pnl_usdt < 0:
            daily_loss_pct = abs(daily_pnl_usdt) / daily_start_equity
            if daily_loss_pct >= daily_loss_limit_pct:
                return True, (
                    f"DAILY_LOSS_LIMIT {daily_loss_pct*100:.1f}% "
                    f">= {daily_loss_limit_pct*100:.0f}%"
                )

        # 2. Max drawdown total
        if self._max_drawdown >= max_dd_limit_pct:
            return True, (
                f"MAX_DRAWDOWN {self._max_drawdown*100:.1f}% "
                f">= {max_dd_limit_pct*100:.0f}%"
            )

        # 3. Loss streak
        if abs(min(self._streak, 0)) >= max_loss_streak:
            return True, f"LOSS_STREAK {abs(self._streak)} consecutive losses"

        return False, ""

    # ── Aliases de compatibilidad con el engine ───────────────────────────────

    def record_equity(self, equity: float, btc_price: float = 0.0) -> None:
        """Alias de update_equity para compatibilidad con llamadas existentes."""
        self.update_equity(equity, btc_price)

    def get_report(self) -> "PerformanceSnapshot":
        """Alias de snapshot() para compatibilidad con llamadas existentes."""
        return self.snapshot()

    # Atributos para exportación de datos en engine._export_data()
    @property
    def _equity_series(self) -> list:
        """Lista de equity points para exportar CSV."""
        return list(self._tick_returns)  # aproximación; el engine exporta esto

    @property
    def _trades(self) -> list:
        """Lista de trade PnLs para exportar CSV."""
        return list(self._trade_pnls)
