"""Unified runtime config loader with priority ENV > DB > file defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from darwin_agent.config import DarwinConfig, load_config

SUPPORTED_ENV_KEYS = (
    "DARWIN_EXCHANGE",
    "DARWIN_TESTNET",
    "DARWIN_LEVERAGE",
    "DARWIN_RISK_PERCENT",
    "DARWIN_SYMBOLS",
    "DARWIN_MODE",
)


@dataclass
class RuntimeConfigLoader:
    config_path: Optional[str] = None
    db_overrides: Optional[Dict[str, Any]] = None

    def load(self) -> DarwinConfig:
        cfg = load_config(self.config_path or os.getenv("DARWIN_CONFIG", "config.yaml"))
        self._apply_db_overrides(cfg)
        self._apply_env_overrides(cfg)
        return cfg

    def _apply_db_overrides(self, cfg: DarwinConfig) -> None:
        if not self.db_overrides:
            return
        exchange = self.db_overrides.get("exchange")
        if exchange and cfg.exchanges:
            cfg.exchanges[0].exchange_id = str(exchange).strip().lower()
        if "testnet" in self.db_overrides and cfg.exchanges:
            cfg.exchanges[0].testnet = bool(self.db_overrides.get("testnet"))
        if "leverage" in self.db_overrides and cfg.exchanges:
            cfg.exchanges[0].leverage = int(self.db_overrides.get("leverage") or cfg.exchanges[0].leverage)
        if "risk_percent" in self.db_overrides:
            cfg.risk.max_position_pct = float(self.db_overrides.get("risk_percent") or cfg.risk.max_position_pct)
        if "symbols" in self.db_overrides and cfg.exchanges:
            cfg.exchanges[0].symbols = self._parse_symbols(self.db_overrides.get("symbols"))
        if "mode" in self.db_overrides:
            cfg.mode = str(self.db_overrides.get("mode") or cfg.mode).lower()

    def _apply_env_overrides(self, cfg: DarwinConfig) -> None:
        exchange = os.getenv("DARWIN_EXCHANGE")
        if exchange and cfg.exchanges:
            cfg.exchanges[0].exchange_id = exchange.strip().lower()

        testnet = os.getenv("DARWIN_TESTNET")
        if testnet is not None and cfg.exchanges:
            cfg.exchanges[0].testnet = testnet.strip().lower() in {"1", "true", "yes", "on"}

        lev_raw = os.getenv("DARWIN_LEVERAGE")
        if lev_raw is not None and cfg.exchanges:
            try:
                cfg.exchanges[0].leverage = int(lev_raw)
            except ValueError:
                pass

        risk_raw = os.getenv("DARWIN_RISK_PERCENT")
        if risk_raw is not None:
            try:
                cfg.risk.max_position_pct = float(risk_raw)
            except ValueError:
                pass

        symbols = os.getenv("DARWIN_SYMBOLS")
        if symbols is not None and cfg.exchanges:
            parsed = self._parse_symbols(symbols)
            if parsed:
                cfg.exchanges[0].symbols = parsed

        mode = os.getenv("DARWIN_MODE")
        if mode:
            cfg.mode = mode.strip().lower()

    @staticmethod
    def _parse_symbols(raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [s.strip().upper() for s in raw.split(",") if s.strip()]
        if isinstance(raw, Iterable):
            return [str(s).strip().upper() for s in raw if str(s).strip()]
        return []
