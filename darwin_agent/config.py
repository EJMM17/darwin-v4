"""
Darwin v4 — Configuration loader.

Loads from YAML file with environment variable overrides.
All secrets can be set via env vars (never hardcode API keys).

Usage:
    config = load_config("config.yaml")
    config = load_config()  # defaults only (testnet)
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("darwin.config")


# ═════════════════════════════════════════════════════════════
# Config dataclasses
# ═════════════════════════════════════════════════════════════

@dataclass
class ExchangeConfig:
    exchange_id: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    futures_type: str = "USDT-M"
    enabled: bool = True
    leverage: int = 5
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
    ])

@dataclass
class RiskConfig:
    max_position_pct: float = 2.0
    max_open_positions: int = 3
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 5.0
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 3.0
    trailing_stop_pct: float = 1.0
    defensive_drawdown_pct: float = 8.0
    critical_drawdown_pct: float = 15.0
    halted_drawdown_pct: float = 25.0
    max_consecutive_losses: int = 5
    max_total_exposure_pct: float = 80.0

@dataclass
class EvolutionConfig:
    pool_size: int = 5
    survival_rate: float = 0.5
    elitism_count: int = 1
    tournament_size: int = 3
    mutation_rate: float = 0.15
    mutation_decay: float = 0.995
    crossover_rate: float = 0.7
    generation_trade_limit: int = 50
    incubation_trades: int = 10

@dataclass
class CapitalConfig:
    starting_capital: float = 50.0
    # Phase thresholds
    bootstrap_target: float = 100.0
    scaling_target: float = 500.0
    acceleration_target: float = 2000.0
    # Phase risk parameters
    bootstrap_risk_pct: float = 2.0
    scaling_risk_pct: float = 3.0
    acceleration_risk_pct: float = 4.0
    consolidation_risk_pct: float = 1.5

@dataclass
class InfraConfig:
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False
    # Postgres
    postgres_dsn: str = "postgresql://darwin:darwin@localhost:5432/darwin"
    postgres_enabled: bool = False
    # Dashboard
    dashboard_port: int = 8080
    dashboard_enabled: bool = True
    # Logging
    log_level: str = "INFO"
    log_file: str = "darwin.log"
    # Tick interval (seconds)
    tick_interval: float = 60.0
    # Signal diagnostics (observability only — does not affect trading logic)
    enable_signal_diagnostics: bool = True
    telegram_debug: bool = False

@dataclass
class DarwinConfig:
    """Top-level configuration."""
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    exchanges: List[ExchangeConfig] = field(default_factory=lambda: [
        ExchangeConfig(exchange_id="binance"),
    ])
    mode: str = "live"  # "paper" or "live"

    @property
    def starting_capital(self) -> float:
        return self.capital.starting_capital

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.capital.starting_capital <= 0:
            errors.append("starting_capital must be > 0")
        if self.mode == "live":
            for ex in self.exchanges:
                if ex.enabled and not ex.api_key:
                    errors.append(f"Exchange {ex.exchange_id}: api_key required for live mode")
                if ex.enabled and not ex.api_secret:
                    errors.append(f"Exchange {ex.exchange_id}: api_secret required for live mode")
        if self.evolution.pool_size < 2:
            errors.append("evolution.pool_size must be >= 2")
        if not (0 < self.evolution.survival_rate < 1):
            errors.append("evolution.survival_rate must be in (0, 1)")
        return errors


# ═════════════════════════════════════════════════════════════
# Loader
# ═════════════════════════════════════════════════════════════

def load_config(path: str | None = None) -> DarwinConfig:
    """
    Load config from YAML file with env var overrides.

    Priority: env vars > YAML file > defaults
    """
    raw: Dict[str, Any] = {}

    if path and Path(path).exists():
        try:
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            logger.info("loaded config from %s", path)
        except ImportError:
            logger.warning("pyyaml not installed, using defaults")
        except Exception as exc:
            logger.warning("failed to load %s: %s, using defaults", path, exc)

    config = DarwinConfig()

    def _env_bool(name: str, default: bool) -> bool:
        raw_val = os.getenv(name)
        if raw_val is None:
            return default
        return raw_val.strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        raw_val = os.getenv(name)
        if raw_val is None or raw_val.strip() == "":
            return default
        try:
            return float(raw_val)
        except ValueError:
            logger.warning("invalid float in %s=%r; using default=%s", name, raw_val, default)
            return default

    def _env_int(name: str, default: int) -> int:
        raw_val = os.getenv(name)
        if raw_val is None or raw_val.strip() == "":
            return default
        try:
            return int(raw_val)
        except ValueError:
            logger.warning("invalid int in %s=%r; using default=%s", name, raw_val, default)
            return default

    # Capital
    cap = raw.get("capital", raw)  # support flat or nested
    config.capital.starting_capital = float(
        os.getenv("DARWIN_CAPITAL", cap.get("starting_capital", 50.0)))

    # Risk
    risk = raw.get("risk", {})
    for field_name in ("max_position_pct", "max_open_positions", "stop_loss_pct",
                       "take_profit_pct", "trailing_stop_pct", "defensive_drawdown_pct",
                       "critical_drawdown_pct", "halted_drawdown_pct",
                       "max_consecutive_losses", "max_total_exposure_pct"):
        if field_name in risk:
            setattr(config.risk, field_name, type(getattr(config.risk, field_name))(risk[field_name]))
    config.risk.max_position_pct = _env_float("DARWIN_RISK_PERCENT", config.risk.max_position_pct)

    # Evolution
    evo = raw.get("evolution", {})
    for field_name in ("pool_size", "survival_rate", "mutation_rate",
                       "generation_trade_limit", "incubation_trades"):
        if field_name in evo:
            setattr(config.evolution, field_name,
                    type(getattr(config.evolution, field_name))(evo[field_name]))

    # Infra
    infra = raw.get("infra", {})
    config.infra.redis_url = os.getenv("REDIS_URL", infra.get("redis_url", config.infra.redis_url))
    config.infra.redis_enabled = infra.get("redis_enabled", config.infra.redis_enabled)
    config.infra.postgres_dsn = os.getenv("DATABASE_URL",
        infra.get("postgres_dsn", config.infra.postgres_dsn))
    config.infra.postgres_enabled = infra.get("postgres_enabled", config.infra.postgres_enabled)
    config.infra.dashboard_port = int(os.getenv("DASHBOARD_PORT",
        infra.get("dashboard_port", 8080)))
    config.infra.tick_interval = float(infra.get("tick_interval", 60.0))
    config.infra.log_level = os.getenv("LOG_LEVEL", infra.get("log_level", "INFO"))
    config.infra.enable_signal_diagnostics = _env_bool(
        "ENABLE_SIGNAL_DIAGNOSTICS",
        infra.get("enable_signal_diagnostics", True),
    )
    config.infra.telegram_debug = _env_bool(
        "TELEGRAM_DEBUG",
        infra.get("telegram_debug", False),
    )

    # Exchanges
    exchanges_raw = raw.get("exchanges", [])
    if exchanges_raw:
        config.exchanges = []
        for ex in exchanges_raw:
            exchange_id = os.getenv("DARWIN_EXCHANGE", ex.get("exchange_id", "binance"))
            exc = ExchangeConfig(
                exchange_id=exchange_id,
                api_key=os.getenv(f"{exchange_id.upper()}_API_KEY",
                                  ex.get("api_key", "")),
                api_secret=os.getenv(f"{exchange_id.upper()}_API_SECRET",
                                     ex.get("api_secret", "")),
                testnet=_env_bool("DARWIN_TESTNET", bool(ex.get("testnet", False))),
                futures_type=os.getenv("DARWIN_FUTURES_TYPE", ex.get("futures_type", "USDT-M")),
                enabled=ex.get("enabled", True),
                leverage=max(1, min(5, _env_int("DARWIN_LEVERAGE", int(ex.get("leverage", 5))))),
                symbols=ex.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
            )
            config.exchanges.append(exc)
    else:
        # Default: check env vars
        default_exchange = os.getenv("DARWIN_EXCHANGE", config.exchanges[0].exchange_id)
        config.exchanges[0].exchange_id = default_exchange
        config.exchanges[0].testnet = _env_bool("DARWIN_TESTNET", config.exchanges[0].testnet)
        config.exchanges[0].futures_type = os.getenv("DARWIN_FUTURES_TYPE", config.exchanges[0].futures_type)
        config.exchanges[0].api_key = os.getenv(f"{default_exchange.upper()}_API_KEY", "")
        config.exchanges[0].api_secret = os.getenv(f"{default_exchange.upper()}_API_SECRET", "")
        config.exchanges[0].leverage = max(1, min(5, _env_int("DARWIN_LEVERAGE", config.exchanges[0].leverage)))

    config.mode = os.getenv("DARWIN_MODE", raw.get("mode", "live"))

    return config
