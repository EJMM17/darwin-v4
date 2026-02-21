# CLAUDE.md — Darwin v4 Contributor Guide

## Project Overview

Darwin v4 is an autonomous algorithmic trading system for crypto futures markets. It uses genetic algorithms and rolling re-evolution to adapt strategies across symbols (BTC, ETH, SOL). The system trades on Binance USDT-M Futures with a multi-layer risk management stack.

**Key concepts**: evolutionary strategy optimization, population-based agents, deterministic reproducibility, regime-adaptive risk management.

**License**: Proprietary. All rights reserved.

## Repository Structure

```
darwin-v4/
├── darwin_agent/                # Core Python package
│   ├── main.py                  # Entry point (--v5, --test flags)
│   ├── config.py                # YAML/env config loader
│   ├── determinism.py           # Seed locking, deterministic IDs, genome hashing
│   ├── freeze.py                # Genome freeze/export/reproduction verification
│   ├── credentials_loader.py    # Runtime credential loading (env vars / stdin)
│   ├── interfaces/              # Layer 0: zero-dependency contracts
│   │   ├── enums.py             # All enumerations (OrderSide, GrowthPhase, etc.)
│   │   ├── types.py             # Shared dataclasses (Candle, Signal, Position, etc.)
│   │   ├── protocols.py         # Abstract Protocol contracts (IExchangeAdapter, etc.)
│   │   └── events.py            # Event definitions for the event bus
│   ├── core/                    # Core engine (pure logic, no I/O dependencies)
│   │   ├── agent.py             # Individual trading agent
│   │   └── engine.py            # DarwinCoreEngine — position sizing, peak tracking
│   ├── v5/                      # V5 engine (production trading pipeline)
│   │   ├── engine.py            # DarwinV5Engine orchestrator (7-layer pipeline)
│   │   ├── execution_engine.py  # Order validation, placement, retry logic
│   │   ├── market_data.py       # OHLCV fetching, feature engineering
│   │   ├── regime_detector.py   # Market regime classification
│   │   ├── signal_generator.py  # Multi-factor signal generation
│   │   ├── position_sizer.py    # Dynamic sizing with vol/dd scaling
│   │   ├── monte_carlo.py       # Monte Carlo strategy validation
│   │   ├── portfolio_constructor.py
│   │   ├── portfolio_risk.py    # Portfolio-level risk manager
│   │   ├── order_flow.py        # Order flow intelligence (anti-manipulation)
│   │   ├── performance_analytics.py  # Sharpe, Kelly, PnL analytics
│   │   ├── visa_order_manager.py     # Server-side SL/TP via Binance
│   │   ├── websocket_stream.py  # Real-time market data stream
│   │   ├── telemetry.py         # Structured observability
│   │   ├── time_sync.py         # Exchange time synchronization
│   │   └── health_monitor.py    # System health checks
│   ├── evolution/               # Genetic algorithm engine
│   │   ├── engine.py            # GA: tournament selection, crossover, mutation
│   │   ├── fitness.py           # Risk-aware fitness scoring
│   │   ├── rolling_engine.py    # RQRE: 30-day re-evolution cycles
│   │   ├── archive.py           # Genome archive management
│   │   └── diagnostics.py       # Evolution diagnostics
│   ├── portfolio/               # Portfolio management layer
│   │   ├── allocation_engine.py # PAE: volatility-weighted allocation
│   │   ├── risk_budget_engine.py # RBE: drawdown-adaptive sizing
│   │   ├── global_regime.py     # GMRT: BTC regime detection/throttle
│   │   ├── sdl.py               # Strategy diversification logic
│   │   └── vapc.py              # Volatility-adjusted position control
│   ├── risk/                    # Portfolio-level risk engine
│   │   └── portfolio_engine.py  # PortfolioRiskEngine (state machine: NORMAL→DEFENSIVE→CRITICAL→HALTED)
│   ├── exchanges/               # Exchange adapters
│   │   ├── binance.py           # Binance futures adapter
│   │   ├── bybit.py             # Bybit adapter
│   │   └── router.py            # Multi-exchange order routing
│   ├── infrastructure/          # External service clients
│   │   ├── binance_client.py    # Low-level Binance API client
│   │   └── telegram_notifier.py # Telegram notifications
│   ├── infra/                   # Internal infrastructure
│   │   ├── event_bus.py         # Async pub/sub event bus
│   │   ├── redis_cache.py       # Redis checkpoint persistence
│   │   └── postgres.py          # PostgreSQL generation history
│   ├── capital/                 # Capital management
│   │   ├── allocator.py         # Capital allocation across agents
│   │   └── phases.py            # Growth phase transitions
│   ├── orchestrator/            # Agent lifecycle management
│   │   └── agent_manager.py     # Spawn, graduate, kill agents
│   ├── runtime/                 # Runtime service layer
│   │   └── runtime_service.py   # Polling loop, retry, graceful shutdown
│   ├── simulation/              # Backtesting and experiments
│   │   ├── harness.py           # Backtesting harness
│   │   ├── experiment.py        # Experiment framework
│   │   └── scorecard.py         # Performance scorecard generation
│   ├── monitoring/              # Observability
│   │   ├── execution_audit.py   # Order intent vs fill tracking
│   │   ├── audit_logger.py      # JSONL audit trail
│   │   └── integration_snippet.py
│   ├── diagnostics/             # Trading diagnostics
│   │   ├── pre_trade_validator.py
│   │   ├── signal_diagnostic.py
│   │   ├── signal_diagnostics_logger.py
│   │   ├── rejection_reason.py
│   │   └── telegram_debug.py
│   ├── analysis/                # Asset analysis
│   │   └── asset_profiler.py
│   ├── strategies/              # Strategy implementations (placeholder)
│   ├── ml/                      # ML components (placeholder)
│   ├── claudbot/                # Claude bot integration (placeholder)
│   └── dashboard/               # Internal dashboard hooks
├── dashboard/                   # Production web dashboard (FastAPI + SQLite)
│   ├── app.py                   # API endpoints, WebSocket, auth
│   ├── bot_controller.py        # Thread-safe bot lifecycle
│   ├── bot_runtime.py           # Bot runtime management
│   ├── config_loader.py         # Dashboard config
│   ├── crypto_vault.py          # Fernet encryption for API keys
│   ├── database.py              # SQLite schema + CRUD
│   ├── alerts.py                # Alert system
│   ├── dash_logger.py           # Dashboard logging
│   └── templates/index.html     # Dark UI with real-time metrics
├── artifacts/                   # Frozen run artifacts for reproducibility
├── .github/workflows/
│   └── deploy.yml               # CI/CD: test → deploy to DigitalOcean
├── test_*.py                    # Test files (34 files, ~12K lines at root)
├── config_example.yaml          # Full configuration template
├── .env.example                 # Environment variable template
├── requirements.txt             # Core dependencies
├── requirements-deterministic.txt
├── docker-compose.yml           # Agent + Redis + Postgres stack
├── Dockerfile                   # Agent container
├── Dockerfile.dashboard         # Dashboard container
├── Dockerfile.deterministic     # Deterministic verification container
├── deploy.sh                    # DigitalOcean deployment script
├── reproduce_run.py             # Determinism verification tool
├── analyze_dna.py               # Genome analysis utility
└── phase2_risk_compression.py   # Risk compression analysis
```

## Architecture

### Layered Dependency Model

```
Layer 0: interfaces/ (enums, types, protocols, events) — zero dependencies
Layer 1: core/, determinism, freeze — depends only on Layer 0
Layer 2: evolution/, portfolio/, risk/ — depends on Layers 0-1
Layer 3: exchanges/, infrastructure/ — external I/O adapters
Layer 4: v5/ engine — integrates all layers into trading pipeline
Layer 5: runtime/, main.py — application entry points
```

### V5 Engine Pipeline (Production)

The V5 engine (`darwin_agent/v5/engine.py`) runs a 7-stage pipeline per tick:

1. **Market Data** — fetch OHLCV, compute technical features
2. **Feature Engineering** — technical indicators
3. **Regime Detection** — classify market state (trending/ranging/volatile)
4. **Multi-Factor Signals** — generate trade signals with confidence scores
5. **Risk & Position Sizing** — dynamic sizing with volatility/drawdown scaling
6. **Execution** — validated order placement with retry and VISA server-side SL/TP
7. **Telemetry** — structured logging and metrics

### Risk State Machine

The portfolio risk engine uses a state machine:
`NORMAL` → `DEFENSIVE` (8% drawdown) → `CRITICAL` (15% drawdown) → `HALTED` (25% drawdown)

### Growth Phases

Capital management follows phases:
`BOOTSTRAP` ($50-100) → `SCALING` ($100-500) → `ACCELERATION` ($500-2000) → `CONSOLIDATION` (>$2000)

Each phase has different risk percentages and position sizing rules.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for full stack)

### Local Development

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run tests
python -m pytest test_*.py -q --asyncio-mode=auto

# Run specific test suites (as CI does)
python -m pytest test_evolution_fitness.py test_scorecard.py \
    test_simulation.py test_fitness_model.py -q --asyncio-mode=auto

# Run standalone test files (non-pytest)
python test_risk_engine.py
python test_integration.py
```

### Docker

```bash
# Full stack (agent + Redis + Postgres)
docker-compose up -d

# Dashboard only
docker build -t darwin-v4:prod -f Dockerfile.dashboard .
docker run -d -p 8000:8000 -e DASHBOARD_SECRET_KEY="dev-key" darwin-v4:prod
```

### Entry Points

- `python -m darwin_agent.main --v5` — production V5 engine
- `python -m darwin_agent.main --test` — test mode (no live trades)
- `python reproduce_run.py --seed 42` — determinism verification

## Build & Test Commands

```bash
# Run the full test suite
python -m pytest test_*.py -q --asyncio-mode=auto

# Run CI test subset (what GitHub Actions runs)
python -m pytest test_evolution_fitness.py test_scorecard.py \
    test_simulation.py test_fitness_model.py -q --asyncio-mode=auto

# Run a single test file
python -m pytest test_v5_execution_engine.py -q --asyncio-mode=auto

# Run standalone async test suites (not pytest-based)
python test_risk_engine.py
python test_integration.py
python test_fitness_model.py
```

## Testing Patterns

The codebase uses two testing styles:

### 1. Pytest-based Tests (preferred for new tests)
- Uses `pytest` with `pytest-asyncio` for async tests
- Class-based organization: `class TestOrderValidation`
- Async tests use `@pytest.mark.asyncio` decorator
- Mocking via `unittest.mock.MagicMock`, `AsyncMock`, and `patch`
- Example: `test_v5_execution_engine.py`, `test_v5_signal_generator.py`

### 2. Standalone Async Tests (legacy pattern)
- Uses `asyncio.run(main())` with custom test runners
- Print-based output with pass/fail counters
- Hand-rolled mock classes (e.g., `MockExchange`, `MockStrategy`)
- Factory helper functions for test data (e.g., `sig()`, `pos()`, `trade()`)
- Example: `test_risk_engine.py`, `test_integration.py`

### Test Conventions
- Test files are at the repository root, prefixed with `test_`
- Custom mock classes implement Protocol interfaces from `darwin_agent/interfaces/protocols.py`
- Use `sys.path.insert(0, os.path.dirname(__file__))` for local imports in standalone tests
- Tests use factory helpers for common domain objects (Signal, Position, TradeResult)
- Prefer descriptive test names: `test_dry_run_no_exchange_call`, `test_position_limits`

## CI/CD Pipeline

**GitHub Actions** (`.github/workflows/deploy.yml`):
1. **Test job**: Python 3.11, installs deps, runs 4 key test files with pytest
2. **Deploy job** (only on `main`, only if tests pass): SSH to DigitalOcean, `git pull` + `docker-compose build --no-cache` + `docker-compose up -d`, health check
3. **Notifications**: Telegram alerts on success/failure

Tests must pass before any deployment proceeds.

## Code Conventions

### Style
- Python 3.11+ features (`type | None` union syntax, `slots=True` dataclasses)
- `from __future__ import annotations` at the top of all modules
- `@dataclass(slots=True)` or `@dataclass(frozen=True, slots=True)` for value types
- `Protocol` classes with `@runtime_checkable` for interfaces
- snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- No formal linter config — follow the existing implicit style
- Bilingual comments/docstrings (English and Spanish are both used)

### Imports
- Group: stdlib → third-party → local (`darwin_agent.*`)
- Use absolute imports: `from darwin_agent.interfaces.enums import OrderSide`
- Standalone test files use `sys.path.insert(0, os.path.dirname(__file__))`

### Architecture Patterns
- **Protocol-based interfaces**: All external boundaries use `Protocol` ABCs
- **Dataclass value objects**: Domain types are frozen/slotted dataclasses
- **Event bus**: Async pub/sub for decoupled communication between components
- **State machines**: Risk engine, agent lifecycle, growth phases
- **Determinism-first**: All randomness seeded, dicts sorted, UUIDs replaced with hash IDs
- **Dependency injection**: Core engine accepts callables for swappable components

### Commit Messages
Use semantic prefixes with em-dash separators:
- `feat: <description> — <details>`
- `fix: <description> — <details>`
- `ci: <description>`

## Configuration

### Environment Variables (`.env`)
```
BINANCE_API_KEY=...        # Required for live trading
BINANCE_API_SECRET=...     # Required for live trading
TELEGRAM_BOT_TOKEN=...     # Optional notifications
TELEGRAM_CHAT_ID=...       # Optional notifications
DASHBOARD_PORT=8080        # Dashboard port
DASHBOARD_SECRET_KEY=...   # Required for dashboard encryption
```

### YAML Config (`config.yaml`)
Copy `config_example.yaml` to `config.yaml`. Environment variables override YAML values:
- `DARWIN_CAPITAL` → `capital.starting_capital`
- `DARWIN_MODE` → `mode`
- `DARWIN_EXCHANGE` → exchange selection
- `DARWIN_TESTNET` → testnet toggle
- `DARWIN_SYMBOLS` → comma-separated symbol list

### Key Config Sections
- `capital`: Growth phase thresholds and risk percentages
- `risk`: Position limits, stop-loss, drawdown thresholds, circuit breakers
- `evolution`: GA parameters (pool size, mutation rate, crossover rate)
- `exchanges`: API credentials and symbol configuration
- `infra`: Redis, Postgres, dashboard, logging settings

## Important Files to Never Commit

Per `.gitignore`:
- `.env` — contains API keys and secrets
- `config.yaml` — contains credentials
- `*.pem`, `*.key` — private keys
- `data/`, `logs/`, `*.db`, `*.csv` — runtime data

## Dependencies

Core (in `requirements.txt`):
- `aiohttp` — async HTTP for exchange APIs
- `numpy`, `pandas` — numerical computation and data manipulation
- `pybit` — Bybit API SDK
- `prometheus_client` — metrics export
- `requests` — HTTP utilities
- `pyyaml` — configuration parsing

Dashboard: `fastapi`, `uvicorn`, `jinja2`, `cryptography` (see `dashboard/requirements.txt`)

## Domain Glossary

| Acronym | Meaning |
|---------|---------|
| GA | Genetic Algorithm |
| RQRE | Rolling Quarterly Re-Evolution (30-day retrain cycles) |
| RBE | Risk Budget Engine (drawdown-adaptive position sizing) |
| GMRT | Global Market Regime Throttle (BTC regime detection) |
| PAE | Portfolio Allocation Engine (volatility-weighted allocation) |
| SDL | Strategy Diversification Logic |
| VAPC | Volatility-Adjusted Position Control |
| VISA | Virtual Instrument Server-side Attachment (exchange-side SL/TP) |
| DDT | Drawdown Throttle |
| R:R | Risk-to-Reward ratio |
