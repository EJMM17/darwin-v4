# CLAUDE.md — Darwin v4 AI Assistant Guide

## Project Overview

Darwin v4 is an autonomous algorithmic trading system for crypto futures markets. It uses genetic algorithms with rolling re-evolution to adapt strategies across Binance USDT-M futures. The system trades BTC, ETH, and SOL with multi-layered risk management.

**Language:** Python 3.11+ (3.12 for deterministic builds)
**License:** Proprietary

## Repository Structure

```
darwin-v4/
├── darwin_agent/              # Core trading engine package
│   ├── interfaces/            # Layer 0: enums, types, protocols (zero deps)
│   ├── core/                  # Core engine (pure logic, no I/O)
│   ├── evolution/             # GA engine, fitness scoring, rolling re-evolution
│   ├── portfolio/             # PAE allocation, RBE risk budget, GMRT regime throttle, SDL, VAPC
│   ├── simulation/            # Backtesting harness, historical replay, experiments
│   ├── monitoring/            # Execution audit, structured logging, Prometheus metrics
│   ├── diagnostics/           # Signal diagnostics, pre-trade validation, rejection tracking
│   ├── exchanges/             # Exchange adapters (Binance, Bybit) + router
│   ├── infrastructure/        # Binance client, Telegram notifier
│   ├── infra/                 # Event bus, Redis cache, Postgres persistence
│   ├── capital/               # Capital allocation, growth phase management
│   ├── orchestrator/          # Agent lifecycle manager
│   ├── runtime/               # Runtime service (v4 live trading loop)
│   ├── risk/                  # Portfolio-level risk engine
│   ├── analysis/              # Asset profiler (cluster detection)
│   ├── v5/                    # v5 engine (production, tick-based pipeline)
│   ├── strategies/            # Strategy implementations
│   ├── config.py              # YAML config loader with env var overrides
│   ├── determinism.py         # Global determinism lock (seed, RNG, float64)
│   ├── freeze.py              # Genome freeze/export/reproduction verification
│   ├── main.py                # CLI entry point (--test, --v4, --v5, --dry-run)
│   └── credentials_loader.py  # Secure credential loading
├── dashboard/                 # Production web dashboard
│   ├── app.py                 # FastAPI endpoints, WebSocket, auth, Prometheus
│   ├── database.py            # SQLite schema + CRUD
│   ├── bot_controller.py      # Thread-safe bot lifecycle
│   ├── bot_runtime.py         # Runtime manager for dashboard-launched bots
│   ├── crypto_vault.py        # Fernet encryption for API credentials
│   ├── config_loader.py       # Dashboard config
│   ├── alerts.py              # Alert system
│   ├── dash_logger.py         # Dashboard-specific logging
│   └── templates/index.html   # Dark UI, real-time metrics
├── artifacts/                 # Frozen run artifacts for reproducibility
├── test_*.py                  # Test suite (~30 test files at root level)
├── reproduce_run.py           # Determinism verification (dual-run SHA256 match)
├── analyze_dna.py             # DNA analysis utility
├── phase2_risk_compression.py # Risk compression analysis
├── config_example.yaml        # Full configuration reference
├── .env.example               # Environment variable template
├── requirements.txt           # Runtime dependencies
├── requirements-deterministic.txt  # Pinned versions for reproducibility
├── Dockerfile                 # Main agent container
├── Dockerfile.dashboard       # Dashboard container
├── Dockerfile.deterministic   # Reproducibility-verified container
├── docker-compose.yml         # Full stack (agent + Redis + Postgres)
├── deploy.sh                  # DigitalOcean deployment script
└── .github/workflows/deploy.yml  # CI/CD pipeline
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

### Run Tests

```bash
# Full suite
DASHBOARD_SECRET_KEY="dev-key" python -m pytest test_*.py -q --tb=short

# CI subset (what deploy.yml runs)
python -m pytest test_evolution_fitness.py test_scorecard.py \
  test_simulation.py test_fitness_model.py \
  -q --asyncio-mode=auto
```

### Run the Engine

```bash
# Test mode (validates connectivity, no trades)
python -m darwin_agent.main --test

# v5 engine (default, production)
python -m darwin_agent.main --v5

# v5 dry-run (signals only, no orders)
python -m darwin_agent.main --dry-run

# Legacy v4 engine
python -m darwin_agent.main --v4
```

### Run the Dashboard

```bash
DASHBOARD_SECRET_KEY="dev-key" uvicorn dashboard.app:app --port 8000
```

## Architecture

### Layered Dependency Graph

The codebase follows a strict layered architecture where dependencies only flow downward:

```
Layer 0: interfaces/          (enums, types, protocols — zero deps)
Layer 1: determinism, freeze  (stdlib only)
Layer 2: evolution/fitness    (depends on Layer 0)
Layer 3: portfolio/           (PAE, RBE, GMRT, SDL, VAPC — depends on Layer 0)
Layer 4: simulation/          (depends on Layers 0-3)
Layer 5: evolution/engine     (depends on Layers 0, 2)
Layer 6: core/, exchanges/    (depends on Layers 0-1)
Layer 7: runtime/, v5/        (depends on all above)
Layer 8: dashboard/           (FastAPI, depends on monitoring/)
```

### Two Engine Versions

- **v4 engine** (`darwin_agent/core/engine.py` + `runtime/runtime_service.py`): Legacy engine with `DarwinCoreEngine` as a pure-logic pipeline. Functional composition via callable hooks (evolutionary_logic, rqre, gmrt, pae, risk_management).
- **v5 engine** (`darwin_agent/v5/engine.py`): Production engine. Tick-based pipeline: Market Data -> Features -> Regime Detection -> Signal Generation -> Position Sizing -> Execution. Includes order flow intelligence, Monte Carlo validation, trailing stops, kill switches, and institutional analytics.

### Key Subsystems

| Acronym | Full Name | Module | Purpose |
|---------|-----------|--------|---------|
| GA | Genetic Algorithm | `evolution/engine.py` | Population-based strategy optimization |
| RQRE | Rolling Quarterly Re-Evolution | `evolution/rolling_engine.py` | 30-day cycles replacing underperformers |
| RBE | Risk Budget Engine | `portfolio/risk_budget_engine.py` | Drawdown-adaptive position sizing multiplier |
| PAE | Portfolio Allocation Engine | `portfolio/allocation_engine.py` | Volatility-weighted capital allocation |
| GMRT | Global Market Regime Throttle | `portfolio/global_regime.py` | BTC-based macro regime scaling |
| SDL | Signal Diagnostics Layer | `portfolio/sdl.py` | Signal quality monitoring |
| VAPC | Volatility-Adjusted Position Control | `portfolio/vapc.py` | Vol-scaled position management |

### Evolution Pipeline

1. `EvolutionEngine.seed_initial_population()` creates generation-0 DNA (optionally cluster-biased)
2. Agents trade for N bars using their DNA genes
3. `evaluate_generation()` re-scores all agents via `RiskAwareFitness`
4. `select_survivors()` picks top performers (elitism + tournament selection)
5. `breed()` (two-point crossover) + `mutate()` (Gaussian with adaptive decay) creates offspring
6. `create_next_generation()` fills the next population

### Risk Management Stack

Position sizing flows through multiple multiplicative layers:
```
final_size = capital * risk_pct * pae_weight * gmrt_mult * rbe_mult
```

Kill switches in v5:
- **Daily Loss Limit**: Halts trading if daily loss exceeds configured threshold (resets on UTC midnight)
- **Max Drawdown Halt**: Halts trading if peak-to-trough drawdown exceeds limit (requires manual restart)

### Determinism

Byte-level reproducibility is a core requirement. The determinism system:
- Seeds all RNGs (stdlib `random`, `numpy`, `torch`) via `lock_determinism(seed)`
- Replaces `uuid4()` with `deterministic_id()` (SHA256 from seed + counter)
- Sorts all dict iterations via `sorted_items()`
- Uses SHA256 tie-breaking in fitness ranking via `stable_sort_by_fitness()`
- Verifies with `reproduce_run.py`: runs full pipeline twice, compares SHA256 of equity curves and genomes

### Dashboard

FastAPI application providing:
- Session-based auth (PBKDF2-SHA256 passwords, httponly cookies)
- CSRF token protection on mutations
- Encrypted credential storage (Fernet/AES-128-CBC via `CryptoVault`)
- WebSocket real-time metrics
- Prometheus `/metrics` endpoint
- Bot lifecycle control (start/stop/restart)
- SQLite persistence

## Key Data Types

Defined in `darwin_agent/interfaces/types.py`:

- `Candle` — OHLCV bar (frozen dataclass)
- `Signal` — Trade signal with strategy, direction, confidence (frozen)
- `DNAData` — Genome genes dict + metadata
- `AgentEvalData` — Evaluation packet: metrics + pnl_series + exposure + DNA
- `OrderRequest` / `OrderResult` — Order lifecycle
- `Position` — Open position tracking
- `TradeResult` — Closed trade record
- `PortfolioRiskMetrics` — Portfolio-level risk snapshot
- `GenerationSnapshot` — Full generation record for persistence

Key enums in `darwin_agent/interfaces/enums.py`:
- `AgentPhase`: INCUBATION -> LIVE -> DYING -> DEAD
- `GrowthPhase`: BOOTSTRAP -> SCALING -> ACCELERATION -> CONSOLIDATION
- `PortfolioRiskState`: NORMAL -> DEFENSIVE -> CRITICAL -> HALTED
- `MarketRegime`: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY
- `StrategyID`: MOMENTUM, MEAN_REVERSION, SCALPING, BREAKOUT

## Configuration

Configuration priority: **environment variables > YAML file > defaults**

Key env vars:
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` — Exchange credentials
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — Notifications
- `DARWIN_MODE` — `live` or `paper`
- `DARWIN_SYMBOLS` — Comma-separated symbol list (default: BTCUSDT,ETHUSDT,SOLUSDT)
- `DARWIN_LEVERAGE` — Leverage multiplier (clamped 1-20)
- `DARWIN_RISK_PERCENT` — Base risk percentage
- `DASHBOARD_SECRET_KEY` — Required for dashboard (Fernet encryption key)
- `DASHBOARD_ADMIN_PASSWORD` — Dashboard admin password

Config is loaded via `darwin_agent/config.py:load_config()` which returns a `DarwinConfig` dataclass.

## Testing Conventions

- **Framework:** pytest + pytest-asyncio
- **Location:** All test files at repository root, named `test_*.py`
- **Pattern:** Tests use inline mock Protocol implementations (no external mocking library)
- **Async tests:** Use `asyncio.run()` or `pytest-asyncio` auto mode
- **Determinism:** Tests seed RNGs explicitly for reproducibility
- **Dashboard tests:** Require `DASHBOARD_SECRET_KEY` env var
- **No external deps:** Tests never call real exchanges; all I/O is mocked

Test file organization:
- `test_evolution_fitness.py` — Fitness model scoring
- `test_scorecard.py` — Simulation scorecard
- `test_simulation.py` — Simulation harness
- `test_integration.py` — Multi-agent lifecycle with mock protocols
- `test_risk_engine.py` — Portfolio risk engine
- `test_dashboard.py` — Dashboard auth, CRUD, lifecycle, WebSocket, CSRF
- `test_bot_runtime.py` — Bot runtime manager
- `test_extreme_scenarios.py` — Edge cases and stress tests
- `test_v5_*.py` — v5 engine component tests

Run a specific test file:
```bash
python -m pytest test_evolution_fitness.py -v
```

## CI/CD

GitHub Actions workflow (`.github/workflows/deploy.yml`):
1. **Test job:** Runs on `ubuntu-latest`, Python 3.11, executes core test subset
2. **Deploy job:** SSHs into DigitalOcean server, runs `git pull` + `docker-compose build --no-cache` + `docker-compose up -d`
3. **Notifications:** Telegram notifications on deploy success/failure

Triggered on: push to `main`, manual `workflow_dispatch`

## Docker

Three Dockerfiles:
- `Dockerfile` — Main agent image
- `Dockerfile.dashboard` — Dashboard with encrypted credentials
- `Dockerfile.deterministic` — Pinned deps for reproducibility verification

docker-compose.yml runs: `darwin-agent` + `redis` (checkpoint persistence) + `postgres` (generation history)

## Coding Conventions

- **Type hints:** All function signatures use type hints (`from __future__ import annotations`)
- **Dataclasses:** Prefer `@dataclass(slots=True)` or `@dataclass(frozen=True, slots=True)` for value types
- **Protocols:** Abstract contracts via `typing.Protocol` with `@runtime_checkable`
- **Logging:** Use `logging.getLogger("darwin.<module>")` hierarchy
- **Determinism:** Always iterate dicts with `sorted()`, use `deterministic_id()` not `uuid4()`
- **Layer isolation:** Modules declare their layer in docstrings; dependencies only flow downward
- **No side effects in constructors:** I/O happens in explicit `initialize()` or `run()` methods
- **Module docstrings:** Every module has a docstring documenting its purpose, layer, dependencies, and usage

## Security Considerations

- Never commit `.env`, `config.yaml`, `*.pem`, `*.key`, or `*.db` files
- API keys must come from environment variables, never hardcoded
- Dashboard uses Fernet encryption (AES-128-CBC) for stored credentials
- Passwords hashed with PBKDF2-SHA256 (100k iterations)
- CORS restricted to localhost
- All dashboard mutations require CSRF tokens

## Common Workflows

### Adding a New Gene

1. Add range to `DEFAULT_GENE_RANGES` in `darwin_agent/evolution/engine.py`
2. Use the gene in `SimulatedAgent` logic in `darwin_agent/simulation/harness.py`
3. Optionally add cluster bias in `EvolutionEngine.CLUSTER_BIASES`
4. Add test coverage in `test_evolution_fitness.py`

### Adding a New Exchange

1. Create adapter in `darwin_agent/exchanges/` implementing `IExchangeAdapter` protocol
2. Add `ExchangeID` variant to `darwin_agent/interfaces/enums.py`
3. Register in `darwin_agent/exchanges/router.py`
4. Add config support in `darwin_agent/config.py`

### Modifying Risk Parameters

1. Adjust `RiskConfig` defaults in `darwin_agent/config.py`
2. Or override via `config.yaml` or environment variables
3. Run simulation suite to validate: `python -m pytest test_simulation.py test_risk_engine.py -v`

### Running Determinism Verification

```bash
python reproduce_run.py --seed 42
```

Verifies: equity curve SHA256, genome SHA256, trade count, bar-by-bar equity match across two identical runs.
