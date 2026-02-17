# Darwin v4 — Evolutionary Algorithmic Trading System

Autonomous trading system that uses genetic algorithms and rolling re-evolution to adapt strategies across crypto futures markets.

## Architecture

```
darwin_agent/
├── evolution/          # GA engine, fitness scoring, rolling re-evolution (RQRE)
├── portfolio/          # PAE allocation, RBE risk budget, GMRT regime throttle
├── simulation/         # Backtesting harness, historical replay, experiment framework
├── monitoring/         # Execution audit, structured logging, Prometheus metrics
├── determinism.py      # Global determinism lock (seed, RNG, float64)
├── freeze.py           # Genome freeze/export/reproduction verification
└── ...

dashboard/              # Production web dashboard (FastAPI + SQLite)
├── app.py              # API endpoints, WebSocket, auth
├── crypto_vault.py     # Fernet encryption for credentials
├── database.py         # SQLite schema + CRUD
├── bot_controller.py   # Thread-safe bot lifecycle
└── templates/
    └── index.html      # Dark UI, real-time metrics
```

## Key Features

- **Genetic Algorithm Evolution**: Population-based strategy optimization with tournament selection, two-point crossover, and adaptive mutation
- **Rolling Quarterly Re-Evolution (RQRE)**: 30-day cycles that retrain and replace underperforming genomes while preserving winners via fitness gate
- **Deterministic Reproducibility**: Same seed → identical genomes → identical trades → identical equity curve (SHA256 verified)
- **Risk Management Stack**: RBE (drawdown-adaptive position sizing) + GMRT (BTC regime detection) + PAE (volatility-weighted allocation) + Circuit Breaker
- **Execution Audit Layer**: Order intent vs fill tracking, slippage/latency distributions, drift monitoring, 26 Prometheus-style metrics
- **Production Dashboard**: Secure web UI with encrypted credential storage, bot lifecycle control, real-time WebSocket metrics

## Performance (Config C — Production)

| Metric | Value |
|--------|-------|
| CAGR | 184.8% |
| Sharpe | 1.34 |
| Max Drawdown | 54.2% |
| Avg Leverage | 2.04x |
| Period | 2022-2024 (30 months) |

## Determinism Verification

```bash
python reproduce_run.py --seed 42
```

Runs the full pipeline twice and verifies byte-identical output:
- Equity curve SHA256 match
- Genome SHA256 match (per symbol)
- Trade count match
- Bar-by-bar equity match

## Quick Start

### Docker (Recommended)

```bash
# Generate encryption key
export DASHBOARD_SECRET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export DASHBOARD_ADMIN_PASSWORD="your-password"

# Build and run
docker build -t darwin-v4:prod -f Dockerfile.dashboard .
docker run -d --name darwin \
  -p 8000:8000 \
  -e DASHBOARD_SECRET_KEY=$DASHBOARD_SECRET_KEY \
  -e DASHBOARD_ADMIN_PASSWORD=$DASHBOARD_ADMIN_PASSWORD \
  -v darwin-data:/app/data \
  -v darwin-logs:/app/logs \
  darwin-v4:prod
```

### Local Development

```bash
pip install -r requirements-deterministic.txt
pip install -r dashboard/requirements.txt

# Run tests
DASHBOARD_SECRET_KEY="dev-key" python -m pytest test_*.py -q

# Run dashboard
DASHBOARD_SECRET_KEY="dev-key" uvicorn dashboard.app:app --port 8000
```

## Tests

```bash
# Full suite (103 sync tests)
python -m pytest test_*.py -q --tb=short
```

Coverage areas: evolution engine, fitness model, risk budget, portfolio allocation, drawdown throttle, diagnostics, extreme scenarios, VAPC, SDL, execution audit, dashboard (auth, CRUD, lifecycle, WebSocket, CSRF).

## Configuration

Production config (Config C):
- Leverage multiplier: 0.75x of computed value
- RBE dd_limit: 0.28
- Timeframe: 4H
- Exchange: Binance USDT-M Futures
- RQRE interval: 30 days
- Symbols: BTC, SOL, DOGE

## Security

- API keys encrypted with Fernet (AES-128-CBC)
- Passwords hashed with PBKDF2-SHA256 (100k iterations)
- Session cookies: httponly, samesite=strict
- CSRF tokens on all mutations
- CORS restricted to localhost
- All actions logged to JSONL audit trail

## License

Proprietary. All rights reserved.

## Headless Runtime (v4 production profile)

Example first-run interaction:

```text
$ python -m darwin_agent.main --test
Enter runtime credentials for Darwin Engine:
binance_api_key: <your-key>
binance_api_secret: ********
telegram_bot_token: ********
telegram_chat_id: 123456789
equity: 1000.00000000
leverage confirmation: {'BTCUSDT': True, 'ETHUSDT': True, 'SOLUSDT': True}
risk percent: 1.0
test mode completed: no trades opened
```

Example Dockerfile CMD:

```dockerfile
CMD ["python", "-m", "darwin_agent.main"]
```
