#!/bin/bash
# ═══════════════════════════════════════════════════════════
# 🧬 Darwin Agent v4.0 — DigitalOcean Deployment
# ═══════════════════════════════════════════════════════════
#
# ONE-COMMAND DEPLOY:
#   curl -sSL https://raw.githubusercontent.com/YOU/darwin/main/deploy.sh | bash
#
# OR manual:
#   bash deploy.sh
#
# REQUIREMENTS:
#   - Ubuntu 22.04+ droplet ($6/mo minimum, $12/mo recommended)
#   - Root or sudo access
#
# WHAT IT DOES:
#   1. Installs Docker + Docker Compose
#   2. Creates project directory
#   3. Sets up config from template
#   4. Starts Darwin Agent + Redis + Postgres
#   5. Enables auto-restart on reboot
#
# ═══════════════════════════════════════════════════════════

set -euo pipefail

INSTALL_DIR="$HOME/darwin_agent"
REPO_URL=""  # Set if using git

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  🧬 Darwin Agent v4.0 — Deployment                   ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Docker ────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  ✅ Docker installed"
else
    echo "  ✅ Docker already installed"
fi

if ! docker compose version &>/dev/null; then
    echo "📦 Installing Docker Compose plugin..."
    sudo apt-get update -qq
    sudo apt-get install -y docker-compose-plugin
    echo "  ✅ Docker Compose installed"
else
    echo "  ✅ Docker Compose already installed"
fi

# ── 2. Project directory ────────────────────────────────
echo ""
echo "📁 Setting up project at $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# ── 3. Configuration ────────────────────────────────────
if [ ! -f config.yaml ]; then
    if [ -f config_example.yaml ]; then
        cp config_example.yaml config.yaml
    else
        echo "  ⚠️  No config_example.yaml found. Creating minimal config..."
        cat > config.yaml << 'CFGEOF'
mode: test
capital:
  starting_capital: 50.0
exchanges:
  - exchange_id: bybit
    api_key: ""
    api_secret: ""
    testnet: true
    enabled: true
    leverage: 20
    symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
infra:
  redis_enabled: true
  postgres_enabled: true
  tick_interval: 60.0
CFGEOF
    fi
    echo ""
    echo "  ⚠️  IMPORTANT: Edit config.yaml with your API keys!"
    echo "     nano $INSTALL_DIR/config.yaml"
    echo ""
fi

# ── 4. Environment file ─────────────────────────────────
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
# Darwin Agent Environment Variables
# These override config.yaml values

# Exchange API Keys (REQUIRED for live trading)
# BYBIT_API_KEY=your_api_key_here
# BYBIT_API_SECRET=your_api_secret_here

# Trading Mode: test or live
DARWIN_MODE=test

# Starting Capital
DARWIN_CAPITAL=50

# Dashboard Port
DASHBOARD_PORT=8080

# Log Level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
ENVEOF
    echo "  📝 Created .env file. Edit with your API keys:"
    echo "     nano $INSTALL_DIR/.env"
fi

# ── 5. Start services ───────────────────────────────────
echo ""
echo "🚀 Starting Darwin Agent..."

docker compose pull 2>/dev/null || true
docker compose up -d --build

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Darwin Agent v4.0 is running!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  📊 Dashboard:  http://$(hostname -I | awk '{print $1}'):8080"
echo "  📋 Logs:       docker compose logs -f darwin"
echo "  🛑 Stop:       docker compose down"
echo "  🔄 Restart:    docker compose restart darwin"
echo "  📝 Config:     nano $INSTALL_DIR/config.yaml"
echo "  🔑 API Keys:   nano $INSTALL_DIR/.env"
echo ""
echo "  ⚠️  NEXT STEPS:"
echo "  1. Set your Bybit API keys in .env"
echo "  2. Restart: docker compose restart darwin"
echo "  3. Watch:   docker compose logs -f darwin"
echo ""
echo "  ⚠️  Start with TESTNET before going live!"
echo "═══════════════════════════════════════════════════════"
