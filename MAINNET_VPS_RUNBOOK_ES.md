# Darwin v4 — Guía clara para correr 24/7 en VPS (Binance Mainnet)

## 0) Requisitos mínimos de VPS

Recomendado para correr Darwin 24/7 de forma estable:

- CPU: 2 vCPU (mínimo), 4 vCPU recomendado.
- RAM: 4 GB mínimo, 8 GB recomendado.
- Disco: 30 GB SSD mínimo.
- SO: Ubuntu 22.04 LTS o 24.04 LTS.
- Red: latencia estable y salida HTTPS sin bloqueos a APIs Binance.
- Hora del sistema: NTP activo (muy importante para evitar errores de timestamp).
- Seguridad:
  - firewall activo (UFW)
  - acceso SSH por llave
  - fail2ban recomendado

## 1) Antes de poner API keys (checklist obligatorio)

- [ ] Ejecutar en **testnet** mínimo 48h sin errores críticos.
- [ ] Confirmar que `mode=live` solo se activará cuando todo esté estable.
- [ ] Crear API key de Binance Futures con:
  - **Enable Futures**
  - **NO** habilitar retiro (withdrawals).
  - Restringir por IP al VPS.
- [ ] Definir límites de riesgo realistas en `config.yaml`:
  - `max_total_exposure_pct`
  - `halted_drawdown_pct`
  - `max_consecutive_losses`

## 2) Preparar VPS

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv git
mkdir -p /opt/darwin && cd /opt/darwin
git clone <tu-repo> darwin-v4
cd darwin-v4
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Variables de entorno seguras

Crear `/opt/darwin/darwin-v4/.env`:

```bash
DARWIN_MODE=live
BINANCE_API_KEY=TU_API_KEY
BINANCE_API_SECRET=TU_API_SECRET
LOG_LEVEL=INFO
```

> Nunca subas `.env` al repo.

## 4) Config mínima recomendada para Binance mainnet

Archivo `config.yaml` (ejemplo base):

```yaml
mode: live
capital:
  starting_capital: 100.0
risk:
  max_total_exposure_pct: 40.0
  defensive_drawdown_pct: 6.0
  critical_drawdown_pct: 10.0
  halted_drawdown_pct: 14.0
  max_consecutive_losses: 4
infra:
  tick_interval: 30
  log_level: INFO
exchanges:
  - exchange_id: binance
    enabled: true
    testnet: false
    leverage: 10
    symbols: ["BTCUSDT", "ETHUSDT"]
```

## 5) Preflight antes de dejarlo solo

```bash
cd /opt/darwin/darwin-v4
source .venv/bin/activate
pytest -q test_binance_adapter_safety.py test_router_set_leverage.py test_main_runtime_compat.py
python -m compileall -q darwin_agent
python -m darwin_agent.main --config config.yaml --mode live --tick 30
```

Observa 15-30 minutos logs antes de dejarlo 24/7.

## 6) Ejecutar 24/7 con systemd

Crear `/etc/systemd/system/darwin.service`:

```ini
[Unit]
Description=Darwin Agent v4
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/darwin/darwin-v4
EnvironmentFile=/opt/darwin/darwin-v4/.env
ExecStart=/opt/darwin/darwin-v4/.venv/bin/python -m darwin_agent.main --config /opt/darwin/darwin-v4/config.yaml
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Activar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable darwin
sudo systemctl start darwin
sudo systemctl status darwin --no-pager
```

Logs en vivo:

```bash
journalctl -u darwin -f
```

## 7) Operación diaria (runbook corto)

- Revisar cada mañana:
  - PnL
  - drawdown
  - número de errores de exchange en logs
- Si hay errores repetidos (`timestamp`, `recvWindow`, `insufficient margin`):
  1. parar servicio
  2. revisar reloj/NTP, margen y límites
  3. reiniciar con tamaño de riesgo menor

## 8) Kill switch manual

Paro inmediato:

```bash
sudo systemctl stop darwin
```

Si quieres hard stop por riesgo, baja temporalmente:
- `max_total_exposure_pct` a 0-5
- o cambiar `mode: test` mientras investigas.

## 9) Actualizar sin downtime largo

```bash
cd /opt/darwin/darwin-v4
sudo systemctl stop darwin
git pull
source .venv/bin/activate
pip install -r requirements.txt
pytest -q test_binance_adapter_safety.py test_router_set_leverage.py test_main_runtime_compat.py
sudo systemctl start darwin
```

---

Si sigues esta guía, el siguiente paso de meter API keys queda controlado y con salvaguardas operativas.
