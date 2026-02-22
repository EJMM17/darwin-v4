# Darwin v4 — Estudio de Viabilidad para Mercado de Acciones (Stocks)

**Fecha:** 2026-02-22
**Autor:** Analisis automatizado sobre codebase darwin-v4

---

## 1. RESUMEN EJECUTIVO

Darwin v4 es un sistema de trading algoritmico evolutivo disenado **exclusivamente para futuros de criptomonedas** (Binance USDT-M, Bybit). Adaptarlo al mercado de acciones (stocks) es **tecnicamete viable pero requiere cambios sustanciales** en multiples capas del sistema.

**Veredicto:** Se puede reutilizar ~40-50% del codigo (motor evolutivo, gestion de riesgo, simulacion, dashboard). El otro 50-60% (conectores de exchange, senales, datos de mercado, ejecucion) debe reescribirse o adaptarse significativamente.

---

## 2. QUE SE PUEDE REUTILIZAR (sin cambios o con cambios menores)

### 2.1 Motor Evolutivo (~90% reutilizable)
- `darwin_agent/evolution/` — Algoritmo genetico, seleccion por torneo, crossover, mutacion adaptativa
- RQRE (Rolling Quarterly Re-Evolution) — Ciclos de re-entrenamiento
- Fitness scoring — El framework de puntuacion sirve, pero los factores necesitan ajuste

### 2.2 Gestion de Riesgo (~70% reutilizable)
- RBE (Risk Budget Engine) — Position sizing adaptativo al drawdown
- Circuit Breaker — Proteccion ante perdidas consecutivas
- Portfolio heat tracking — Control de exposicion total
- **Excepcion:** Kelly Criterion tiene un bug de doble aplicacion documentado en `REVIEW_QUANT_AUDIT.md` que debe corregirse ANTES de produccion

### 2.3 Simulacion/Backtesting (~80% reutilizable)
- `darwin_agent/simulation/` — Harness de backtesting, replay historico
- Framework de experimentos
- Verificacion determinista (seed -> resultados identicos)

### 2.4 Dashboard (~95% reutilizable)
- `dashboard/` — FastAPI + SQLite + WebSocket
- UI de metricas en tiempo real
- Autenticacion y encriptacion de credenciales

### 2.5 Monitoring (~90% reutilizable)
- `darwin_agent/monitoring/` — Audit trail, Prometheus metrics
- Logging estructurado JSONL

---

## 3. QUE DEBE CAMBIAR (cambios significativos)

### 3.1 Conectores de Exchange — REESCRITURA COMPLETA

**Estado actual:** Adaptadores para Binance Futures y Bybit (`darwin_agent/exchanges/`)
- API firmada con HMAC-SHA256 especifica de Binance
- Endpoints `/fapi/v1/` (futuros perpetuos)
- Conceptos inexistentes en stocks: funding rate, mark price vs last price, leverage configurable

**Necesario para stocks:**
Un nuevo adaptador que implemente `IExchangeAdapter` para un broker de stocks. La interfaz ya esta abstraida en `darwin_agent/interfaces/`, lo cual facilita el cambio.

### 3.2 Generador de Senales — REESCRITURA ~70%

**Estado actual** (`darwin_agent/v5/signal_generator.py`):
- 4 factores: Momentum, Mean Reversion, Residual Alpha, Funding Carry
- El factor **Funding Carry** (20% del peso) NO EXISTE en stocks
- Residual Alpha esta calibrado contra BTC/ETH como proxy de mercado

**Necesario para stocks:**
- Reemplazar Funding Carry por factores relevantes: earnings momentum, sector rotation, o volume profile
- Cambiar proxy de mercado de BTC a SPY/SPX
- Ajustar lookbacks (crypto = 24/7, stocks = solo horario de mercado)

### 3.3 Datos de Mercado — REESCRITURA ~80%

**Estado actual** (`darwin_agent/v5/market_data.py`):
- REST polling cada 5 segundos a APIs de crypto
- Order flow basado en order book de futuros (OBI, TFD)
- Regimen basado en volatilidad de BTC

**Necesario para stocks:**
- Fuente de datos de mercado de acciones (ver seccion 4)
- El order flow en stocks es completamente diferente (dark pools, lit exchanges, NBBO)
- Regimen debe basarse en VIX/SPX, no en BTC

### 3.4 Ejecucion — CAMBIOS SIGNIFICATIVOS

**Estado actual** (`darwin_agent/v5/execution_engine.py`):
- Ordenes MARKET exclusivas (taker fees)
- VISA order manager para SL/TP server-side
- Logica de leverage y margin

**Necesario para stocks:**
- Soporte para ordenes LIMIT (critico para reducir costos)
- Sin leverage configurable por posicion (margin account standard)
- Pattern Day Trader (PDT) rule: minimo $25,000 para day trading en US
- Market hours: solo 09:30-16:00 ET (vs 24/7 crypto)
- T+1 settlement (no acceso inmediato a fondos post-venta)

---

## 4. PLATAFORMAS RECOMENDADAS

### 4.1 Para Trading en Vivo

| Plataforma | Tipo | Comisiones | API | Min. Capital | Mejor para |
|------------|------|-----------|-----|-------------|------------|
| **Interactive Brokers (IBKR)** | Broker institucional | $0.005/accion (tiered) | TWS API / REST | $0 ($25K para day trading) | **RECOMENDADO** — Mas completo |
| **Alpaca** | Broker API-first | $0 (PFOF) / tiered | REST + WebSocket | $0 ($25K PDT) | Prototipo rapido |
| **TD Ameritrade / Schwab** | Broker retail | $0 | thinkorswim API | $0 ($25K PDT) | US retail |
| **Tradier** | Broker API | $0 o $0.35/trade | REST | $0 | Options + stocks |

### 4.2 Para Datos de Mercado

| Fuente | Tipo | Costo/mes | Notas |
|--------|------|-----------|-------|
| **Polygon.io** | Real-time + historico | $29-$199 | Excelente para backtesting |
| **Alpha Vantage** | Historico + delay | Gratis-$50 | Bueno para prototipo |
| **IEX Cloud** | Real-time | $9-$99 | Buena calidad/precio |
| **Yahoo Finance** | Historico (delay) | Gratis | Solo para backtesting |
| **IBKR Market Data** | Real-time incluido | $0-$10 | Si ya usas IBKR |

### 4.3 Recomendacion Principal: **Interactive Brokers + Polygon.io**

**Por que IBKR:**
- API mas robusta del mercado (TWS API o Client Portal REST)
- Acceso a acciones globales (US, EU, Asia)
- Margin rates mas bajos de la industria
- Soporte para ordenes algoritmicas nativas (TWAP, VWAP, Adaptive)
- Paper trading integrado (ideal para validacion)
- Libreria Python oficial: `ib_insync` (async-compatible con Darwin)

**Por que Polygon.io:**
- WebSocket real-time (resuelve el problema de latencia por REST polling)
- Historicos tick-level para backtesting preciso
- API limpia y bien documentada

---

## 5. PRESUPUESTO ESTIMADO

### 5.1 Fase 1: Desarrollo y Backtesting (Mes 1-3)

| Concepto | Costo |
|----------|-------|
| Polygon.io (plan Starter) | $29/mes x 3 = $87 |
| Servidor desarrollo (VPS) | $20/mes x 3 = $60 |
| IBKR Paper Trading | Gratis |
| **Subtotal Fase 1** | **~$150** |

### 5.2 Fase 2: Paper Trading en Vivo (Mes 3-5)

| Concepto | Costo |
|----------|-------|
| Polygon.io (plan Business) | $199/mes x 2 = $398 |
| Servidor produccion (VPS 4GB+) | $40/mes x 2 = $80 |
| IBKR Paper Trading | Gratis |
| **Subtotal Fase 2** | **~$480** |

### 5.3 Fase 3: Trading Real (Mes 5+)

| Concepto | Costo mensual |
|----------|--------------|
| Polygon.io | $199/mes |
| Servidor produccion (dedicado) | $80-$150/mes |
| IBKR Market Data (US stocks) | $0-$10/mes |
| IBKR commissions (estimado 500 trades) | ~$50-$100/mes |
| **Subtotal mensual Fase 3** | **~$330-$460/mes** |

### 5.4 Capital de Trading

| Escenario | Capital minimo | Notas |
|-----------|---------------|-------|
| Sin day trading (swing) | $500-$2,000 | Solo posiciones overnight |
| Day trading US (PDT rule) | **$25,000** | Obligatorio por regulacion FINRA |
| Day trading fuera de US | $1,000-$5,000 | Sin regla PDT (pero otros riesgos) |
| Optimo para el algoritmo evolutivo | **$10,000-$50,000** | Para que las comisiones no erosionen el edge |

### 5.5 Resumen Total

| Fase | Duracion | Costo infra | Capital trading | Total |
|------|----------|-------------|-----------------|-------|
| Backtesting | 3 meses | $150 | $0 | $150 |
| Paper Trading | 2 meses | $480 | $0 | $480 |
| Trading Real (swing) | mensual | ~$330/mes | $2,000-$10,000 | $2,330-$10,330 |
| Trading Real (day trade US) | mensual | ~$460/mes | $25,000+ | $25,460+ |

---

## 6. RIESGOS Y CONSIDERACIONES CRITICAS

### 6.1 Diferencias Fundamentales Crypto vs Stocks

| Aspecto | Crypto (actual) | Stocks |
|---------|----------------|--------|
| Horario | 24/7/365 | 6.5h/dia, L-V |
| Liquidez | Variable, spreads amplios | Alta en large-caps |
| Volatilidad | 3-10% diaria normal | 0.5-2% diaria normal |
| Leverage | 20-125x | 2x (Reg-T) / 4x (intraday) |
| Settlement | Instantaneo | T+1 |
| Comisiones | 0.02-0.08% | $0 o $0.005/accion |
| Regulacion | Minima | SEC, FINRA, PDT |
| Datos | Gratis (exchanges) | Pagados o con delay |

### 6.2 Riesgos del Audit Existente (aplican tambien a stocks)

El `REVIEW_QUANT_AUDIT.md` documenta bugs criticos que deben corregirse **antes** de cualquier migracion:

1. **Double Kelly bug** — Doble aplicacion de Kelly Criterion que causa over-betting
2. **Formula de position sizing incorrecta** — No calcula riesgo real por trade
3. **Monte Carlo validation inutil** — No valida calidad de senales
4. **Thread safety en MarketDataLayer** — Race condition con `requests.Session`

### 6.3 Regulatory (Regulatorio)

- **US:** Si operas >3 day trades en 5 dias con <$25K, tu cuenta se congela (PDT rule)
- **EU:** MiFID II aplica; algunos brokers limitan apalancamiento retail a 5x
- **LATAM:** Regulacion variable; considerar brokers offshore (IBKR permite cuentas desde la mayoria de paises)

---

## 7. PLAN DE MIGRACION PROPUESTO

```
Fase 1 (Backtesting):
  [1] Crear IBKRAdapter que implemente IExchangeAdapter
  [2] Crear PolygonDataAdapter para market_data.py
  [3] Adaptar signal_generator: reemplazar Funding Carry, cambiar proxy a SPX
  [4] Ajustar regime_detector: usar VIX en vez de volatilidad BTC
  [5] Corregir bugs del audit (Kelly double, position sizing)
  [6] Backtest con datos historicos de SPY, AAPL, MSFT, etc.

Fase 2 (Paper Trading):
  [7] Conectar IBKR Paper Trading
  [8] Implementar market hours logic (pre-market, regular, after-hours)
  [9] Implementar T+1 settlement tracking
  [10] Correr 30+ dias de paper trading

Fase 3 (Live):
  [11] Migrar a IBKR live con capital minimo
  [12] Monitorear slippage real vs backtested
  [13] Ajustar parametros evolutivos al regimen de stocks
```

---

## 8. CONCLUSION

**Es viable?** Si, pero no es un cambio trivial. La arquitectura de Darwin v4 tiene buena separacion de capas (interfaces, exchanges, engine), lo que hace la migracion factible. El motor evolutivo y el framework de riesgo son asset-agnostic y se reutilizan directamente.

**Presupuesto minimo realista:**
- Para **backtesting y validacion**: ~$600 (infra 5 meses)
- Para **trading real swing** (sin day trading): ~$2,500-$10,500
- Para **day trading en US**: ~$26,000+ (por la regla PDT de $25K)

**Plataforma recomendada:** Interactive Brokers (broker) + Polygon.io (datos)

**Tiempo estimado de desarrollo:** La complejidad equivale a construir un nuevo adaptador de exchange completo, adaptar las senales de trading, y validar con backtesting extensivo antes de arriesgar capital real.
