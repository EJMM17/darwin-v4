"""
Darwin v4 Dashboard — FastAPI Backend.

Secure operational dashboard. Monitoring + control only.
Does NOT modify trading logic, genomes, or risk parameters.

Run:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Request, Response,
    HTTPException, Depends, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, REGISTRY, generate_latest
from pydantic import BaseModel, Field

# Dashboard modules
from dashboard.crypto_vault import CryptoVault
from dashboard.database import Database
from dashboard.bot_controller import BotController, BotState
from dashboard.bot_runtime import DarwinRuntime
from dashboard.dash_logger import DashboardLogger
from darwin_agent.monitoring.execution_audit import ExecutionAudit
from darwin_agent.exchanges.binance import BinanceAdapter

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

SESSION_COOKIE = "darwin_session"
SESSION_MAX_AGE = 3600 * 8  # 8 hours

# ═══════════════════════════════════════════════════════
# GLOBALS (initialized on startup)
# ═══════════════════════════════════════════════════════

db = Database(os.environ.get("DASHBOARD_DB_PATH", "data/dashboard.db"))
vault = None  # initialized in lifespan
controller = BotController()
dash_log = DashboardLogger(log_dir="logs")
sessions: Dict[str, Dict[str, Any]] = {}
audit_ref = None  # set externally via set_execution_audit()
runtime_ref: Optional[DarwinRuntime] = None
runtime_lock = threading.Lock()
runtime_autostart_enabled = os.environ.get("DARWIN_RUNTIME_AUTOSTART", "0").strip().lower() in {"1", "true", "yes", "on"}
runtime_default_mode = os.environ.get("DARWIN_RUNTIME_DEFAULT_MODE", "paper").strip().lower() or "paper"


def _metric_gauge(name: str, help_text: str) -> Gauge:
    try:
        return Gauge(name, help_text)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


METRIC_EQUITY = _metric_gauge("darwin_equity", "Current Darwin equity")
METRIC_DRAWDOWN_PCT = _metric_gauge("darwin_drawdown_pct", "Current Darwin drawdown pct")
METRIC_OPEN_POSITIONS = _metric_gauge("darwin_open_positions", "Current open positions")
METRIC_SLIPPAGE_MEAN_BPS = _metric_gauge("darwin_slippage_mean_bps", "Mean slippage bps")
METRIC_LATENCY_P95_MS = _metric_gauge("darwin_latency_p95_ms", "Latency p95 ms")
METRIC_CB_TRIGGERS_TOTAL = _metric_gauge("darwin_cb_triggers_total", "Circuit breaker triggers")
METRIC_ALERTS_TOTAL = _metric_gauge("darwin_alerts_total", "Execution alerts total")

# ═══════════════════════════════════════════════════════
# EXTERNAL INTEGRATION
# ═══════════════════════════════════════════════════════

def set_execution_audit(audit_instance) -> None:
    """Call from main.py to wire in the ExecutionAudit instance."""
    global audit_ref
    audit_ref = audit_instance


def _ensure_runtime() -> DarwinRuntime:
    global runtime_ref, audit_ref
    with runtime_lock:
        if audit_ref is None:
            audit_ref = ExecutionAudit(log_dir="logs/audit")
        if runtime_ref is None:
            runtime_ref = DarwinRuntime(controller=controller, audit=audit_ref)
        return runtime_ref


def _update_prometheus_metrics() -> None:
    s = controller.status
    METRIC_EQUITY.set(float(s.equity))
    METRIC_DRAWDOWN_PCT.set(float(s.drawdown_pct))
    METRIC_OPEN_POSITIONS.set(float(len(s.exposure_by_symbol)))

    if audit_ref is not None:
        try:
            m = audit_ref.get_metrics()
            METRIC_SLIPPAGE_MEAN_BPS.set(float(m.get("darwin_slippage_mean_bps", 0.0)))
            METRIC_LATENCY_P95_MS.set(float(m.get("darwin_latency_p95_ms", 0.0)))
            METRIC_CB_TRIGGERS_TOTAL.set(float(m.get("darwin_cb_triggers_total", 0.0)))
            METRIC_ALERTS_TOTAL.set(float(m.get("darwin_alerts_total", 0.0)))
        except Exception:
            pass

set_execution_audit(ExecutionAudit(log_dir="logs/audit"))



# ═══════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vault
    runtime = _ensure_runtime()
    try:
        vault = CryptoVault()
    except RuntimeError:
        print("WARNING: DASHBOARD_SECRET_KEY not set. Credential encryption disabled.")
        vault = None

    # Create default admin user if none exists
    if db.user_count() == 0:
        _create_default_admin()

    if runtime_autostart_enabled:
        runtime.start(mode=runtime_default_mode)

    yield

    if runtime_ref is not None:
        runtime_ref.stop()
    if audit_ref is not None:
        audit_ref.close()
    dash_log.close()


app = FastAPI(
    title="Darwin v4 Dashboard",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

# CORS — restricted to localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════

def _hash_password(password: str) -> str:
    import hashlib
    salt = os.environ.get("DASHBOARD_SALT", "darwin-v4-salt")
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100000
    ).hex()


def _create_default_admin():
    default_pw = os.environ.get("DASHBOARD_ADMIN_PASSWORD", "darwin2026")
    pw_hash = _hash_password(default_pw)
    db.create_user("admin", pw_hash)
    print("Created default admin user. Change password immediately!")


def _get_session(request: Request) -> Optional[Dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token or token not in sessions:
        return None
    sess = sessions[token]
    if time.time() - sess["created_at"] > SESSION_MAX_AGE:
        del sessions[token]
        return None
    return sess


def require_auth(request: Request) -> Dict:
    sess = _get_session(request)
    if sess is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _get_latest_live_binance_credential() -> Optional[Dict[str, Any]]:
    creds = sorted(db.list_credentials(), key=lambda c: c.get("id", 0), reverse=True)
    live_rec = next((c for c in creds if str(c.get("exchange", "")).lower() == "binance" and not bool(c.get("testnet", 1))), None)
    if not live_rec:
        return None
    return db.get_credential(int(live_rec["id"]))


async def _validate_live_binance_credentials_or_raise() -> None:
    if vault is None:
        raise HTTPException(status_code=500, detail="DASHBOARD_SECRET_KEY is required to use live credentials")
    rec = _get_latest_live_binance_credential()
    if not rec:
        raise HTTPException(status_code=422, detail="Live Binance credentials are required before starting in LIVE mode")
    try:
        api_key = vault.decrypt(rec["encrypted_api_key"])
        api_secret = vault.decrypt(rec["encrypted_secret_key"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to decrypt Binance credentials: {exc}")

    adapter = BinanceAdapter(api_key=api_key, api_secret=api_secret, testnet=False)
    try:
        await adapter.validate_live_credentials()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Live Binance credential check failed: {exc}")
    finally:
        await adapter.close()


# ── CSRF ──

def _generate_csrf() -> str:
    return secrets.token_hex(32)


def _check_csrf(request: Request, sess: Dict) -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    token = request.headers.get("x-csrf-token", "")
    if not token or token != sess.get("csrf_token", ""):
        raise HTTPException(status_code=403, detail="CSRF token invalid")


# ═══════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


class CredentialCreate(BaseModel):
    exchange: str = "binance"
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None
    testnet: bool = True


class BotStartRequest(BaseModel):
    mode: str = "paper"


# ═══════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.post("/api/login")
async def login(body: LoginRequest, request: Request):
    user = db.get_user(body.username)
    pw_hash = _hash_password(body.password)

    if not user or user["password_hash"] != pw_hash:
        dash_log.log(body.username, "LOGIN_FAILED", _get_client_ip(request), False)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_hex(32)
    csrf = _generate_csrf()
    sessions[token] = {
        "username": body.username,
        "created_at": time.time(),
        "csrf_token": csrf,
    }

    dash_log.log(body.username, "LOGIN", _get_client_ip(request))

    response = JSONResponse({
        "ok": True,
        "username": body.username,
        "csrf_token": csrf,
    })
    response.set_cookie(
        SESSION_COOKIE, token,
        httponly=True, samesite="strict", max_age=SESSION_MAX_AGE,
    )
    return response


@app.post("/api/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in sessions:
        user = sessions[token].get("username", "unknown")
        del sessions[token]
        dash_log.log(user, "LOGOUT", _get_client_ip(request))

    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/auth/check")
async def auth_check(request: Request):
    sess = _get_session(request)
    if sess is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    return {"authenticated": True, "username": sess["username"],
            "csrf_token": sess["csrf_token"]}


# ═══════════════════════════════════════════════════════
# CREDENTIAL ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.post("/api/credentials")
async def create_credential(body: CredentialCreate, request: Request,
                             sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    if vault is None:
        raise HTTPException(500, "Encryption not configured")

    enc_key = vault.encrypt(body.api_key)
    enc_secret = vault.encrypt(body.secret_key)
    enc_pass = vault.encrypt(body.passphrase) if body.passphrase else None

    cred_id = db.save_credential(
        exchange=body.exchange,
        encrypted_api_key=enc_key,
        encrypted_secret_key=enc_secret,
        encrypted_passphrase=enc_pass,
        testnet=body.testnet,
    )

    dash_log.log(sess["username"], "CREDENTIAL_ADD",
                 _get_client_ip(request), details={"exchange": body.exchange, "id": cred_id})

    return {"ok": True, "id": cred_id}


@app.get("/api/credentials")
async def list_credentials(request: Request, sess: Dict = Depends(require_auth)):
    creds = db.list_credentials()
    return {"credentials": creds}


@app.delete("/api/credentials/{cred_id}")
async def delete_credential(cred_id: int, request: Request,
                              sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    deleted = db.delete_credential(cred_id)
    if not deleted:
        raise HTTPException(404, "Credential not found")

    dash_log.log(sess["username"], "CREDENTIAL_DELETE",
                 _get_client_ip(request), details={"id": cred_id})
    return {"ok": True}


# ═══════════════════════════════════════════════════════
# BOT LIFECYCLE ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.get("/bot/status")
async def bot_status(request: Request, sess: Dict = Depends(require_auth)):
    s = controller.status.to_dict()

    # Merge ExecutionAudit metrics if available
    if audit_ref is not None:
        try:
            m = audit_ref.get_metrics()
            s["reject_rate_pct"] = round(m.get("darwin_orders_reject_rate_pct", 0), 2)
            s["slippage_mean_bps"] = round(m.get("darwin_slippage_mean_bps", 0), 1)
            s["latency_p95_ms"] = round(m.get("darwin_latency_p95_ms", 0), 0)
            s["cb_triggers"] = int(m.get("darwin_cb_triggers_total", 0))
            s["alerts_total"] = int(m.get("darwin_alerts_total", 0))
        except Exception:
            pass

    _update_prometheus_metrics()
    return s




@app.get("/bot/runtime-status")
async def bot_runtime_status(request: Request, sess: Dict = Depends(require_auth)):
    runtime = _ensure_runtime()
    return runtime.get_runtime_status()

@app.post("/bot/start")
async def bot_start(body: BotStartRequest, request: Request,
                     sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    requested_mode = (body.mode or "paper").strip().lower()
    if requested_mode == "live":
        await _validate_live_binance_credentials_or_raise()

    runtime = _ensure_runtime()
    started = runtime.start(mode=requested_mode)
    result = {
        "ok": started,
        "state": "running" if started else "already_running",
        "mode": requested_mode,
    }
    if not started and runtime.last_start_error:
        raise HTTPException(status_code=422, detail=runtime.last_start_error)
    dash_log.log(sess["username"], "BOT_START", _get_client_ip(request),
                 result["ok"], details=result)
    db.log_event(sess["username"], "BOT_START", _get_client_ip(request),
                 result["ok"], result)
    return JSONResponse(result, status_code=200 if result.get("ok") else 409)


@app.post("/bot/stop")
async def bot_stop(request: Request, sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    runtime = _ensure_runtime()
    stopped = runtime.stop()
    result = {
        "ok": True,
        "state": "stopped" if stopped else "not_running",
    }
    dash_log.log(sess["username"], "BOT_STOP", _get_client_ip(request),
                 result["ok"], details=result)
    db.log_event(sess["username"], "BOT_STOP", _get_client_ip(request),
                 result["ok"], result)
    return JSONResponse(result, status_code=200)


@app.post("/bot/emergency-close")
async def bot_emergency_close(request: Request,
                                sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    runtime = _ensure_runtime()
    runtime_handled = runtime.emergency_close()
    result = controller.emergency_close()
    result["runtime_handled"] = bool(runtime_handled)

    # Log to execution audit if available
    if audit_ref is not None:
        try:
            audit_ref._fire_alert("DASHBOARD_EMERGENCY_CLOSE", {
                "user": sess["username"],
                "ip": _get_client_ip(request),
                "runtime_handled": bool(runtime_handled),
            })
        except Exception:
            pass

    dash_log.log(sess["username"], "EMERGENCY_CLOSE", _get_client_ip(request),
                 result["ok"], details=result)
    db.log_event(sess["username"], "EMERGENCY_CLOSE", _get_client_ip(request),
                 result["ok"], result)
    return result


@app.post("/bot/unlock")
async def bot_unlock(request: Request, sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    result = controller.unlock()
    dash_log.log(sess["username"], "EMERGENCY_UNLOCK", _get_client_ip(request),
                 result["ok"], details=result)
    return result


# ═══════════════════════════════════════════════════════
# EQUITY HISTORY
# ═══════════════════════════════════════════════════════

@app.get("/api/equity-history")
async def equity_history(request: Request, sess: Dict = Depends(require_auth)):
    return {"history": db.get_equity_history(500)}


@app.get("/api/events")
async def event_log(request: Request, sess: Dict = Depends(require_auth)):
    return {"events": db.recent_events(50)}



@app.get("/metrics")
async def metrics():
    _update_prometheus_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ═══════════════════════════════════════════════════════
# WEBSOCKET — REAL-TIME METRICS
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    # Auth check via cookie
    token = websocket.cookies.get(SESSION_COOKIE)
    if not token or token not in sessions:
        await websocket.close(code=4001)
        return

    await websocket.accept()

    try:
        while True:
            s = controller.status.to_dict()

            # Merge audit metrics
            if audit_ref is not None:
                try:
                    m = audit_ref.get_metrics()
                    s["reject_rate_pct"] = round(m.get("darwin_orders_reject_rate_pct", 0), 2)
                    s["slippage_mean_bps"] = round(m.get("darwin_slippage_mean_bps", 0), 1)
                    s["latency_p95_ms"] = round(m.get("darwin_latency_p95_ms", 0), 0)
                    s["cb_triggers"] = int(m.get("darwin_cb_triggers_total", 0))
                    s["alerts_total"] = int(m.get("darwin_alerts_total", 0))

                    recent_alerts = audit_ref.get_recent_alerts(3)
                    s["recent_alerts"] = [
                        {"type": a.get("alert_type", ""), "details": str(a.get("details", ""))[:100]}
                        for a in recent_alerts
                    ]
                except Exception:
                    pass

            s["circuit_breaker_status"] = "LOCKED" if controller.is_locked else (
                "ACTIVE" if s.get("running") else "IDLE"
            )

            _update_prometheus_metrics()
            await websocket.send_json(s)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════
# FRONTEND — Serve index.html
# ═══════════════════════════════════════════════════════

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(DASHBOARD_DIR, "templates", "index.html")
    with open(html_path) as f:
        return HTMLResponse(f.read())
