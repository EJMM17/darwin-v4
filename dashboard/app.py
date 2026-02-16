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
from pydantic import BaseModel, Field

# Dashboard modules
from dashboard.crypto_vault import CryptoVault
from dashboard.database import Database
from dashboard.bot_controller import BotController, BotState
from dashboard.dash_logger import DashboardLogger

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

# ═══════════════════════════════════════════════════════
# EXTERNAL INTEGRATION
# ═══════════════════════════════════════════════════════

def set_execution_audit(audit_instance) -> None:
    """Call from main.py to wire in the ExecutionAudit instance."""
    global audit_ref
    audit_ref = audit_instance


def set_bot_runner(runner_fn) -> None:
    """Call from main.py to set the bot runner function."""
    controller.set_runner(runner_fn)


# ═══════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vault
    try:
        vault = CryptoVault()
    except RuntimeError:
        print("WARNING: DASHBOARD_SECRET_KEY not set. Credential encryption disabled.")
        vault = None

    # Create default admin user if none exists
    if db.user_count() == 0:
        _create_default_admin()

    yield

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

    return s


@app.post("/bot/start")
async def bot_start(body: BotStartRequest, request: Request,
                     sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    result = controller.start(mode=body.mode)
    dash_log.log(sess["username"], "BOT_START", _get_client_ip(request),
                 result["ok"], details=result)
    db.log_event(sess["username"], "BOT_START", _get_client_ip(request),
                 result["ok"], result)
    return result


@app.post("/bot/stop")
async def bot_stop(request: Request, sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    result = controller.stop()
    dash_log.log(sess["username"], "BOT_STOP", _get_client_ip(request),
                 result["ok"], details=result)
    db.log_event(sess["username"], "BOT_STOP", _get_client_ip(request),
                 result["ok"], result)
    return result


@app.post("/bot/emergency-close")
async def bot_emergency_close(request: Request,
                                sess: Dict = Depends(require_auth)):
    _check_csrf(request, sess)
    result = controller.emergency_close()

    # Log to execution audit if available
    if audit_ref is not None:
        try:
            audit_ref._fire_alert("DASHBOARD_EMERGENCY_CLOSE", {
                "user": sess["username"],
                "ip": _get_client_ip(request),
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
