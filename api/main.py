"""FastAPI main application entrypoint for SEVERANCE.

ZERO ACCESS DECISIONS INSIDE THIS FILE OR ANY FILE UNDER api/.
All security decisions are made exclusively in trust/labels.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contracts import Principal
from auth.deps import current_principal, get_db_path
from auth.session import COOKIE_NAME, verify_session_token
from auth.sponsors import init_grants_tables, verify_sponsors
from auth.users import get_principal, init_users_table
from ingest.seed import seed_initial_admin
from trust.conversations import init_conversations_table
from trust.ledger import init_ledger_table
from trust.reports import init_reports_table
from api.routes import admin, ask, auth, conversations, documents, grants, ledger

UI_DIR = Path(__file__).parent.parent / "ui"
STATIC_DIR = UI_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize database schemas and bootstrap seed on startup."""
    db_path = get_db_path()
    init_ledger_table(db_path)
    init_users_table(db_path)
    init_grants_tables(db_path)
    init_reports_table(db_path)
    init_conversations_table(db_path)

    # Seed initial administrator account if database is fresh
    seeded = seed_initial_admin(db_path)
    if seeded:
        print("[STARTUP] Fresh environment: Seeded initial administrator account.")

    # Startup sponsor verification check
    try:
        verify_sponsors(db_path)
        print("[STARTUP] All declared compartment sponsors verified successfully.")
    except Exception as exc:
        print(f"[STARTUP WARNING] Sponsor verification check: {exc}")
        print("[STARTUP NOTICE] If this is a fresh bootstrap, please log in as admin to provision the sponsor accounts.")

    yield


app = FastAPI(
    title="SEVERANCE — Sovereign Air-Gapped Document Trust Workbench",
    description="Automated RTI Section 10 Severability for Mangalore Refinery and Petrochemicals Limited (MRPL)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for air-gapped web client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register API Routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(grants.router)
app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(ledger.router)
app.include_router(conversations.router)


@app.get("/me")
def get_current_user_profile(
    principal: Principal = Depends(current_principal),
):
    """Retrieve the currently authenticated caller's profile."""
    return principal


# ---------------------------------------------------------------------------
# HTML UI Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    """Redirect to /app if authenticated, otherwise to /login."""
    token = request.cookies.get(COOKIE_NAME)
    if token and verify_session_token(token):
        return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")


@app.get("/login")
def login_page():
    """Serve authentication UI."""
    login_path = UI_DIR / "login.html"
    return FileResponse(str(login_path))


@app.get("/app")
def app_page(request: Request):
    """Serve main workbench UI."""
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        return RedirectResponse(url="/login")
    app_path = UI_DIR / "app.html"
    return FileResponse(str(app_path))


@app.get("/admin")
def admin_page(request: Request):
    """Serve administrative and sponsor console UI."""
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        return RedirectResponse(url="/login")
    admin_path = UI_DIR / "admin.html"
    return FileResponse(str(admin_path))
