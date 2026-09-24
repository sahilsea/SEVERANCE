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
from auth.sponsors import init_grants_tables, seed_sponsor_accounts, verify_sponsors
from auth.users import get_principal, init_users_table
from ingest.seed import seed_initial_admin
from trust.conversations import init_conversations_table
from trust.ledger import init_ledger_table
from trust.network_monitor import init_network_log_table, mark_process_start
from trust.reports import init_reports_table
from api.routes import admin, ask, auth, conversations, documents, grants, ledger, network

UI_DIR = Path(__file__).parent.parent / "ui"
STATIC_DIR = UI_DIR / "static"

# These HTML pages are auth-gated and change with every UI edit -- a browser
# caching them (even briefly, via heuristic caching in the absence of any
# Cache-Control header) means a user can be looking at a stale build after a
# real fix ships, with no visible indication anything is wrong. Force
# revalidation on every load.
NO_CACHE_HEADERS = {"Cache-Control": "no-store, must-revalidate"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize database schemas and bootstrap seed on startup."""
    db_path = get_db_path()
    init_ledger_table(db_path)
    init_users_table(db_path)
    init_grants_tables(db_path)
    init_reports_table(db_path)
    init_conversations_table(db_path)
    init_network_log_table(db_path)
    mark_process_start()

    # Seed initial administrator account if database is fresh
    seeded = seed_initial_admin(db_path)
    if seeded:
        print("[STARTUP] Fresh environment: Seeded initial administrator account.")

    # Bootstrap any declared compartment sponsor account that doesn't exist yet
    # (idempotent -- a no-op once they're all provisioned). Without this, the
    # sole sponsor of a compartment could never receive it through the app's
    # own UI: they can't grant it to themselves, and no one else is
    # authorized to grant it either until they hold it. See
    # auth/sponsors.py::seed_sponsor_accounts for the full reasoning.
    newly_seeded_sponsors = seed_sponsor_accounts(db_path)
    if newly_seeded_sponsors:
        print(f"[STARTUP] Bootstrapped compartment sponsor account(s): {', '.join(newly_seeded_sponsors)}.")

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


@app.middleware("http")
async def no_store_static_assets(request: Request, call_next):
    """StaticFiles has no per-response header hook, so this applies the same
    no-store policy to /static/* as NO_CACHE_HEADERS does for the HTML pages
    above -- otherwise a JS edit (e.g. app.js) can silently fail to reach a
    browser that cached the old file, with no visible sign anything is wrong.
    This app has no build step or content-hashed filenames to cache-bust
    with, so "always revalidate" is the only option that keeps edits live.

    The exception is /static/vendor/: third-party files (Tailwind, fonts)
    whose filenames carry a version or content hash and never change in
    place, so browsers may keep them indefinitely instead of re-downloading
    ~2.4 MB on every page load."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/vendor/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

# Register API Routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(grants.router)
app.include_router(ask.router)
app.include_router(documents.router)
app.include_router(ledger.router)
app.include_router(conversations.router)
app.include_router(network.router)


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
    return FileResponse(str(login_path), headers=NO_CACHE_HEADERS)


@app.get("/app")
def app_page(request: Request):
    """Serve main workbench UI."""
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        return RedirectResponse(url="/login")
    app_path = UI_DIR / "app.html"
    return FileResponse(str(app_path), headers=NO_CACHE_HEADERS)


@app.get("/admin")
def admin_page(request: Request):
    """Serve administrative and sponsor console UI."""
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        return RedirectResponse(url="/login")
    admin_path = UI_DIR / "admin.html"
    return FileResponse(str(admin_path), headers=NO_CACHE_HEADERS)
