"""Real network sovereignty enforcement AND monitoring -- not a dashboard
that just trusts the app to behave, an actual guard every outbound HTTP
call in this codebase passes through before the request is made.

DESIGN, AND WHY EVERY LOGGED ROW HERE IS REAL, NOT DEMO DATA:
- check_and_record() is called BEFORE any outbound httpx call in
  agents/real.py (the only place this app itself makes outbound network
  calls -- ingest/ocr.py is local-only; tools/sandbox.py blocks every
  network attempt by generated code and logs it via record_blocked_attempt).
  It checks whether the destination host is local (loopback, or an
  explicitly allowlisted on-machine host -- never via DNS); if not, it
  RAISES before the caller ever opens a socket, and records the attempt as
  blocked. If it is loopback, it records the attempt as allowed and lets
  the caller proceed.
- The one exception -- a call that is EXPECTED to be blocked -- is the
  on-demand /network/test-probe endpoint (api/routes/network.py), which
  deliberately tries to reach a real external address through this SAME
  guard, so a viewer can prove the block is real by triggering it
  themselves, rather than trusting a static claim.
- Nothing here is fabricated after the fact: every row in network_log
  corresponds to a real attempted connection at the time shown.
"""

from __future__ import annotations

import ipaddress
import os
import sqlite3
import threading
import time
from typing import Optional
from urllib.parse import urlparse
from trust.ledger import get_db_connection

LOOPBACK_HOSTNAMES = {"localhost"}


def _extra_allowed_hosts() -> set[str]:
    """Hostnames explicitly declared as on-machine, beyond loopback -- e.g.
    `host.docker.internal` when this app runs in a container and reaches
    the Ollama server on the same Mac. Read per call so it's never stale,
    and matched literally (never DNS-resolved): an entry here is an
    operator's explicit, auditable decision, not something a DNS answer
    can widen."""
    raw = os.getenv("SEVERANCE_LOCAL_HOST_ALLOWLIST", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}

# Process start time, for the monitor's real "uptime" stat -- set once by
# api/main.py's lifespan on startup, read here rather than plumbed through
# every caller.
_process_start_time: Optional[float] = None


def mark_process_start() -> None:
    global _process_start_time
    _process_start_time = time.time()


def get_uptime_seconds() -> float:
    if _process_start_time is None:
        return 0.0
    return time.time() - _process_start_time


class ExternalConnectionBlocked(Exception):
    """Raised by check_and_record() when a destination is not loopback.
    The caller must never proceed to the actual network call when this is
    raised -- the check happens strictly before any socket is opened."""


def _default_db_path() -> str:
    return os.getenv("SEVERANCE_DB_PATH", "severance.db")


# Every model call records a row and the Live Socket Stream polls once a
# second, so these paths must stay cheap: create the table once per database
# per process, then use a plain connection (the ledger's get_db_connection
# re-runs DDL and a commit on every open).
_initialized_paths: set[str] = set()
_init_lock = threading.Lock()


def _connect(db_path: str) -> sqlite3.Connection:
    if db_path not in _initialized_paths:
        with _init_lock:
            if db_path not in _initialized_paths:
                init_network_log_table(db_path)
                _initialized_paths.add(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_network_log_table(db_path: str) -> None:
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS network_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    process_label TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    allowed INTEGER NOT NULL
                );
                """
            )
    finally:
        conn.close()


def _is_allowed(host: str) -> bool:
    """True only for "localhost", a literal loopback IP, or an entry in the
    explicit allowlist above. Deliberately does NO DNS resolution: looking
    up an external hostname would itself send a query off the machine, so
    a guard that resolved names to decide would leak the very destination
    it's about to block."""
    host = host.lower()
    if host in LOOPBACK_HOSTNAMES or host in _extra_allowed_hosts():
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_and_record(process_label: str, url: str, db_path: Optional[str] = None) -> None:
    """Call BEFORE making any outbound request with this URL. Records the
    attempt either way. Raises ExternalConnectionBlocked and does NOT let
    the caller proceed if the destination isn't loopback."""
    db_path = db_path or _default_db_path()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    allowed = bool(host) and _is_allowed(host)
    _record(db_path, process_label, host, port, allowed)
    if not allowed:
        raise ExternalConnectionBlocked(
            f"Outbound call to '{host}:{port}' blocked -- not a local address. "
            "This build has no cloud LLM backend compiled in; see agents/real.py."
        )


def record_blocked_attempt(process_label: str, host: str, port: int, db_path: Optional[str] = None) -> None:
    """Log a connection attempt that was already stopped elsewhere -- used by
    tools/sandbox.py, whose block happens inside the sandboxed child process
    (which has no access to this database), so the parent records each
    real attempt here after the run."""
    _record(db_path or _default_db_path(), process_label, host, port, False)


def _record(db_path: str, process_label: str, host: str, port: int, allowed: bool) -> None:
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO network_log (timestamp, process_label, host, port, allowed) VALUES (?, ?, ?, ?, ?);",
                (time.time(), process_label, host, port, 1 if allowed else 0),
            )
    finally:
        conn.close()


def get_stats(db_path: Optional[str] = None) -> dict:
    db_path = db_path or _default_db_path()
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(allowed), 0) FROM network_log;")
        total, allowed = cursor.fetchone()
        return {
            "total": total,
            "allowed": allowed,
            "blocked": total - allowed,
            "active_host": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            "uptime_seconds": get_uptime_seconds(),
        }
    finally:
        conn.close()


def get_recent_events(db_path: Optional[str] = None, limit: int = 100, only_blocked: bool = False) -> list[dict]:
    db_path = db_path or _default_db_path()
    conn = _connect(db_path)
    try:
        cursor = conn.cursor()
        query = "SELECT id, timestamp, process_label, host, port, allowed FROM network_log"
        if only_blocked:
            query += " WHERE allowed = 0"
        query += " ORDER BY id DESC LIMIT ?;"
        cursor.execute(query, (limit,))
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "process_label": row["process_label"],
                "host": row["host"],
                "port": row["port"],
                "allowed": bool(row["allowed"]),
            }
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()
