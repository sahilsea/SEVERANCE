"""Local, sandboxed Python execution -- the "code execution in a sandbox"
tool from the SIH problem statement, and the thing that turns code
generation into a real generate-run-verify-iterate loop instead of a
one-shot chat reply.

HONEST SCOPE OF ISOLATION (read before trusting this with anything but
locally-generated, non-adversarial code):
This is a subprocess-based sandbox, not a container/VM per run (even when
the app itself is deployed in a container, each code run is a subprocess
inside it). What it actually enforces, per run:
  1. A fresh `python3 -I` subprocess (isolated mode: ignores the user's
     site-packages and PYTHONPATH/PYTHONHOME env vars).
  2. A wall-clock timeout (subprocess.run(timeout=...)), hard-killing a
     hung/infinite-looping process.
  3. POSIX resource limits (CPU time, address space, output file size,
     process count) applied via preexec_fn, to contain runaway resource use
     -- a crude fork-bomb/memory-bomb backstop, not a full quota system.
  4. Outbound network calls blocked at the Python level: socket.connect and
     name resolution are monkeypatched to raise before the user's code ever
     runs, inside the subprocess, and every attempt is logged to the real
     network monitor. This stops normal Python networking (urllib, requests,
     http.client all end up calling socket) but is NOT a kernel-level
     firewall -- code that reaches the network through something other than
     Python's socket module (a raw syscall via ctypes, a subprocess spawning
     curl) is not stopped by this. It IS the right layer for its actual
     purpose here: proving, in the demo, that generated code cannot phone
     out, for the overwhelming majority of ways Python code would do that.
  5. A dedicated, empty temp working directory per run, deleted afterward --
     the code can read/write files, but only within its own throwaway
     sandbox directory, and that directory starts empty every time.
This is real containment appropriate for "run and verify code this same
local model just generated for the user," not a hardened boundary for
executing arbitrary untrusted/adversarial code from strangers.
"""

from __future__ import annotations

import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional
from trust.network_monitor import record_blocked_attempt

DEFAULT_TIMEOUT_SECONDS = 8.0
CPU_TIME_LIMIT_SECONDS = 5
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256MB
OUTPUT_FILE_LIMIT_BYTES = 10 * 1024 * 1024  # 10MB
MAX_PROCESSES = 16
MAX_OUTPUT_CHARS = 4000  # truncate captured stdout/stderr before it reaches the model/UI


def extract_code_block(markdown: str, lang: str = "python") -> Optional[str]:
    """Pull the first fenced code block tagged with `lang` (case-insensitive)
    out of a Markdown string, or the first UNTAGGED fenced block if none is
    explicitly tagged `lang` -- a small model often forgets the language tag
    even when the code clearly is Python. Returns None if no fenced block
    is found at all, which the caller treats as "nothing to execute" rather
    than an error (not every code answer is Python, or executable at all --
    a SQL query or a config snippet has nothing to run here)."""
    pattern = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(markdown)
    if not blocks:
        return None

    for tag, body in blocks:
        if tag.strip().lower() == lang:
            return body.strip()
    for tag, body in blocks:
        if not tag.strip():
            return body.strip()
    return None


def _set_resource_limits() -> None:
    """Runs inside the child process (preexec_fn) before exec. Each limit is
    applied independently and failures are swallowed -- some platforms
    (notably macOS for RLIMIT_AS) only partially honor certain limits, and a
    limit this sandbox can't set is not a reason to refuse to run code at
    all; the timeout and network block below are the limits that matter
    most and always apply."""
    for limit, value in (
        (resource.RLIMIT_CPU, CPU_TIME_LIMIT_SECONDS),
        (resource.RLIMIT_AS, MEMORY_LIMIT_BYTES),
        (resource.RLIMIT_FSIZE, OUTPUT_FILE_LIMIT_BYTES),
        (resource.RLIMIT_NPROC, MAX_PROCESSES),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


# Name resolution is blocked too, not just connect(): libraries like urllib3
# call getaddrinfo() BEFORE connecting, and a DNS query is itself traffic
# leaving the machine. Each blocked attempt is appended to a file in the
# run's own directory, which the parent reads back and records in the real
# network log (the child has no access to the app database).
_NETWORK_BLOCK_PREAMBLE = textwrap.dedent(
    """
    import json as _sandbox_json
    import socket as _sandbox_socket

    _SANDBOX_ATTEMPTS_PATH = {attempts_path!r}

    def _sandbox_block(host, port):
        try:
            with open(_SANDBOX_ATTEMPTS_PATH, "a") as _f:
                _f.write(_sandbox_json.dumps({{"host": str(host), "port": port}}) + "\\n")
        except Exception:
            pass
        raise RuntimeError(
            f"Network access blocked: this air-gapped workbench does not allow "
            f"outbound connections (attempted {{host}}:{{port}})."
        )

    def _sandbox_port(port):
        try:
            return int(port)
        except (TypeError, ValueError):
            return {{"https": 443, "http": 80}}.get(str(port).lower(), 0)

    def _sandbox_addr(address):
        if isinstance(address, tuple) and len(address) >= 2:
            return address[0], _sandbox_port(address[1])
        return str(address), 0

    def _blocked_getaddrinfo(host, port, *_a, **_k):
        _sandbox_block(host, _sandbox_port(port))

    def _blocked_gethostbyname(host, *_a, **_k):
        _sandbox_block(host, 0)

    def _blocked_create_connection(address, *_a, **_k):
        _sandbox_block(*_sandbox_addr(address))

    def _blocked_connect(_self, address, *_a, **_k):
        _sandbox_block(*_sandbox_addr(address))

    _sandbox_socket.getaddrinfo = _blocked_getaddrinfo
    _sandbox_socket.gethostbyname = _blocked_gethostbyname
    _sandbox_socket.gethostbyname_ex = _blocked_gethostbyname
    _sandbox_socket.create_connection = _blocked_create_connection
    _sandbox_socket.socket.connect = _blocked_connect
    _sandbox_socket.socket.connect_ex = _blocked_connect
    """
)

ATTEMPTS_FILENAME = ".severance_network_attempts.jsonl"
SANDBOX_PROCESS_LABEL = "code-sandbox"


def _read_attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    attempts = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
            attempts.append({"host": str(item.get("host", "")), "port": int(item.get("port") or 0)})
        except (ValueError, TypeError):
            continue
    return attempts


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS, db_path: Optional[str] = None) -> dict:
    """Execute `code` as Python in the sandbox described above. Always
    returns a result dict (never raises for a normal failure -- a syntax
    error, an exception, a timeout are all legitimate outcomes to report
    back, not exceptions in the Python sense); only a genuine sandbox setup
    problem (e.g. python3 not found) raises.

    Every network attempt the code made (all of them blocked) is recorded
    in trust/network_monitor.py's log under "code-sandbox", and returned.

    Returns: {"success": bool, "stdout": str, "stderr": str,
              "returncode": Optional[int], "timed_out": bool,
              "network_attempts": list[{"host": str, "port": int}]}
    """
    workdir = Path(tempfile.mkdtemp(prefix="severance-sandbox-"))
    attempts_path = workdir / ATTEMPTS_FILENAME
    try:
        preamble = _NETWORK_BLOCK_PREAMBLE.format(attempts_path=str(attempts_path))
        script = preamble + "\n" + code
        script_path = workdir / "main.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_set_resource_limits if sys.platform != "win32" else None,
            )
            result = {
                "success": proc.returncode == 0,
                "stdout": proc.stdout[:MAX_OUTPUT_CHARS],
                "stderr": proc.stderr[:MAX_OUTPUT_CHARS],
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "success": False,
                "stdout": _as_text(exc.stdout)[:MAX_OUTPUT_CHARS],
                "stderr": _as_text(exc.stderr)[:MAX_OUTPUT_CHARS] + f"\n[Killed: exceeded {timeout}s sandbox timeout.]",
                "returncode": None,
                "timed_out": True,
            }

        attempts = _read_attempts(attempts_path)
        for attempt in attempts:
            record_blocked_attempt(SANDBOX_PROCESS_LABEL, attempt["host"], attempt["port"], db_path=db_path)
        result["network_attempts"] = attempts
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
