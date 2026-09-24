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
  4. No network access. On macOS (the app not already sandboxed) the macOS
     kernel refuses every network operation -- raw sockets, ctypes and
     subprocesses included -- and file writes are limited to the run's own
     directories. Elsewhere, name resolution and connect() are
     blocked at the Python level only. Every attempt is logged to the real
     network monitor. See the mode notes above _NETWORK_PREAMBLE.
  5. A dedicated, empty temp working directory per run, deleted afterward.
This is real containment appropriate for "run and verify code this same
local model just generated for the user." Reads are not restricted, and the
Python-only fallback mode is not a hardened boundary for adversarial code.
"""

from __future__ import annotations

import json
import os
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


# Two enforcement modes, chosen per run by _os_sandbox_available():
#
# OS MODE (macOS, app not already inside a sandbox): the child runs under
# sandbox-exec with a profile in which the KERNEL denies every network
# operation (the call fails with "Operation not permitted") and file writes
# outside its own directories. Nothing inside the process -- raw _socket,
# ctypes, reloading socket -- can get past that. The Python preamble only
# REPORTS the destination it is about to try and then lets the real call
# through, so the kernel is what stops it. The kernel denies rather than kills
# (SIGKILL): a sandbox kill makes macOS show a "Python quit unexpectedly"
# dialog every time, and it offers no silent, unforgeable record of a denial.
# So the LOG is best-effort, as in Python mode: code could fake or hide its own
# entries, but never reach the network.
#
# PYTHON MODE (fallback: already sandboxed -- macOS refuses to nest
# sandbox-exec -- or not macOS): the preamble blocks name resolution and
# connect() itself and raises. This stops normal Python networking but is not
# kernel-enforced, and reports cannot be corroborated, so they are validated
# and capped. When the app runs under scripts/run_sandboxed.sh the outer
# kernel profile still denies all non-loopback traffic for this child.
_NETWORK_PREAMBLE = textwrap.dedent(
    """
    import json as _sandbox_json
    import socket as _sandbox_socket

    _SANDBOX_REPORT_PATH = {report_path!r}
    _SANDBOX_OS_ENFORCED = {os_enforced!r}

    def _sandbox_report(host, port):
        try:
            with open(_SANDBOX_REPORT_PATH, "a") as _f:
                _f.write(_sandbox_json.dumps({{"host": str(host), "port": port}}) + "\\n")
        except Exception:
            pass
        if not _SANDBOX_OS_ENFORCED:
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

    def _sandbox_wrap(original, describe):
        def wrapper(*args, **kwargs):
            _sandbox_report(*describe(*args))
            return original(*args, **kwargs)
        return wrapper

    _sandbox_socket.getaddrinfo = _sandbox_wrap(_sandbox_socket.getaddrinfo, lambda h, p, *a: (h, _sandbox_port(p)))
    _sandbox_socket.gethostbyname = _sandbox_wrap(_sandbox_socket.gethostbyname, lambda h, *a: (h, 0))
    _sandbox_socket.gethostbyname_ex = _sandbox_wrap(_sandbox_socket.gethostbyname_ex, lambda h, *a: (h, 0))
    _sandbox_socket.create_connection = _sandbox_wrap(_sandbox_socket.create_connection, lambda a, *r: _sandbox_addr(a))
    _sandbox_socket.socket.connect = _sandbox_wrap(_sandbox_socket.socket.connect, lambda s, a, *r: _sandbox_addr(a))
    _sandbox_socket.socket.connect_ex = _sandbox_wrap(_sandbox_socket.socket.connect_ex, lambda s, a, *r: _sandbox_addr(a))
    """
)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_CHILD_PROFILE = """
(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
  (subpath (param "WORKDIR"))
  (subpath (param "REPORT_DIR"))
  (literal "/dev/null")
  (regex #"^/dev/tty")
  (regex #"^/dev/fd/"))
"""

REPORT_FILENAME = "network_attempts.jsonl"
SANDBOX_PROCESS_LABEL = "code-sandbox"
MAX_REPORTED_ATTEMPTS = 10
_VALID_HOST = re.compile(r"^[A-Za-z0-9._:%\[\]-]{1,253}$")
_os_sandbox_cache: Optional[bool] = None


def _already_sandboxed() -> bool:
    try:
        import ctypes
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        return libsystem.sandbox_check(os.getpid(), None, 0) == 1
    except Exception:
        return False


def _os_sandbox_available() -> bool:
    global _os_sandbox_cache
    if _os_sandbox_cache is None:
        _os_sandbox_cache = (
            sys.platform == "darwin" and os.path.exists(_SANDBOX_EXEC) and not _already_sandboxed()
        )
    return _os_sandbox_cache


def _read_reports(path: Path) -> list[dict]:
    """Well-formed destination reports, de-duplicated and capped."""
    if not path.exists():
        return []
    reports: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                item = json.loads(line)
                host, port = str(item.get("host", "")), int(item.get("port") or 0)
            except (ValueError, TypeError, AttributeError):
                continue
            entry = {"host": host, "port": port}
            if _VALID_HOST.match(host) and 0 <= port <= 65535 and entry not in reports:
                reports.append(entry)
                if len(reports) >= MAX_REPORTED_ATTEMPTS:
                    break
    return reports


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
              "network_attempts": list[{"host": str, "port": int}],
              "network_enforcement": "os" | "python"}
    """
    os_mode = _os_sandbox_available()
    # Resolved paths: the kernel profile matches real paths, and macOS temp
    # dirs live behind the /var -> /private/var symlink.
    workdir = Path(tempfile.mkdtemp(prefix="severance-sandbox-")).resolve()
    report_dir = Path(tempfile.mkdtemp(prefix="severance-sandbox-report-")).resolve()
    report_path = report_dir / REPORT_FILENAME
    try:
        preamble = _NETWORK_PREAMBLE.format(report_path=str(report_path), os_enforced=os_mode)
        script_path = workdir / "main.py"
        script_path.write_text(preamble + "\n" + code, encoding="utf-8")

        command = [sys.executable, "-I", str(script_path)]
        if os_mode:
            command = [
                _SANDBOX_EXEC, "-p", _CHILD_PROFILE,
                "-D", f"WORKDIR={workdir}", "-D", f"REPORT_DIR={report_dir}",
            ] + command

        try:
            proc = subprocess.run(
                command,
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

        attempts = _read_reports(report_path)
        if os_mode and attempts:
            where = ", ".join(f"{a['host']}:{a['port']}" if a["port"] else a["host"] for a in attempts)
            result["stderr"] += (
                f"\n[Network access blocked by the OS sandbox: the code tried to reach {where}; "
                f"the kernel refused the connection.]"
            )

        for attempt in attempts:
            record_blocked_attempt(SANDBOX_PROCESS_LABEL, attempt["host"], attempt["port"], db_path=db_path)
        result["network_attempts"] = attempts
        result["network_enforcement"] = "os" if os_mode else "python"
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
