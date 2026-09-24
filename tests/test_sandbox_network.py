"""The code sandbox must block every network attempt the generated code makes
-- including DNS lookups -- and record each attempt in the network log.

Both enforcement modes are exercised: the macOS kernel sandbox ("os", when
available) and the Python-level fallback ("python")."""

import os
from pathlib import Path

import pytest

import tools.sandbox as sandbox
from tools.sandbox import run_python
from trust.network_monitor import get_recent_events, get_stats

OS_AVAILABLE = sandbox._os_sandbox_available()
MODES = [pytest.param("python", id="python")] + [
    pytest.param("os", id="os", marks=pytest.mark.skipif(not OS_AVAILABLE, reason="macOS sandbox-exec unavailable"))
]


@pytest.fixture
def mode(request, monkeypatch):
    monkeypatch.setattr(sandbox, "_os_sandbox_cache", request.param == "os")
    return request.param


@pytest.mark.parametrize("mode", MODES, indirect=True)
def test_urllib_request_blocked_and_logged(tmp_path: Path, mode):
    db_path = str(tmp_path / "netmon.db")
    result = run_python(
        "import urllib.request\nurllib.request.urlopen('https://example.com/data', timeout=3)",
        db_path=db_path,
    )
    assert result["success"] is False
    assert result["network_enforcement"] == mode
    assert "network access blocked" in result["stderr"].lower()
    assert result["network_attempts"] == [{"host": "example.com", "port": 443}]

    events = get_recent_events(db_path)
    assert len(events) == 1
    assert events[0]["process_label"] == "code-sandbox"
    assert events[0]["host"] == "example.com"
    assert events[0]["allowed"] is False


@pytest.mark.parametrize("mode", MODES, indirect=True)
def test_dns_lookup_blocked_and_logged(tmp_path: Path, mode):
    db_path = str(tmp_path / "netmon.db")
    result = run_python("import socket\nsocket.getaddrinfo('api.example.org', 80)", db_path=db_path)
    assert result["success"] is False
    assert result["network_attempts"] == [{"host": "api.example.org", "port": 80}]
    assert get_stats(db_path)["blocked"] == 1


@pytest.mark.parametrize("mode", MODES, indirect=True)
def test_socket_connect_blocked(tmp_path: Path, mode):
    db_path = str(tmp_path / "netmon.db")
    result = run_python(
        "import socket\ns = socket.socket()\ns.connect(('93.184.216.34', 443))",
        db_path=db_path,
    )
    assert result["success"] is False
    assert result["network_attempts"] == [{"host": "93.184.216.34", "port": 443}]


@pytest.mark.parametrize("mode", MODES, indirect=True)
def test_code_without_network_logs_nothing(tmp_path: Path, mode):
    db_path = str(tmp_path / "netmon.db")
    result = run_python("print(sum(range(10)))", db_path=db_path)
    assert result["success"] is True
    assert result["stdout"].strip() == "45"
    assert result["network_attempts"] == []
    assert get_stats(db_path)["total"] == 0


@pytest.mark.parametrize("mode", MODES, indirect=True)
def test_malformed_reports_ignored_and_capped(tmp_path: Path, mode):
    code = (
        "import socket, json, __main__ as m\n"
        "with open(m._SANDBOX_REPORT_PATH, 'a') as f:\n"
        "    f.write('not json\\n' + json.dumps({'host': 'bad host!', 'port': 1}) + '\\n')\n"
        "    for i in range(50): f.write(json.dumps({'host': f'h{i}.example', 'port': 80}) + '\\n')\n"
        "socket.socket().connect(('93.184.216.34', 443))"
    )
    result = run_python(code, db_path=str(tmp_path / "netmon.db"))
    assert result["success"] is False
    assert all(" " not in a["host"] for a in result["network_attempts"])
    assert len(result["network_attempts"]) <= sandbox.MAX_REPORTED_ATTEMPTS


@pytest.mark.skipif(not OS_AVAILABLE, reason="macOS sandbox-exec unavailable")
@pytest.mark.parametrize("mode", [pytest.param("os", id="os")], indirect=True)
def test_os_mode_stops_raw_socket_bypass(tmp_path: Path, mode):
    """The raw C socket module skips the Python-level wrappers, but the
    kernel still refuses the connection -- nothing reaches the network."""
    result = run_python(
        "import _socket\ns = _socket.socket()\ns.connect(('127.0.0.1', 9))\nprint('reached')",
        db_path=str(tmp_path / "netmon.db"),
    )
    assert result["success"] is False
    assert "reached" not in result["stdout"]
    assert "Operation not permitted" in result["stderr"]
    assert result["returncode"] == 1  # refused, not killed: no crash dialog


@pytest.mark.skipif(not OS_AVAILABLE, reason="macOS sandbox-exec unavailable")
@pytest.mark.parametrize("mode", [pytest.param("os", id="os")], indirect=True)
def test_os_mode_denies_writes_outside_workdir(tmp_path: Path, mode):
    target = tmp_path / "escape.txt"
    result = run_python(f"open({str(target)!r}, 'w').write('x')", db_path=str(tmp_path / "netmon.db"))
    assert result["success"] is False
    assert "Operation not permitted" in result["stderr"]
    assert not os.path.exists(target)
