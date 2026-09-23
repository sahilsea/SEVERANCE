"""The code sandbox must block every network attempt the generated code
makes -- including DNS lookups -- and record each one in the real network
log, so the Sovereignty Monitor shows genuine blocked attempts."""

from pathlib import Path

from tools.sandbox import run_python
from trust.network_monitor import get_recent_events, get_stats


def test_urllib_request_blocked_and_logged(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    result = run_python(
        "import urllib.request\nurllib.request.urlopen('https://example.com/data', timeout=3)",
        db_path=db_path,
    )
    assert result["success"] is False
    assert "Network access blocked" in result["stderr"]
    assert result["network_attempts"] == [{"host": "example.com", "port": 443}]

    events = get_recent_events(db_path)
    assert len(events) == 1
    assert events[0]["process_label"] == "code-sandbox"
    assert events[0]["host"] == "example.com"
    assert events[0]["allowed"] is False


def test_dns_lookup_blocked_and_logged(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    result = run_python("import socket\nsocket.getaddrinfo('api.example.org', 80)", db_path=db_path)
    assert result["success"] is False
    assert result["network_attempts"] == [{"host": "api.example.org", "port": 80}]
    assert get_stats(db_path)["blocked"] == 1


def test_raw_socket_connect_blocked(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    result = run_python(
        "import socket\ns = socket.socket()\ns.connect(('93.184.216.34', 443))",
        db_path=db_path,
    )
    assert result["success"] is False
    assert result["network_attempts"] == [{"host": "93.184.216.34", "port": 443}]


def test_code_without_network_logs_nothing(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    result = run_python("print(sum(range(10)))", db_path=db_path)
    assert result["success"] is True
    assert result["stdout"].strip() == "45"
    assert result["network_attempts"] == []
    assert get_stats(db_path)["total"] == 0
