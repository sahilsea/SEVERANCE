"""Unit tests for trust/network_monitor.py -- the real loopback-only guard
every outbound HTTP call in this app passes through, and the monitoring
read side built on the same log."""

import socket
from pathlib import Path

import pytest
from trust.network_monitor import (
    ExternalConnectionBlocked,
    check_and_record,
    get_recent_events,
    get_stats,
    record_blocked_attempt,
)


def test_guard_never_resolves_dns(tmp_path: Path, monkeypatch):
    def fail_if_called(*_a, **_k):
        raise AssertionError("guard performed a DNS lookup")

    monkeypatch.setattr(socket, "gethostbyname", fail_if_called)
    monkeypatch.setattr(socket, "getaddrinfo", fail_if_called)
    db_path = str(tmp_path / "netmon.db")
    with pytest.raises(ExternalConnectionBlocked):
        check_and_record("x", "https://example.com/", db_path=db_path)
    check_and_record("y", "http://127.0.0.1:11434/", db_path=db_path)


def test_allowlisted_host_allowed_only_when_configured(tmp_path: Path, monkeypatch):
    db_path = str(tmp_path / "netmon.db")
    monkeypatch.delenv("SEVERANCE_LOCAL_HOST_ALLOWLIST", raising=False)
    with pytest.raises(ExternalConnectionBlocked):
        check_and_record("a", "http://host.docker.internal:11434/api/chat", db_path=db_path)

    monkeypatch.setenv("SEVERANCE_LOCAL_HOST_ALLOWLIST", "host.docker.internal")
    check_and_record("b", "http://host.docker.internal:11434/api/chat", db_path=db_path)
    with pytest.raises(ExternalConnectionBlocked):
        check_and_record("c", "https://example.com/", db_path=db_path)

    stats = get_stats(db_path)
    assert stats["allowed"] == 1
    assert stats["blocked"] == 2


def test_record_blocked_attempt(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    record_blocked_attempt("code-sandbox", "example.com", 443, db_path=db_path)
    events = get_recent_events(db_path)
    assert events[0]["allowed"] is False
    assert events[0]["process_label"] == "code-sandbox"


def test_loopback_call_allowed_and_recorded(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    check_and_record("test-ollama-call", "http://127.0.0.1:11434/api/chat", db_path=db_path)

    stats = get_stats(db_path)
    assert stats["total"] == 1
    assert stats["allowed"] == 1
    assert stats["blocked"] == 0

    events = get_recent_events(db_path)
    assert len(events) == 1
    assert events[0]["allowed"] is True
    assert events[0]["host"] == "127.0.0.1"
    assert events[0]["port"] == 11434
    assert events[0]["process_label"] == "test-ollama-call"


def test_localhost_hostname_treated_as_loopback(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    check_and_record("test-call", "http://localhost:11434/api/tags", db_path=db_path)
    stats = get_stats(db_path)
    assert stats["allowed"] == 1
    assert stats["blocked"] == 0


def test_external_call_blocked_and_recorded_not_performed(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    with pytest.raises(ExternalConnectionBlocked):
        check_and_record("test-external-call", "https://example.com/api", db_path=db_path)

    stats = get_stats(db_path)
    assert stats["total"] == 1
    assert stats["allowed"] == 0
    assert stats["blocked"] == 1

    events = get_recent_events(db_path)
    assert events[0]["allowed"] is False
    assert events[0]["host"] == "example.com"
    assert events[0]["port"] == 443


def test_mixed_calls_produce_correct_aggregate_stats(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    check_and_record("a", "http://127.0.0.1:11434/x", db_path=db_path)
    check_and_record("b", "http://127.0.0.1:11434/y", db_path=db_path)
    try:
        check_and_record("c", "https://evil.example/z", db_path=db_path)
    except ExternalConnectionBlocked:
        pass

    stats = get_stats(db_path)
    assert stats["total"] == 3
    assert stats["allowed"] == 2
    assert stats["blocked"] == 1


def test_only_blocked_filter(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    check_and_record("a", "http://127.0.0.1:11434/x", db_path=db_path)
    try:
        check_and_record("b", "https://evil.example/z", db_path=db_path)
    except ExternalConnectionBlocked:
        pass

    all_events = get_recent_events(db_path, only_blocked=False)
    blocked_events = get_recent_events(db_path, only_blocked=True)
    assert len(all_events) == 2
    assert len(blocked_events) == 1
    assert blocked_events[0]["allowed"] is False


def test_events_newest_first(tmp_path: Path):
    db_path = str(tmp_path / "netmon.db")
    check_and_record("first", "http://127.0.0.1:11434/1", db_path=db_path)
    check_and_record("second", "http://127.0.0.1:11434/2", db_path=db_path)

    events = get_recent_events(db_path)
    assert events[0]["process_label"] == "second"
    assert events[1]["process_label"] == "first"
