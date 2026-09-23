"""Sovereignty Monitor endpoints -- read side of trust/network_monitor.py's
real connection log, plus an on-demand probe so a viewer can trigger a
genuine blocked external call themselves rather than trusting a static claim.

ZERO ACCESS DECISIONS INSIDE.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from contracts import Principal
from auth.deps import current_principal, get_db_path
from trust.network_monitor import ExternalConnectionBlocked, check_and_record, get_recent_events, get_stats

router = APIRouter(prefix="/network", tags=["Sovereignty Monitor"])


@router.get("/stats")
def network_stats(principal: Principal = Depends(current_principal)):
    """Real aggregate counts from network_log -- every row corresponds to an
    actual outbound call attempt made by this process, not a demo figure."""
    return get_stats(get_db_path())


@router.get("/events")
def network_events(
    limit: int = Query(default=100, ge=1, le=1000),
    only_blocked: bool = Query(default=False),
    principal: Principal = Depends(current_principal),
):
    """Real recent connection attempts, newest first."""
    return get_recent_events(get_db_path(), limit=limit, only_blocked=only_blocked)


@router.post("/test-probe")
def network_test_probe(principal: Principal = Depends(current_principal)):
    """Deliberately attempt a real outbound call to a real external host,
    through the SAME check_and_record() guard every other call in this app
    passes through. This is expected to be blocked -- that's the point: it
    lets a viewer watch a genuine ExternalConnectionBlocked happen live and
    see the real logged row appear, instead of taking the claim on faith.
    """
    probe_url = "https://example.com/severance-sovereignty-test-probe"
    try:
        check_and_record("sovereignty-test-probe", probe_url, db_path=get_db_path())
        # Should be unreachable: check_and_record() always raises for a
        # non-loopback host before returning.
        return {"blocked": False, "url": probe_url}
    except ExternalConnectionBlocked as exc:
        return {"blocked": True, "url": probe_url, "detail": str(exc)}
