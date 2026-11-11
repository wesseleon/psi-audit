"""Offline tests for the psi-audit workflow (no network, no real key)."""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/worker/workflows/psi_audit.py"
spec = importlib.util.spec_from_file_location("psi_audit", MODULE_PATH)
psi = importlib.util.module_from_spec(spec)
sys.modules["psi_audit"] = psi
spec.loader.exec_module(psi)

pytestmark = pytest.mark.asyncio


FAKE_KEY = "test-key"


def _fake_response(status_code: int = 200, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    return response


def _lighthouse_payload(strategy: str = "mobile") -> dict:
    return {
        "lighthouseResult": {
            "_strategy": strategy,
            "categories": {
                "performance": {"score": 0.44},
                "accessibility": {"score": 0.92},
                "best-practices": {"score": 0.93},
                "seo": {"score": 0.83},
            },
            "audits": {
                "first-contentful-paint": {"numericValue": 900, "score": 0.9},
                "largest-contentful-paint": {"numericValue": 4700, "score": 0.2},
                "total-blocking-time": {"numericValue": 12123, "score": 0.0},
                "cumulative-layout-shift": {"numericValue": 0.003, "score": 1.0},
                "speed-index": {"numericValue": 12500, "score": 0.3},
                "unused-javascript": {
                    "title": "Reduce unused JavaScript",
                    "score": 0.5,
                    "details": {"type": "opportunity", "overallSavingsMs": 600},
                },
                "uses-https": {"title": "Use HTTPS", "score": 1.0},
                "render-blocking-resources": {
                    "title": "Eliminate render-blocking resources",
                    "score": 0.1,
                    "displayValue": "Potential savings of 1,200 ms",
                    "description": "Resources are blocking the first paint.",
                    "details": {"type": "opportunity", "overallSavingsMs": 1200},
                },
                "third-party": {"details": {"items": [{"entity": "example.com", "blockingTime": 250}]}},
                "largest-contentful-paint-element": {"details": {"items": [{"node": {"snippet": "<img>"}}]}},
                "diagnostics": {"details": {"items": [{"numRequests": 120}]}},
            },
            "full-page-screenshot": {"data": "aGVsbG8="},
        },
        "loadingExperience": {
            "overall_category": "FAST",
            "metrics": {"cumulative_layout_shift": {"percentile": 0.007, "category": "AVERAGE"}},
        },
    }


async def test_digest_has_all_categories_and_audits(monkeypatch):
    """Parsing yields four category scores, opportunities, passed and failed audits."""

    monkeypatch.setenv("PSI_API_KEY", FAKE_KEY)
    payload = _lighthouse_payload()
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.get = AsyncMock(return_value=_fake_response(payload=payload))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake_client)

    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is True
    assert result["categories"] == {
        "performance": 44,
        "accessibility": 92,
        "best-practices": 93,
        "seo": 83,
    }
    assert result["performance_score"] == 44
    assert result["total_blocking_time_ms"] == 12123
    assert result["opportunities"][0]["audit"] == "render-blocking-resources"
    assert any(a["id"] == "uses-https" for a in result["passed_audits"])
    assert any(o["audit"] == "unused-javascript" for o in result["opportunities"])
    assert result["screenshot"].startswith("data:image/")
    assert result["field_data"]["overall_category"] == "FAST"


async def test_missing_key_returns_graceful_error(monkeypatch):
    """Without a bound key the activity returns a readable error, no exception."""

    monkeypatch.delenv("PSI_API_KEY", raising=False)
    fake_client = MagicMock()
    fake_client.get = AsyncMock()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake_client)

    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "PSI_API_KEY" in result["error"]


async def test_quota_error_is_graceful(monkeypatch):
    """A 429 becomes an error string, never an exception."""

    monkeypatch.setenv("PSI_API_KEY", FAKE_KEY)
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.get = AsyncMock(return_value=_fake_response(status_code=429))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake_client)

    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "429" in result["error"]


async def test_no_lighthouse_result_is_graceful(monkeypatch):
    """An empty lighthouse section becomes an error string."""

    monkeypatch.setenv("PSI_API_KEY", FAKE_KEY)
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.get = AsyncMock(return_value=_fake_response(payload={"lighthouseResult": {}}))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: fake_client)

    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "no lighthouse result" in result["error"]


def test_input_defaults_and_validation():
    """Defaults to both strategies; unknown strategies fall back to the defaults."""

    full = psi.PsiInput(url="https://example.com")
    assert full.strategies == ["mobile", "desktop"]

    filtered = psi.PsiInput(url="https://example.com", strategies=["mobile", "nonsense"])
    assert filtered.strategies == ["mobile"]
    empty = psi.PsiInput(url="https://example.com", strategies=["nonsense"])
    assert empty.strategies == ["mobile", "desktop"]
