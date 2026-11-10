"""Offline tests for the psi-audit workflow: every HTTP call is mocked."""

import json
import os
from pathlib import Path

import httpx
import pytest

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "src" / "worker" / "workflows"
PSI_MODULE_PATH = WORKFLOWS_DIR / "psi_audit.py"

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def psi(monkeypatch: pytest.MonkeyPatch):
    """Import the workflow module fresh, with PSI_API_KEY controlled by the test."""

    os.environ.pop("PSI_API_KEY", None)
    import importlib.util

    spec = importlib.util.spec_from_file_location("psi_audit", PSI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lighthouse_payload(score: float = 0.92) -> dict:
    audit = lambda title, val: {  # noqa: E731
        "id": title,
        "title": title,
        "numericValue": val,
        "score": 0.95,
        "details": {},
    }
    return {
        "audits": {
            "first-contentful-paint": audit("First Contentful Paint", 1200),
            "largest-contentful-paint": audit("Largest Contentful Paint", 2100),
            "total-blocking-time": audit("Total Blocking Time", 150),
            "cumulative-layout-shift": audit("Cumulative Layout Shift", 0.05),
            "speed-index": audit("Speed Index", 1800),
            "max-potential-fid": audit("Max Potential FID", 120),
            "largest-contentful-paint-element": {"details": {"items": [{"node": {"snippet": "<img>"}}]}},
            "diagnostics": {"details": {"items": [{"numRequests": 42, "totalByteWeight": 900000}]}},
            "render-blocking-resources": {
                "title": "Eliminate render-blocking resources",
                "numericValue": 900,
                "score": 0.5,
                "details": {"type": "opportunity", "overallSavingsMs": 800},
            },
            "third-party": {
                "details": {
                    "items": [
                        {"entity": "googletagmanager.com", "blockingTime": 90},
                        {"entity": "fonts.example.com", "blockingTime": 0},
                    ]
                }
            },
        },
        "categories": {"performance": {"score": score}},
    }


async def test_parses_lighthouse_result(psi, monkeypatch):
    async def fake_send(self, request, **kwargs):
        payload = {
            "lighthouseResult": _lighthouse_payload(0.92),
            "loadingExperience": {
                "overall_category": "AVERAGE",
                "metrics": {"largest_contentful_paint": {"percentile": 2500, "category": "NEEDS_IMPROVEMENT"}},
            },
        }
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setenv("PSI_API_KEY", "test-key")
    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is True
    assert result["performance_score"] == 92
    assert result["metrics"]["largest-contentful-paint"]["assessment"] == "good"
    assert result["field_data"]["overall_category"] == "AVERAGE"
    assert result["opportunities"][0]["audit"] == "render-blocking-resources"
    assert result["third_parties"][0]["origin"] == "googletagmanager.com"
    assert result["lcp_element"]


async def test_missing_api_key(psi):
    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "PSI_API_KEY" in result["error"]


async def test_quota_error(psi, monkeypatch):
    async def fake_send(self, request, **kwargs):
        return httpx.Response(429, json={"error": {"message": "Quota exceeded"}}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setenv("PSI_API_KEY", "test-key")
    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "429" in result["error"]


async def test_no_lighthouse_result(psi, monkeypatch):
    async def fake_send(self, request, **kwargs):
        return httpx.Response(200, json={"lighthouseResult": {}}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setenv("PSI_API_KEY", "test-key")
    result = await psi.collect_psi("https://example.com", "mobile")
    assert result["ok"] is False
    assert "no lighthouse result" in result["error"]


async def test_workflow_input_defaults(psi):
    """Input defaults to both strategies and drops unknown ones."""

    data = json.loads(psi.PsiInput(url="https://example.com").model_dump_json())
    assert data["strategies"] == ["mobile", "desktop"]
    data2 = json.loads(psi.PsiInput(url="https://example.com", strategies=["tablet", "desktop"]).model_dump_json())
    assert "desktop" in data2["strategies"]
    assert "tablet" not in data2["strategies"]
