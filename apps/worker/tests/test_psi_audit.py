"""Offline tests for the psi-audit workflow: digest correctness with mocked PSI responses."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from worker.workflows.psi_audit import AuditSiteInput, collect_psi

SAMPLE: dict[str, Any] = {
    "id": "https://example.com/",
    "captchaResult": "CAPTCHA_NOT_NEEDED",
    "loadingExperience": {
        "overall_category": "AVERAGE",
        "metrics": {
            "LARGEST_CONTENTFUL_PAINT_MS": {"percentile": 2500, "category": "AVERAGE"},
        },
    },
    "lighthouseResult": {
        "lighthouseVersion": "13.6.0",
        "categories": {
            "performance": {
                "score": 0.68,
                "auditRefs": [
                    {"id": "render-blocking-resources"},
                    {"id": "unused-javascript"},
                    {"id": "uses-optimized-images"},
                ],
            }
        },
        "audits": {
            "first-contentful-paint": {"displayValue": "1.2 s", "score": 0.8},
            "speed-index": {"displayValue": "2.8 s", "score": 0.6},
            "largest-contentful-paint": {"displayValue": "3.1 s", "score": 0.45},
            "total-blocking-time": {"displayValue": "480 ms", "score": 0.4},
            "cumulative-layout-shift": {"displayValue": "0.05", "score": 1.0},
            "interactive": {"displayValue": "4.0 s", "score": 0.86},
            "render-blocking-resources": {
                "score": 0.5,
                "details": {"items": [{"overallSavingsMs": 400, "overallSavingsBytes": 51200}]},
            },
            "unused-javascript": {
                "score": 0.5,
                "details": {"items": [{"overallSavingsMs": 200, "overallSavingsBytes": 204800}]},
            },
            "third-party-summary": {
                "details": {
                    "items": [
                        {"entity": "Google Fonts", "totalBytes": 40960, "blockingTime": 120},
                        "https://fonts.gstatic.com",  # PSI sometimes returns bare strings
                    ]
                }
            },
            "mainthread-work-breakdown": {"displayValue": "1.4 s"},
            "total-byte-weight": {"displayValue": "Total size was 700 KiB"},
            "largest-contentful-paint-element": {
                "details": {"items": [{"items": [{"node": {"selector": "body > img.hero", "nodeLabel": "hero.jpg"}}]}]}
            },
        },
    },
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, **kwargs: Any):
        pass

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def get(self, url: str, params: dict[str, str]) -> _FakeResponse:
        assert url == "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
        assert params["strategy"] in ("mobile", "desktop")
        assert params["key"] == "test-key"
        return _FakeResponse(SAMPLE)


class _QuotaClient(_FakeClient):
    async def get(self, url: str, params: dict[str, str]) -> _FakeResponse:
        return _FakeResponse(SAMPLE, status_code=429)


@pytest.mark.asyncio
async def test_collect_psi_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSI_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    result = await collect_psi("https://example.com/", "mobile")

    assert result.error is None
    assert result.performance_score == 68.0
    assert result.lighthouse_version == "13.6.0"
    assert len(result.vitals) == 6
    assert result.vitals[2].audit == "largest-contentful-paint"
    assert result.vitals[2].display_value == "3.1 s"
    assert result.field_category == "AVERAGE"
    assert result.field_data[0].percentile == "2.50s"
    # Opportunities sorted by estimated savings, string items skipped.
    assert [o.audit for o in result.opportunities] == ["render-blocking-resources", "unused-javascript"]
    assert result.opportunities[0].savings_ms == 400
    assert len(result.third_parties) == 1
    assert result.third_parties[0].entity == "Google Fonts"
    assert result.third_parties[0].blocking_ms == 120
    assert result.diagnostics["mainthread-work-breakdown"] == "1.4 s"
    assert result.lcp_element == "body > img.hero (hero.jpg)"


@pytest.mark.asyncio
async def test_collect_psi_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    result = await collect_psi("https://example.com/", "desktop")

    assert result.error == "PSI_API_KEY is not set on the deployment"
    assert result.performance_score is None


@pytest.mark.asyncio
async def test_collect_psi_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSI_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", _QuotaClient)

    result = await collect_psi("https://example.com/", "mobile")

    assert result.error is not None and "429" in result.error


def test_input_defaults() -> None:
    parsed = AuditSiteInput.model_validate({"url": "https://rivilo.com"})
    assert parsed.strategies == ["mobile", "desktop"]

    single = AuditSiteInput.model_validate({"url": "https://rivilo.com", "strategies": ["mobile"]})
    assert single.strategies == ["mobile"]


def test_sample_fixture_is_json() -> None:
    assert isinstance(json.dumps(SAMPLE), str)
