"""PSI performance audit workflow.

Runs Google Lighthouse audits through the PageSpeed Insights API for one or
more strategies and returns a structured report. The Google API key is a fixed
service credential bound from the workspace secret ``psi_api_key`` into the
``PSI_API_KEY`` environment variable; it is resolved inside the activity only.
"""

from __future__ import annotations

import os
from datetime import timedelta

import structlog
from pydantic import BaseModel, Field

import mistralai.workflows as workflows

logger = structlog.get_logger("psi_audit")

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

METRIC_AUDITS = (
    "first-contentful-paint",
    "speed-index",
    "largest-contentful-paint",
    "total-blocking-time",
    "cumulative-layout-shift",
    "interactive",
)


class Vital(BaseModel):
    audit: str
    display_value: str | None = None
    score: float | None = None


class Opportunity(BaseModel):
    audit: str
    savings_ms: float = 0
    savings_bytes: float = 0
    display_value: str | None = None


class ThirdPartyImpact(BaseModel):
    entity: str
    total_bytes: float = 0
    blocking_ms: float = 0


class FieldMetric(BaseModel):
    metric_id: str
    percentile: str | None = None
    category: str | None = None


class StrategyResult(BaseModel):
    strategy: str
    performance_score: float | None = None
    lighthouse_version: str | None = None
    vitals: list[Vital] = Field(default_factory=list)
    field_data: list[FieldMetric] = Field(default_factory=list)
    field_category: str | None = None
    opportunities: list[Opportunity] = Field(default_factory=list)
    third_parties: list[ThirdPartyImpact] = Field(default_factory=list)
    lcp_element: str | None = None
    diagnostics: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class AuditSiteInput(BaseModel):
    """Audit one URL with Lighthouse via the PageSpeed Insights API."""

    url: str
    strategies: list[str] = Field(default_factory=lambda: ["mobile", "desktop"])


class AuditSiteOutput(BaseModel):
    url: str
    results: list[StrategyResult] = Field(default_factory=list)


def _human_ms(n: float) -> str:
    return f"{n / 1000:.2f}s" if n >= 1000 else f"{n:.0f}ms"


@workflows.activity(start_to_close_timeout=timedelta(seconds=300), retry_policy_max_attempts=1)
async def collect_psi(url: str, strategy: str) -> StrategyResult:
    """Fetch one PSI run and reduce it to a compact, business-level result.

    The Google API key never crosses the activity boundary: it is read from the
    deployment's runtime environment here, sent as a request parameter to the
    PSI API, and kept out of logs and results. ``retry_policy_max_attempts=1``
    because a retry would re-bill the key's daily quota for the same run and
    a partial failure is surfaced, not replayed.
    """
    import httpx

    api_key = os.environ.get("PSI_API_KEY", "").strip()
    if not api_key:
        return StrategyResult(strategy=strategy, error="PSI_API_KEY is not set on the deployment")

    params = {
        "url": url,
        "strategy": strategy,
        "category": "performance",
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.get(PSI_ENDPOINT, params=params)

    if response.status_code == 429:
        return StrategyResult(strategy=strategy, error="PSI quota exceeded (HTTP 429); retry later or check the Google Cloud quota")
    if response.status_code in (400, 403):
        return StrategyResult(strategy=strategy, error=f"PSI rejected the request (HTTP {response.status_code}); check the URL and the API key restrictions")

    data = response.json()
    lighthouse = data.get("lighthouseResult") or {}
    audits = lighthouse.get("audits") or {}
    categories = lighthouse.get("categories") or {}
    performance = categories.get("performance") or {}
    score = performance.get("score")

    result = StrategyResult(
        strategy=strategy,
        performance_score=round(score * 100, 1) if score is not None else None,
        lighthouse_version=lighthouse.get("lighthouseVersion"),
    )

    for audit_id in METRIC_AUDITS:
        audit = audits.get(audit_id) or {}
        if audit:
            result.vitals.append(
                Vital(
                    audit=audit_id,
                    display_value=audit.get("displayValue"),
                    score=audit.get("score"),
                )
            )

    loading = data.get("loadingExperience") or {}
    result.field_category = loading.get("overall_category")
    for metric_id, metric in (loading.get("metrics") or {}).items():
        percentile = metric.get("percentile")
        result.field_data.append(
            FieldMetric(
                metric_id=metric_id,
                percentile=_human_ms(percentile) if percentile is not None else None,
                category=metric.get("category"),
            )
        )

    for ref in performance.get("auditRefs", []):
        audit = audits.get(ref.get("id")) or {}
        details = audit.get("details") or {}
        savings_ms = 0.0
        savings_bytes = 0.0
        for item in details.get("items") or []:
            if not isinstance(item, dict):
                continue
            savings_ms += item.get("overallSavingsMs") or 0
            savings_bytes += item.get("overallSavingsBytes") or 0
        if audit.get("score") is not None and audit["score"] < 1 and (savings_ms or savings_bytes):
            result.opportunities.append(
                Opportunity(
                    audit=ref["id"],
                    savings_ms=savings_ms,
                    savings_bytes=savings_bytes,
                    display_value=audit.get("displayValue"),
                )
            )
    result.opportunities.sort(key=lambda o: (o.savings_ms, o.savings_bytes), reverse=True)

    third_party = (audits.get("third-party-summary") or {}).get("details") or {}
    for item in third_party.get("items") or []:
        if not isinstance(item, dict):
            continue
        result.third_parties.append(
            ThirdPartyImpact(
                entity=item.get("entity") or item.get("url") or "?",
                total_bytes=item.get("totalBytes") or 0,
                blocking_ms=item.get("blockingTime") or item.get("blocking") or 0,
            )
        )

    for diagnostic in ("mainthread-work-breakdown", "bootup-time", "dom-size", "server-response-time", "total-byte-weight"):
        audit = audits.get(diagnostic) or {}
        if audit.get("displayValue"):
            result.diagnostics[diagnostic] = audit["displayValue"]

    lcp = audits.get("largest-contentful-paint-element") or {}
    items = (lcp.get("details") or {}).get("items") or []
    if items:
        try:
            node = items[0]["items"][0]["node"]
            result.lcp_element = f"{node.get('selector', '?')} ({str(node.get('nodeLabel', '')).strip()})"
        except (KeyError, IndexError, TypeError):
            pass

    logger.info(
        "psi run complete",
        url=url,
        strategy=strategy,
        performance_score=result.performance_score,
    )
    return result


@workflows.workflow.define(
    name="psi-audit",
    workflow_display_name="PSI Audit",
    workflow_description="Run Google Lighthouse audits (PageSpeed Insights API) for a URL, mobile and desktop, and return a structured performance report.",
)
class PsiAuditWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: AuditSiteInput) -> AuditSiteOutput:
        results: list[StrategyResult] = []
        for strategy in input.strategies:
            results.append(await collect_psi(input.url, strategy))
        return AuditSiteOutput(url=input.url, results=results)
