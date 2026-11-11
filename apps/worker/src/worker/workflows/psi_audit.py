"""Run Google Lighthouse audits through the PageSpeed Insights (PSI) API.

The API key never appears in code: the platform injects the secret bound to the
deployment as the ``PSI_API_KEY`` environment variable. All four Lighthouse
categories are requested, matching what the PageSpeed Insights site shows.
"""

import asyncio
import os
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from pydantic import BaseModel, Field, field_validator

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
DEFAULT_STRATEGIES = ("mobile", "desktop")
# Every category the PSI site reports.
CATEGORIES = ("performance", "accessibility", "best-practices", "seo")


class PsiInput(BaseModel):
    """Workflow input: the page to audit and the device strategies to run."""

    url: str = Field(description="Absolute page URL to audit, e.g. https://example.com")
    strategies: list[str] = Field(
        default_factory=lambda: list(DEFAULT_STRATEGIES),
        description='Device strategies to audit: "mobile", "desktop" or both.',
    )

    @field_validator("strategies")
    @classmethod
    def _keep_known_strategies(cls, value: list[str]) -> list[str]:
        known = [s for s in value if s in DEFAULT_STRATEGIES]
        return known or list(DEFAULT_STRATEGIES)


class Metric(BaseModel):
    """A single lab metric: value plus its category assessment."""

    value: float | None = None
    assessment: str | None = None


class AuditOpportunity(BaseModel):
    """A savings estimate an audit opportunity reports."""

    audit: str
    title: str
    savings_ms: float | None = None


class ThirdParty(BaseModel):
    """A third-party origin the page loads and its blocking time."""

    origin: str
    blocking_ms: float | None = None


class AuditResult(BaseModel):
    """A single audit in brief: weight and, for failures, what to fix."""

    id: str
    title: str
    state: str  # pass / fail / informative
    display: str | None = None
    description: str | None = None
    savings_ms: float | None = None


class StrategyResult(BaseModel):
    """Structured PSI report for one strategy, all categories."""

    strategy: str
    ok: bool = False
    performance_score: int | None = None
    error: str | None = None
    categories: dict[str, int] = Field(default_factory=dict)  # all four scores 0-100
    first_contentful_paint_ms: float | None = None
    largest_contentful_paint_ms: float | None = None
    total_blocking_time_ms: float | None = None
    cumulative_layout_shift: float | None = None
    speed_index_ms: float | None = None
    interaction_to_next_paint_ms: float | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)
    field_data: dict[str, Any] = Field(default_factory=dict)
    opportunities: list[AuditOpportunity] = Field(default_factory=list)
    passed_audits: list[AuditResult] = Field(default_factory=list)
    failed_audits: list[AuditResult] = Field(default_factory=list)
    third_parties: list[ThirdParty] = Field(default_factory=list)
    lcp_element: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    screenshot: str | None = None


def _metric_value(audit: dict[str, Any] | None) -> tuple[float | None, str | None]:
    """Read numeric value and assessment from a PSI audit item, tolerating gaps."""

    if not isinstance(audit, dict):
        return None, None
    raw = audit.get("numericValue")
    value = float(raw) if isinstance(raw, (int, float)) else None
    score = audit.get("score")
    if score is None:
        assessment = "informative"
    elif score >= 0.9:
        assessment = "good"
    elif score >= 0.5:
        assessment = "needs improvement"
    else:
        assessment = "poor"
    return value, assessment


def _category_score(audits: dict[str, Any], category: dict[str, Any]) -> int | None:
    """A category score can come with scoreOverrideMode=informative; use its audits then."""

    raw = category.get("score")
    if isinstance(raw, (int, float)):
        return round(float(raw) * 100)
    # Manual mode: average the referenced audits' scores.
    refs = category.get("auditRefs", [])
    scores = []
    for ref in refs:
        audit = audits.get(ref.get("id") if isinstance(ref, dict) else None, {})
        s = audit.get("score") if isinstance(audit, dict) else None
        if isinstance(s, (int, float)):
            scores.append(s)
    return round(sum(scores) / len(scores) * 100) if scores else None


def _audit_state(audit: dict[str, Any]) -> str:
    score = audit.get("score")
    if score is None:
        return "informative"
    return "pass" if score >= 0.9 else "fail"


def _parse_lightning(result: dict[str, Any]) -> StrategyResult:
    """Turn one PSI `lighthouseResult` into a StrategyResult."""

    audits: dict[str, Any] = result.get("audits", {})
    categories = result.get("categories", {})
    out = StrategyResult(strategy=result.get("_strategy", ""), ok=True)

    scores: dict[str, int] = {}
    for cat_id, cat in categories.items():
        if not isinstance(cat, dict):
            continue
        s = _category_score(audits, cat)
        if s is not None:
            scores[cat_id] = s
    out.categories = scores
    out.performance_score = scores.get("performance")

    for key, audit_id in (
        ("first_contentful_paint_ms", "first-contentful-paint"),
        ("largest_contentful_paint_ms", "largest-contentful-paint"),
        ("total_blocking_time_ms", "total-blocking-time"),
        ("cumulative_layout_shift", "cumulative-layout-shift"),
        ("speed_index_ms", "speed-index"),
        ("interaction_to_next_paint_ms", "interactive" if "interactive" in audits else "max-potential-fid"),
    ):
        value, assessment = _metric_value(audits.get(audit_id))
        setattr(out, key, value)
        if value is not None:
            out.metrics[audit_id] = Metric(value=value, assessment=assessment)

    opportunities: list[AuditOpportunity] = []
    passed: list[AuditResult] = []
    failed: list[AuditResult] = []
    for audit_id, audit in audits.items():
        if not isinstance(audit, dict):
            continue
        details = audit.get("details", {}) if isinstance(audit.get("details"), dict) else {}
        state = _audit_state(audit)
        if details.get("type") == "opportunity":
            savings = details.get("overallSavingsMs")
            if isinstance(savings, (int, float)) and savings >= 10:
                opportunities.append(
                    AuditOpportunity(audit=audit_id, title=audit.get("title", audit_id), savings_ms=float(savings))
                )
            continue
        # Keep informative audits out; track scored pass/fail for parity with the PSI site.
        if state == "pass":
            passed.append(AuditResult(id=audit_id, title=audit.get("title", audit_id), state=state))
        elif state == "fail":
            failed.append(
                AuditResult(
                    id=audit_id,
                    title=audit.get("title", audit_id),
                    state=state,
                    display=audit.get("displayValue"),
                    description=audit.get("description"),
                )
            )
    opportunities.sort(key=lambda o: o.savings_ms or 0, reverse=True)
    out.opportunities = opportunities[:10]
    out.failed_audits = [
        a for a in failed
        if a.id not in {o.audit for o in out.opportunities}
    ][:20]
    out.passed_audits = passed

    third_parties: list[ThirdParty] = []
    tpa = audits.get("third-party", {}).get("details", {}).get("items", [])
    for item in tpa:
        if not isinstance(item, dict):
            continue
        blocking = item.get("blockingTime")
        if isinstance(blocking, (int, float)) and blocking > 0:
            third_parties.append(ThirdParty(origin=item.get("entity", ""), blocking_ms=float(blocking)))
    third_parties.sort(key=lambda t: t.blocking_ms or 0, reverse=True)
    out.third_parties = third_parties[:10]

    lcp_item = audits.get("largest-contentful-paint-element", {}).get("details", {}).get("items", [])
    if lcp_item:
        out.lcp_element = str(lcp_item[0])

    diagnostics = audits.get("diagnostics", {}).get("details", {}).get("items", [])
    if diagnostics:
        out.diagnostics = diagnostics[0]

    shot = result.get("full-page-screenshot", {})
    if isinstance(shot, dict):
        data = shot.get("data")
        if isinstance(data, str):
            out.screenshot = f"data:image/jpeg;base64,{data}" if not data.startswith("data:") else data

    return out


def _parse_field_data(section: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize the CrUX field data section, if the page has traffic."""

    if not isinstance(section, dict) or not section.get("metrics"):
        return {}
    metrics: dict[str, Any] = {}
    for key, metric in section.get("metrics", {}).items():
        if not isinstance(metric, dict):
            continue
        metrics[key] = {"value": metric.get("percentile"), "assessment": metric.get("category")}
    return {"overall_category": section.get("overall_category"), "metrics": metrics}


def _error_message(status: int, body: str) -> str:
    """Map an HTTP failure to a readable, non-leaking error string."""

    if status == 429:
        return "PSI quota exceeded (HTTP 429): retry later or request a higher quota."
    if status in (400, 403):
        return f"PSI rejected the request (HTTP {status}): check the URL and the API key restrictions."
    return f"PSI request failed with HTTP {status}."


@workflows.activity(start_to_close_timeout=timedelta(seconds=420), retry_policy_max_attempts=1)
async def collect_psi(url: str, strategy: str) -> dict[str, Any]:
    """Fetch one PSI report for `url` under `strategy` and structure it."""

    # Imported here so the module stays importable without httpx (worker startup,
    # offline tests) and semgrep's import rules are satisfied.
    import httpx

    api_key = os.environ.get("PSI_API_KEY", "").strip()
    if not api_key:
        return StrategyResult(
            strategy=strategy, ok=False, error="PSI_API_KEY is not set: bind the psi_api_key secret to the deployment."
        ).model_dump()

    params: dict[str, Any] = {"url": url, "strategy": strategy, "key": api_key}
    params["category"] = list(CATEGORIES)
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.get(PSI_ENDPOINT, params=params)

    if response.status_code != 200:
        return StrategyResult(strategy=strategy, ok=False, error=_error_message(response.status_code, response.text)).model_dump()

    payload = response.json()
    lighthouse = payload.get("lighthouseResult", {})
    if not lighthouse:
        return StrategyResult(
            strategy=strategy, ok=False, error="PSI returned no lighthouse result for this URL."
        ).model_dump()

    parsed = _parse_lightning(lighthouse)
    parsed.strategy = strategy
    parsed.field_data = _parse_field_data(payload.get("loadingExperience", {}))
    return parsed.model_dump()


@workflows.workflow.define(name="psi-audit")
class PsiAuditWorkflow:
    """Audit `url` with Lighthouse on the requested strategies, in parallel."""

    @workflows.workflow.entrypoint
    async def run(self, audit: PsiInput) -> dict[str, Any]:
        strategies = [s for s in audit.strategies if s in DEFAULT_STRATEGIES] or list(DEFAULT_STRATEGIES)

        futures = [collect_psi(audit.url, strategy) for strategy in strategies]
        results = await asyncio.gather(*futures)
        by_strategy = {r["strategy"]: r for r in results if isinstance(r, dict) and r.get("strategy")}
        return {"url": audit.url, "results": by_strategy}
