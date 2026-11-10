# psi-audit

A Mistral Workflow that runs Google Lighthouse audits through the PageSpeed
Insights API and returns a structured performance report (performance score,
Core Web Vitals, field data, ranked opportunities, third-party impact, LCP
element, diagnostics) for a URL, mobile and desktop.

## Workflow

- `psi-audit` (`apps/worker/src/worker/workflows/psi_audit.py`): input
  `{ "url": "...", "strategies": ["mobile", "desktop"] }`, output a structured
  `AuditSiteOutput`.

## Secret

The Google API key is bound at deploy time from the workspace secret
`psi_api_key` into the `PSI_API_KEY` environment variable. No credentials are
committed to this repository.

## Local development

```bash
bash tools/uv.sh sync --all-packages
bash tools/uv.sh run --all-packages pytest apps/worker/tests -q
bash tools/uv.sh tool run semgrep@1.176.0 \
  --config .agents/skills/capability-workflows/scripts/linting/rules \
  --metrics=off apps/worker/src/worker/workflows
```
