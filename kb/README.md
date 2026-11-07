# Phase 3 corpus: two fictional companies

A small, permission-aware test corpus for DocuMind: two orgs with parallel
documents (each has an expense policy, an on-call runbook, salary bands, and
so on) but different facts. If row-level security ever leaked, an eval
question would come back with the other org's number, so leakage shows up as
a wrong answer, not just a log line.

Everything here is fictional: the companies, people, numbers and products.

## Orgs and teams

| Org id | Company | What it does | Teams |
|---|---|---|---|
| `aurora` | Aurora Robotics | Builds and services autonomous warehouse robots (the "Pika" picker and "Atlas" mover) | `platform`, `firmware`, `field-ops`, `people` |
| `bluefin` | Bluefin Analytics | A data-analytics SaaS for retail chains ("Tide" dashboards, "Current" forecasting) | `data-eng`, `ml`, `support`, `finance` |

## Clearance levels (both orgs)

| Level | Meaning | Examples |
|---|---|---|
| 0 | Everyone in the org | Expense policy, onboarding guide, holiday calendar |
| 1 | Internal working documents | Runbooks, product specs, meeting notes |
| 2 | Confidential | Incident postmortems with customer names, pricing, vendor contracts |
| 3 | Restricted | Salary bands, board notes, acquisition plans |

## Documents

`kb/docs/<org>/<file>.txt`, 15 per org, plain text so they upload as they
are. Lengths vary on purpose, from a short memo to a long runbook. Each
holds a few specific, checkable facts (numbers, names, dates, thresholds)
for eval questions to target. Access assignments live in
`kb/manifest.yaml`.

| Aurora Robotics | Bluefin Analytics |
|---|---|
| expense-policy | expense-policy |
| onboarding-guide | onboarding-guide |
| holiday-calendar-2027 | remote-work-policy |
| oncall-runbook-platform | oncall-runbook-data-eng |
| firmware-release-process | ml-model-release-checklist |
| pika-picker-spec | current-forecasting-spec |
| atlas-battery-spec | tide-dashboards-spec |
| field-service-runbook | support-escalation-runbook |
| meeting-notes-platform-2026-09-14 | meeting-notes-ml-2026-10-02 |
| meeting-notes-field-ops-2026-10-08 | meeting-notes-finance-2026-09-30 |
| postmortem-2026-08-atlas-fleet-stall | postmortem-2026-07-forecast-outage |
| vendor-contract-cellcore | pricing-2027 |
| security-policy | security-policy |
| salary-bands-2027 | salary-bands-2027 |
| board-notes-2026-q3 | acquisition-plan-gullwing |
