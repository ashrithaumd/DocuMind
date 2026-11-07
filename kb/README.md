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

`kb/docs/<org>/<file>.txt`, 25 per org, plain text so they upload as they
are. Every document states its year and version in its text.

- **30 short, hand-written documents** (15 per org): policies, runbooks,
  specs, meeting notes, short postmortems, contracts and pricing, salary
  bands, board and deal notes.
- **20 generated documents** (`kb/tools/build_long_docs.py`, fixed seed, so
  `python kb/tools/build_long_docs.py` rebuilds them identically): employee
  handbooks for 2025 and 2026, six service or operations manuals, full
  contracts, full incident reports, 2026 release notes, and older versions
  of the expense policy and salary bands. 13 of them are 3,000 to 8,000
  words.

What the generated documents plant on purpose (listed in `kb/facts.yaml`,
with a suggested question for each):

- **Exact-term facts** full-text search should win: error codes (`E-2417`,
  `CUR-4107`, `TD-3309`), part numbers (`AUR-PK3-4471-B`), config keys,
  ticket ids (`FOPS-88213`, `SUP-55102`), contract numbers and clause
  numbers.
- **Paraphrase facts** dense search should win, written in words a question
  wouldn't use (caregiver leave as "step away on full pay to look after a
  relative").
- **Versioned distractors**: 2025 and 2026 handbooks and expense policies
  with changed numbers; Pika generation 2 (2024) next to generation 3, and
  Current v3 (2025) next to v4, which reuse error codes with different
  thresholds and fixes; 2026 and 2027 salary bands.
- **Mirrored facts across the orgs**: the same topics with different
  numbers (meal allowance 75 vs 60 USD, gift limit, parental leave, on-call
  stipend, and so on).

Facts stay consistent across documents: the long incident reports match
their one-page summaries, the full CellCore contract matches its summary,
and an error code quoted in a postmortem is the one in the manual.

## Size

At the app's default chunking (600 tokens, 90 overlap), the corpus is
303 chunks; the largest documents split into about 25 to 29
chunks each.
