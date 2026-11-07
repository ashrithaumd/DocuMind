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
284 chunks: 151 for Aurora and 133 for Bluefin, as loaded into the
Phase 3 database. The largest documents split into about 22 to 26 chunks
each. (An earlier count of 303 read a Windows checkout, where git turns
LF into CRLF: the extra carriage returns add tokens. The files are stored
with LF, and the loader reads them that way on every platform.)

## Access design

Assignments are in `kb/manifest.yaml`. Some documents are paired on
purpose, so each rule of the RLS policy (same org, clearance at or below the
user's, team empty or one of the user's) can fail on its own and show up in
an eval:

| Test | Documents (team, clearance) | What it proves |
|---|---|---|
| Clearance alone, org-wide | #4 CellCore full contract (org-wide, 2) vs #25 its summary (org-wide, 1) | A clearance-1 user in Aurora gets the summary's numbers but never the full contract; neither document is team-scoped, so only the clearance rule is in play. |
| Clearance alone, top level | #3 board notes (org-wide, 3) | Only clearance-3 users see it, whatever their team. |
| Team and clearance, same incident | #20 Atlas full report (platform, 2) vs #21 summary (org-wide, 1); #40 forecast full report (data-eng, 2) vs #41 summary (org-wide, 1) | Everyone in the org can read the summary; only the owning team at clearance 2 or above sees the full report's tickets and log details. A detail that only the full report contains, answered for anyone else, is a leak. |
| Team alone, same clearance, versioned exact terms | #17 Pika gen 2 manual (field-ops, 1) vs #18 gen 3 (org-wide, 1); #28 Current v3 (ml, 1) vs #29 v4 (org-wide, 1) | Same clearance, so only the team rule separates them. The older manual reuses error codes (E-2417, CUR-4107) with different fixes: outside the team, a question about the code must get the current version's answer only. |
| Team alone, at clearance 0 | #13 field-ops meeting notes (field-ops, 0); #35 ML meeting notes (ml, 0) | The lowest clearance, so only the team rule applies: a clearance-0 member of the team sees them, while a clearance-3 user outside the team doesn't. High clearance never substitutes for team membership. |
| Team and clearance 3 together | Salary bands #22, #23 (people, 3) and #45, #46 (finance, 3); #26 Gullwing acquisition plan (finance, 3) | Both rules must hold: clearance 3 outside the team, or the team below clearance 3, sees nothing. Finance also has documents at clearance 2 (#34 finance review, #37 Northgate MSA), so a finance member at clearance 2 sees those but not #26, #45 or #46: a clearance step inside one team. |
| Org isolation, mirrored facts | Expense policies #8 vs #33, on-call runbooks #16 (platform, 1) vs #39 (data-eng, 1), handbooks #5, #6 vs #30, #31, salary bands, postmortems | The same topics with different numbers in each org (meal allowance 75 vs 60 USD, on-call stipend 400 vs 350 USD). An answer with the other org's number is a cross-org leak. |
| Not an access test: versions at the same access | 2025 vs 2026 handbooks (#5/#6, #30/#31) and expense policies (#7/#8, #32/#33); 2026 vs 2027 salary bands | Both versions are visible to the same users, so these test version disambiguation in retrieval and answering, without being confounded with access. |

Counts: Aurora has 8 / 12 / 2 / 3 documents at clearance 0 / 1 / 2 / 3, 14
org-wide; Bluefin has 8 / 10 / 4 / 3, 13 org-wide. Every team owns at least
two documents.
