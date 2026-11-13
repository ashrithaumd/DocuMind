# FinPilot progress log

The plan, the architecture checklist and the commit budget are in
[FINPILOT_PLAN.md](FINPILOT_PLAN.md). DocuMind's own log (Phases 0–2 of the
enterprise build, its eval baselines and decisions) is in git history: see
`docs/PROGRESS.md` at commit 51ba57a.

**Commits: 3 of 22** (budget pending your choice of 20 or 22; see the plan).

## Phase 0: cleanup and rename (done)

| # | Commit | What |
|---|---|---|
| 1 | `0fccfd7` | Remove upload and the access-change API; rename DocuMind to FinPilot; migration 0008 moves the app role to `finpilot_app` |
| spare 1 | `00fbfda` | Integration tests loaded documents through `/upload`; they now use the store (CI caught it) |
| 2 | this commit | Plan, embedding pacing from `phase3-corpus`, `docs/baseline/` deleted, fresh progress log, README |

### Notes

- **Role rename without `ALTER ROLE ... RENAME`.** Roles are cluster-wide and
  every test database migrates on one server, so 0008 creates `finpilot_app`
  if missing, moves the grants to it and strips them from `documind_app` in
  that database. A test checks both directions.
- **`@documind.test` stays** in the eval-account guard: the real Firebase
  eval account and the Firebase project (`documind-29040`) use that name.
  New test accounts come with the Phase 1 users.
- **Local `.env` and dev volume untouched.** `.env` names the DB user and
  database `documind` explicitly, which still works; compose now builds
  `finpilot_app` URLs, and the migrate step runs 0008 on the existing volume.
- **From `phase3-corpus`**, only the generic parts came over: rate limiting
  by texts and tokens, paced embedding, a clean stop on the daily quota
  (`backend/ingestion/pacing.py`), and per-model embedding text counts. The
  `kb/` corpus, its manifest loader and test users stay on that branch.
- **Testing rule from here on** (your instruction, 2026-11-13): no Docker,
  no local Postgres, no Gemini calls on this machine. Light tests locally
  (backend tests without Postgres, frontend lint and build); GitHub CI runs
  the full Postgres suite on every push.
