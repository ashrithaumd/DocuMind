# Enterprise eval set

`items.jsonl` holds the permission-aware eval questions for the Phase 3
corpus (`kb/`), one JSON object per line. The target is about 60 items,
covering every type and both orgs. Three examples are in place; check the
file with:

    backend/.venv/Scripts/python backend/eval/enterprise/validate_items.py

It runs in CI too (`backend/tests/unit/test_enterprise_items.py`), so an
invalid item fails the build.

## Fields

| Field | Meaning |
|---|---|
| `id` | Unique, e.g. `aurora-exact-001`. |
| `org` | `aurora` or `bluefin`: the org the question is asked in. |
| `type` | One of the types below. |
| `question` | What the user asks, in their own words. |
| `as_user` | Who asks: `teams` (a list, possibly empty, of that org's team ids), `clearance` (0-3) and `role` (`viewer`, `editor`, `admin`). The eval signs in as a user with exactly this access. |
| `expected_outcome` | `answer` or `refuse`. Fixed by the type. |
| `expected_answer` | The answer a grader should accept. For refusals, say what the refusal should be, and add the hidden answer in brackets if there is one, so a leak is easy to spot. |
| `gold` | The text that answers it: a list of `{"doc": path under kb/docs/, "span": exact text from that document}`. Whitespace may differ (documents wrap lines); everything else must match. |
| `distractors` | For `version-trap`: the other version's text that must *not* win, same shape as `gold`. Otherwise an empty list. |
| `notes` | Why the item exists, for reviewers. Free text. |

## Types

| Type | Outcome | What it tests | Validator rule |
|---|---|---|---|
| `exact-term` | answer | An identifier (error code, part number, ticket id) that full-text search should find. | The user can see every gold document. |
| `paraphrase` | answer | A fact asked in different words from the document's; dense search should find it. | The user can see every gold document. |
| `version-trap` | answer | Two versions (2025 and 2026) both visible; the current one must win. | Needs a distractor, visible to the user, in a different document. |
| `multi-chunk` | answer | The answer needs text from more than one chunk. | At least two gold spans that don't all fit in one chunk at the app's default chunking. |
| `unanswerable` | refuse | Nothing in the corpus answers it. | No gold. (The validator can't prove absence: check by hand.) |
| `permission-restricted` | refuse | The answer exists, but this user can't see it. | The user can see no gold document, and no document they can see contains a gold span. |

Visibility follows the manifest exactly as the database does (RLS): same
org, the document's clearance at or below the user's, and the document
org-wide or in one of the user's teams (`kb/tools/manifest.py`, `can_see`).
