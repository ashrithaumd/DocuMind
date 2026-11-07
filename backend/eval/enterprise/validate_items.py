"""Validates the enterprise eval set (items.jsonl) against the corpus.

    python backend/eval/enterprise/validate_items.py [items.jsonl]

Checks, for every item:
  - the fields and their values (see README.md here);
  - every gold and distractor span appears in its document (whitespace
    normalised, since documents wrap lines);
  - the asking user's access, by the manifest's rules (the RLS policy),
    matches the expected outcome:
      answer types: every gold document is visible to the user;
      permission-restricted: no gold document is visible, and no visible
        document contains a gold span (a summary the user can see would
        make it answerable);
      unanswerable: no gold;
      version-trap: the distractor (the other version) is visible too,
        otherwise it isn't a trap;
      multi-chunk: the gold spans don't all fall inside one chunk at the
        app's default chunking.
Prints counts by org and type, and exits 1 if any item is invalid.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "kb" / "tools"))
from manifest import DOCS, Manifest, User, can_see, doc_text, load_manifest  # noqa: E402

ITEMS = HERE / "items.jsonl"
ANSWER_TYPES = {"exact-term", "paraphrase", "version-trap", "multi-chunk"}
REFUSE_TYPES = {"unanswerable", "permission-restricted"}
TYPES = ANSWER_TYPES | REFUSE_TYPES
ROLES = {"viewer", "editor", "admin"}
FIELDS = {"id", "org", "type", "question", "as_user", "expected_outcome", "expected_answer",
          "gold", "distractors", "notes"}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_items(path: Path) -> list[tuple[int, dict | None, str | None]]:
    """(line number, item or None, parse error or None) per non-empty line."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            out.append((n, None, f"line {n}: not valid JSON: {exc}"))
            continue
        out.append((n, item, None))
    return out


class ChunkIndex:
    """Which chunks of a document contain a span, at the app's default chunking."""

    def __init__(self):
        self._chunks: dict[str, list[str]] = {}

    def containing(self, file: str, span: str) -> set[int]:
        if file not in self._chunks:
            sys.path.insert(0, str(REPO / "backend"))
            from config import Settings
            from ingestion.chunking import chunk_pages
            from ingestion.extract import PageText

            settings = Settings(_env_file=None)
            pieces = chunk_pages([PageText(page_number=1, text=doc_text(file))], file, settings)
            self._chunks[file] = [normalise(c.text) for c in pieces]
        needle = normalise(span)
        return {i for i, text in enumerate(self._chunks[file]) if needle in text}


def check_item(item: dict, manifest: Manifest, chunks: ChunkIndex, docs_dir: Path = DOCS) -> list[str]:
    errors: list[str] = []
    if set(item) != FIELDS:
        missing, extra = FIELDS - set(item), set(item) - FIELDS
        return [f"fields: missing {sorted(missing)}, unexpected {sorted(extra)}"]

    kind, org = item["type"], item["org"]
    if kind not in TYPES:
        errors.append(f"type {kind!r} is not one of {sorted(TYPES)}")
    if org not in manifest.orgs:
        return errors + [f"unknown org {org!r}"]
    for key in ("question", "expected_answer"):
        if not isinstance(item[key], str) or not item[key].strip():
            errors.append(f"{key} must be a non-empty string")

    u = item["as_user"]
    if not isinstance(u, dict) or set(u) != {"teams", "clearance", "role"}:
        return errors + ["as_user needs exactly teams, clearance, role"]
    bad_teams = [t for t in u["teams"] if t not in manifest.teams(org)]
    if bad_teams:
        errors.append(f"as_user.teams {bad_teams} are not teams of {org}")
    if not isinstance(u["clearance"], int) or not 0 <= u["clearance"] <= 3:
        errors.append("as_user.clearance must be an integer 0-3")
    if u["role"] not in ROLES:
        errors.append(f"as_user.role must be one of {sorted(ROLES)}")
    if errors:
        return errors
    user = User(org=org, teams=tuple(u["teams"]), clearance=u["clearance"])

    expected = "answer" if kind in ANSWER_TYPES else "refuse"
    if item["expected_outcome"] != expected:
        errors.append(f"expected_outcome must be {expected!r} for type {kind}")

    # Every span must be real text in its document.
    for field in ("gold", "distractors"):
        for ref in item[field]:
            if not isinstance(ref, dict) or set(ref) != {"doc", "span"}:
                errors.append(f"{field} entries need exactly doc and span: {ref!r}")
                continue
            if ref["doc"] not in manifest.docs:
                errors.append(f"{field}: {ref['doc']} is not in the manifest")
            elif normalise(ref["span"]) not in normalise(doc_text(ref["doc"], docs_dir)):
                errors.append(f"{field}: span not found in {ref['doc']}: {ref['span'][:80]!r}")
    if errors:
        return errors

    gold = [(manifest.docs[r["doc"]], r["span"]) for r in item["gold"]]
    visible = [d for d in manifest.docs.values() if can_see(user, d)]

    if kind in ANSWER_TYPES:
        if not gold:
            errors.append("an answerable item needs at least one gold span")
        for doc, _ in gold:
            if not can_see(user, doc):
                errors.append(f"the user can't see gold document {doc.file} "
                              f"(team {doc.team}, clearance {doc.clearance}); use permission-restricted?")
    if kind == "unanswerable" and gold:
        errors.append("unanswerable items have no gold")
    if kind == "permission-restricted":
        if not gold:
            errors.append("permission-restricted needs the gold the user must not see")
        for doc, span in gold:
            if can_see(user, doc):
                errors.append(f"the user can see gold document {doc.file}, so it isn't restricted")
            for other in visible:
                if normalise(span) in normalise(doc_text(other.file, docs_dir)):
                    errors.append(f"visible document {other.file} also contains the gold span, "
                                  "so the user could answer it")
    if kind == "version-trap":
        if not item["distractors"]:
            errors.append("version-trap needs the other version as a distractor")
        for r in item["distractors"]:
            d = manifest.docs[r["doc"]]
            if not can_see(user, d):
                errors.append(f"distractor {d.file} isn't visible to the user, so it's no trap")
            if d.file in {g.file for g, _ in gold}:
                errors.append("the distractor must be a different document from the gold")
    if kind == "multi-chunk":
        if len(gold) < 2:
            errors.append("multi-chunk needs at least two gold spans")
        else:
            sets = []
            for doc, span in gold:
                found = chunks.containing(doc.file, span)
                if not found:
                    errors.append(f"span crosses a chunk boundary in {doc.file}; shorten it: {span[:60]!r}")
                sets.append({(doc.file, i) for i in found})
            if sets and all(sets) and set.intersection(*sets):
                errors.append("all gold spans fit in one chunk; that's not multi-chunk")
    return errors


def validate(path: Path = ITEMS, manifest: Manifest | None = None,
             docs_dir: Path = DOCS) -> tuple[list[str], Counter]:
    manifest = manifest or load_manifest()
    chunks, problems, counts, ids = ChunkIndex(), [], Counter(), set()
    for n, item, parse_error in load_items(path):
        if parse_error:
            problems.append(parse_error)
            continue
        label = f"line {n} ({item.get('id', '?')})"
        if item.get("id") in ids:
            problems.append(f"{label}: duplicate id")
        ids.add(item.get("id"))
        for e in check_item(item, manifest, chunks, docs_dir):
            problems.append(f"{label}: {e}")
        counts[(item.get("org"), item.get("type"))] += 1
    return problems, counts


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ITEMS
    problems, counts = validate(path)
    total = sum(counts.values())
    print(f"{total} items in {path.name}")
    for org in sorted({o for o, _ in counts}):
        row = ", ".join(f"{t} {counts[(org, t)]}" for t in sorted(TYPES) if counts[(org, t)])
        print(f"  {org}: {row}")
    missing = sorted(t for t in TYPES if not any(counts[(o, t)] for o, _ in counts))
    if missing:
        print(f"  types with no items yet: {', '.join(missing)}")
    for p in problems:
        print(f"INVALID {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
