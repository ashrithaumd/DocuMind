"""kb/manifest.yaml: loading, validation, and who may see each document.

Shared by the eval-set validator (backend/eval/enterprise/validate_items.py)
and the corpus loader (scripts/load_corpus.py), so both read the manifest
the same way. can_see() is the RLS policy (migration 0004) written in
Python: same org, clearance at or below the user's, and the document
org-wide or in one of the user's teams.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

KB = Path(__file__).resolve().parents[1]
DOCS = KB / "docs"
MANIFEST = KB / "manifest.yaml"
CLEARANCES = range(0, 4)


class ManifestError(ValueError):
    """The manifest is malformed or disagrees with kb/docs/. Lists every problem."""


@dataclass(frozen=True)
class Doc:
    file: str            # path under kb/docs/, e.g. "aurora/expense-policy.txt"
    org: str
    team: str | None     # None = org-wide
    clearance: int


@dataclass(frozen=True)
class Manifest:
    orgs: dict[str, dict]          # org id -> {"name": ..., "teams": [...]}
    docs: dict[str, Doc]           # file -> Doc, in manifest order

    def teams(self, org: str) -> list[str]:
        return list(self.orgs[org]["teams"])


@dataclass(frozen=True)
class User:
    org: str
    teams: tuple[str, ...]
    clearance: int


def can_see(user: User, doc: Doc) -> bool:
    """The RLS policy: same org, clearance at or below, team empty or the user's."""
    return (user.org == doc.org and doc.clearance <= user.clearance
            and (doc.team is None or doc.team in user.teams))


def load_manifest(path: Path = MANIFEST, docs_dir: Path = DOCS) -> Manifest:
    """Parse and validate. Raises ManifestError listing every problem found."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: not valid YAML: {exc}") from exc
    problems: list[str] = []
    if not isinstance(raw, dict) or set(raw) != {"orgs", "documents"}:
        raise ManifestError(f"{path}: needs exactly the keys 'orgs' and 'documents'")

    orgs = raw["orgs"] or {}
    seen_teams: dict[str, str] = {}
    for org, spec in orgs.items():
        if not isinstance(spec, dict) or set(spec) != {"name", "teams"}:
            problems.append(f"org {org}: needs exactly 'name' and 'teams'")
            continue
        for team in spec["teams"] or []:
            # teams.team_id is the primary key: unique across all orgs.
            if team in seen_teams:
                problems.append(f"team {team} is in both {seen_teams[team]} and {org}; "
                                "team ids must be unique across orgs")
            seen_teams[team] = org

    docs: dict[str, Doc] = {}
    for i, entry in enumerate(raw["documents"] or [], start=1):
        where = f"document #{i}"
        if not isinstance(entry, dict) or set(entry) != {"file", "org", "team", "clearance"}:
            problems.append(f"{where}: needs exactly file, org, team, clearance; got {entry!r}")
            continue
        file, org, team, clearance = entry["file"], entry["org"], entry["team"], entry["clearance"]
        where = f"document #{i} ({file})"
        if file in docs:
            problems.append(f"{where}: listed twice")
        if org not in orgs:
            problems.append(f"{where}: unknown org {org!r}")
        elif not str(file).startswith(f"{org}/"):
            problems.append(f"{where}: file is not under its org's folder {org}/")
        elif team is not None and team not in (orgs[org].get("teams") or []):
            problems.append(f"{where}: team {team!r} is not a team of {org}")
        if not isinstance(clearance, int) or isinstance(clearance, bool) or clearance not in CLEARANCES:
            problems.append(f"{where}: clearance must be an integer 0-3, got {clearance!r}")
        if not (Path(docs_dir) / str(file)).is_file():
            problems.append(f"{where}: no such file under {docs_dir}")
        docs[file] = Doc(file=file, org=org, team=team, clearance=clearance)

    on_disk = {p.relative_to(docs_dir).as_posix() for p in Path(docs_dir).rglob("*.txt")}
    for missing in sorted(on_disk - set(docs)):
        problems.append(f"{missing}: in kb/docs/ but not in the manifest")
    if problems:
        raise ManifestError("\n".join(problems))
    return Manifest(orgs=orgs, docs=docs)


def doc_text(file: str, docs_dir: Path = DOCS) -> str:
    """The document's text with LF line endings, whatever the checkout did:
    a CRLF copy must chunk, hash and embed exactly like the LF original."""
    return normalise_newlines((Path(docs_dir) / file).read_bytes().decode("utf-8"))


def normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
