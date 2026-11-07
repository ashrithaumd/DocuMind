"""Load the Phase 3 corpus (kb/) into Postgres, as the owner. Idempotent.

    backend/.venv/Scripts/python scripts/load_corpus.py --database-url OWNER_URL
        [--dry-run] [--prune] [--per-minute 90] [--tokens-per-minute 27000]

Reads and validates kb/manifest.yaml, then:
  1. upserts each org and its teams (team ids are unique across orgs);
  2. for each document, in its own transaction, decides by comparing with
     what the database already holds:
       insert          not loaded yet;
       skip            same content, same access;
       update_access   same content, the manifest changed its team or
                       clearance: the document and all its chunks change in
                       one statement, as in HybridStore.set_document_access;
       replace         the content changed: the old document and chunks are
                       deleted and the new ones inserted;
  3. reports loaded documents that are no longer in the manifest, and deletes
     them only with --prune.
Re-running with nothing changed skips everything and writes nothing.

Writes as the owner (MIGRATION_DATABASE_URL, never from .env), like the
migrate step and setup_eval_admin.py: the app role can only write documents
into its own access, and this loads every org. The owner must be able to
bypass RLS (true for the Docker Postgres owner, a superuser). Run the migrate
step first.

Documents go in like an upload: the documents row first, then the chunks,
which copy org, team and clearance from that row in SQL (the same statements
as HybridStore.add_document), so a chunk can't differ from its document.
Chunking is the app's (backend/ingestion), and embeddings use the configured
provider (EMBEDDING_PROVIDER etc., as for the backend): loading the whole
corpus is one embedding call per chunk, about 300.
Document ids are "kb-" plus a hash of the path and content, so the same
file always gets the same id and an edited file gets a new one.

Pacing and quota. Gemini's free tier limits embedding requests (about 100 a
minute, every text counting as one) and, the tighter limit in practice,
embedding tokens (30,000 a minute, measured in the Phase 3 load):
  - texts are paced to at most --per-minute (default 90) and their tokens to
    at most --tokens-per-minute (default 27,000) in any rolling minute,
    before they are sent, counted with the chunker's tokenizer (cl100k_base,
    an estimate of Gemini's own count, hence the margin);
  - a per-minute 429 that outlasts the provider's own retries waits a
    minute and tries the document again (up to 3 times);
  - a daily-quota error stops the run cleanly (exit code 3) with how far it
    got. Embedding happens before a document's write transaction opens, and
    each document is written in one transaction, so nothing half-loaded is
    ever left: re-run the same command, and loaded documents are skipped.
At the end it reports the corpus's chunks per org and, from the embedding
client's own usage tracking (so its internal retries are included), the
texts and requests sent and the retries, plus the tokens paced.
Text is read with LF line endings whatever the checkout did, so a CRLF copy
chunks and hashes exactly like the original.
"""
import argparse
import asyncio
import hashlib
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kb" / "tools"))
sys.path.insert(0, str(REPO / "backend"))
from manifest import DOCS, Doc, Manifest, doc_text, load_manifest, normalise_newlines  # noqa: E402

PREFIX = "kb-"
EXIT_QUOTA = 3
TOKENS_PER_MINUTE = 27_000  # under the measured free-tier limit of 30,000
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 60


class QuotaStop(Exception):
    """The provider's daily quota is used up; the run stopped cleanly.
    `counts` holds the usage up to that point."""
    model: str | None = None
    counts: dict = {}


class RateLimiter:
    """At most `per_minute` units (texts, or tokens) in any rolling 60-second window."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep):
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.per_minute = per_minute
        self._clock, self._sleep = clock, sleep
        self._sent: deque[tuple[float, int]] = deque()

    def acquire(self, n: int) -> None:
        if n > self.per_minute:
            raise ValueError(f"{n} can never fit in {self.per_minute} a minute; split them")
        while True:
            now = self._clock()
            while self._sent and self._sent[0][0] <= now - 60:
                self._sent.popleft()
            if sum(count for _, count in self._sent) + n <= self.per_minute:
                self._sent.append((now, n))
                return
            self._sleep(self._sent[0][0] + 60 - now)


def count_tokens(text: str) -> int:
    """The chunker's tokenizer (cl100k_base), with the prefix the Gemini
    embedder adds to every document text."""
    import tiktoken

    from embeddings import GEMINI_DOCUMENT_PREFIX

    return len(tiktoken.get_encoding("cl100k_base").encode(GEMINI_DOCUMENT_PREFIX + text))


class PacedEmbedder:
    """Sends texts in slices that fit both limits (texts and tokens per
    minute), waiting for each. tokens_paced counts the tokens it let through
    (first sends; retries resend them)."""

    def __init__(self, embeddings, limiter: RateLimiter, token_limiter: RateLimiter | None = None,
                 tokens=count_tokens):
        self._embeddings, self._limiter = embeddings, limiter
        self._token_limiter = token_limiter or RateLimiter(TOKENS_PER_MINUTE)
        self._tokens = tokens
        self.model, self.dimension = embeddings.model, embeddings.dimension
        self.tokens_paced = 0

    def _slices(self, texts: list[str]):
        part, part_tokens = [], 0
        for text in texts:
            n = self._tokens(text)
            if part and (len(part) == self._limiter.per_minute
                         or part_tokens + n > self._token_limiter.per_minute):
                yield part, part_tokens
                part, part_tokens = [], 0
            part.append(text)
            part_tokens += n
        if part:
            yield part, part_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for part, part_tokens in self._slices(texts):
            self._limiter.acquire(len(part))
            self._token_limiter.acquire(part_tokens)
            self.tokens_paced += part_tokens
            vectors += asyncio.run(self._embeddings.embed_documents(part))
        return vectors


@dataclass(frozen=True)
class Loaded:
    """What the database holds for one corpus file."""
    doc_id: str
    team: str | None
    clearance: int


def doc_id_for(file: str, text: str) -> str:
    return PREFIX + hashlib.sha256(f"{file}\0{text}".encode("utf-8")).hexdigest()[:16]


def filename_for(doc: Doc) -> str:
    """The name users see. Unique within an org: kb/docs/<org>/ has no subfolders."""
    return Path(doc.file).name


def plan(doc: Doc, doc_id: str, loaded: list[Loaded]) -> str:
    """insert | skip | update_access | replace, for one manifest entry."""
    if not loaded:
        return "insert"
    if len(loaded) == 1 and loaded[0].doc_id == doc_id:
        same = (loaded[0].team, loaded[0].clearance) == (doc.team, doc.clearance)
        return "skip" if same else "update_access"
    return "replace"


def chunk(doc: Doc, doc_id: str, text: str):
    """The app's chunking, with ids derived from doc_id (not a random one)."""
    from config import Settings
    from ingestion.chunking import Chunk, chunk_pages
    from ingestion.extract import PageText

    pieces = chunk_pages([PageText(page_number=1, text=text)], filename_for(doc), Settings())
    return [Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename=filename_for(doc),
                  page_number=p.page_number, chunk_index=i, text=p.text)
            for i, p in enumerate(pieces)]


# --- database (owner) ------------------------------------------------------------

def _check_owner(conn, embeddings) -> None:
    import sqlalchemy as sa

    if not conn.execute(sa.text("SELECT rolsuper OR rolbypassrls FROM pg_roles "
                                "WHERE rolname = current_user")).scalar():
        raise SystemExit("This role can't bypass row-level security. Use the owner's URL.")
    if conn.execute(sa.text("SELECT to_regclass('documents')")).scalar() is None:
        raise SystemExit("No documents table: run the migrate step (backend/migrate.py) first.")
    if embeddings is not None:
        column = conn.execute(sa.text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'")).scalar()
        if column != f"vector({embeddings.dimension})":
            raise SystemExit(f"chunks.embedding is {column}, but the configured embedding model "
                             f"produces vector({embeddings.dimension}).")


def _upsert_orgs(conn, manifest: Manifest) -> None:
    import sqlalchemy as sa

    for org, spec in manifest.orgs.items():
        conn.execute(sa.text("INSERT INTO orgs VALUES (:org, :name) "
                             "ON CONFLICT (org_id) DO UPDATE SET name = EXCLUDED.name"),
                     {"org": org, "name": spec["name"]})
        for team in spec["teams"]:
            owner = conn.execute(sa.text("SELECT org_id FROM teams WHERE team_id = :t"),
                                 {"t": team}).scalar()
            if owner is None:
                conn.execute(sa.text("INSERT INTO teams VALUES (:t, :org, :t)"),
                             {"t": team, "org": org})
            elif owner != org:
                raise SystemExit(f"Team {team} already exists in org {owner}, not {org}.")


def _loaded(conn, doc: Doc) -> list[Loaded]:
    import sqlalchemy as sa

    rows = conn.execute(sa.text(
        "SELECT doc_id, team_id, clearance FROM documents "
        "WHERE org_id = :org AND filename = :name AND doc_id LIKE :prefix"),
        {"org": doc.org, "name": filename_for(doc), "prefix": PREFIX + "%"}).all()
    return [Loaded(doc_id=r[0], team=r[1], clearance=r[2]) for r in rows]


def _delete(conn, doc_ids: list[str]) -> None:
    import sqlalchemy as sa

    for doc_id in doc_ids:
        conn.execute(sa.text("DELETE FROM chunks WHERE doc_id = :d"), {"d": doc_id})
        conn.execute(sa.text("DELETE FROM documents WHERE doc_id = :d"), {"d": doc_id})


def _insert(conn, doc: Doc, doc_id: str, chunks, vectors, model: str) -> None:
    """The documents row, then its chunks copying access from it (as HybridStore.add_document)."""
    import sqlalchemy as sa

    conn.execute(sa.text("""
        INSERT INTO documents (doc_id, filename, org_id, team_id, clearance, uploaded_by)
        VALUES (:doc_id, :filename, :org, :team, :clearance, NULL)
    """), {"doc_id": doc_id, "filename": filename_for(doc), "org": doc.org, "team": doc.team,
           "clearance": doc.clearance})
    rows = [{"chunk_id": c.chunk_id, "doc_id": doc_id, "page_number": c.page_number,
             "chunk_index": c.chunk_index, "text": c.text, "embedding_model": model,
             "embedding": "[" + ",".join(repr(float(x)) for x in v) + "]"}
            for c, v in zip(chunks, vectors)]
    inserted = conn.execute(sa.text("""
        INSERT INTO chunks (chunk_id, doc_id, filename, page_number, chunk_index, text,
                            embedding_model, embedding, org_id, team_id, clearance)
        SELECT :chunk_id, d.doc_id, d.filename, :page_number, :chunk_index, :text,
               :embedding_model, CAST(:embedding AS vector), d.org_id, d.team_id, d.clearance
        FROM documents d WHERE d.doc_id = :doc_id
    """), rows).rowcount
    if inserted != len(rows):
        raise RuntimeError(f"Inserted {inserted} of {len(rows)} chunks for {doc.file}")


def _update_access(conn, doc: Doc, doc_id: str) -> None:
    """Document and every chunk in one statement (as HybridStore.set_document_access)."""
    import sqlalchemy as sa

    conn.execute(sa.text("""
        WITH d AS (
            UPDATE documents SET team_id = :team, clearance = :clearance WHERE doc_id = :doc_id
            RETURNING doc_id, org_id, team_id, clearance
        )
        UPDATE chunks SET org_id = d.org_id, team_id = d.team_id, clearance = d.clearance
        FROM d WHERE chunks.doc_id = d.doc_id
    """), {"doc_id": doc_id, "team": doc.team, "clearance": doc.clearance})


def _not_in_manifest(conn, manifest: Manifest) -> list[tuple[str, str, str]]:
    import sqlalchemy as sa

    wanted = {(d.org, filename_for(d)) for d in manifest.docs.values()}
    rows = conn.execute(sa.text("SELECT doc_id, org_id, filename FROM documents "
                                "WHERE doc_id LIKE :prefix"), {"prefix": PREFIX + "%"}).all()
    return [tuple(r) for r in rows if (r[1], r[2]) not in wanted]


def _embed_with_retries(embedder: "PacedEmbedder", texts: list[str], sleep) -> list[list[float]]:
    from llm import classify_error

    for attempt in range(1, RATE_LIMIT_RETRIES + 2):
        try:
            return embedder.embed(texts)
        except Exception as exc:
            kind = classify_error(exc)
            if kind == "daily_quota":
                raise QuotaStop(str(type(exc).__name__)) from exc
            if kind in ("rate_limited", "unavailable") and attempt <= RATE_LIMIT_RETRIES:
                print(f"  {kind}: waiting {RATE_LIMIT_WAIT_S} s, then retry {attempt} of "
                      f"{RATE_LIMIT_RETRIES}", flush=True)
                sleep(RATE_LIMIT_WAIT_S)
                continue
            raise


def chunks_by_org(conn) -> dict[str, int]:
    import sqlalchemy as sa

    rows = conn.execute(sa.text("SELECT org_id, count(*) FROM chunks WHERE doc_id LIKE :p "
                                "GROUP BY org_id ORDER BY org_id"), {"p": PREFIX + "%"}).all()
    return {org: n for org, n in rows}


def usage_counts(usage: dict, model: str | None) -> dict:
    """Texts, requests and retries for `model`, from the embedding client's
    own tracking (llm.track_usage), so internal retries are included."""
    if model is None:
        return {"embedding_texts_sent": 0, "embedding_requests": 0, "embedding_retries": 0}
    return {
        "embedding_texts_sent": usage.get("embedding_texts_per_model", {}).get(model, 0),
        "embedding_requests": usage["attempts_per_model"].get(model, 0),
        "embedding_retries": sum(1 for f in usage["failures"] if f["model"] == model),
    }


def load(owner_url: str, manifest: Manifest, dry_run: bool = False, prune: bool = False,
         docs_dir: Path = DOCS, embeddings=None, limiter: RateLimiter | None = None,
         sleep=time.sleep, token_limiter: RateLimiter | None = None) -> dict:
    """Load the manifest's documents. Raises QuotaStop (after printing how far
    it got) if the daily quota runs out; a re-run resumes."""
    from llm import track_usage

    with track_usage() as usage:
        try:
            counts = _load(owner_url, manifest, dry_run, prune, docs_dir, embeddings, limiter,
                           sleep, token_limiter, usage)
        except QuotaStop as stop:
            stop.counts = usage_counts(usage, stop.model)
            raise
    counts.update(usage_counts(usage, counts.pop("model")))
    return counts


def _load(owner_url, manifest, dry_run, prune, docs_dir, embeddings, limiter, sleep,
          token_limiter, usage) -> dict:
    import sqlalchemy as sa

    engine = sa.create_engine(owner_url)
    counts = {"insert": 0, "skip": 0, "update_access": 0, "replace": 0, "pruned": 0}
    embedder: PacedEmbedder | None = None
    total, done = len(manifest.docs), 0
    try:
        with engine.begin() as conn:
            _check_owner(conn, None)
            if not dry_run:
                _upsert_orgs(conn, manifest)
        for doc in manifest.docs.values():
            text = normalise_newlines(doc_text(doc.file, docs_dir))  # LF, whatever the checkout
            doc_id = doc_id_for(doc.file, text)
            with engine.begin() as conn:
                action = plan(doc, doc_id, _loaded(conn, doc))
            counts[action] += 1
            print(f"{action:13s} {doc.file} -> {doc_id} (team {doc.team or 'org-wide'}, "
                  f"clearance {doc.clearance})", flush=True)
            if dry_run or action == "skip":
                done += 1
                continue
            if action == "update_access":
                with engine.begin() as conn:
                    _update_access(conn, doc, doc_id)
                done += 1
                continue
            if embedder is None:  # built only when something needs embedding
                if embeddings is None:
                    from config import Settings
                    from embeddings import build_embeddings

                    embeddings = build_embeddings(Settings())
                embedder = PacedEmbedder(embeddings, limiter or RateLimiter(90),
                                         token_limiter or RateLimiter(TOKENS_PER_MINUTE))
                with engine.begin() as conn:
                    _check_owner(conn, embedder)
            pieces = chunk(doc, doc_id, text)
            try:
                # Before the write transaction opens: no transaction waits on
                # the network, and a failure here leaves nothing behind.
                vectors = _embed_with_retries(embedder, [c.text for c in pieces], sleep)
            except QuotaStop as stop:
                counts[action] -= 1
                sent = usage_counts(usage, embedder.model)
                print(f"Daily embedding quota used up: {done} of {total} documents are loaded; "
                      f"{sent['embedding_texts_sent']} texts in {sent['embedding_requests']} requests "
                      f"sent this run ({sent['embedding_retries']} retries). Re-run the same command "
                      "to resume; loaded documents are skipped.", flush=True)
                stop.model = embedder.model
                raise
            with engine.begin() as conn:
                # Re-read: another run may have loaded it meanwhile.
                loaded = _loaded(conn, doc)
                if plan(doc, doc_id, loaded) == "skip":
                    done += 1
                    continue
                _delete(conn, [row.doc_id for row in loaded])
                _insert(conn, doc, doc_id, pieces, vectors, embedder.model)
            done += 1
        with engine.begin() as conn:
            stale = _not_in_manifest(conn, manifest)
            for doc_id, org, name in stale:
                print(f"{'prune' if prune and not dry_run else 'not in manifest'}: {org}/{name} ({doc_id})")
            if prune and not dry_run:
                _delete(conn, [s[0] for s in stale])
                counts["pruned"] = len(stale)
            counts["chunks_by_org"] = chunks_by_org(conn)
    finally:
        engine.dispose()
    counts["model"] = embedder.model if embedder else None
    counts["embedding_tokens_paced"] = embedder.tokens_paced if embedder else 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--database-url", default=os.environ.get("MIGRATION_DATABASE_URL"),
                        help="Owner Postgres URL (default: MIGRATION_DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan; write nothing")
    parser.add_argument("--prune", action="store_true",
                        help="Delete loaded corpus documents that are no longer in the manifest")
    parser.add_argument("--per-minute", type=int, default=90,
                        help="Most embedding texts sent in any minute (free tier: about 100)")
    parser.add_argument("--tokens-per-minute", type=int, default=TOKENS_PER_MINUTE,
                        help="Most embedding tokens sent in any minute (free tier: 30,000)")
    args = parser.parse_args()
    if not args.database_url:
        print("Pass --database-url (the owner) or set MIGRATION_DATABASE_URL.", file=sys.stderr)
        return 2
    manifest = load_manifest()
    try:
        counts = load(args.database_url, manifest, dry_run=args.dry_run, prune=args.prune,
                      limiter=RateLimiter(args.per_minute),
                      token_limiter=RateLimiter(args.tokens_per_minute))
    except QuotaStop:
        return EXIT_QUOTA
    by_org = counts.pop("chunks_by_org")
    print(", ".join(f"{k} {v}" for k, v in counts.items()) + (" (dry run)" if args.dry_run else ""))
    print("corpus chunks per org: " + ", ".join(f"{o} {n}" for o, n in by_org.items())
          + f" (total {sum(by_org.values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
