"""Load the Phase 3 corpus (kb/) into Postgres, as the owner. Idempotent.

    backend/.venv/Scripts/python scripts/load_corpus.py --database-url OWNER_URL [--dry-run] [--prune]

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
"""
import argparse
import asyncio
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kb" / "tools"))
sys.path.insert(0, str(REPO / "backend"))
from manifest import DOCS, Doc, Manifest, doc_text, load_manifest  # noqa: E402

PREFIX = "kb-"


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


def load(owner_url: str, manifest: Manifest, dry_run: bool = False, prune: bool = False,
         docs_dir: Path = DOCS) -> dict[str, int]:
    import sqlalchemy as sa

    engine = sa.create_engine(owner_url)
    counts = {"insert": 0, "skip": 0, "update_access": 0, "replace": 0, "pruned": 0}
    embeddings = None
    try:
        with engine.begin() as conn:
            _check_owner(conn, None)
            if not dry_run:
                _upsert_orgs(conn, manifest)
        for doc in manifest.docs.values():
            text = doc_text(doc.file, docs_dir)
            doc_id = doc_id_for(doc.file, text)
            with engine.begin() as conn:
                loaded = _loaded(conn, doc)
                action = plan(doc, doc_id, loaded)
                counts[action] += 1
                print(f"{action:13s} {doc.file} -> {doc_id} (team {doc.team or 'org-wide'}, "
                      f"clearance {doc.clearance})", flush=True)
                if dry_run or action == "skip":
                    continue
                if action == "update_access":
                    _update_access(conn, doc, doc_id)
                    continue
                if embeddings is None:  # built only when something needs embedding
                    from config import Settings
                    from embeddings import build_embeddings

                    embeddings = build_embeddings(Settings())
                    _check_owner(conn, embeddings)
                chunks = chunk(doc, doc_id, text)
                vectors = asyncio.run(embeddings.embed_documents([c.text for c in chunks]))
                _delete(conn, [row.doc_id for row in loaded])
                _insert(conn, doc, doc_id, chunks, vectors, embeddings.model)
        with engine.begin() as conn:
            stale = _not_in_manifest(conn, manifest)
            for doc_id, org, name in stale:
                print(f"{'prune' if prune and not dry_run else 'not in manifest'}: {org}/{name} ({doc_id})")
            if prune and not dry_run:
                _delete(conn, [s[0] for s in stale])
                counts["pruned"] = len(stale)
    finally:
        engine.dispose()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--database-url", default=os.environ.get("MIGRATION_DATABASE_URL"),
                        help="Owner Postgres URL (default: MIGRATION_DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan; write nothing")
    parser.add_argument("--prune", action="store_true",
                        help="Delete loaded corpus documents that are no longer in the manifest")
    args = parser.parse_args()
    if not args.database_url:
        print("Pass --database-url (the owner) or set MIGRATION_DATABASE_URL.", file=sys.stderr)
        return 2
    manifest = load_manifest()
    counts = load(args.database_url, manifest, dry_run=args.dry_run, prune=args.prune)
    print(", ".join(f"{k} {v}" for k, v in counts.items()) + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
