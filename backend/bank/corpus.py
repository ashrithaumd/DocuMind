"""The FinPilot Bank corpus (data/bank/) as chunks: manifest, parse, noise
removal, structure-aware chunking; and its loader.

    python -m bank.corpus                  # chunks, token range and removed noise per document
    python -m bank.corpus load             # + chunk metadata (LLM, cached), then the database
    python -m bank.corpus load --no-metadata

Document ids are derived from the file path, so they stay the same across
runs and chunk ids can key a cache. Metadata (type, department, version,
effective date, clearance) comes from the manifest, not from the text.
"""
import argparse
import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from config import Settings
from ingestion.chunking import Chunk, chunk_blocks, tokens
from ingestion.noise import strip_boilerplate
from ingestion.parse import parse_file

BANK = Path(__file__).resolve().parents[2] / "data" / "bank"


@dataclass
class CorpusDocument:
    file: str
    meta: dict            # the manifest entry
    chunks: list[Chunk]
    noise_share: float    # share of the document's characters removed as noise


def doc_id_for(file: str) -> str:
    return hashlib.sha1(file.encode()).hexdigest()[:8]


def load_corpus(settings: Settings, root: Path = BANK) -> list[CorpusDocument]:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    entries = {d["file"]: d for d in manifest["documents"]}
    parsed, removed = {}, {}
    for file in entries:
        parsed[file], removed[file] = parse_file((root / file).read_bytes(), file, settings)
    # Characters before any removal: what parsing kept plus what it dropped.
    original = {f: sum(len(b.text) for b in parsed[f]) + removed[f] for f in entries}
    for file, n in strip_boilerplate(parsed).items():
        removed[file] += n
    return [CorpusDocument(file, meta, chunk_blocks(parsed[file], file, doc_id_for(file)),
                           removed[file] / max(1, original[file]))
            for file, meta in entries.items()]


async def load(store, docs: list[CorpusDocument], *, llm=None, model: str = "",
               cache: dict | None = None) -> dict[str, int]:
    """Write every document, replacing any with the same id, through `store`
    (scoped to an admin of the bank org who may write clearance 2). With
    `llm`, chunks get their metadata first (ingestion/metadata.py)."""
    from ingestion.metadata import annotate
    from ingestion.store import DOCUMENT_META
    from ingestion.tables import extract_tables

    calls = 0
    if llm is not None:
        calls = await annotate([c for d in docs for c in d.chunks], llm, model,
                               cache if cache is not None else {})
    for doc in docs:
        await store.add_document(doc.chunks, team_id=None, clearance=doc.meta["clearance"],
                                 uploaded_by=None, tables=extract_tables(doc.chunks),
                                 meta={k: doc.meta.get(k) for k in DOCUMENT_META}, replace=True)
    return {"documents": len(docs), "chunks": sum(len(d.chunks) for d in docs), "metadata_calls": calls}


def _load_command(no_metadata: bool) -> None:
    from bank.synthetic import BANK_ORG
    from config import get_settings
    from db.access import Access
    from deps import get_llm, get_store
    from ingestion.metadata import load_cache, save_cache

    settings = get_settings()
    store = get_store().scoped(Access(org_id=BANK_ORG, clearance=2, role="admin"))
    cache = load_cache()
    try:
        counts = asyncio.run(load(store, load_corpus(settings), llm=None if no_metadata else get_llm(),
                                  model=settings.grading_model, cache=cache))
    finally:
        save_cache(cache)  # keep what was paid for, even if a later step failed
    print(", ".join(f"{k} {v}" for k, v in counts.items()))


def main() -> None:
    from config import get_settings

    parser = argparse.ArgumentParser(description="Show or load the FinPilot Bank corpus.")
    parser.add_argument("command", nargs="?", choices=["load"])
    parser.add_argument("--no-metadata", action="store_true", help="skip the LLM metadata calls")
    args = parser.parse_args()
    if args.command == "load":
        _load_command(args.no_metadata)
        return
    docs = load_corpus(get_settings())
    print(f"{'document':42} {'chunks':>6} {'tokens':>9} {'noise':>6}")
    for d in docs:
        sizes = [tokens(c.text) for c in d.chunks]
        print(f"{d.file:42} {len(sizes):6} {min(sizes):4}-{max(sizes):<4} {d.noise_share:6.1%}")
    print(f"{'total':42} {sum(len(d.chunks) for d in docs):6}")


if __name__ == "__main__":
    main()
