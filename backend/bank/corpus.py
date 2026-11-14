"""The FinPilot Bank corpus (data/bank/) as chunks: manifest, parse, noise
removal, structure-aware chunking.

    python -m bank.corpus    # chunks, token range and removed noise per document

Document ids are derived from the file path, so they stay the same across
runs and chunk ids can key a cache. Metadata (type, department, version,
effective date, clearance) comes from the manifest, not from the text.
"""
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


def main() -> None:
    from config import get_settings

    docs = load_corpus(get_settings())
    print(f"{'document':42} {'chunks':>6} {'tokens':>9} {'noise':>6}")
    for d in docs:
        sizes = [tokens(c.text) for c in d.chunks]
        print(f"{d.file:42} {len(sizes):6} {min(sizes):4}-{max(sizes):<4} {d.noise_share:6.1%}")
    print(f"{'total':42} {sum(len(d.chunks) for d in docs):6}")


if __name__ == "__main__":
    main()
