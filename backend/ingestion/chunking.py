"""Chunking.

chunk_blocks (FinPilot's corpus): structure-aware. Parsers (parse.py) cut a
document into blocks (paragraphs, list items, whole tables, API
operations); this packs neighbouring blocks into chunks of 256-512 tokens:

  - A chunk starts with its section path, and a new path line wherever it
    crosses into another section, so every chunk says where it came from.
  - Blocks of the same section and page are packed together up to 512
    tokens. A chunk still under 256 also takes in the next section.
  - A table is never cut mid-row: one bigger than 512 splits between rows,
    each part repeating the header. Other oversized blocks split on
    paragraph, line, then sentence boundaries.
  - An API operation is always a chunk of its own.

chunk_pages (DocuMind's PDF path, kept for the eval tooling): pages split
independently at ~600 tokens with 15% overlap, so every chunk maps to
exactly one page.
"""
import uuid
from dataclasses import dataclass, replace
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from ingestion.extract import PageText
from ingestion.parse import Block

MIN_TOKENS, MAX_TOKENS = 256, 512


@dataclass
class Chunk:
    chunk_id: str      # "<doc_id>-<chunk_index>" — stable, human-readable
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int   # position within the whole document
    text: str
    section: str = ""  # heading path of the chunk's first block


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _split_oversized(block: Block, budget: int) -> list[Block]:
    whole = f"{block.header}\n{block.text}" if block.header else block.text
    if tokens(whole) <= budget:
        return [block]
    if block.kind == "table":
        # shortcut: a single row bigger than the budget stays whole; no corpus table has one.
        groups: list[list[str]] = [[]]
        for row in block.text.split("\n"):
            if groups[-1] and tokens(block.header + "\n" + "\n".join(groups[-1] + [row])) > budget:
                groups.append([])
            groups[-1].append(row)
        return [replace(block, text="\n".join(g)) for g in groups]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=budget, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""])
    return [replace(block, text=t) for t in splitter.split_text(block.text)]


def _render(group: list[Block]) -> str:
    out, prev = "", None
    for b in group:
        body = f"{b.header}\n{b.text}" if b.header else b.text
        if prev is None or b.section != prev.section:
            out += ("\n\n" if out else "") + (f"{b.section}\n\n" if b.section else "") + body
        else:
            out += ("\n" if prev.kind == b.kind == "item" else "\n\n") + body
        prev = b
    return out


def _can_join(group: list[Block], block: Block) -> bool:
    if "operation" in (group[-1].kind, block.kind):
        return False
    same_place = (group[-1].section, group[-1].page_number) == (block.section, block.page_number)
    return (same_place or tokens(_render(group)) < MIN_TOKENS) and tokens(_render(group + [block])) <= MAX_TOKENS


def chunk_blocks(blocks: list[Block], filename: str, doc_id: str) -> list[Chunk]:
    pieces = [p for b in blocks if b.text.strip()
              for p in _split_oversized(b, MAX_TOKENS - tokens(b.section) - 4)]
    groups: list[list[Block]] = []
    for piece in pieces:
        if groups and _can_join(groups[-1], piece):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    # A short last chunk joins the one before it when they fit together.
    if (len(groups) > 1 and tokens(_render(groups[-1])) < MIN_TOKENS
            and all(b.kind != "operation" for b in groups[-2] + groups[-1])
            and tokens(_render(groups[-2] + groups[-1])) <= MAX_TOKENS):
        groups[-2:] = [groups[-2] + groups[-1]]
    return [Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename=filename,
                  page_number=g[0].page_number, chunk_index=i, text=_render(g), section=g[0].section)
            for i, g in enumerate(groups)]


def chunk_pages(pages: list[PageText], filename: str, settings: Settings) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    doc_id = uuid.uuid4().hex[:8]
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{index:04d}",
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
