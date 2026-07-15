"""Token-aware chunking.

Design decision: pages are split independently, so every chunk maps to exactly
one page number. That keeps citations precise ("p. 12" is always correct) at
the cost of never having a chunk that spans a page break — an acceptable trade
for a citation-first system. Chunks target ~600 tokens with 15% (90-token)
overlap so sentences cut at a boundary still appear whole in one chunk.
"""
import uuid
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from ingestion.extract import PageText


@dataclass
class Chunk:
    chunk_id: str      # "<doc_id>-<chunk_index>" — stable, human-readable
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int   # position within the whole document
    text: str


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
