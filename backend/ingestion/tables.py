"""Tables in documents become relational rows (doc_tables, doc_table_rows,
migration 0010), each row linked to the chunk that holds it, so a
calculation can cite exactly where a number came from.

Cells stay text ("$15.00 per item"): the calculator parses what it needs
and cites the row. A table split across chunks (parts 1, 2, ...) is one
table whose rows point at different chunks.
"""
from dataclasses import dataclass, field

from ingestion.chunking import Chunk


@dataclass
class DocTable:
    table_id: str                 # "<doc_id>-t<n>"
    doc_id: str
    section: str
    columns: list[str]
    rows: list[tuple[str, list[str]]] = field(default_factory=list)  # (chunk_id, cells)


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def extract_tables(chunks: list[Chunk]) -> list[DocTable]:
    tables: list[DocTable] = []
    for chunk in chunks:
        for block in chunk.blocks:
            if block.kind != "table":
                continue
            if block.part == 0:
                tables.append(DocTable(f"{chunk.doc_id}-t{len(tables):02d}", chunk.doc_id,
                                       block.section, cells(block.header.split("\n")[0])))
            tables[-1].rows += [(chunk.chunk_id, cells(line)) for line in block.text.split("\n") if line.strip()]
    return tables
