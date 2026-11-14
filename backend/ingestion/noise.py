"""Noise removal before chunking: text that repeats on every page or in
every document carries no answer and only dilutes retrieval.

  - Page furniture (paged formats): a line that opens or closes at least
    half the pages of a document of 3+ pages is a running header or footer.
    Page-number lines ("12", "Page 3 of 9", "- 4 -") go on any page.
  - Navigation text from saved web pages ("Skip to main content", "Back to
    top", breadcrumbs).
  - Corpus boilerplate: a paragraph that appears word for word in 3+
    documents (the bank's draft banner, for one).

Every function returns how many characters it removed, so the loader can
report the removed share per document.
"""
import re
from collections import Counter

from ingestion.extract import PageText
from ingestion.parse import Block

_PAGE_NUMBER = re.compile(r"^\s*[-–]?\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*[-–]?\s*$", re.I)
_NAV = re.compile(r"^\s*(skip to (main )?content|back to top|print this page|share this page"
                  r"|was this page helpful\??|home\s*[>›»/|].*)\s*$", re.I)
_EDGE_LINES = 2  # header/footer candidates: this many lines at each end of a page


def strip_page_furniture(pages: list[PageText]) -> tuple[list[PageText], int]:
    def lines_of(page: PageText) -> list[str]:
        return [line.strip() for line in page.text.splitlines()]

    edges: Counter[str] = Counter()
    if len(pages) >= 3:
        for page in pages:
            lines = [line for line in lines_of(page) if line]
            edges.update(set(lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]))
    repeated = {line for line, n in edges.items() if n >= len(pages) / 2}

    cleaned, removed = [], 0
    for page in pages:
        lines = page.text.splitlines()
        filled = [i for i, line in enumerate(lines) if line.strip()]
        edge = set(filled[:_EDGE_LINES] + filled[-_EDGE_LINES:])
        kept = []
        for i, line in enumerate(lines):
            # Page numbers only at the edges: a lone "12" mid-page may be a table cell.
            if line.strip() in repeated or (i in edge and _PAGE_NUMBER.match(line)):
                removed += len(line)
            else:
                kept.append(line)
        cleaned.append(PageText(page.page_number, "\n".join(kept)))
    return cleaned, removed


def strip_nav_lines(blocks: list[Block]) -> tuple[list[Block], int]:
    out, removed = [], 0
    for block in blocks:
        if block.kind != "text":
            out.append(block)
            continue
        kept = [line for line in block.text.splitlines() if not _NAV.match(line)]
        removed += len(block.text) - len("\n".join(kept))
        if any(line.strip() for line in kept):
            block.text = "\n".join(kept)
            out.append(block)
    return out, removed


def _normal(text: str) -> str:
    return " ".join(text.split()).lower()


def strip_boilerplate(docs: dict[str, list[Block]], min_docs: int = 3) -> dict[str, int]:
    """Drop, in place, text blocks found in `min_docs` or more documents,
    except in the first document that has them, so the text (the draft
    banner says the bank isn't insured) stays retrievable once. Returns
    characters removed per document."""
    seen = Counter(t for blocks in docs.values()
                   for t in {_normal(b.text) for b in blocks if b.kind == "text"})
    common = {t for t, n in seen.items() if n >= min_docs}
    kept_once: set[str] = set()
    removed = {}
    for name, blocks in docs.items():
        mine = {_normal(b.text) for b in blocks if b.kind == "text"} & common
        drop, kept_once = kept_once & mine, kept_once | mine
        keep = [b for b in blocks if not (b.kind == "text" and _normal(b.text) in drop)]
        removed[name] = sum(len(b.text) for b in blocks) - sum(len(b.text) for b in keep)
        blocks[:] = keep
    return removed
