"""Parsers, noise removal and the structure-aware chunker (Phase 3)."""
from ingestion.chunking import MAX_TOKENS, MIN_TOKENS, chunk_blocks, tokens
from ingestion.extract import PageText
from ingestion.noise import strip_boilerplate, strip_nav_lines, strip_page_furniture
from ingestion.parse import Block, parse_csv, parse_markdown, parse_openapi

DOC = """# Guide

Intro paragraph.

## 1. Fees

| Service | Fee |
|---|---|
| Overdraft | $15.00 |
| Stop payment | $30.00 |

- first item
  continues here
- second item

### 1.1 Detail

Detail text.
"""


def test_markdown_sections_tables_and_list_items():
    blocks = parse_markdown(DOC)
    assert [(b.kind, b.section) for b in blocks] == [
        ("text", "Guide"),
        ("table", "Guide > 1. Fees"),
        ("item", "Guide > 1. Fees"),
        ("item", "Guide > 1. Fees"),
        ("text", "Guide > 1. Fees > 1.1 Detail"),
    ]
    table = blocks[1]
    assert table.header == "| Service | Fee |\n|---|---|"
    assert table.text.count("\n") == 1  # two rows, header kept apart
    assert blocks[2].text == "- first item\n  continues here"


def test_a_skipped_heading_level_keeps_the_path_aligned():
    blocks = parse_markdown("# A\n\n### C\n\ntext")
    assert blocks[0].section == "A > C"


def test_small_document_is_one_chunk_with_its_section_path_first():
    chunks = chunk_blocks(parse_markdown(DOC), "guide.md", "d1")
    assert len(chunks) == 1
    text = chunks[0].text
    assert text.startswith("Guide\n\nIntro paragraph.")
    assert "Guide > 1. Fees\n\n| Service | Fee |" in text
    assert "- first item\n  continues here\n- second item" in text  # items stay single-spaced
    assert chunks[0].chunk_id == "d1-0000" and chunks[0].section == "Guide"


def test_a_big_table_splits_between_rows_and_repeats_its_header():
    header = "| Code | Meaning |\n|---|---|"
    rows = [f"| C{i:03d} | the meaning of code number {i} in plain words |" for i in range(120)]
    chunks = chunk_blocks([Block("\n".join(rows), "Codes", "table", header=header)], "codes.md", "d2")
    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        assert tokens(chunk.text) <= MAX_TOKENS
        assert chunk.text.startswith(f"Codes\n\n{header}\n")
        seen += [line for line in chunk.text.splitlines() if line.startswith("| C") and line not in header]
    assert seen == rows  # every row once, in order, none cut


def test_sections_pack_up_to_the_limit_and_small_ones_merge():
    para = "This sentence is about deposits and how they are made available. " * 12
    blocks = [Block(para, f"Doc > {i}") for i in range(12)]
    chunks = chunk_blocks(blocks, "doc.md", "d3")
    sizes = [tokens(c.text) for c in chunks]
    assert all(s <= MAX_TOKENS for s in sizes)
    assert all(s >= MIN_TOKENS for s in sizes[:-1])
    assert len(chunks) < len(blocks)


def test_oversized_prose_splits_below_the_limit():
    para = " ".join(f"Clause {i} says something specific about interest." for i in range(300))
    chunks = chunk_blocks([Block(para, "Doc > Interest")], "doc.md", "d4")
    assert len(chunks) > 2
    assert all(tokens(c.text) <= MAX_TOKENS for c in chunks)
    assert all(c.text.startswith("Doc > Interest\n\n") for c in chunks)


OPENAPI = """
openapi: 3.1.0
info: {title: Bank API, version: "1.0"}
paths:
  /accounts:
    get: {operationId: listAccounts, summary: List accounts.}
  /accounts/{id}:
    get: {operationId: getAccount, summary: One account.}
    delete: {operationId: closeAccount, summary: Close it.}
components:
  schemas:
    Account: {type: object}
"""


def test_every_openapi_operation_is_a_chunk_of_its_own():
    chunks = chunk_blocks(parse_openapi(OPENAPI), "api.yaml", "d5")
    ops = [c for c in chunks if "operationId" in c.text]
    assert [c.section for c in ops] == ["Bank API > GET /accounts", "Bank API > GET /accounts/{id}",
                                        "Bank API > DELETE /accounts/{id}"]
    assert all(c.text.count("operationId") == 1 for c in ops)


def test_csv_versions_never_share_a_table():
    text = "effective_date,product,apy\n2025-07-01,Savings,3.45\n2026-02-01,Savings,3.10\n"
    blocks = parse_csv(text, "rates")
    assert [b.section for b in blocks] == ["rates > effective 2025-07-01", "rates > effective 2026-02-01"]
    assert all(b.header.startswith("| effective_date | product | apy |") for b in blocks)


def test_running_headers_footers_and_page_numbers_are_removed():
    pages = [PageText(n, f"FinPilot Bank Annual Report\nBody text of page {n} with facts.\n"
                         f"More body.\nConfidential draft\nPage {n} of 4") for n in range(1, 5)]
    cleaned, removed = strip_page_furniture(pages)
    for page in cleaned:
        assert page.text.splitlines() == [f"Body text of page {page.page_number} with facts.", "More body."]
    assert removed == sum(len("FinPilot Bank Annual Report") + len("Confidential draft") + len(f"Page {n} of 4")
                          for n in range(1, 5))


def test_a_number_inside_the_page_is_kept():
    pages = [PageText(n, f"Header\nRate\n12\nmore text\nmore\nFooter") for n in range(1, 4)]
    cleaned, _ = strip_page_furniture(pages)
    assert "12" in cleaned[0].text.splitlines()


def test_navigation_lines_are_removed():
    blocks, removed = strip_nav_lines([Block("Skip to main content\nHome > Guides > Fees\nReal text.")])
    assert [b.text for b in blocks] == ["Real text."]
    assert removed > 0


def test_boilerplate_in_three_documents_is_kept_once():
    banner = "DRAFT. Fictional bank; not insured."
    docs = {name: [Block(banner), Block(f"{name} content")] for name in ("a", "b", "c")}
    docs["d"] = [Block("only d")]
    removed = strip_boilerplate(docs)
    assert [b.text for b in docs["a"]] == [banner, "a content"]
    assert [b.text for b in docs["b"]] == ["b content"]
    assert removed == {"a": 0, "b": len(banner), "c": len(banner), "d": 0}
