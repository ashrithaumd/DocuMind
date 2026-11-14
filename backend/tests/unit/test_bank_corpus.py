"""The FinPilot Bank corpus (data/bank/): the manifest covers every file,
drafts are marked, fee documents charge what the generator charges, and the
staff memo's canary appears nowhere else."""
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from bank.synthetic import FEE_SCHEDULES, SCHEDULE_CHANGE

REPO = Path(__file__).resolve().parents[3]
BANK = REPO / "data" / "bank"
NOT_DOCUMENTS = {"README.md", "manifest.yaml"}


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load((BANK / "manifest.yaml").read_text(encoding="utf-8"))


def test_the_manifest_lists_every_document_and_nothing_else(manifest):
    on_disk = {p.relative_to(BANK).as_posix() for p in BANK.rglob("*")
               if p.is_file() and p.name not in NOT_DOCUMENTS
               and not {"tools", "receipts"} & set(p.parts)}  # receipts are customer data
    assert {d["file"] for d in manifest["documents"]} == on_disk


def test_every_text_document_is_marked_draft(manifest):
    for doc in manifest["documents"]:
        assert doc["status"] == "draft"
        if doc["file"].endswith((".md", ".yaml")):
            assert "DRAFT, pending review" in (BANK / doc["file"]).read_text(encoding="utf-8"), doc


def test_only_internal_documents_are_staff_only(manifest):
    for doc in manifest["documents"]:
        assert doc["clearance"] == (2 if doc["file"].startswith("internal/") else 0), doc["file"]


def _row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {label} "))


@pytest.mark.parametrize("file,effective", [("fee-schedule-2025.md", date(2025, 1, 1)),
                                            ("fee-schedule-2026.md", SCHEDULE_CHANGE)])
def test_fee_schedules_match_the_generator(file, effective):
    text = (BANK / file).read_text(encoding="utf-8")
    sched = FEE_SCHEDULES[effective]

    def dollars(cents):
        return f"${cents / 100:,.2f}"

    assert dollars(sched.overdraft) in _row(text, "Overdraft (paid item)")
    assert f"{sched.overdraft_max_per_day} per day" in _row(text, "Overdraft fee limit")
    assert dollars(sched.maintenance) in _row(text, "Monthly maintenance")
    assert dollars(sched.maintenance_waiver_deposits) in _row(text, "Monthly maintenance")
    assert dollars(sched.atm_out_of_network) in _row(text, "Out-of-network ATM withdrawal")
    assert dollars(sched.late_payment) in _row(text, "Late payment")
    nsf = _row(text, "Returned item (NSF)")
    assert (dollars(sched.nsf) in nsf) if sched.nsf else ("No fee" in nsf)
    if sched.overdraft_cushion:
        assert dollars(sched.overdraft_cushion) in _row(text, "Overdraft cushion")


def test_the_canary_appears_only_in_the_staff_memo():
    memo = BANK / "internal" / "fraud-rules-memo.md"
    found = re.findall(r"FINPILOT-CANARY-[0-9a-f]{12}", memo.read_text(encoding="utf-8"))
    assert len(found) == 1
    canary = found[0]
    elsewhere = [p for root in (REPO / "data", REPO / "backend", REPO / "docs", REPO / "frontend" / "src")
                 for p in root.rglob("*")
                 if p.is_file() and p != memo and ".venv" not in p.parts
                 and "node_modules" not in p.parts and p.suffix in {".md", ".py", ".yaml", ".csv",
                                                                     ".json", ".ts", ".tsx"}
                 and canary in p.read_text(encoding="utf-8", errors="ignore")]
    assert elsewhere == []


def test_the_openapi_spec_parses_and_never_names_another_customer():
    spec = yaml.safe_load((BANK / "api" / "openapi.yaml").read_text(encoding="utf-8"))
    assert spec["openapi"].startswith("3.1")
    paths = " ".join(spec["paths"])
    assert "customer_id" not in paths  # every operation is for the signed-in customer only


# --- Phase 3: the corpus as chunks -------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    from bank.corpus import load_corpus
    from config import Settings

    return {d.file: d for d in load_corpus(Settings(anthropic_api_key="x"))}


def test_every_document_chunks_within_the_token_limit(corpus, manifest):
    from ingestion.chunking import MAX_TOKENS, tokens

    assert set(corpus) == {d["file"] for d in manifest["documents"]}
    for doc in corpus.values():
        assert doc.chunks, doc.file
        assert all(tokens(c.text) <= MAX_TOKENS for c in doc.chunks), doc.file


def test_enough_documents_span_several_chunks(corpus):
    # Chunking and retrieval need documents whose answers sit in different chunks.
    multi = [d.file for d in corpus.values() if len(d.chunks) >= 3]
    assert len(multi) >= 6, multi
    assert len(corpus["deposit-agreement.md"].chunks) >= 15


def test_the_deposit_agreement_is_full_length():
    text = (BANK / "deposit-agreement.md").read_text(encoding="utf-8")
    assert 4000 <= len(text.split()) <= 6000
    assert len(re.findall(r"^## \d+\. ", text, re.M)) >= 20
    assert text.count("|---|") >= 1


def test_the_canary_reaches_only_staff_chunks(corpus):
    for doc in corpus.values():
        hits = [c for c in doc.chunks if "FINPILOT-CANARY-" in c.text]
        assert not hits or doc.meta["clearance"] == 2, doc.file


def test_every_rate_row_lands_in_a_chunk(corpus):
    import csv

    for name in ("deposit-rates.csv", "card-rates.csv"):
        text = "\n".join(c.text for c in corpus[f"rates/{name}"].chunks)
        with open(BANK / "rates" / name, encoding="utf-8", newline="") as f:
            for row in csv.reader(list(f)[1:]):
                assert "| " + " | ".join(row) + " |" in text
    sheet = "\n".join(c.text for c in corpus["rates/rate-sheet.xlsx"].chunks)
    assert sheet.count("| Product | Term (months) | Balance from ($) |") == 2  # header found under the title rows
    assert "| High-Yield Savings |  | 10000 | 49999.99 | 3.2 | 3.25 | 0 |" in sheet


def test_each_api_operation_is_exactly_one_chunk(corpus):
    spec = yaml.safe_load((BANK / "api" / "openapi.yaml").read_text(encoding="utf-8"))
    ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()
           if isinstance(op, dict) and "operationId" in op]
    chunks = corpus["api/openapi.yaml"].chunks
    for op_id in ids:
        assert sum(f"operationId: {op_id}\n" in c.text for c in chunks) == 1, op_id


def test_the_draft_banner_is_kept_once(corpus):
    with_banner = [d.file for d in corpus.values() if any("DRAFT, pending review" in c.text for c in d.chunks)]
    # The memo's banner differs (it adds the staff-only warning), so it isn't boilerplate.
    assert [f for f in with_banner if f.endswith(".md")] == ["fee-schedule-2025.md", "internal/fraud-rules-memo.md"]
    assert all(d.noise_share < 0.1 for d in corpus.values())


def test_chunk_ids_are_stable(corpus):
    from bank.corpus import load_corpus
    from config import Settings

    again = {d.file: [c.chunk_id for c in d.chunks] for d in load_corpus(Settings(anthropic_api_key="x"))}
    assert again == {f: [c.chunk_id for c in d.chunks] for f, d in corpus.items()}
