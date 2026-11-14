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
               if p.is_file() and p.name not in NOT_DOCUMENTS and "tools" not in p.parts}
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
