"""The enterprise eval set and its validator (backend/eval/enterprise/), on the
real corpus in kb/: the shipped items are valid, and each rule rejects the
item it should. No database, no model calls."""
import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend" / "eval" / "enterprise"))
import validate_items as v  # noqa: E402

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "manifest.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")


@pytest.fixture(scope="module")
def manifest():
    return v.load_manifest()


@pytest.fixture(scope="module")
def examples():
    return {item["id"]: item for _, item, _ in v.load_items(v.ITEMS)}


def errors(item, manifest):
    return v.check_item(item, manifest, v.ChunkIndex())


def test_shipped_items_are_valid():
    problems, counts = v.validate()
    assert problems == []
    assert sum(counts.values()) >= 3


def test_a_span_that_is_not_in_the_document_is_rejected(examples, manifest):
    item = copy.deepcopy(examples["aurora-exact-001"])
    item["gold"][0]["span"] = "E-2417 GRIPPER_VACUUM_LOW. Raised when vacuum below 99 kPa"
    assert any("span not found" in e for e in errors(item, manifest))


def test_whitespace_differences_are_ignored(examples, manifest):
    item = copy.deepcopy(examples["aurora-version-001"])
    item["gold"][0]["span"] = item["gold"][0]["span"].replace(" on top", "\n   on top")
    assert errors(item, manifest) == []


def test_an_answerable_item_the_user_cannot_see_is_rejected(examples, manifest):
    item = copy.deepcopy(examples["aurora-exact-001"])
    item["as_user"]["clearance"] = 0  # the gen 3 manual is at clearance 1
    assert any("can't see gold document" in e for e in errors(item, manifest))


def test_permission_restricted_must_really_be_hidden(examples, manifest):
    item = copy.deepcopy(examples["bluefin-perm-001"])
    item["as_user"]["teams"] = ["finance"]  # finance at clearance 3 can see it
    assert any("can see gold document" in e for e in errors(item, manifest))


def test_permission_restricted_fails_if_a_visible_document_has_the_answer(manifest):
    # The CellCore price is in the full contract (clearance 2) and in its
    # summary (clearance 1): a clearance-1 user can answer from the summary.
    item = {
        "id": "x", "org": "aurora", "type": "permission-restricted", "question": "Pack price?",
        "as_user": {"teams": [], "clearance": 1, "role": "viewer"}, "expected_outcome": "refuse",
        "expected_answer": "Refuses.", "distractors": [], "notes": "",
        "gold": [{"doc": "aurora/cellcore-supply-agreement.txt", "span": "2,150 US dollars per pack"}],
    }
    assert errors(item, manifest) == []  # "US dollars" wording is only in the full contract
    item["gold"][0]["span"] = "2,150"
    assert any("also contains the gold span" in e for e in errors(item, manifest))


def test_version_trap_needs_a_visible_distractor(examples, manifest):
    item = copy.deepcopy(examples["aurora-version-001"])
    item["distractors"] = []
    assert any("needs the other version" in e for e in errors(item, manifest))


def test_unanswerable_has_no_gold(examples, manifest):
    item = copy.deepcopy(examples["aurora-exact-001"])
    item["type"], item["expected_outcome"] = "unanswerable", "refuse"
    assert any("no gold" in e for e in errors(item, manifest))


def test_multi_chunk_spans_must_not_share_a_chunk(manifest):
    base = {"id": "x", "org": "aurora", "type": "multi-chunk", "question": "q",
            "as_user": {"teams": ["field-ops"], "clearance": 1, "role": "viewer"},
            "expected_outcome": "answer", "expected_answer": "a", "distractors": [], "notes": ""}
    doc = "aurora/pika-gen3-service-manual.txt"
    near = dict(base, gold=[{"doc": doc, "span": "Maximum payload: 12 kg"},
                            {"doc": doc, "span": "Pick rate: up to 600 items per hour"}])
    assert any("one chunk" in e for e in errors(near, manifest))
    far = dict(base, gold=[{"doc": doc, "span": "Maximum payload: 12 kg"},
                           {"doc": doc, "span": "E-2417 GRIPPER_VACUUM_LOW"}])
    assert errors(far, manifest) == []


def test_bad_fields_and_values_are_rejected(examples, manifest):
    item = copy.deepcopy(examples["aurora-exact-001"])
    item["type"] = "trick"
    item["as_user"]["teams"] = ["finance"]  # a bluefin team
    found = errors(item, manifest)
    assert any("is not one of" in e for e in found)
    assert any("not teams of aurora" in e for e in found)
    del item["notes"]
    assert any("fields" in e for e in errors(item, manifest))


def test_duplicate_ids_are_reported(tmp_path, examples):
    line = json.dumps(examples["aurora-exact-001"])
    path = tmp_path / "items.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    problems, _ = v.validate(path)
    assert any("duplicate id" in p for p in problems)
