"""kb/manifest.yaml parsing and validation (kb/tools/manifest.py), and the
corpus loader's database-free parts (scripts/load_corpus.py): document ids
and the insert/skip/update/replace plan that makes re-runs idempotent. No
database, no embedding calls."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "kb" / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "manifest.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")

import load_corpus  # noqa: E402
from manifest import Doc, ManifestError, User, can_see, load_manifest  # noqa: E402

ORGS = """
orgs:
  acme:
    name: Acme
    teams: [ops, legal]
  zed:
    name: Zed
    teams: [eng]
documents:
"""


def _kb(tmp_path, entries: str, files=("acme/a.txt", "acme/b.txt", "zed/c.txt")):
    docs = tmp_path / "docs"
    for f in files:
        (docs / f).parent.mkdir(parents=True, exist_ok=True)
        (docs / f).write_text(f"text of {f}", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(ORGS + entries, encoding="utf-8")
    return manifest, docs


VALID = """
  - {file: acme/a.txt, org: acme, team: null, clearance: 0}
  - {file: acme/b.txt, org: acme, team: legal, clearance: 3}
  - {file: zed/c.txt, org: zed, team: eng, clearance: 1}
"""


def test_the_real_manifest_is_valid_and_covers_every_document():
    m = load_manifest()
    assert len(m.docs) == 50
    assert set(m.orgs) == {"aurora", "bluefin"}


def test_a_valid_manifest_parses(tmp_path):
    m = load_manifest(*_kb(tmp_path, VALID))
    assert m.docs["acme/b.txt"] == Doc(file="acme/b.txt", org="acme", team="legal", clearance=3)
    assert m.teams("acme") == ["ops", "legal"]


@pytest.mark.parametrize("entries,problem", [
    (VALID + "  - {file: acme/a.txt, org: acme, team: null, clearance: 0}\n", "listed twice"),
    (VALID.replace("org: zed, team: eng", "org: acme, team: eng"), "not under its org's folder"),
    (VALID.replace("team: legal", "team: eng"), "is not a team of acme"),
    (VALID.replace("org: zed,", "org: nowhere,"), "unknown org"),
    (VALID.replace("clearance: 3", "clearance: 4"), "clearance must be an integer 0-3"),
    (VALID.replace("clearance: 3", "clearance: '3'"), "clearance must be an integer 0-3"),
    (VALID.replace("clearance: 3", "clearance: true"), "clearance must be an integer 0-3"),
    (VALID.replace("acme/b.txt", "acme/missing.txt"), "no such file"),
    (VALID.replace(", clearance: 1}", "}"), "needs exactly file, org, team, clearance"),
    (VALID.replace(", clearance: 1}", ", clearance: 1, owner: me}"), "needs exactly file, org, team, clearance"),
])
def test_invalid_manifests_are_rejected(tmp_path, entries, problem):
    with pytest.raises(ManifestError, match=problem):
        load_manifest(*_kb(tmp_path, entries))


def test_a_document_missing_from_the_manifest_is_reported(tmp_path):
    entries = VALID.replace("  - {file: zed/c.txt, org: zed, team: eng, clearance: 1}\n", "")
    with pytest.raises(ManifestError, match="zed/c.txt: in kb/docs/ but not in the manifest"):
        load_manifest(*_kb(tmp_path, entries))


def test_team_ids_must_be_unique_across_orgs(tmp_path):
    manifest, docs = _kb(tmp_path, VALID)
    manifest.write_text(manifest.read_text().replace("teams: [eng]", "teams: [eng, ops]"))
    with pytest.raises(ManifestError, match="team ops is in both acme and zed"):
        load_manifest(manifest, docs)


def test_every_problem_is_reported_at_once(tmp_path):
    entries = VALID.replace("team: legal", "team: eng").replace("clearance: 1", "clearance: 9")
    with pytest.raises(ManifestError) as err:
        load_manifest(*_kb(tmp_path, entries))
    assert "is not a team of acme" in str(err.value) and "clearance must be" in str(err.value)


def test_can_see_follows_the_rls_rules():
    doc = Doc(file="acme/b.txt", org="acme", team="legal", clearance=2)
    assert can_see(User("acme", ("legal",), 2), doc)
    assert not can_see(User("acme", ("legal",), 1), doc)  # clearance too low
    assert not can_see(User("acme", ("ops",), 3), doc)  # not in the team
    assert not can_see(User("zed", ("legal",), 3), doc)  # another org
    assert can_see(User("acme", (), 2), Doc(file="acme/a.txt", org="acme", team=None, clearance=2))


# --- the loader's idempotency, without a database -------------------------------

DOC = Doc(file="acme/b.txt", org="acme", team="legal", clearance=3)


def test_doc_ids_are_stable_and_change_with_content_or_path():
    a = load_corpus.doc_id_for("acme/b.txt", "hello")
    assert a == load_corpus.doc_id_for("acme/b.txt", "hello")
    assert a.startswith("kb-") and len(a) == 19
    assert a != load_corpus.doc_id_for("acme/b.txt", "hello!")
    assert a != load_corpus.doc_id_for("zed/b.txt", "hello")


@pytest.mark.parametrize("loaded,action", [
    ([], "insert"),
    ([load_corpus.Loaded("kb-x", "legal", 3)], "skip"),
    ([load_corpus.Loaded("kb-x", None, 3)], "update_access"),
    ([load_corpus.Loaded("kb-x", "legal", 1)], "update_access"),
    ([load_corpus.Loaded("kb-old", "legal", 3)], "replace"),
    ([load_corpus.Loaded("kb-x", "legal", 3), load_corpus.Loaded("kb-y", "legal", 3)], "replace"),
])
def test_plan_decides_what_a_re_run_does(loaded, action):
    assert load_corpus.plan(DOC, "kb-x", loaded) == action


def test_chunks_get_ids_from_the_document_id_not_a_random_one():
    text = load_corpus.doc_text("aurora/pika-gen3-service-manual.txt")
    doc = Doc(file="aurora/pika-gen3-service-manual.txt", org="aurora", team=None, clearance=1)
    first = load_corpus.chunk(doc, "kb-abc", text)
    again = load_corpus.chunk(doc, "kb-abc", text)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert first[0].chunk_id == "kb-abc-0000" and len(first) > 10
    assert {c.filename for c in first} == {"pika-gen3-service-manual.txt"}


def test_a_crlf_copy_chunks_and_hashes_exactly_like_the_lf_original(tmp_path):
    from manifest import doc_text

    source = "aurora/pika-gen3-service-manual.txt"
    lf = load_corpus.doc_text(source)
    assert "\r" not in lf
    copies = {"lf": lf.encode("utf-8"), "crlf": lf.replace("\n", "\r\n").encode("utf-8")}
    for name, data in copies.items():
        (tmp_path / name / "aurora").mkdir(parents=True)
        (tmp_path / name / source).write_bytes(data)
    assert b"\r\n" in (tmp_path / "crlf" / source).read_bytes()
    texts = {n: doc_text(source, tmp_path / n) for n in copies}
    assert texts["crlf"] == texts["lf"] == lf

    doc = Doc(file=source, org="aurora", team=None, clearance=1)
    ids = {n: load_corpus.doc_id_for(source, t) for n, t in texts.items()}
    chunks = {n: load_corpus.chunk(doc, ids[n], t) for n, t in texts.items()}
    assert ids["crlf"] == ids["lf"]
    assert [(c.chunk_id, c.text) for c in chunks["crlf"]] == [(c.chunk_id, c.text) for c in chunks["lf"]]
    assert len(chunks["lf"]) == 26  # as loaded into the Phase 3 database
