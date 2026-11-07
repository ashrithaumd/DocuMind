"""Leakage tests on the real Phase 3 corpus, through the API.

Each test loads all 50 documents of kb/ with the real loader (fake
embeddings, no quota), seeds the 29 test users of kb/test-users.yaml, and
signs in as them with the fake token checker. What each user may see always
comes from the manifest (kb/tools/manifest.py, can_see), the same rule as
the RLS policy, never from a second hand-written list. A test fails if the
API shows a user anything the manifest hides, or (positive controls) hides
anything it allows.
"""
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

from tests.helpers import FakeEmbeddings

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "kb" / "tools"))

pytestmark = pytest.mark.skipif(not (REPO / "kb" / "manifest.yaml").exists(),
                                reason="needs the Phase 3 corpus (kb/)")

import load_corpus  # noqa: E402
import seed_test_users  # noqa: E402
from manifest import User, can_see, load_manifest  # noqa: E402

QUESTION = "What does the document say about pumps, salary, price, policy and the incident?"


class Corpus:
    def __init__(self, client, pg_url):
        self.client = client
        self.manifest = load_manifest()
        fast = load_corpus.RateLimiter(10**12)
        load_corpus.load(pg_url, self.manifest, embeddings=FakeEmbeddings(), limiter=fast,
                         token_limiter=fast)
        self.users = {u.id: u for u in seed_test_users.load_users(manifest=self.manifest)
                      if not u.email}
        seed_test_users.seed(pg_url, list(self.users.values()), self.manifest)
        engine = sa.create_engine(pg_url)
        with engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT doc_id, org_id, filename FROM documents "
                                        "WHERE doc_id LIKE 'kb-%'")).all()
        engine.dispose()
        self.file_of = {doc_id: f"{org}/{name}" for doc_id, org, name in rows}  # doc_id -> kb path
        self.id_of = {f: d for d, f in self.file_of.items()}
        assert len(self.id_of) == 50

    def headers(self, user_id):
        return {"Authorization": f"Bearer token:{self.users[user_id].fake_uid}"}

    def expected(self, user_id) -> set[str]:
        u = self.users[user_id]
        user = User(org=u.org, teams=u.teams, clearance=u.clearance)
        return {d.file for d in self.manifest.docs.values() if can_see(user, d)}

    def listed(self, user_id) -> set[str]:
        response = self.client.get("/documents", headers=self.headers(user_id))
        assert response.status_code == 200
        org = self.users[user_id].org
        return {f"{org}/{d['filename']}" for d in response.json()["documents"]}

    def ask(self, user_id, files=None, question=QUESTION) -> dict:
        ids = [self.id_of[f] for f in (files or self.id_of)]
        response = self.client.post("/query", headers=self.headers(user_id),
                                    json={"question": question, "doc_ids": ids})
        assert response.status_code == 200, response.text
        return response.json()

    def retrieved(self, body) -> set[str]:
        """kb paths of every chunk retrieved and every citation."""
        chunk_ids = [c["chunk_id"] for c in body["chunks"]] + [c["chunk_id"] for c in body["citations"]]
        return {self.file_of[cid.rsplit("-", 1)[0]] for cid in chunk_ids}


@pytest.fixture
def corpus(client, pg_url):
    return Corpus(client, pg_url)


def _sees(corpus, user_id, file) -> bool:
    return file in corpus.listed(user_id)


# --- every user, every document -------------------------------------------------------

def test_documents_lists_exactly_what_the_manifest_allows_for_every_user(corpus):
    for user_id in corpus.users:
        assert corpus.listed(user_id) == corpus.expected(user_id), user_id


def test_query_over_all_50_documents_retrieves_only_visible_ones(corpus):
    for user_id in corpus.users:
        body = corpus.ask(user_id)  # asks for every document of both orgs, by id
        got = corpus.retrieved(body)
        assert got <= corpus.expected(user_id), (user_id, got - corpus.expected(user_id))
        assert got, f"{user_id} retrieved nothing: the negative check would pass vacuously"


# --- named boundaries --------------------------------------------------------------

def test_mirrored_facts_never_cross_orgs(corpus):
    meal = "What is the daily meal allowance while travelling?"
    stipend = "How much is the on-call stipend per week?"
    aurora_meal, bluefin_meal = "75 USD per person", "up to 60 USD a day"
    aurora_stipend, bluefin_stipend = "stipend of 400 USD per week", "350 USD per week"
    for user_id, own, other in (("aurora-admin", (aurora_meal, aurora_stipend), (bluefin_meal, bluefin_stipend)),
                                ("bluefin-admin", (bluefin_meal, bluefin_stipend), (aurora_meal, aurora_stipend))):
        texts = ""
        for question in (meal, stipend):
            body = corpus.ask(user_id, question=question)  # every document of both orgs
            assert {f.split("/")[0] for f in corpus.retrieved(body)} == {user_id.split("-")[0]}
            texts += " ".join(" ".join(c["text"].split()) for c in body["chunks"]) + body["answer"]
        for phrase in other:
            assert phrase not in texts, (user_id, phrase)
        assert any(phrase in texts for phrase in own), f"{user_id}: own facts not retrieved"


def test_another_orgs_documents_by_id_retrieve_nothing(corpus):
    bluefin_files = [f for f in corpus.id_of if f.startswith("bluefin/")]
    body = corpus.ask("aurora-admin", files=bluefin_files)
    assert body["chunks"] == [] and body["citations"] == []


def test_team_only_at_clearance_0_pair(corpus):
    # #13 field-ops notes and #35 ML notes: clearance 0, so only the team decides.
    for file, member, outsiders in (
        ("aurora/meeting-notes-field-ops-2026-10-08.txt", "aurora-field-ops-c0",
         ["aurora-viewer-c3", "aurora-platform-c3", "aurora-people-c3"]),
        ("bluefin/meeting-notes-ml-2026-10-02.txt", "bluefin-ml-c0",
         ["bluefin-viewer-c3", "bluefin-finance-c3", "bluefin-data-eng-c3"]),
    ):
        assert _sees(corpus, member, file)
        assert corpus.retrieved(corpus.ask(member, files=[file])) == {file}
        for outsider in outsiders:  # even at clearance 3
            assert not _sees(corpus, outsider, file), (outsider, file)
            assert corpus.ask(outsider, files=[file])["chunks"] == [], (outsider, file)


@pytest.mark.parametrize("summary,full,sees_summary_only,sees_both,sees_neither", [
    # Atlas incident: summary org-wide at 1, full report platform at 2.
    ("aurora/postmortem-2026-08-atlas-fleet-stall.txt",
     "aurora/postmortem-2026-08-atlas-fleet-stall-full-report.txt",
     ["aurora-viewer-c1", "aurora-viewer-c3", "aurora-field-ops-c3"], ["aurora-platform-c3"],
     ["aurora-viewer-c0", "aurora-platform-c0"]),
    # Forecast incident: summary org-wide at 1, full report data-eng at 2.
    ("bluefin/postmortem-2026-07-forecast-outage.txt",
     "bluefin/postmortem-2026-07-forecast-outage-full-report.txt",
     ["bluefin-viewer-c1", "bluefin-ml-c3", "bluefin-finance-c3"], ["bluefin-data-eng-c3"],
     ["bluefin-viewer-c0", "bluefin-data-eng-c0"]),
    # CellCore: summary org-wide at 1, full contract org-wide at 2 (clearance alone).
    ("aurora/vendor-contract-cellcore.txt", "aurora/cellcore-supply-agreement.txt",
     ["aurora-viewer-c1"],
     ["aurora-viewer-c2", "aurora-people-c3"], ["aurora-viewer-c0", "aurora-firmware-c0"]),
])
def test_summary_and_full_versions_by_clearance_and_team(corpus, summary, full, sees_summary_only,
                                                         sees_both, sees_neither):
    for u in sees_summary_only:
        assert (_sees(corpus, u, summary), _sees(corpus, u, full)) == (True, False), u
        assert corpus.retrieved(corpus.ask(u, files=[summary, full])) == {summary}, u
    for u in sees_both:
        assert (_sees(corpus, u, summary), _sees(corpus, u, full)) == (True, True), u
        assert corpus.retrieved(corpus.ask(u, files=[full])) == {full}, u
    for u in sees_neither:
        assert corpus.ask(u, files=[summary, full])["chunks"] == [], u


def test_salary_bands_need_the_team_and_clearance_3(corpus):
    for files, member, refused in (
        (["aurora/salary-bands-2026.txt", "aurora/salary-bands-2027.txt"], "aurora-people-c3",
         ["aurora-people-c0", "aurora-viewer-c3", "aurora-platform-c3", "aurora-editor"]),
        (["bluefin/salary-bands-2026.txt", "bluefin/salary-bands-2027.txt"], "bluefin-finance-c3",
         ["bluefin-finance-c2", "bluefin-finance-c0", "bluefin-viewer-c3", "bluefin-ml-c3"]),
    ):
        assert corpus.retrieved(corpus.ask(member, files=files)) == set(files)
        for u in refused:
            assert corpus.ask(u, files=files)["chunks"] == [], u
            assert not set(files) & corpus.listed(u), u


def test_gullwing_needs_finance_and_clearance_3(corpus):
    gullwing = "bluefin/acquisition-plan-gullwing.txt"
    allowed = {"bluefin-finance-c3", "bluefin-admin"}
    for user_id in [u for u in corpus.users if u.startswith("bluefin-")]:
        visible = _sees(corpus, user_id, gullwing)
        assert visible == (user_id in allowed), user_id
        got = corpus.retrieved(corpus.ask(user_id, files=[gullwing]))
        assert got == ({gullwing} if user_id in allowed else set()), user_id
    # finance at clearance 2 still sees finance's clearance-2 documents
    assert {"bluefin/meeting-notes-finance-2026-09-30.txt",
            "bluefin/northgate-master-services-agreement.txt"} <= corpus.listed("bluefin-finance-c2")


@pytest.mark.parametrize("old,current,outsider,member,old_only_term", [
    ("aurora/pika-gen2-service-manual.txt", "aurora/pika-gen3-service-manual.txt",
     "aurora-viewer-c1", "aurora-field-ops-c3", "AUR-PK2-3307-A"),
    ("bluefin/current-v3-operations-manual.txt", "bluefin/current-v4-operations-manual.txt",
     "bluefin-viewer-c1", "bluefin-ml-c3", "CUR3-CFG-scoring.max_feature_age_hours"),
])
def test_old_versions_that_are_team_only(corpus, old, current, outsider, member, old_only_term):
    # Outside the team, the reused error code finds only the current manual.
    code = "E-2417" if "pika" in old else "CUR-4107"
    body = corpus.ask(outsider, files=[old, current], question=f"What does error {code} mean?")
    assert corpus.retrieved(body) == {current}
    assert old_only_term not in " ".join(c["text"] for c in body["chunks"]) + body["answer"]
    # The team sees both versions.
    assert {old, current} <= corpus.listed(member)
    assert corpus.retrieved(corpus.ask(member, files=[old])) == {old}


def test_changing_access_on_an_invisible_document_is_404(corpus):
    for user_id, file in (("aurora-editor", "bluefin/pricing-2027.txt"),
                          ("aurora-editor", "aurora/salary-bands-2027.txt")):
        response = corpus.client.patch(f"/documents/{corpus.id_of[file]}/access",
                                       headers=corpus.headers(user_id),
                                       json={"team_id": None, "clearance": 0})
        assert response.status_code == 404, (user_id, file)
