"""The documents table is the write path for access (Phase 2 commit 6).

Uploads write the documents row (org, team, clearance) and its chunks in one
transaction, the chunks copying access from that row; only editors and
admins upload, never into a team they're not in or above their clearance;
set_document_access changes a document and every chunk at once.

Users (org A, signed in explicitly, never the default admin fixture):
    editor  role editor, team t1, clearance 1
    viewer  role viewer, team t1, clearance 1
    other   role viewer, team t2, clearance 1
"""
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, ProgrammingError

import deps
from ingestion.chunking import Chunk
from ingestion.store import Access

EDITOR = Access(org_id="A", clearance=1, team_ids=("t1",))


def _as(uid):
    return {"Authorization": f"Bearer token:{uid}"}


@pytest.fixture
def owner(pg_url):
    engine = sa.create_engine(pg_url)
    yield engine
    engine.dispose()


@pytest.fixture
def users(client, owner):
    """Needs `client` first: its startup runs the migrations. Returns user ids."""
    ids = {}
    with owner.begin() as conn:
        conn.execute(sa.text("INSERT INTO orgs VALUES ('A', 'Org A'), ('B', 'Org B')"))
        conn.execute(sa.text("INSERT INTO teams VALUES ('t1', 'A', 'One'), ('t2', 'A', 'Two')"))
        for uid, role, team in (("editor", "editor", "t1"), ("viewer", "viewer", "t1"),
                                ("other", "viewer", "t2")):
            ids[uid] = conn.execute(sa.text(
                "INSERT INTO users (firebase_uid, org_id, role, clearance) "
                "VALUES (:uid, 'A', :role, 1) RETURNING user_id"), {"uid": uid, "role": role}).scalar()
            conn.execute(sa.text("INSERT INTO team_memberships VALUES (:u, :t, 'A')"),
                         {"u": ids[uid], "t": team})
    return ids


def _upload(client, uid, text=b"Pump notes. The pump runs at 400 RPM.", **form):
    return client.post("/upload", headers=_as(uid), data=form,
                       files={"file": ("notes.txt", text, "text/plain")})


def _access(owner, doc_id):
    """(document's access, set of its chunks' access)."""
    with owner.connect() as conn:
        doc = conn.execute(sa.text(
            "SELECT org_id, team_id, clearance FROM documents WHERE doc_id = :d"), {"d": doc_id}).one()
        chunks = {tuple(r) for r in conn.execute(sa.text(
            "SELECT org_id, team_id, clearance FROM chunks WHERE doc_id = :d"), {"d": doc_id})}
    return tuple(doc), chunks


def _divergent_chunks(owner) -> int:
    with owner.connect() as conn:
        return conn.execute(sa.text("""
            SELECT count(*) FROM chunks c JOIN documents d USING (doc_id)
            WHERE (c.org_id, c.team_id, c.clearance) IS DISTINCT FROM (d.org_id, d.team_id, d.clearance)
        """)).scalar()


def _documents(owner) -> int:
    with owner.connect() as conn:
        return conn.execute(sa.text("SELECT count(*) FROM documents")).scalar()


# --- uploads ------------------------------------------------------------------

def test_upload_records_access_on_the_document_and_copies_it_to_every_chunk(client, users, owner):
    many_pages = b"\n\n".join(b"Pump section %d. " % i * 60 for i in range(6))
    response = _upload(client, "editor", text=many_pages, team_id="t1", clearance="1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["team_id"], body["clearance"]) == ("t1", 1)
    assert body["chunks_created"] > 1

    doc, chunks = _access(owner, body["doc_id"])
    assert doc == ("A", "t1", 1)
    assert chunks == {doc}
    with owner.connect() as conn:
        assert conn.execute(sa.text("SELECT uploaded_by FROM documents WHERE doc_id = :d"),
                            {"d": body["doc_id"]}).scalar() == users["editor"]


def test_upload_without_a_team_is_org_wide(client, users, owner):
    body = _upload(client, "editor", clearance="0").json()
    assert _access(owner, body["doc_id"]) == (("A", None, 0), {("A", None, 0)})


def test_only_editors_and_admins_can_upload(client, users, owner):
    assert _upload(client, "viewer", clearance="0").status_code == 403
    assert _documents(owner) == 0


@pytest.mark.parametrize("form,why", [
    ({"team_id": "t2", "clearance": "0"}, "a team the editor isn't in"),
    ({"team_id": "t9", "clearance": "0"}, "a team that doesn't exist"),
    ({"clearance": "2"}, "a clearance above the editor's"),
])
def test_upload_cannot_reach_beyond_the_uploaders_access(client, users, owner, form, why):
    response = _upload(client, "editor", **form)
    assert response.status_code == 403, why
    assert _documents(owner) == 0


@pytest.mark.parametrize("form", [{}, {"clearance": "-1"}, {"clearance": "high"}])
def test_upload_needs_an_explicit_valid_clearance(client, users, owner, form):
    assert _upload(client, "editor", **form).status_code == 422
    assert _documents(owner) == 0


def test_uploaded_document_is_visible_only_within_its_access(client, users):
    doc_id = _upload(client, "editor", team_id="t1", clearance="1").json()["doc_id"]

    def listed(uid):
        return [d["doc_id"] for d in client.get("/documents", headers=_as(uid)).json()["documents"]]

    assert listed("viewer") == [doc_id]  # same team, same clearance
    assert listed("other") == []  # team t2


# --- the store, below the API checks ------------------------------------------

def _chunks(doc_id, n):
    return [Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename="f.txt",
                  page_number=1, chunk_index=i, text=f"Pump part {i}.") for i in range(n)]


async def test_rls_refuses_a_team_or_clearance_beyond_the_caller_even_without_the_api(users, owner):
    store = deps.get_store().scoped(EDITOR)
    for team, clearance in (("t2", 0), (None, 2)):
        with pytest.raises(ProgrammingError, match="row-level security"):
            await store.add_document(_chunks("d", 2), team_id=team, clearance=clearance,
                                     uploaded_by=None)
    assert _documents(owner) == 0


async def test_a_failed_chunk_insert_leaves_no_documents_row(users, owner):
    chunks = _chunks("d", 3)
    chunks[2] = chunks[0]  # duplicate chunk_id: the last insert fails
    with pytest.raises(IntegrityError):
        await deps.get_store().scoped(EDITOR).add_document(chunks, team_id="t1", clearance=1,
                                                           uploaded_by=None)
    assert _documents(owner) == 0
    with owner.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM chunks")).scalar() == 0


# --- set_document_access ------------------------------------------------------

async def test_access_change_updates_the_document_and_every_chunk_at_once(users, owner):
    store = deps.get_store().scoped(EDITOR)
    await store.add_document(_chunks("d", 40), team_id=None, clearance=0, uploaded_by=None)
    assert _access(owner, "d") == (("A", None, 0), {("A", None, 0)})

    store.set_document_access("d", team_id="t1", clearance=1)

    doc, chunks = _access(owner, "d")
    assert doc == ("A", "t1", 1)
    assert chunks == {doc}  # all 40, not some
    with owner.connect() as conn:
        assert conn.execute(sa.text(
            "SELECT count(*) FROM chunks WHERE doc_id = 'd' AND team_id = 't1' AND clearance = 1"
        )).scalar() == 40
    assert _divergent_chunks(owner) == 0


@pytest.mark.parametrize("team,clearance", [("t2", 0), (None, 2)])
async def test_a_refused_access_change_changes_nothing(users, owner, team, clearance):
    store = deps.get_store().scoped(EDITOR)
    await store.add_document(_chunks("d", 5), team_id="t1", clearance=1, uploaded_by=None)

    with pytest.raises(ProgrammingError, match="row-level security"):
        store.set_document_access("d", team_id=team, clearance=clearance)

    assert _access(owner, "d") == (("A", "t1", 1), {("A", "t1", 1)})


async def test_access_change_on_a_document_the_caller_cant_see_is_refused(users, owner):
    store = deps.get_store().scoped(EDITOR)
    await store.add_document(_chunks("d", 2), team_id="t1", clearance=1, uploaded_by=None)

    org_b = deps.get_store().scoped(Access(org_id="B", clearance=9))
    with pytest.raises(LookupError):
        org_b.set_document_access("d", team_id=None, clearance=0)
    assert _access(owner, "d") == (("A", "t1", 1), {("A", "t1", 1)})


async def test_access_change_takes_effect_on_the_next_read(client, users):
    store = deps.get_store().scoped(EDITOR)
    await store.add_document(_chunks("d", 2), team_id="t1", clearance=1, uploaded_by=None)
    assert [d["doc_id"] for d in client.get("/documents", headers=_as("viewer")).json()["documents"]] == ["d"]

    store.set_document_access("d", team_id=None, clearance=0)  # org-wide now
    assert [d["doc_id"] for d in client.get("/documents", headers=_as("other")).json()["documents"]] == ["d"]


async def test_no_chunk_ever_differs_from_its_document(client, users, owner):
    store = deps.get_store().scoped(EDITOR)
    for i, (team, clearance) in enumerate([(None, 0), ("t1", 1), ("t1", 0)]):
        await store.add_document(_chunks(f"d{i}", 4), team_id=team, clearance=clearance,
                                 uploaded_by=None)
    _upload(client, "editor", team_id="t1", clearance="1")
    store.set_document_access("d0", team_id="t1", clearance=1)
    store.set_document_access("d1", team_id=None, clearance=0)

    assert _divergent_chunks(owner) == 0
