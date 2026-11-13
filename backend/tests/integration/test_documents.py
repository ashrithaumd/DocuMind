"""The documents table is the write path for access.

add_document (the corpus loader's write) stores the documents row (org,
team, clearance) and its chunks in one transaction, the chunks copying
access from that row; only editors and admins write, and RLS refuses a team
they're not in or a clearance above theirs. There is no upload or
access-change API.

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

EDITOR = Access(org_id="A", clearance=1, team_ids=("t1",), role="editor")


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


def _access(owner, doc_id):
    """(document's access, set of its chunks' access)."""
    with owner.connect() as conn:
        doc = conn.execute(sa.text(
            "SELECT org_id, team_id, clearance FROM documents WHERE doc_id = :d"), {"d": doc_id}).one()
        chunks = {tuple(r) for r in conn.execute(sa.text(
            "SELECT org_id, team_id, clearance FROM chunks WHERE doc_id = :d"), {"d": doc_id})}
    return tuple(doc), chunks


def _documents(owner) -> int:
    with owner.connect() as conn:
        return conn.execute(sa.text("SELECT count(*) FROM documents")).scalar()


# --- add_document -------------------------------------------------------------

def _chunks(doc_id, n):
    return [Chunk(chunk_id=f"{doc_id}-{i:04d}", doc_id=doc_id, filename="f.txt",
                  page_number=1, chunk_index=i, text=f"Pump part {i}.") for i in range(n)]


async def test_rls_refuses_a_team_or_clearance_beyond_the_caller(users, owner):
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


VIEWER = Access(org_id="A", clearance=1, team_ids=("t1",), role="viewer")


async def test_add_document_records_access_and_copies_it_to_every_chunk(users, owner):
    await deps.get_store().scoped(EDITOR).add_document(_chunks("d", 40), team_id="t1",
                                                       clearance=1, uploaded_by=users["editor"])
    doc, chunks = _access(owner, "d")
    assert doc == ("A", "t1", 1)
    assert chunks == {doc}  # all 40, not some
    with owner.connect() as conn:
        assert conn.execute(sa.text("SELECT uploaded_by FROM documents WHERE doc_id = 'd'")
                            ).scalar() == users["editor"]


async def test_a_document_is_visible_only_within_its_access(client, users):
    await deps.get_store().scoped(EDITOR).add_document(_chunks("d", 2), team_id="t1",
                                                       clearance=1, uploaded_by=None)

    def listed(uid):
        return [d["doc_id"] for d in client.get("/documents", headers=_as(uid)).json()["documents"]]

    assert listed("viewer") == ["d"]  # same team, same clearance
    assert listed("other") == []  # team t2


async def test_the_store_refuses_a_viewer(users, owner):
    with pytest.raises(PermissionError):
        await deps.get_store().scoped(VIEWER).add_document(_chunks("e", 1), team_id=None,
                                                           clearance=0, uploaded_by=None)
    assert _documents(owner) == 0
