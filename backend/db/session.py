"""Database wiring + the small query-log API used by main.py.

The same Postgres holds the chunk store (ingestion/store.py), so the app
needs it to serve at all. Query logging itself stays fail-open: every log
call is individually guarded, so a failed log write never fails a query.

Two separate steps, so the app process never holds owner credentials:
  - migrate_database(): the one-shot migrate step (backend/migrate.py; the
    `migrate` service in docker-compose, a Cloud Run job in Phase 9). As the
    owner (MIGRATION_DATABASE_URL): Alembic to head, finpilot_app's password
    from DATABASE_URL, and the embedding-model check across every org.
  - init_db(): the app's startup. Connects as finpilot_app (DATABASE_URL)
    for the query log; migrates nothing.
"""
import logging
import statistics
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, make_url, select, text
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from db.access import set_access
from db.models import Base, Feedback, QueryLog

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
APP_ROLE = "finpilot_app"  # created by migration 0004
# The revision matching the schema create_all built before Alembic.
BASELINE_REVISION = "0001"
BASELINE_TABLES = ("query_logs", "feedback")
# Model columns that later revisions add: adopting a pre-Alembic database
# must leave them to those revisions.
POST_BASELINE_COLUMNS = {"user_id", "org_id"}  # 0007

_session_factory: sessionmaker | None = None


def migrate_database(settings: Settings) -> None:
    """The migrate step, as the owner (MIGRATION_DATABASE_URL; DATABASE_URL
    when unset, for single-role databases such as the SQLite log tests).
    Raises on any failure: the app must not start on a half-migrated
    database."""
    owner = create_engine(settings.migration_database_url or settings.database_url)
    try:
        migrate(owner, settings)
    finally:
        owner.dispose()


def init_db(settings: Settings) -> bool:
    """The app's query-log connection (DATABASE_URL, finpilot_app). The schema
    must already be migrated (migrate_database). Returns True if logging is
    active."""
    global _session_factory
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — query logging is DISABLED.")
        return False
    try:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        logger.info("Query logging enabled (Postgres).")
        return True
    except Exception:
        logger.exception("Could not connect to Postgres — query logging is DISABLED.")
        return False


def migrate(engine, settings: Settings) -> None:
    """Run Alembic to head on this engine. Migrations read `settings` (the
    chunk vector size comes from EMBEDDING_DIMENSIONS), not the global ones.

    A database created by create_all before Alembic existed (tables but no
    alembic_version) is adopted first: columns a newer Phase 0 model added
    are filled in, then it's stamped at the baseline revision, whose schema
    it now matches.
    """
    config = Config(str(ALEMBIC_INI))
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        config.attributes["settings"] = settings
        inspector = inspect(conn)
        if inspector.has_table("query_logs") and not inspector.has_table("alembic_version"):
            _add_missing_columns(conn)
            command.stamp(config, BASELINE_REVISION)
            logger.info("Adopted a pre-Alembic database at revision %s.", BASELINE_REVISION)
        command.upgrade(config, "head")
        _sync_app_role_password(conn, settings)
        _check_embedding_models(conn, settings)


def _check_embedding_models(conn, settings: Settings) -> None:
    """Refuse to go on if the chunks hold vectors from another embedding
    model: vectors from two models aren't comparable. It reads every org's
    chunks, so the migration role must be able to bypass RLS."""
    if conn.dialect.name != "postgresql" or not inspect(conn).has_table("chunks"):
        return
    # shortcut: true for the Docker Postgres owner (a superuser). On managed
    # Postgres (Phase 9) the migration role needs BYPASSRLS for this check.
    if not conn.execute(text("SELECT rolsuper OR rolbypassrls FROM pg_roles "
                             "WHERE rolname = current_user")).scalar():
        raise RuntimeError("The migration role can't bypass row-level security, so it can't "
                           "check every org's embedding model. Give it BYPASSRLS.")
    models = conn.execute(text("SELECT DISTINCT embedding_model FROM chunks")).scalars().all()
    foreign = [m for m in models if m != settings.embedding_model]
    if foreign:
        raise RuntimeError(
            f"The chunks table holds vectors from {foreign}, but the configured embedding "
            f"model is '{settings.embedding_model}'. Vectors from different models aren't "
            "comparable: use a fresh database and re-upload your documents."
        )


def _sync_app_role_password(conn, settings: Settings) -> None:
    """Give finpilot_app the password in DATABASE_URL, so one setting holds
    it. Only when migrating as a different (owner) role, on Postgres."""
    if conn.dialect.name != "postgresql" or not settings.migration_database_url:
        return
    app = make_url(settings.database_url)
    if app.username != APP_ROLE or not app.password:
        return
    # format(%L) quotes the password as a literal; it's never logged.
    statement = conn.execute(text(f"SELECT format('ALTER ROLE {APP_ROLE} PASSWORD %L', :pw)"),
                             {"pw": app.password}).scalar()
    # The raw cursor with no parameters: SQLAlchemy would %-format the
    # statement (breaking on a '%' in the password) and text() would read
    # ':x' in it as a bind parameter. Same transaction as the migration.
    cursor = conn.connection.cursor()
    try:
        cursor.execute(statement)
    finally:
        cursor.close()


def _add_missing_columns(conn) -> None:
    """ADD COLUMN for any baseline-table column missing from a pre-Alembic
    database (all columns added after v1 are nullable, which makes this
    safe). Only used when adopting such a database; every later schema
    change is an Alembic revision.
    """
    inspector = inspect(conn)
    for name in BASELINE_TABLES:
        table = Base.metadata.tables[name]
        if not inspector.has_table(name):
            continue
        existing = {c["name"] for c in inspector.get_columns(name)}
        for column in table.columns:
            if column.name not in existing and column.name not in POST_BASELINE_COLUMNS:
                conn.execute(text(
                    f"ALTER TABLE {name} ADD COLUMN "
                    f"{column.name} {column.type.compile(conn.dialect)}"
                ))
                logger.info("Schema migrated: added %s.%s", name, column.name)


@contextmanager
def _session(access):
    """A log session whose transaction runs under `access` (RLS by org)."""
    with _session_factory() as session:
        set_access(session, access)
        yield session


def log_query(record: dict, *, access) -> int | None:
    """Insert one QueryLog row in the caller's org; swallow (but log) any
    failure.

    Returns the new row id — the handle the feedback endpoint links to —
    or None when logging is disabled or the write failed.
    """
    if _session_factory is None:
        return None
    try:
        with _session(access) as session:
            row = QueryLog(**record)
            session.add(row)
            session.commit()
            return row.id
    except Exception:
        logger.exception("Failed to write query log")
        return None


def log_feedback(query_log_id: int, thumbs_up: bool, comment: str | None, *,
                 access, user_id: str | None) -> str:
    """Upsert the user's verdict for one query of their own org. Returns a
    status string the API maps to a response code: ok | unknown_query |
    disabled | error. Another org's query is unknown_query: RLS hides it."""
    if _session_factory is None:
        return "disabled"
    try:
        with _session(access) as session:
            if session.get(QueryLog, query_log_id) is None:
                return "unknown_query"
            existing = session.scalars(
                select(Feedback).where(Feedback.query_log_id == query_log_id)
            ).first()
            if existing:
                existing.thumbs_up = thumbs_up
                existing.user_id = user_id
                if comment is not None:
                    existing.comment = comment
            else:
                session.add(
                    Feedback(
                        query_log_id=query_log_id,
                        thumbs_up=thumbs_up,
                        comment=comment,
                        user_id=user_id,
                        org_id=access.org_id,
                    )
                )
            session.commit()
            return "ok"
    except Exception:
        logger.exception("Failed to write feedback")
        return "error"


def fetch_logs(limit: int = 20, *, access) -> list[dict]:
    """The caller's org's most recent query logs (RLS by org)."""
    if _session_factory is None:
        return []
    with _session(access) as session:
        rows = session.scalars(
            select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
        ).all()
        return [row.as_dict() for row in rows]


def fetch_stats(window: int = 200, *, access) -> dict:
    """Outcome counts + latency distribution over the caller's org's most
    recent `window` queries (RLS by org)."""
    if _session_factory is None:
        return {"logging_enabled": False}

    with _session(access) as session:
        rows = session.scalars(
            select(QueryLog).order_by(QueryLog.id.desc()).limit(window)
        ).all()
        feedback_rows = session.scalars(
            select(Feedback).order_by(Feedback.id.desc()).limit(window)
        ).all()
        recent_negative = session.execute(
            select(Feedback, QueryLog.question)
            .join(QueryLog, Feedback.query_log_id == QueryLog.id)
            .where(Feedback.thumbs_up.is_(False))
            .order_by(Feedback.id.desc())
            .limit(5)
        ).all()

    outcomes: dict[str, int] = {}
    routes: dict[str, int] = {}
    for row in rows:
        outcomes[row.outcome] = outcomes.get(row.outcome, 0) + 1
        if row.route:
            routes[row.route] = routes.get(row.route, 0) + 1

    thumbs_down = sum(1 for f in feedback_rows if not f.thumbs_up)

    def summarize(values: list[float]) -> dict | None:
        values = [v for v in values if v is not None]
        if not values:
            return None
        values.sort()
        p95_index = max(0, round(0.95 * (len(values) - 1)))
        return {
            "avg_ms": round(statistics.mean(values), 1),
            "p50_ms": round(statistics.median(values), 1),
            "p95_ms": round(values[p95_index], 1),
        }

    return {
        "logging_enabled": True,
        "window": len(rows),
        "outcomes": outcomes,
        "routes": routes,
        # The quality-drift signal: a climbing thumbs_down_pct or a cluster of
        # recent negative comments says answers got worse before any metric
        # in `outcomes` would show it.
        "feedback": {
            "total": len(feedback_rows),
            "thumbs_down": thumbs_down,
            "thumbs_down_pct": (
                round(100 * thumbs_down / len(feedback_rows), 1)
                if feedback_rows
                else None
            ),
            "recent_negative": [
                {
                    "query_id": f.query_log_id,
                    "question": question,
                    "comment": f.comment,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f, question in recent_negative
            ],
        },
        "latency": {
            "retrieval": summarize([r.retrieval_ms for r in rows]),
            "grading": summarize([r.grading_ms for r in rows]),
            "generation": summarize([r.generation_ms for r in rows]),
            "groundedness": summarize([r.groundedness_ms for r in rows]),
            "total": summarize([r.total_ms for r in rows]),
        },
    }
