"""Database wiring + the small query-log API used by main.py.

Deliberate behavior: observability must never take the product down. If
DATABASE_URL is unset or Postgres is unreachable, we warn and keep serving —
every log call is individually guarded.
"""
import logging
import statistics

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from db.models import Base, Feedback, QueryLog

logger = logging.getLogger(__name__)

_session_factory: sessionmaker | None = None


def init_db(settings: Settings) -> bool:
    """Create the engine and tables. Returns True if logging is active."""
    global _session_factory
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — query logging is DISABLED.")
        return False
    try:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        Base.metadata.create_all(engine)
        _add_missing_columns(engine)
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        logger.info("Query logging enabled (Postgres).")
        return True
    except Exception:
        logger.exception("Could not connect to Postgres — query logging is DISABLED.")
        return False


def _add_missing_columns(engine) -> None:
    """Forward-only micro-migration: ADD COLUMN for any model column missing
    from an existing table. create_all only creates whole tables, so without
    this a database from an earlier version would silently keep the old
    schema. All columns added after v1 are nullable, which is what makes this
    safe. A real migration tool (Alembic) is the upgrade path if schema
    changes ever get more interesting than additive nullable columns.
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE {table.name} ADD COLUMN "
                        f"{column.name} {column.type.compile(engine.dialect)}"
                    ))
                    logger.info("Schema migrated: added %s.%s", table.name, column.name)


def log_query(record: dict) -> int | None:
    """Insert one QueryLog row; swallow (but log) any failure.

    Returns the new row id — the handle the feedback endpoint links to —
    or None when logging is disabled or the write failed.
    """
    if _session_factory is None:
        return None
    try:
        with _session_factory() as session:  # type: Session
            row = QueryLog(**record)
            session.add(row)
            session.commit()
            return row.id
    except Exception:
        logger.exception("Failed to write query log")
        return None


def log_feedback(query_log_id: int, thumbs_up: bool, comment: str | None) -> str:
    """Upsert the user's verdict for one query. Returns a status string the
    API maps to a response code: ok | unknown_query | disabled | error."""
    if _session_factory is None:
        return "disabled"
    try:
        with _session_factory() as session:
            if session.get(QueryLog, query_log_id) is None:
                return "unknown_query"
            existing = session.scalars(
                select(Feedback).where(Feedback.query_log_id == query_log_id)
            ).first()
            if existing:
                existing.thumbs_up = thumbs_up
                if comment is not None:
                    existing.comment = comment
            else:
                session.add(
                    Feedback(
                        query_log_id=query_log_id,
                        thumbs_up=thumbs_up,
                        comment=comment,
                    )
                )
            session.commit()
            return "ok"
    except Exception:
        logger.exception("Failed to write feedback")
        return "error"


def fetch_logs(limit: int = 20) -> list[dict]:
    if _session_factory is None:
        return []
    with _session_factory() as session:
        rows = session.scalars(
            select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
        ).all()
        return [row.as_dict() for row in rows]


def fetch_stats(window: int = 200) -> dict:
    """Outcome counts + latency distribution over the most recent `window` queries."""
    if _session_factory is None:
        return {"logging_enabled": False}

    with _session_factory() as session:
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
