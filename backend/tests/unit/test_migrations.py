"""Alembic at startup: a fresh database is migrated to head, a database
created by create_all before Alembic is adopted (missing columns added,
stamped at the baseline), re-running is a no-op, and the migrations
produce exactly the schema the models describe."""
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.config import Config

from config import Settings
from db import session as db_session
from db.models import Base


def _settings(tmp_path):
    return Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'log.db'}")


def _head() -> str:
    return ScriptDirectory.from_config(Config(str(db_session.ALEMBIC_INI))).get_current_head()


def _revision(engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def test_fresh_database_is_migrated_to_head(tmp_path):
    settings = _settings(tmp_path)

    assert db_session.init_db(settings)

    engine = sa.create_engine(settings.database_url)
    assert _revision(engine) == _head()
    assert {"query_logs", "feedback"} <= set(sa.inspect(engine).get_table_names())


def test_migrations_match_the_models(tmp_path):
    settings = _settings(tmp_path)
    db_session.init_db(settings)

    engine = sa.create_engine(settings.database_url)
    with engine.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diffs == []


def test_pre_alembic_database_is_adopted(tmp_path):
    settings = _settings(tmp_path)
    engine = sa.create_engine(settings.database_url)
    # A database from before Alembic: built by create_all, and from a version
    # that didn't have the `route` column yet, with a row in it.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.text("ALTER TABLE query_logs DROP COLUMN route"))
        conn.execute(sa.text(
            "INSERT INTO query_logs (question, outcome, regenerated) VALUES ('q', 'answered', 0)"
        ))

    assert db_session.init_db(settings)

    assert _revision(engine) == _head()
    columns = {c["name"] for c in sa.inspect(engine).get_columns("query_logs")}
    assert "route" in columns
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM query_logs")).scalar() == 1


def test_migrating_twice_is_a_no_op(tmp_path):
    settings = _settings(tmp_path)

    assert db_session.init_db(settings)
    assert db_session.init_db(settings)

    assert _revision(sa.create_engine(settings.database_url)) == _head()
