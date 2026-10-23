"""Alembic environment.

Two ways in:
  - the app's startup (db.session.init_db) passes its own connection in
    config.attributes["connection"];
  - the CLI (`alembic upgrade head` from backend/) connects with the app's
    DATABASE_URL.
Logging config is left to the app: no fileConfig() here, so a migration at
startup doesn't reset the server's log handlers.
"""
from alembic import context
from sqlalchemy import create_engine

from config import get_settings
from db.models import Base

target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    _run(connection)
else:
    url = get_settings().database_url
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    with create_engine(url).connect() as conn:
        _run(conn)
