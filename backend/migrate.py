"""The one-shot migrate step: run it before the backend starts.

As the owner role (MIGRATION_DATABASE_URL): migrate the schema to head, set
documind_app's password from DATABASE_URL, and check that the stored vectors
come from the configured embedding model. Exits non-zero on any failure.

docker-compose runs it as the `migrate` service, which the backend waits for
(service_completed_successfully); Phase 9 runs it as a Cloud Run job. The
backend itself gets only DATABASE_URL and refuses to start with
MIGRATION_DATABASE_URL set.

    MIGRATION_DATABASE_URL=... DATABASE_URL=... python migrate.py
"""
import logging
import sys

from config import get_settings
from db.session import migrate_database


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not settings.migration_database_url or not settings.database_url:
        print("migrate.py needs MIGRATION_DATABASE_URL (the owner) and DATABASE_URL "
              "(documind_app, for its password).", file=sys.stderr)
        return 2
    migrate_database(settings)
    print("Migrations done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
