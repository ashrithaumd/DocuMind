"""The per-transaction access context that row-level security reads.

Every policy (chunks, documents: migration 0004; query_logs, feedback:
0007) compares a row's own columns with app.org_id, app.clearance and
app.team_ids; the customer-data policies (0009) compare customer_id with
app.customer_id. set_access sets them for the current transaction only
(set_config(..., true)), so nothing outlives it on a pooled connection; a
transaction that sets nothing sees no rows.
"""
from dataclasses import dataclass

import sqlalchemy as sa

# Roles that may write documents or change their access.
WRITER_ROLES = frozenset({"editor", "admin"})


@dataclass(frozen=True)
class Access:
    """Whose rows a transaction may see. auth.Principal has the same fields."""
    org_id: str
    clearance: int
    team_ids: tuple[str, ...] = ()
    role: str = "viewer"
    customer_id: str | None = None  # the bank customer this user is; None for staff


_SET_ACCESS = sa.text("""
    SELECT set_config('app.org_id', :org_id, true),
           set_config('app.clearance', CAST(:clearance AS text), true),
           set_config('app.team_ids', CAST(CAST(:team_ids AS text[]) AS text), true),
           set_config('app.customer_id', :customer_id, true)
""")


def set_access(conn, access) -> None:
    """Bind this transaction (a Connection or Session) to `access`: anything
    with org_id, clearance and team_ids (and optionally customer_id; none
    or None sets '', which no policy matches). Postgres only: the SQLite log
    stand-in has no RLS."""
    # A Session has get_bind(); a Connection has the dialect itself.
    dialect = conn.get_bind().dialect if hasattr(conn, "get_bind") else conn.dialect
    if dialect.name != "postgresql":
        return
    conn.execute(_SET_ACCESS, {"org_id": access.org_id, "clearance": access.clearance,
                               "team_ids": list(access.team_ids),
                               "customer_id": getattr(access, "customer_id", None) or ""})
