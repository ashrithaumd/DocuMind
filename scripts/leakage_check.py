"""Read-only leakage check of a loaded database, as the app role. Exits 1 on any leak.

    backend/.venv/Scripts/python scripts/neon.py run -- python ../scripts/leakage_check.py

Connects with DATABASE_URL (finpilot_app; a superuser, BYPASSRLS or owner
connection is refused, since it would see everything). Writes nothing, so
it is safe on a live database (Neon now, the Phase 12 red-team later);
never point the pytest fixtures at one, they create and drop roles and
databases. Checks:

  - the role: not superuser, no BYPASSRLS, owns no table; every table with
    a customer_id or clearance column (except users) has row-level
    security, forced;
  - no access context: every such table shows zero rows;
  - each customer (bank.synthetic's, same seed): every customer-owned table
    shows only that customer's rows, and at least one of them;
  - the same customer (clearance 0) sees no row above clearance 0 in any
    table with a clearance column (chunks, documents, doc_tables,
    doc_table_rows) and no staff canary, while staff (clearance 2) does
    see clearance-2 rows: the check isn't passing on an empty database.

The tables are read from the catalog, so tables added later are covered.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import sqlalchemy as sa  # noqa: E402

from bank.synthetic import BANK_ORG, generate  # noqa: E402
from config import sqlalchemy_url  # noqa: E402
from db.access import Access, set_access  # noqa: E402

CANARY = "%FINPILOT-CANARY-%"


# The sign-in directory: no RLS by design (the app role can only read it).
NO_RLS = {"users"}


def _tables(conn) -> tuple[dict[str, set[str]], list[str]]:
    """(protected table -> its columns, problems). Protected: has row-level
    security, or should have: a customer_id or clearance column. A table
    whose RLS was switched off stays in the check instead of dropping out."""
    rows = conn.execute(sa.text("""
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, a.attname
        FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'r'
    """)).all()
    columns: dict[str, set[str]] = {}
    flags: dict[str, tuple[bool, bool]] = {}
    for table, rls, forced, column in rows:
        columns.setdefault(table, set()).add(column)
        flags[table] = (rls, forced)
    protected = {t: cols for t, cols in columns.items()
                 if flags[t][0] or (cols & {"customer_id", "clearance"} and t not in NO_RLS)}
    problems = [f"{t}: row-level security is off" if not flags[t][0] else f"{t}: row-level security not forced"
                for t in sorted(protected) if not all(flags[t])]
    return protected, problems


def check(url: str, customer_ids: list[str]) -> list[str]:
    """Every problem found; empty means no leak."""
    problems: list[str] = []
    engine = sa.create_engine(sqlalchemy_url(url))
    try:
        with engine.begin() as conn:
            name, superuser, bypass, owned = conn.execute(sa.text("""
                SELECT current_user, r.rolsuper, r.rolbypassrls,
                       (SELECT count(*) FROM pg_tables WHERE tableowner = current_user)
                FROM pg_roles r WHERE r.rolname = current_user""")).one()
            if superuser or bypass or owned:
                return [f"{name} would bypass row-level security (superuser={superuser}, "
                        f"bypassrls={bypass}, owns {owned} tables): connect as finpilot_app"]
            tables, problems = _tables(conn)
            # No context set in this transaction: nothing may be visible.
            for table in sorted(tables):
                if n := conn.execute(sa.text(f'SELECT count(*) FROM "{table}"')).scalar():
                    problems.append(f"no context: {table} shows {n} rows")

        owned_tables = sorted(t for t, cols in tables.items() if "customer_id" in cols)
        graded = sorted(t for t, cols in tables.items() if "clearance" in cols)
        per_customer = " UNION ALL ".join(
            f"SELECT '{t}', count(*) FILTER (WHERE customer_id IS DISTINCT FROM :c), count(*) FROM \"{t}\""
            for t in owned_tables)
        above = " UNION ALL ".join(
            f"SELECT '{t}', count(*) FILTER (WHERE clearance > :clearance), count(*) FROM \"{t}\"" for t in graded)

        with engine.begin() as conn:
            set_access(conn, Access(org_id=BANK_ORG, clearance=2, role="viewer"))
            staff = conn.execute(sa.text("SELECT count(*) FROM chunks WHERE clearance = 2")).scalar()
        if not staff:
            problems.append("staff see no clearance-2 chunks: load the corpus first, or the "
                            "clearance check proves nothing")

        for cid in customer_ids:
            with engine.begin() as conn:
                set_access(conn, Access(org_id=BANK_ORG, clearance=0, customer_id=cid))
                for table, others, total in conn.execute(sa.text(per_customer), {"c": cid}):
                    if others:
                        problems.append(f"{cid}: sees {others} rows of other customers in {table}")
                    if table in ("customers", "accounts") and not total:
                        problems.append(f"{cid}: sees none of its own {table} (not loaded?)")
                for table, leaked, _ in conn.execute(sa.text(above), {"clearance": 0}):
                    if leaked:
                        problems.append(f"{cid}: sees {leaked} rows above clearance 0 in {table}")
                if n := conn.execute(sa.text("SELECT count(*) FROM chunks WHERE text LIKE :p"),
                                     {"p": CANARY}).scalar():
                    problems.append(f"{cid}: sees the staff canary in {n} chunks")
        print(f"checked {len(tables)} protected tables, {len(customer_ids)} customers, "
              f"{len(owned_tables)} customer-owned tables, {len(graded)} tables with clearance; "
              f"staff see {staff} clearance-2 chunks")
    finally:
        engine.dispose()
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                        help="finpilot_app's URL (default: DATABASE_URL)")
    parser.add_argument("--seed", type=int, default=7, help="bank.synthetic seed the data was loaded with")
    parser.add_argument("--customers", type=int, default=50)
    args = parser.parse_args()
    if not args.database_url:
        print("Pass --database-url or set DATABASE_URL (finpilot_app).", file=sys.stderr)
        return 2
    ids = [c["customer_id"] for c in generate(args.seed, args.customers)["customers"]]
    problems = check(args.database_url, ids)
    for p in problems:
        print("LEAK" if "sees" in p or "shows" in p else "FAIL", p)
    print("no leaks found" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
