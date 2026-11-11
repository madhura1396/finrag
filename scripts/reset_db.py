"""Drop every table and recreate the schema, marked and seeded (ADR 0005).

    python -m scripts.reset_db --env production|test --yes --confirm-name <db> [--allow-production] [--database-url URL]

Guards, all checked before anything is dropped:
- --yes, and --confirm-name must equal the target database's name;
- --env is required, so a database is never marked `production` by default;
- the target's environment marker: a database marked `production`, or an
  unmarked one that already holds any data (statements, trips, rules or
  label memory), needs --allow-production;
- the target is explicit: --database-url, else the app's DATABASE_URL loaded by
  this process from .env. Only host and database name are printed, never the
  password.

Uploaded PDFs in uploads/ are never touched.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.db_setup import create_tables, drop_tables


def describe(url) -> str:
    return f"{url.database} on {url.host or 'localhost'}:{url.port or 5432}"


DATA_TABLES = ("statements", "trips", "category_rules", "label_memory")


def inspect_target(engine) -> tuple[str | None, int]:
    """(environment marker or None, rows in tables that hold user data)."""
    with engine.connect() as conn:
        has_marker = conn.execute(text("SELECT to_regclass('public.db_environment') IS NOT NULL")).scalar()
        marker = conn.execute(text("SELECT environment FROM db_environment WHERE id = 1")).scalar() if has_marker else None
        rows = 0
        for table in DATA_TABLES:
            if conn.execute(text(f"SELECT to_regclass('public.{table}') IS NOT NULL")).scalar():
                rows += conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return marker, rows


def refusal(args, url, marker, statements) -> str | None:
    if not args.yes:
        return "refusing without --yes"
    if args.confirm_name != url.database:
        return f"--confirm-name must be exactly the target database name ({url.database!r})"
    if marker is not None and marker != args.env:
        return f"target is marked {marker!r}; refusing to re-create it as {args.env!r}"
    if (marker == "production" or (marker is None and statements)) and not args.allow_production:
        return (f"target holds data ({'marked production' if marker else f'{statements} rows of user data, unmarked'}); "
                "add --allow-production only if you mean to destroy it")
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", help="target; defaults to DATABASE_URL from .env")
    parser.add_argument("--env", choices=("production", "test"), required=True,
                        help="environment marker to write; required so nothing is marked production by default")
    parser.add_argument("--confirm-name", help="must equal the target database name")
    parser.add_argument("--allow-production", action="store_true",
                        help="allow resetting a database that holds or is marked as real data")
    parser.add_argument("--yes", action="store_true", help="required")
    args = parser.parse_args(argv)

    load_dotenv()
    raw_url = args.database_url or os.getenv("DATABASE_URL")
    if not raw_url:
        print("no target: pass --database-url or set DATABASE_URL", file=sys.stderr)
        return 2
    url = make_url(raw_url)

    engine = create_engine(url)      # unguarded on purpose: this writes the marker
    try:
        marker, statements = inspect_target(engine)
        print(f"target: {describe(url)} | marker: {marker or 'none'} | rows of user data: {statements}")
        reason = refusal(args, url, marker, statements)
        if reason:
            print(f"refused: {reason}", file=sys.stderr)
            return 1
        drop_tables(engine)
        with engine.begin() as conn:
            # Tables from before ADR 0005 that the current models no longer define.
            conn.execute(text("DROP TABLE IF EXISTS statements, transactions, trips CASCADE"))
        create_tables(engine, args.env)
        print(f"reset complete: {describe(url)} marked {args.env!r} and seeded")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
