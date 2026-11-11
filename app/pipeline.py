from app.extractor import (
    extract_raw_text,
    parse_statement_period,
    detect_structure,
    parse_transactions_from_text,
)
from sqlalchemy.exc import IntegrityError

from app import categorization, category_store
from app.categories import SECTION_CATEGORY_KEYS
from app.models import Statement, Transaction
from app.sessions import new_session
from app.statement import DuplicateStatement, StatementData, StatementRejected, validate

PERIOD_CONSTRAINT = "uq_statement_period"


def _raise_if_duplicate(db, period_start, period_end) -> None:
    existing = db.query(Statement).filter_by(
        period_start=period_start,
        period_end=period_end,
    ).first()
    if existing:
        raise DuplicateStatement(
            f"Statement for {period_start} → {period_end} already uploaded"
        )


def _is_period_conflict(error: IntegrityError) -> bool:
    diag = getattr(error.orig, "diag", None)
    return getattr(diag, "constraint_name", None) == PERIOD_CONSTRAINT


def persist_statement(data: StatementData, filename: str) -> dict:
    """Validate a parsed statement and store it, or store nothing.

    Validation runs before any database work, so a rejected statement leaves no
    rows and can be uploaded again. The statement and its rows are inserted in
    one transaction.
    """
    mismatches = validate(data)
    if mismatches:
        raise StatementRejected(mismatches)

    db = new_session()
    try:
        # Uploads bump data_version, so a preview taken before this upload is stale.
        categorization.lock_version(db)
        _raise_if_duplicate(db, data.period_start, data.period_end)
        section_ids = {section: category_store.system_category_id(db, key)
                       for section, key in SECTION_CATEGORY_KEYS.items()}

        statement = Statement(
            filename=filename,
            period_start=data.period_start,
            period_end=data.period_end,
        )
        db.add(statement)
        db.flush()
        # Read the id now: after commit() the object is expired, and reloading it
        # could fail and report a 500 for a statement that was in fact stored.
        statement_id = statement.id

        for row in data.rows:
            tx = Transaction(
                statement_id=statement_id,
                trans_date=row.trans_date,
                raw_description=row.description,
                merchant=row.description,
                amount=row.amount,
                # Derived from the sign, never set independently (ADR 0003, 5a).
                # Kept because query_router's SQL prompt filters on 'true'/'false'.
                # A 0.00 row (printed zero interest) counts as a charge.
                is_credit="true" if row.amount > 0 else "false",
                section=row.section,
                # Fee and interest rows are labelled by their section (ADR 0003,
                # 2b; ADR 0005). Everything else starts pending for enrichment.
                category_id=section_ids.get(row.section),
                category_source="section" if row.section in section_ids else "pending",
                merchant_source="section" if row.section in section_ids else "pending",
            )
            db.add(tx)
            db.flush()
            categorization.open_history(db, tx, None)

        categorization.bump_version(db)
        db.commit()

        return {
            "statement_id":       statement_id,
            "period_start":       data.period_start,
            "period_end":         data.period_end,
            "total_transactions": len(data.rows),
        }

    except IntegrityError as e:
        db.rollback()
        # Two uploads of the same period racing past the check above: the
        # unique constraint is the real guard, so report it as the duplicate it is.
        if _is_period_conflict(e):
            raise DuplicateStatement(
                f"Statement for {data.period_start} → {data.period_end} already uploaded"
            ) from e
        raise

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def extract_from_pdf(filepath: str, filename: str) -> dict:
    pages = extract_raw_text(filepath)
    period_start, period_end = parse_statement_period(pages)

    db = new_session()

    try:
        _raise_if_duplicate(db, period_start, period_end)

        structure        = detect_structure(pages)
        raw_transactions = parse_transactions_from_text(pages, structure, period_start, period_end)

        statement = Statement(
            filename=filename,
            period_start=period_start,
            period_end=period_end,
        )
        db.add(statement)
        db.flush()

        db_transactions = []
        for t in raw_transactions:
            tx = Transaction(
                statement_id=statement.id,
                trans_date=t["trans_date"],
                raw_description=t["raw_description"],
                merchant=t["raw_description"],
                amount=t["amount"],
                is_credit=str(t["is_credit"]).lower(),
            )
            db.add(tx)
            db_transactions.append(tx)

        db.commit()

        return {
            "statement_id":       statement.id,
            "period_start":       period_start,
            "period_end":         period_end,
            "total_transactions": len(db_transactions),
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def enrich_transactions(statement_id: int) -> dict:
    """Label the statement's pending rows before /upload replies (ADR 0005 fixes,
    upload option A). Model failures stay pending and are retried later."""
    db = new_session()
    try:
        return categorization.enrich_pending(db, statement_id=statement_id)
    finally:
        db.close()
