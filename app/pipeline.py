from app.extractor import (
    extract_raw_text,
    parse_statement_period,
    detect_structure,
    parse_transactions_from_text,
)
from sqlalchemy.exc import IntegrityError

from app.llm import call_llm_batch
from app.embedder import build_embedded_text, generate_embedding
from app.models import SessionLocal, Statement, Transaction
from app.statement import DuplicateStatement, StatementData, StatementRejected, validate


# Rows in these sections get a fixed category at insert (ADR 0003, decision 2b),
# in the same insert, so an enrichment failure can never leave them as the
# column default "Other". PR2 makes enrich_transactions skip them; until then
# nothing calls persist_statement.
FIXED_CATEGORIES = {"fees": "Fees", "interest": "Interest"}

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

    db = SessionLocal()
    try:
        _raise_if_duplicate(db, data.period_start, data.period_end)

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
            )
            if row.section in FIXED_CATEGORIES:
                tx.category = FIXED_CATEGORIES[row.section]
            db.add(tx)

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

    db = SessionLocal()

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
    db = SessionLocal()

    try:
        db_transactions = db.query(Transaction).filter_by(statement_id=statement_id).all()

        tx_dicts = []
        for tx in db_transactions:
            tx_dicts.append({"id": tx.id, "raw_description": tx.raw_description})

        results, missed_ids = call_llm_batch(tx_dicts)

        result_map = {r["id"]: r for r in results}
        for tx in db_transactions:
            if tx.id in result_map:
                tx.merchant = result_map[tx.id]["merchant"]
                tx.category = result_map[tx.id]["category"]

        for tid in missed_ids:
            tx = next(t for t in db_transactions if t.id == tid)
            tx.needs_review = True

        for tx in db_transactions:
            tx.embedded_text = build_embedded_text(tx)
            tx.embedding     = generate_embedding(tx.embedded_text)

        db.commit()

        return {
            "enriched":     len(results),
            "needs_review": sum(1 for tx in db_transactions if tx.needs_review),
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
