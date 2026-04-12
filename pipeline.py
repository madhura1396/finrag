from extractor import (
    extract_raw_text,
    parse_statement_period,
    detect_structure,
    parse_transactions_from_text,
)
from llm import call_llm_batch, call_llm_single
from embedder import build_embedded_text, generate_embedding
from models import SessionLocal, Statement, Transaction


def extract_from_pdf(filepath: str, filename: str) -> dict:
    pages = extract_raw_text(filepath)
    period_start, period_end = parse_statement_period(pages)

    db = SessionLocal()

    try:
        existing = db.query(Statement).filter_by(
            period_start=period_start,
            period_end=period_end,
        ).first()

        if existing:
            raise ValueError(
                f"Statement for {period_start} → {period_end} already uploaded"
            )

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
            tx     = next(t for t in db_transactions if t.id == tid)
            result = call_llm_single({"id": tx.id, "raw_description": tx.raw_description})
            if result:
                tx.merchant = result["merchant"]
                tx.category = result["category"]
            else:
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
