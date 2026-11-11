import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import categorization, category_store
from app.models import CategoryRule, ChangeLog, LabelMemory, ReviewItem, Statement, Transaction, Trip
from app.pipeline import enrich_transactions, extract_from_pdf
from app.query_router import route
from app.sessions import get_session, new_session
from app.statement import REJECT_REASON, DuplicateStatement, StatementRejected

log = logging.getLogger("finrag")

# The label endpoints write real data and have no authentication (ADR 0005, 4).
# uvicorn's default bind is 127.0.0.1; this rejects anything else regardless.
LOOPBACK_HOSTS = {"127.0.0.1", "::1"}


def run_maintenance() -> None:
    """Retry pending labels, then refresh stale embeddings. Safe to run any time:
    both work from derived state, so a lost run is redone by the next one."""
    from app.embedder import generate_embedding

    db = new_session()
    try:
        categorization.enrich_pending(db)
        categorization.refresh_embeddings(db, generate_embedding)
    except Exception:
        log.exception("background maintenance failed; it will be retried")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Background tasks are lost on restart, so pending work is retried at startup.
    asyncio.get_running_loop().run_in_executor(None, run_maintenance)
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    host = request.client.host if request.client else None
    if host not in LOOPBACK_HOSTS:
        return JSONResponse(status_code=403, content={"detail": "this API only accepts requests from this machine"})
    return await call_next(request)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# A plain `def`: FastAPI runs it in a worker thread, so the labelling pass it
# waits for (ADR 0005, upload option A) never blocks other requests (issue #10).
@app.post("/upload")
def upload_statement(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Only the two outcomes we can name are mapped. Anything else is a bug and
    # stays a 500 rather than being dressed up as a conflict.
    try:
        result = extract_from_pdf(filepath, file.filename)
    except DuplicateStatement as e:
        raise HTTPException(status_code=409, detail=str(e))
    except StatementRejected as e:
        raise HTTPException(
            status_code=422,
            detail={
                "reason":     REJECT_REASON,
                "mismatches": [m.to_dict() for m in e.mismatches],
            },
        )

    # Upload waits for labelling (ADR 0005 fixes, option A); embeddings follow
    # in the background.
    try:
        enrich_result = enrich_transactions(result["statement_id"])
    except Exception:
        # The statement is already stored. Report its rows as pending rather than
        # a 500 that a retry would turn into a misleading 409; maintenance retries.
        log.exception("labelling failed after upload; rows stay pending and will be retried")
        enrich_result = _counts_after_failure(result)
    background.add_task(run_maintenance)

    return {
        "statement_id":       result["statement_id"],
        "period_start":       str(result["period_start"]),
        "period_end":         str(result["period_end"]),
        "total_transactions": result["total_transactions"],
        "enriched":           enrich_result["enriched"],
        "needs_review":       enrich_result["needs_review"],
        "pending":            enrich_result["pending"],
    }


def _counts_after_failure(result: dict) -> dict:
    """Actual counts if the database is reachable (fee and interest rows are
    already labelled); otherwise every row reported as pending, never fewer."""
    db = new_session()
    try:
        return {"enriched": 0, **categorization.label_counts(db, result["statement_id"])}
    except Exception:
        return {"enriched": 0, "needs_review": 0, "pending": result["total_transactions"]}
    finally:
        db.close()


@app.get("/statements")
def list_statements(db: Session = Depends(get_session)):
    statements = db.query(Statement).order_by(Statement.period_start.desc()).all()
    return [
        {
            "id":           s.id,
            "filename":     s.filename,
            "period_start": str(s.period_start),
            "period_end":   str(s.period_end),
            "uploaded_at":  str(s.uploaded_at),
        }
        for s in statements
    ]


@app.get("/statements/{statement_id}/transactions")
def list_transactions(statement_id: int, db: Session = Depends(get_session)):
    txs = db.query(Transaction).filter_by(statement_id=statement_id).order_by(Transaction.trans_date).all()
    if not txs:
        raise HTTPException(status_code=404, detail="No transactions found for this statement")
    names = category_store.names_by_id(db)
    return [
        {
            "id":              tx.id,
            "date":            str(tx.trans_date),
            "merchant":        tx.merchant,
            "merchant_source": tx.merchant_source,
            "amount":          str(tx.amount),
            # null while pending (ADR 0005)
            "category":        names.get(tx.category_id) if tx.category_id is not None else None,
            "category_source": tx.category_source,
            "is_credit":       tx.is_credit,
            "needs_review":    tx.needs_review,
            "raw_description": tx.raw_description,
        }
        for tx in txs
    ]


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = route(request.question)
    return {
        "answer": result["answer"],
        "type":   result["type"],
        "sql":    result.get("sql"),
        "rows":   result.get("rows", []),
    }


class TripRequest(BaseModel):
    name:       str
    start_date: str
    end_date:   str


@app.post("/trips")
def create_trip(request: TripRequest, db: Session = Depends(get_session)):
    from datetime import date
    trip = Trip(
        name=request.name,
        start_date=date.fromisoformat(request.start_date),
        end_date=date.fromisoformat(request.end_date),
    )
    db.add(trip)
    db.commit()
    return {"id": trip.id, "name": trip.name, "start_date": request.start_date, "end_date": request.end_date}


@app.get("/trips")
def list_trips(db: Session = Depends(get_session)):
    trips = db.query(Trip).order_by(Trip.start_date.desc()).all()
    return [
        {
            "id":         t.id,
            "name":       t.name,
            "start_date": str(t.start_date),
            "end_date":   str(t.end_date),
        }
        for t in trips
    ]


# --------------------------------------------------------------------------
# Categories, rules, memory, review and changes (ADR 0005)
# --------------------------------------------------------------------------

class ChangeRequest(BaseModel):
    kind:   Literal[categorization.CHANGE_KINDS]  # type: ignore[valid-type]
    params: dict[str, Any] = Field(default_factory=dict)


class ApplyRequest(ChangeRequest):
    token: str | None = None


def _payload(request: ChangeRequest) -> dict:
    return {"kind": request.kind, "params": request.params}


def _change_error(e: categorization.ChangeRejected) -> HTTPException:
    return HTTPException(status_code=422, detail={"reason": e.reason, "detail": e.detail})


@app.get("/categories")
def list_categories(db: Session = Depends(get_session)):
    return [
        {"category_id": c.category_id, "name": c.name, "archived": c.archived,
         "is_system": c.is_system, "system_key": c.system_key}
        for c in category_store.current_categories(db)
    ]


@app.get("/rules")
def list_rules(include_deleted: bool = False, db: Session = Depends(get_session)):
    query = select(CategoryRule).order_by(CategoryRule.priority.desc(), CategoryRule.rule_id)
    if not include_deleted:
        query = query.where(CategoryRule.deleted.is_(False))
    names = category_store.names_by_id(db)
    return [
        {"rule_id": r.rule_id, "pattern": r.pattern, "kind": r.kind, "priority": r.priority,
         "category_id": r.sets_category_id, "category": names.get(r.sets_category_id),
         "merchant": r.sets_merchant, "deleted": r.deleted}
        for r in db.scalars(query).all()
    ]


@app.get("/memory")
def list_memory(db: Session = Depends(get_session)):
    names = category_store.names_by_id(db)
    return [
        {"description_key": m.description_key, "category_id": m.category_id,
         "category": names.get(m.category_id), "category_source": m.category_source,
         "merchant": m.merchant, "merchant_source": m.merchant_source}
        for m in db.scalars(select(LabelMemory).order_by(LabelMemory.description_key)).all()
    ]


@app.get("/review")
def list_review(db: Session = Depends(get_session)):
    """Open review items grouped by description key, so one fix covers them all."""
    rows = db.execute(
        select(ReviewItem, Transaction)
        .join(Transaction, Transaction.id == ReviewItem.transaction_id)
        .where(ReviewItem.resolved_at.is_(None))
        .order_by(ReviewItem.review_id)
    ).all()
    groups: dict[tuple[str, str], dict] = {}
    for item, tx in rows:
        key = category_store.description_key(tx.raw_description)
        group = groups.setdefault((key, item.kind), {
            "description_key": key, "kind": item.kind, "detail": item.detail,
            "example_description": tx.raw_description, "transaction_ids": [],
        })
        group["transaction_ids"].append(tx.id)
    return list(groups.values())


@app.get("/changes")
def list_changes(limit: int = 50, db: Session = Depends(get_session)):
    changes = db.scalars(select(ChangeLog).order_by(ChangeLog.change_id.desc()).limit(min(limit, 500))).all()
    return [
        {"change_id": c.change_id, "kind": c.kind, "payload": c.payload, "before": c.before,
         "after": c.after, "undoes": c.undoes, "created_at": str(c.created_at)}
        for c in changes
    ]


@app.post("/changes/preview")
def preview_change(request: ChangeRequest, db: Session = Depends(get_session)):
    try:
        return categorization.preview(db, _payload(request))
    except categorization.ChangeRejected as e:
        raise _change_error(e)


@app.post("/changes/apply")
def apply_change(request: ApplyRequest, background: BackgroundTasks, db: Session = Depends(get_session)):
    try:
        result = categorization.apply(db, _payload(request), request.token)
    except categorization.ChangeRejected as e:
        raise _change_error(e)
    except categorization.StalePreview as e:
        raise HTTPException(status_code=409, detail=str(e))
    # Rows that fell to pending get the model, and relabelled rows get re-embedded.
    background.add_task(run_maintenance)
    return result


@app.post("/enrichment/retry")
def retry_enrichment(db: Session = Depends(get_session)):
    return categorization.enrich_pending(db)


@app.post("/embeddings/refresh")
def refresh_embeddings(db: Session = Depends(get_session)):
    from app.embedder import generate_embedding
    return {"refreshed": categorization.refresh_embeddings(db, generate_embedding)}
