import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import extract_from_pdf, enrich_transactions
from query_router import route
from models import SessionLocal, Statement, Transaction, Trip

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_statement(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = extract_from_pdf(filepath, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    enrich_result = enrich_transactions(result["statement_id"])

    return {
        "statement_id":       result["statement_id"],
        "period_start":       str(result["period_start"]),
        "period_end":         str(result["period_end"]),
        "total_transactions": result["total_transactions"],
        "enriched":           enrich_result["enriched"],
        "needs_review":       enrich_result["needs_review"],
    }


@app.get("/statements")
def list_statements():
    db = SessionLocal()
    try:
        statements = db.query(Statement).order_by(Statement.period_start.desc()).all()
        return [
            {
                "id":          s.id,
                "filename":    s.filename,
                "period_start": str(s.period_start),
                "period_end":  str(s.period_end),
                "uploaded_at": str(s.uploaded_at),
            }
            for s in statements
        ]
    finally:
        db.close()


@app.get("/statements/{statement_id}/transactions")
def list_transactions(statement_id: int):
    db = SessionLocal()
    try:
        txs = db.query(Transaction).filter_by(statement_id=statement_id).order_by(Transaction.trans_date).all()
        if not txs:
            raise HTTPException(status_code=404, detail="No transactions found for this statement")
        return [
            {
                "id":              tx.id,
                "date":            str(tx.trans_date),
                "merchant":        tx.merchant,
                "amount":          str(tx.amount),
                "category":        tx.category,
                "is_credit":       tx.is_credit,
                "needs_review":    tx.needs_review,
                "raw_description": tx.raw_description,
            }
            for tx in txs
        ]
    finally:
        db.close()


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
def create_trip(request: TripRequest):
    from datetime import date
    db = SessionLocal()
    try:
        trip = Trip(
            name=request.name,
            start_date=date.fromisoformat(request.start_date),
            end_date=date.fromisoformat(request.end_date),
        )
        db.add(trip)
        db.commit()
        return {"id": trip.id, "name": trip.name, "start_date": request.start_date, "end_date": request.end_date}
    finally:
        db.close()


@app.get("/trips")
def list_trips():
    db = SessionLocal()
    try:
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
    finally:
        db.close()
