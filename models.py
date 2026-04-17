import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Text, DateTime, Date, Numeric, Boolean, ForeignKey, UniqueConstraint, create_engine, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine)

# Parent class for all table models. SQLAlchemy uses this to track
# which Python classes represent real database tables.
Base = declarative_base()


class Statement(Base):
    """One uploaded Wells Fargo PDF statement."""
    __tablename__ = "statements"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    # Statement period pulled from the PDF header e.g. "12/03/2025 to 01/02/2026"
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Enforce that the same statement period can never be inserted twice.
    # This is the database-level duplicate upload guard.
    # Even if Python code has a bug, PostgreSQL will reject the second insert.
    __table_args__ = (
        UniqueConstraint("period_start", "period_end", name="uq_statement_period"),
    )

    # One statement contains many transactions
    transactions = relationship("Transaction", back_populates="statement")


class Transaction(Base):
    """One row from the Purchases or Credits section of a statement."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statements.id"), nullable=False)

    # Parsed directly from the PDF row
    trans_date = Column(Date, nullable=False)
    merchant = Column(String, nullable=False)       # cleaned name e.g. "TRADER JOE S #534"
    raw_description = Column(Text, nullable=False)  # original full text from PDF
    # Numeric(10,2) = up to 10 digits total, 2 decimal places. Right type for money.
    # Never use Float for money — floating point errors cause $19.999999 instead of $20.00
    amount = Column(Numeric(10, 2), nullable=False)
    # Negative amount = credit/refund, positive = charge
    is_credit = Column(String, nullable=False, default="false")

    # Auto-assigned by LLM on upload
    category = Column(String, nullable=False, default="Other")

    # True if LLM could not confidently clean merchant or assign category.
    # Set when: batch LLM call missed this transaction, or retry also failed.
    # These appear in a review queue in the frontend so you can fix them manually.
    needs_review = Column(Boolean, default=False)

    # The human-readable sentence we embed for semantic search
    # e.g. "On 12/06, spent $15.98 at TRADER JOE S in Groceries category"
    embedded_text = Column(Text, nullable=True)

    # 384-dimensional vector from all-MiniLM-L6-v2
    # Used for semantic queries like "trip spending" or "hidden patterns"
    embedding = Column(Vector(384), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Navigate back to parent statement
    statement = relationship("Statement", back_populates="transactions")


class Trip(Base):
    """A named trip defined by the user with a date range."""
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    # User-defined name e.g. "Hawaii Trip", "Vermont Trip"
    name = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_tables():
    # Load pgvector extension first — must exist before Vector(384) columns are created
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables defined above if they don't already exist
    # IF NOT EXISTS is built in — safe to call on every app startup
    Base.metadata.create_all(bind=engine)
