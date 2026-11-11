import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import (
    JSON, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey,
    Index, Integer, Numeric, String, Text, UniqueConstraint, create_engine, func, text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector

from app.db_guard import install_guard

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
# Every connection checks the database's environment marker (app/db_guard.py).
install_guard(engine)

SessionLocal = sessionmaker(bind=engine)

# Parent class for all table models. SQLAlchemy uses this to track
# which Python classes represent real database tables.
Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


CATEGORY_SOURCES = ("section", "pin", "rule", "memory", "model", "none_fit", "pending")
MERCHANT_SOURCES = ("section", "pin", "rule", "memory", "model", "pending")


def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class DbEnvironment(Base):
    """One row saying what this database is: production or test (app/db_guard.py)."""
    __tablename__ = "db_environment"

    id = Column(Integer, primary_key=True)
    environment = Column(String, nullable=False)

    __table_args__ = (
        CheckConstraint("id = 1", name="ck_db_environment_single_row"),
        CheckConstraint("environment IN ('production', 'test')", name="ck_db_environment_value"),
    )


class DataVersion(Base):
    """One row, bumped by every logged change, upload and enrichment write.
    Preview tokens carry it; applies lock it with SELECT ... FOR UPDATE (ADR 0005, 4)."""
    __tablename__ = "data_version"

    id = Column(Integer, primary_key=True)
    version = Column(BigInteger, nullable=False, default=0)

    __table_args__ = (CheckConstraint("id = 1", name="ck_data_version_single_row"),)


class Statement(Base):
    """One uploaded PDF statement."""
    __tablename__ = "statements"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    # Statement period pulled from the PDF header
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    uploaded_at = Column(DateTime, default=_now)

    # Enforce that the same statement period can never be inserted twice.
    # This is the database-level duplicate upload guard.
    __table_args__ = (
        UniqueConstraint("period_start", "period_end", name="uq_statement_period"),
    )

    transactions = relationship("Transaction", back_populates="statement")


class CategoryId(Base):
    """Durable identity of a category: the FK target for every category reference.
    Names live in CategoryVersion (SCD2), so a rename never touches references."""
    __tablename__ = "category_ids"

    category_id = Column(Integer, primary_key=True)
    is_system = Column(Boolean, nullable=False, default=False)
    # Stable handle for system categories ('fees', 'interest', 'other'); null for user ones.
    system_key = Column(String, nullable=True, unique=True)
    created_at = Column(DateTime, nullable=False, default=_now)


class CategoryVersion(Base):
    """SCD2 names of a category. The current version has valid_to IS NULL."""
    __tablename__ = "category_versions"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("category_ids.category_id"), nullable=False)
    name = Column(String, nullable=False)
    archived = Column(Boolean, nullable=False, default=False)
    valid_from = Column(DateTime, nullable=False, default=_now)
    valid_to = Column(DateTime, nullable=True)
    change_id = Column(Integer, ForeignKey("change_log.change_id"), nullable=True)

    __table_args__ = (
        # One open version per category.
        Index("uq_category_open_version", "category_id", unique=True,
              postgresql_where=text("valid_to IS NULL")),
        # Current, non-archived names are unique ignoring case; archived names are free.
        Index("uq_category_current_name", func.lower(name), unique=True,
              postgresql_where=text("valid_to IS NULL AND NOT archived")),
    )


class CategoryRule(Base):
    """A user rule over the printed description (ADR 0005, decision 2). Soft-deleted."""
    __tablename__ = "category_rules"

    rule_id = Column(Integer, primary_key=True)
    pattern = Column(Text, nullable=False)
    kind = Column(String, nullable=False)
    priority = Column(Integer, nullable=False)
    sets_category_id = Column(Integer, ForeignKey("category_ids.category_id"), nullable=True)
    sets_merchant = Column(String, nullable=True)
    deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now)

    __table_args__ = (
        CheckConstraint("kind IN ('contains', 'regex')", name="ck_rule_kind"),
        CheckConstraint("sets_category_id IS NOT NULL OR sets_merchant IS NOT NULL",
                        name="ck_rule_has_target"),
        CheckConstraint("length(pattern) BETWEEN 1 AND 200", name="ck_rule_pattern_length"),
    )


class LabelMemory(Base):
    """The remembered label per normalised description key, per field (decision 3b).
    Never holds a none_fit or failed model result."""
    __tablename__ = "label_memory"

    description_key = Column(String, primary_key=True)
    category_id = Column(Integer, ForeignKey("category_ids.category_id"), nullable=True)
    category_source = Column(String, nullable=True)   # 'model' or 'user'
    merchant = Column(String, nullable=True)
    merchant_source = Column(String, nullable=True)   # 'model' or 'user'
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now)

    __table_args__ = (
        CheckConstraint("category_source IS NULL OR category_source IN ('model', 'user')",
                        name="ck_memory_category_source"),
        CheckConstraint("merchant_source IS NULL OR merchant_source IN ('model', 'user')",
                        name="ck_memory_merchant_source"),
        CheckConstraint("(category_id IS NULL) = (category_source IS NULL)",
                        name="ck_memory_category_pair"),
        CheckConstraint("(merchant IS NULL) = (merchant_source IS NULL)",
                        name="ck_memory_merchant_pair"),
    )


class ChangeLog(Base):
    """Every logged change: rules, categories, memory, pins, undo (decision 5)."""
    __tablename__ = "change_log"

    change_id = Column(Integer, primary_key=True)
    kind = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    before = Column(JSON, nullable=False)
    after = Column(JSON, nullable=False)
    # What the change touched, for the reverse-order undo check.
    touched = Column(JSON, nullable=False)
    undoes = Column(Integer, ForeignKey("change_log.change_id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)


class Transaction(Base):
    """One row from a statement section."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    statement_id = Column(Integer, ForeignKey("statements.id"), nullable=False)

    # Parsed directly from the PDF row
    trans_date = Column(Date, nullable=False)
    raw_description = Column(Text, nullable=False)  # original full text from PDF
    # Numeric(10,2) = up to 10 digits total, 2 decimal places. Right type for money.
    # Never use Float for money.
    amount = Column(Numeric(10, 2), nullable=False)
    # Signed (ADR 0003, 8): positive = payment, refund or credit; negative =
    # charge, fee or interest. is_credit is derived from the sign; nothing on the
    # read path reads it any more (ADR 0004).
    is_credit = Column(String, nullable=False, default="false")

    # Statement section the row was printed under, one of app.statement.SECTIONS.
    section = Column(String, nullable=False)

    # Labels (ADR 0005). Each field records where its value came from; a field
    # is pinned exactly when its source is 'pin'.
    category_id = Column(Integer, ForeignKey("category_ids.category_id"), nullable=True)
    category_source = Column(String, nullable=False, default="pending")
    category_rule_id = Column(Integer, ForeignKey("category_rules.rule_id"), nullable=True)
    merchant = Column(String, nullable=False)       # cleaned name; the description until labelled
    merchant_source = Column(String, nullable=False, default="pending")
    merchant_rule_id = Column(Integer, ForeignKey("category_rules.rule_id"), nullable=True)

    # Set when a label could not be decided: none_fit, a rule conflict, or a
    # model failure. Details live in review_items.
    needs_review = Column(Boolean, nullable=False, default=False)

    # The text we embed for semantic search (app/embedding_text.py)
    embedded_text = Column(Text, nullable=True)

    # 384-dimensional vector from all-MiniLM-L6-v2
    embedding = Column(Vector(384), nullable=True)

    created_at = Column(DateTime, default=_now)

    statement = relationship("Statement", back_populates="transactions")

    __table_args__ = (
        CheckConstraint(_in("category_source", CATEGORY_SOURCES), name="ck_tx_category_source"),
        CheckConstraint(_in("merchant_source", MERCHANT_SOURCES), name="ck_tx_merchant_source"),
        # A category is missing exactly while it is pending.
        CheckConstraint("(category_id IS NULL) = (category_source = 'pending')",
                        name="ck_tx_category_pending"),
        CheckConstraint("(category_rule_id IS NOT NULL) = (category_source = 'rule')",
                        name="ck_tx_category_rule"),
        CheckConstraint("(merchant_rule_id IS NOT NULL) = (merchant_source = 'rule')",
                        name="ck_tx_merchant_rule"),
    )


class TransactionLabelHistory(Base):
    """SCD2 history of a transaction's labels. Mirrors the label columns on
    transactions exactly; the open version (valid_to IS NULL) equals them.
    Append-only: closed versions are never reopened or edited."""
    __tablename__ = "transaction_label_history"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("category_ids.category_id"), nullable=True)
    category_source = Column(String, nullable=False)
    category_rule_id = Column(Integer, ForeignKey("category_rules.rule_id"), nullable=True)
    merchant = Column(String, nullable=False)
    merchant_source = Column(String, nullable=False)
    merchant_rule_id = Column(Integer, ForeignKey("category_rules.rule_id"), nullable=True)
    valid_from = Column(DateTime, nullable=False, default=_now)
    valid_to = Column(DateTime, nullable=True)
    change_id = Column(Integer, ForeignKey("change_log.change_id"), nullable=True)

    __table_args__ = (
        Index("uq_label_history_open_version", "transaction_id", unique=True,
              postgresql_where=text("valid_to IS NULL")),
    )


class ReviewItem(Base):
    """Rows needing the user's attention: none_fit, a rule conflict, or pending."""
    __tablename__ = "review_items"

    review_id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    kind = Column(String, nullable=False)
    detail = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=_now)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by_change_id = Column(Integer, ForeignKey("change_log.change_id"), nullable=True)

    __table_args__ = (
        CheckConstraint("kind IN ('none_fit', 'conflict', 'pending')", name="ck_review_kind"),
    )


class Trip(Base):
    """A named trip defined by the user with a date range."""
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    # User-defined name e.g. "Hawaii Trip", "Vermont Trip"
    name = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=_now)
