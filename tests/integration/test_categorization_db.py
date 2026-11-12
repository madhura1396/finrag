"""ADR 0005 against a real Postgres (finrag_test). Invented descriptors only."""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app import categorization as cz
from app import category_store, pipeline
from app.llm import NONE_FIT, ModelLabel
from app.models import (
    CategoryId, CategoryVersion, ChangeLog, LabelMemory, ReviewItem, Transaction, TransactionLabelHistory,
)
from app.statement import Row, StatementData, SummaryLine

pytestmark = pytest.mark.integration

START, END = date(2025, 12, 3), date(2026, 1, 2)
GROCER = "GROCER MART #101 SPRINGFIELD"
GROCER_2 = "GROCER MART #202 SPRINGFIELD"
CAFE = "CORNER CAFE 4455 RIVERTON"
INTEREST = "INTEREST CHARGE ON PURCHASES"
GROCER_KEY, CAFE_KEY = "GROCER MART SPRINGFIELD", "CORNER CAFE RIVERTON"


def statement(descriptions: list[str], start=START, end=END) -> StatementData:
    """A reconciling statement: each purchase is $10.00, plus one interest row."""
    rows = [Row(start + timedelta(days=2), None, d, Decimal("-10.00"), "purchases") for d in descriptions]
    rows.append(Row(end, None, INTEREST, Decimal("-1.00"), "interest"))
    purchases = Decimal("10.00") * len(descriptions)
    zero = Decimal("0.00")
    totals = {"payments": zero, "other_credits": zero, "purchases": purchases, "fees": zero, "interest": Decimal("1.00")}
    summary = {
        "previous_balance": SummaryLine(zero, 1, None),
        "payments":         SummaryLine(zero, -1, "payments"),
        "other_credits":    SummaryLine(zero, -1, "other_credits"),
        "cash_advances":    SummaryLine(zero, 1, "cash_advances"),
        "purchases":        SummaryLine(purchases, 1, "purchases"),
        "fees":             SummaryLine(zero, 1, "fees"),
        "interest":         SummaryLine(Decimal("1.00"), 1, "interest"),
        "new_balance":      SummaryLine(purchases + Decimal("1.00"), 0, None),
    }
    return StatementData(start, end, rows, totals, summary)


def second_statement(*descriptions):
    pipeline.persist_statement(statement(list(descriptions), date(2026, 1, 3), date(2026, 2, 2)), "second.pdf")


def labeller_from(answers: dict[str, ModelLabel | None], calls: list | None = None):
    def labeller(descriptions, category_names):
        if calls is not None:
            calls.append(dict(descriptions))
        return {key: answers.get(key) for key in descriptions}
    return labeller


def cat_id(db, name):
    return next(c.category_id for c in category_store.current_categories(db) if c.name == name)


def system(db, key):
    return category_store.system_category_id(db, key)


def tx_by_desc(db, description):
    return db.scalar(select(Transaction).where(Transaction.raw_description == description))


def name_of(db, description):
    """The row's current category name (None while pending)."""
    tx = tx_by_desc(db, description)
    return category_store.names_by_id(db).get(tx.category_id) if tx.category_id is not None else None


def count(db, table):
    return db.scalar(text(f"SELECT COUNT(*) FROM {table}"))


def assert_history_matches(db):
    """The open history version equals the current labels, for every row."""
    for tx in db.scalars(select(Transaction)).all():
        open_versions = db.scalars(select(TransactionLabelHistory).where(
            TransactionLabelHistory.transaction_id == tx.id,
            TransactionLabelHistory.valid_to.is_(None))).all()
        assert len(open_versions) == 1
        v = open_versions[0]
        assert (v.category_id, v.category_source, v.category_rule_id, v.merchant, v.merchant_source,
                v.merchant_rule_id) == tuple(vars(cz.labels_of(tx)).values())


def change(db, change_kind, **params):
    payload = {"kind": change_kind, "params": params}
    token = cz.preview(db, payload)["token"]
    return cz.apply(db, payload, token)


def rule(db, pattern, category=None, priority=1, **extra):
    params = {"pattern": pattern, "kind": "contains", "priority": priority, **extra}
    if category:
        params["category_id"] = cat_id(db, category)
    return change(db, "create_rule", **params)


def rule_id_of(db, result):
    return db.scalar(select(ChangeLog.after).where(ChangeLog.change_id == result["change_id"]))["rule_id"]


def pin(db, description, field, value):
    value = cat_id(db, value) if field == "category" else value
    return cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx_by_desc(db, description).id,
                                                   "field": field, "value": value}}, None)


def undo(db, result):
    return change(db, "undo", change_id=result["change_id"])


def undo_blockers(db, result):
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "undo", "params": {"change_id": result["change_id"]}})
    return e.value.detail["blocking_change_ids"]


@pytest.fixture
def uploaded(db):
    pipeline.persist_statement(statement([GROCER, GROCER_2, CAFE]), "synthetic.pdf")
    return db


@pytest.fixture
def labelled(uploaded):
    answers = {GROCER_KEY: ModelLabel("Groceries", "Grocer Mart", None),
               CAFE_KEY: ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    cz.enrich_pending(uploaded, labeller=labeller_from(answers))
    return uploaded


# --- seed and constraints ---------------------------------------------------

def test_seed_creates_system_defaults_and_never_resurrects_a_renamed_default(db):
    from app.db_setup import seed
    cats = category_store.current_categories(db)
    assert {c.system_key for c in cats if c.is_system} == {"fees", "interest", "other"}
    assert cz.current_version(db) is not None
    change(db, "rename_category", category_id=cat_id(db, "Groceries"), name="Food at home")
    seed(db)
    db.flush()
    names = {c.name for c in category_store.current_categories(db)}
    assert "Groceries" not in names and "Food at home" in names


def _second_open_version(db):
    db.add(CategoryVersion(category_id=cat_id(db, "Groceries"), name="Dup"))


def _same_name_ignoring_case(db):
    identity = CategoryId(is_system=False)
    db.add(identity)
    db.flush()
    db.add(CategoryVersion(category_id=identity.category_id, name="groceries"))


def _second_open_history(db):
    tx = tx_by_desc(db, GROCER)
    db.add(TransactionLabelHistory(transaction_id=tx.id, category_source="pending",
                                   merchant=tx.merchant, merchant_source="pending"))


def _duplicate_memory_key(db):
    db.add(LabelMemory(description_key="SAME KEY", merchant="A", merchant_source="user"))
    db.flush()
    db.add(LabelMemory(description_key="SAME KEY", merchant="B", merchant_source="user"))


@pytest.mark.parametrize("break_it", [_second_open_version, _same_name_ignoring_case, _second_open_history,
                                      _duplicate_memory_key])
def test_database_constraints(uploaded, break_it):
    break_it(uploaded)
    with pytest.raises(IntegrityError):
        uploaded.flush()


def test_database_rejects_null_category_unless_pending(uploaded):
    uploaded.execute(text("SAVEPOINT s"))
    with pytest.raises(IntegrityError):
        uploaded.execute(text("UPDATE transactions SET category_source = 'model' WHERE id = :i"),
                         {"i": tx_by_desc(uploaded, GROCER).id})
    uploaded.execute(text("ROLLBACK TO SAVEPOINT s"))


# --- ingest and enrichment --------------------------------------------------

def test_upload_labels_sections_and_leaves_the_rest_pending(uploaded):
    interest = tx_by_desc(uploaded, INTEREST)
    assert (interest.category_source, interest.category_id) == ("section", system(uploaded, "interest"))
    assert tx_by_desc(uploaded, GROCER).category_source == "pending"
    assert_history_matches(uploaded)


def test_enrichment_uses_one_model_call_per_key_and_remembers(uploaded):
    calls = []
    answers = {GROCER_KEY: ModelLabel("Groceries", "Grocer Mart", None),
               CAFE_KEY: ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    result = cz.enrich_pending(uploaded, labeller=labeller_from(answers, calls))

    assert len(calls) == 1 and set(calls[0]) == set(answers)       # #101 and #202 share a key
    for desc in (GROCER, GROCER_2):
        tx = tx_by_desc(uploaded, desc)
        assert (tx.category_source, tx.merchant, tx.merchant_source) == ("model", "Grocer Mart", "model")
        assert name_of(uploaded, desc) == "Groceries"
    memory = uploaded.get(LabelMemory, GROCER_KEY)
    assert memory.category_source == "model" and memory.merchant == "Grocer Mart"
    assert result["pending"] == 0
    assert_history_matches(uploaded)


def test_none_fit_goes_to_other_with_review_and_is_not_remembered(uploaded):
    answers = {GROCER_KEY: ModelLabel(NONE_FIT, "Grocer Mart", "Bulk Food"),
               CAFE_KEY: ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    cz.enrich_pending(uploaded, labeller=labeller_from(answers))
    tx = tx_by_desc(uploaded, GROCER)
    assert (tx.category_source, tx.category_id, tx.needs_review) == ("none_fit", system(uploaded, "other"), True)
    item = uploaded.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "none_fit"))
    assert item.detail == {"suggestion": "Bulk Food"}
    memory = uploaded.get(LabelMemory, GROCER_KEY)
    assert (memory.category_id, memory.merchant) == (None, "Grocer Mart")    # merchant remembered, none_fit never


def test_model_failure_leaves_rows_pending_and_flagged(uploaded):
    result = cz.enrich_pending(uploaded, labeller=labeller_from({}))
    tx = tx_by_desc(uploaded, CAFE)
    assert (tx.category_id, tx.category_source, tx.needs_review, result["pending"]) == (None, "pending", True, 3)
    assert uploaded.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "pending"))
    assert uploaded.get(LabelMemory, CAFE_KEY) is None


def test_enrichment_never_overwrites_a_pin_set_while_the_model_ran(uploaded):
    cafe_id, shopping = tx_by_desc(uploaded, CAFE).id, cat_id(uploaded, "Shopping")

    def labeller(descriptions, names):
        from app.models import new_session
        other = new_session()
        cz.apply(other, {"kind": "pin", "params": {"transaction_id": cafe_id, "field": "category",
                                                    "value": shopping}}, None)
        other.close()
        return {k: ModelLabel("Groceries", "X", None) for k in descriptions}

    cz.enrich_pending(uploaded, labeller=labeller)
    uploaded.expire_all()
    cafe = uploaded.get(Transaction, cafe_id)
    assert (cafe.category_id, cafe.category_source) == (shopping, "pin")


def test_enrichment_with_nothing_to_do_does_not_invalidate_previews(db, labelled):
    version = cz.current_version(db)
    cz.enrich_pending(db, labeller=labeller_from({}))
    assert cz.current_version(db) == version


# --- rules, precedence, tokens --------------------------------------------

def test_rule_relabels_matching_rows_and_never_touches_section_or_pins(db, labelled):
    pin(db, GROCER_2, "category", "Entertainment")
    result = rule(db, "GROCER MART", "Shopping", priority=10)
    assert result["effect"]["rows_changed"] == 1
    assert [tx_by_desc(db, d).category_source for d in (GROCER, GROCER_2, INTEREST)] == ["rule", "pin", "section"]
    assert_history_matches(db)


def test_merchant_pin_does_not_freeze_category(db, labelled):
    pin(db, GROCER, "merchant", "My Grocer")
    rule(db, "GROCER", "Shopping")
    tx = tx_by_desc(db, GROCER)
    assert (tx.merchant, tx.merchant_source, tx.category_source) == ("My Grocer", "pin", "rule")


def test_equal_priority_conflict_is_refused_at_creation(db, labelled):
    rule(db, "GROCER", "Shopping", priority=5)
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "create_rule", "params": {"pattern": "SPRINGFIELD", "kind": "contains",
                                                         "priority": 5, "category_id": cat_id(db, "Entertainment")}})
    assert e.value.detail["total"] >= 1


def test_equal_priority_conflict_at_ingest_is_flagged_not_decided(db, labelled):
    rule(db, "RIVERTON", "Shopping", priority=5)
    rule(db, "ZEBRA", "Entertainment", priority=5)
    # Test harness: sessions share one connection, so release this session's open
    # savepoint before another session commits, or a later rollback here would
    # discard that commit. Real sessions each have their own connection.
    db.commit()
    second_statement("ZEBRA SHOP RIVERTON")
    cz.enrich_pending(db, labeller=labeller_from({}))
    tx = tx_by_desc(db, "ZEBRA SHOP RIVERTON")
    assert tx.category_source == "pending" and tx.needs_review
    assert db.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "conflict"))


def test_preview_token_goes_stale_after_an_upload(db, labelled):
    payload = {"kind": "create_rule", "params": {"pattern": "CAFE", "kind": "contains", "priority": 1,
                                                 "category_id": cat_id(db, "Shopping")}}
    token = cz.preview(db, payload)["token"]
    second_statement("NEW PLACE")
    with pytest.raises(cz.StalePreview):
        cz.apply(db, payload, token)


def test_preview_leaves_no_trace_anywhere(db, labelled):
    tables = ("transaction_label_history", "review_items", "category_rules", "category_versions",
              "label_memory", "change_log")
    counts = lambda: {t: count(db, t) for t in tables}
    rows = lambda: [(t.needs_review, cz.labels_of(t)) for t in db.scalars(select(Transaction).order_by(Transaction.id))]
    before, before_rows, version = counts(), rows(), cz.current_version(db)
    for payload in (
        {"kind": "create_rule", "params": {"pattern": "GROCER", "kind": "contains", "priority": 1,
                                           "category_id": cat_id(db, "Shopping")}},
        {"kind": "merge_category", "params": {"source_id": cat_id(db, "Groceries"),
                                              "target_id": cat_id(db, "Shopping")}},
        {"kind": "set_memory", "params": {"description_key": CAFE_KEY, "category_id": None}},
    ):
        cz.preview(db, payload)
    assert counts() == before and rows() == before_rows and cz.current_version(db) == version


def test_deleting_a_rule_falls_back_to_memory_without_calling_the_model(db, labelled):
    result = change(db, "delete_rule", rule_id=rule_id_of(db, rule(db, "GROCER", "Shopping")))
    assert result["effect"]["to_memory"] == 2
    assert (tx_by_desc(db, GROCER).category_source, name_of(db, GROCER)) == ("memory", "Groceries")


def test_rule_delete_resets_merchant_to_the_printed_description(db, labelled):
    db.execute(text("UPDATE label_memory SET merchant = NULL, merchant_source = NULL"))
    created = rule(db, "CAFE", merchant="Rule Cafe")
    assert tx_by_desc(db, CAFE).merchant == "Rule Cafe"
    change(db, "delete_rule", rule_id=rule_id_of(db, created))
    tx = tx_by_desc(db, CAFE)
    assert (tx.merchant, tx.merchant_source) == (CAFE, "pending")


# --- undo, rename, merge ------------------------------------------------------

def test_undo_is_a_compensating_change_and_history_is_append_only(db, labelled):
    versions_before = count(db, "transaction_label_history")
    undo(db, rule(db, "GROCER", "Shopping"))
    assert name_of(db, GROCER) == "Groceries"
    # Two relabels per matching row (rule, then undo), nothing rewritten.
    assert count(db, "transaction_label_history") == versions_before + 4
    assert_history_matches(db)


def test_undo_order_is_enforced_newest_first(db, labelled):
    first = rule(db, "GROCER", "Shopping", priority=1)
    second = rule(db, "GROCER", "Entertainment", priority=2)
    assert undo_blockers(db, first) == [second["change_id"]]
    undo(db, second)
    undo(db, first)          # no longer stuck behind the undone pair
    assert name_of(db, GROCER) == "Groceries"
    assert_history_matches(db)


def test_rename_changes_no_labels_and_keeps_old_name_in_time(db, labelled):
    groceries = cat_id(db, "Groceries")
    history_before = count(db, "transaction_label_history")
    change(db, "rename_category", category_id=groceries, name="Food at home")
    assert count(db, "transaction_label_history") == history_before
    assert name_of(db, GROCER) == "Food at home"
    old = db.scalar(select(CategoryVersion.valid_from).where(CategoryVersion.category_id == groceries)
                    .order_by(CategoryVersion.valid_from))
    assert category_store.name_at(db, groceries, old) == "Groceries"


def test_merge_repoints_and_archives_and_undo_restores(db, labelled):
    dining = cat_id(db, "Dining & DoorDash")
    merged = change(db, "merge_category", source_id=dining, target_id=cat_id(db, "Shopping"))
    assert name_of(db, CAFE) == "Shopping"
    assert next(c for c in category_store.current_categories(db) if c.category_id == dining).archived
    undo(db, merged)
    assert name_of(db, CAFE) == "Dining & DoorDash"
    assert_history_matches(db)


def test_merge_effect_counts_pinned_rows(db, labelled):
    dining = cat_id(db, "Dining & DoorDash")
    pin(db, GROCER, "category", "Dining & DoorDash")
    preview = cz.preview(db, {"kind": "merge_category", "params": {"source_id": dining,
                                                                  "target_id": cat_id(db, "Shopping")}})
    assert preview["effect"]["rows_changed"] == 2              # the pinned GROCER row and the CAFE row


def test_archived_name_can_be_reused(db, labelled):
    change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash"))
    change(db, "create_category", name="Dining & DoorDash")
    assert sum(1 for c in category_store.current_categories(db) if c.name == "Dining & DoorDash") == 2


@pytest.mark.parametrize("kind, target, into", [
    ("rename_category", "fees", None),
    ("delete_category", "fees", None),
    ("delete_category", "Shopping", "fees"),
    ("delete_category", "Shopping", "other"),      # never silently Other
])
def test_system_categories_are_protected(db, labelled, kind, target, into):
    params = {"category_id": system(db, target) if target == "fees" else cat_id(db, target)}
    if kind == "rename_category":
        params["name"] = "Charges"
    if into:
        params["into"] = system(db, into)
    with pytest.raises(cz.ChangeRejected):
        cz.preview(db, {"kind": kind, "params": params})


@pytest.mark.parametrize("field, value", [("category", "Shopping"), ("merchant", "Bank")])
def test_pin_on_a_section_row_is_refused(db, labelled, field, value):
    with pytest.raises(cz.ChangeRejected):
        pin(db, INTEREST, field, value)


# --- embeddings ----------------------------------------------------------------

def test_refresh_embeds_stale_rows_and_rename_makes_them_stale_again(db, labelled):
    embed = lambda text: [0.0] * 384
    assert (cz.refresh_embeddings(db, embed), cz.refresh_embeddings(db, embed)) == (4, 0)
    change(db, "rename_category", category_id=cat_id(db, "Groceries"), name="Food at home")
    assert cz.refresh_embeddings(db, embed) == 2


# --- undo dependencies and the other change kinds (review fixes) ----------------

def test_undo_of_a_category_is_blocked_while_a_rule_targets_it(db, labelled):
    created = change(db, "create_category", name="Bulk Food")
    targeting = rule(db, "GROCER", "Bulk Food")
    assert undo_blockers(db, created) == [targeting["change_id"]]


def test_undoing_a_rule_delete_is_blocked_after_its_category_was_merged(db, labelled):
    deleted = change(db, "delete_rule", rule_id=rule_id_of(db, rule(db, "CAFE", "Dining & DoorDash")))
    merged = change(db, "merge_category", source_id=cat_id(db, "Dining & DoorDash"),
                    target_id=cat_id(db, "Shopping"))
    assert merged["change_id"] in undo_blockers(db, deleted)


def test_unpin_falls_back_and_undo_restores_the_pin(db, labelled):
    pinned = pin(db, GROCER, "category", "Shopping")
    assert pinned["effect"]["rows_changed"] == 1
    unpinned = cz.apply(db, {"kind": "unpin", "params": {"transaction_id": tx_by_desc(db, GROCER).id,
                                                         "field": "category"}}, None)
    assert (name_of(db, GROCER), tx_by_desc(db, GROCER).category_source) == ("Groceries", "memory")
    undo(db, unpinned)
    assert (name_of(db, GROCER), tx_by_desc(db, GROCER).category_source) == ("Shopping", "pin")
    assert_history_matches(db)


def test_set_memory_normalises_the_key_and_relabels_and_undo_restores(db, labelled):
    result = change(db, "set_memory", description_key="grocer mart #555 springfield",
                    category_id=cat_id(db, "Shopping"))
    assert (db.get(LabelMemory, GROCER_KEY).category_source, name_of(db, GROCER)) == ("user", "Shopping")
    undo(db, result)
    assert (db.get(LabelMemory, GROCER_KEY).category_source, name_of(db, GROCER)) == ("model", "Groceries")


def test_edit_rule_changes_its_target_and_undo_restores(db, labelled):
    edited = change(db, "edit_rule", rule_id=rule_id_of(db, rule(db, "CAFE", "Shopping")),
                    category_id=cat_id(db, "Entertainment"))
    assert name_of(db, CAFE) == "Entertainment"
    undo(db, edited)
    assert name_of(db, CAFE) == "Shopping"


def test_undo_of_category_creation_archives_it(db, labelled):
    undo(db, change(db, "create_category", name="Bulk Food"))
    assert next(c for c in category_store.current_categories(db) if c.name == "Bulk Food").archived


# --- delete without a target: labels withdrawn, rows re-decided (user decision A) ---

def test_delete_withdraws_the_category_and_the_model_re_decides(db, labelled):
    only_category = rule(db, "CORNER", "Dining & DoorDash", priority=1)
    with_merchant = rule(db, "CAFE", "Dining & DoorDash", priority=2, merchant="Corner Cafe")
    result = change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash"))

    assert result["effect"]["to_pending"] == 1
    cafe = tx_by_desc(db, CAFE)
    assert (cafe.category_id, cafe.category_source, cafe.merchant) == (None, "pending", "Corner Cafe")
    assert db.get(cz.CategoryRule, rule_id_of(db, only_category)).deleted
    kept = db.get(cz.CategoryRule, rule_id_of(db, with_merchant))
    assert (kept.deleted, kept.sets_category_id) == (False, None)            # merchant part kept
    memory = db.get(LabelMemory, CAFE_KEY)
    assert memory is None or memory.category_id is None

    # The background pass re-decides from the remaining categories; nothing is silently Other.
    offered = []

    def labeller(descriptions, names):
        offered.extend(names)
        return {k: ModelLabel("Shopping", "Corner Cafe", None) for k in descriptions}
    cz.enrich_pending(db, labeller=labeller)
    assert "Dining & DoorDash" not in offered and name_of(db, CAFE) == "Shopping"
    assert_history_matches(db)


def test_delete_with_into_is_a_merge(db, labelled):
    change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash"), into=cat_id(db, "Groceries"))
    assert name_of(db, CAFE) == "Groceries"


def test_undo_of_delete_restores_rules_memory_and_labels(db, labelled):
    created = rule(db, "CORNER", "Dining & DoorDash")
    undo(db, change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash")))
    assert not db.get(cz.CategoryRule, rule_id_of(db, created)).deleted
    assert name_of(db, CAFE) == "Dining & DoorDash"
    assert_history_matches(db)


def test_delete_then_enrichment_then_undo_restores_the_category(db, labelled):
    pin(db, GROCER_2, "category", "Dining & DoorDash")
    deleted = change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash"))
    # CAFE has nothing else to decide it: pending. The pinned GROCER_2 row loses
    # its pin but memory knows its store: back to Groceries.
    assert (deleted["effect"]["to_pending"], deleted["effect"]["to_memory"]) == (1, 1)
    cz.enrich_pending(db, labeller=labeller_from({CAFE_KEY: ModelLabel("Shopping", "Cafe", None),
                                                  GROCER_KEY: ModelLabel("Groceries", "GM", None)}))
    assert name_of(db, CAFE) == "Shopping"
    undo(db, deleted)
    assert name_of(db, CAFE) == "Dining & DoorDash"
    assert (name_of(db, GROCER_2), tx_by_desc(db, GROCER_2).category_source) == ("Dining & DoorDash", "pin")
    assert_history_matches(db)


# --- memory healing (review fixes) ------------------------------------------------

def test_undo_of_a_category_the_model_learned_heals_memory(db, labelled):
    created = change(db, "create_category", name="Bulk Food")
    bulk = cat_id(db, "Bulk Food")
    # Simulate a later model pass filing GROCER rows into the new category (an
    # unlogged enrichment write), then undo the creation.
    grocer_rows = {"c": bulk, "a": GROCER, "b": GROCER_2}
    db.execute(text("UPDATE label_memory SET category_id = :c WHERE description_key = :k"), {"c": bulk, "k": GROCER_KEY})
    db.execute(text("UPDATE transactions SET category_id = :c WHERE raw_description IN (:a, :b)"), grocer_rows)
    db.execute(text("UPDATE transaction_label_history SET category_id = :c WHERE valid_to IS NULL AND transaction_id IN "
                    "(SELECT id FROM transactions WHERE raw_description IN (:a, :b))"), grocer_rows)
    db.commit()     # harness: raw edits must be committed before code that rolls back its read
    undo(db, created)                                            # nothing logged blocks it
    assert tx_by_desc(db, GROCER).category_source == "pending"
    assert db.get(LabelMemory, GROCER_KEY).category_id is None   # forgotten, not pointing at the archived id

    calls = []
    cz.enrich_pending(db, labeller=labeller_from({GROCER_KEY: ModelLabel("Groceries", "Grocer Mart", None)}, calls))
    assert db.get(LabelMemory, GROCER_KEY).category_id == cat_id(db, "Groceries")
    cz.enrich_pending(db, labeller=labeller_from({}, calls))
    assert len(calls) == 1                                       # stabilised: not re-asked


def test_remember_replaces_an_archived_category(db, labelled):
    change(db, "delete_category", category_id=cat_id(db, "Shopping"))
    shopping_archived = next(c.category_id for c in category_store.current_categories(db)
                             if c.name == "Shopping" and c.archived)
    db.execute(text("UPDATE label_memory SET category_id = :c, category_source = 'model' "
                    "WHERE description_key = :k"), {"c": shopping_archived, "k": CAFE_KEY})
    db.execute(text("UPDATE transactions SET category_id = NULL, category_source = 'pending' "
                    "WHERE raw_description = :d"), {"d": CAFE})
    db.commit()     # harness: raw edits must be committed before code that rolls back its read
    cz.enrich_pending(db, labeller=labeller_from({CAFE_KEY: ModelLabel("Groceries", "Cafe", None)}))
    assert db.get(LabelMemory, CAFE_KEY).category_id == cat_id(db, "Groceries")
