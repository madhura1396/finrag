"""ADR 0005 against a real Postgres (finrag_test). Invented descriptors only."""

from dataclasses import replace
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


def statement(descriptions: list[tuple[str, str]], start=START, end=END) -> StatementData:
    """A reconciling statement: each purchase is $10.00, plus one interest row."""
    rows = [Row(start + timedelta(days=2), None, d, Decimal("-10.00"), section) for d, section in descriptions]
    rows.append(Row(end, None, "INTEREST CHARGE ON PURCHASES", Decimal("-1.00"), "interest"))
    purchases = Decimal("10.00") * len([r for r in rows if r.section == "purchases"])
    totals = {"payments": Decimal("0.00"), "other_credits": Decimal("0.00"), "purchases": purchases,
              "fees": Decimal("0.00"), "interest": Decimal("1.00")}
    summary = {
        "previous_balance": SummaryLine(Decimal("0.00"), 1, None),
        "payments":         SummaryLine(Decimal("0.00"), -1, "payments"),
        "other_credits":    SummaryLine(Decimal("0.00"), -1, "other_credits"),
        "cash_advances":    SummaryLine(Decimal("0.00"), 1, "cash_advances"),
        "purchases":        SummaryLine(purchases, 1, "purchases"),
        "fees":             SummaryLine(Decimal("0.00"), 1, "fees"),
        "interest":         SummaryLine(Decimal("1.00"), 1, "interest"),
        "new_balance":      SummaryLine(purchases + Decimal("1.00"), 0, None),
    }
    return StatementData(start, end, rows, totals, summary)


def labeller_from(answers: dict[str, ModelLabel | None], calls: list | None = None):
    def labeller(descriptions, category_names):
        if calls is not None:
            calls.append(dict(descriptions))
        return {key: answers.get(key) for key in descriptions}
    return labeller


def cat_id(db, name):
    return next(c.category_id for c in category_store.current_categories(db) if c.name == name)


def tx_by_desc(db, description):
    return db.scalar(select(Transaction).where(Transaction.raw_description == description))


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


GROCER = "GROCER MART #101 SPRINGFIELD"
GROCER_2 = "GROCER MART #202 SPRINGFIELD"
CAFE = "CORNER CAFE 4455 RIVERTON"


@pytest.fixture
def uploaded(db):
    pipeline.persist_statement(statement([(GROCER, "purchases"), (GROCER_2, "purchases"), (CAFE, "purchases")]),
                               "synthetic.pdf")
    return db


# --- seed and constraints ---------------------------------------------------

def test_seed_creates_system_defaults_and_version(db):
    cats = category_store.current_categories(db)
    assert {c.system_key for c in cats if c.is_system} == {"fees", "interest", "other"}
    assert "Groceries" in {c.name for c in cats}
    assert cz.current_version(db) is not None


def test_seed_never_resurrects_a_renamed_default(db):
    from app.db_setup import seed
    change(db, "rename_category", category_id=cat_id(db, "Groceries"), name="Food at home")
    seed(db)
    db.flush()
    names = {c.name for c in category_store.current_categories(db)}
    assert "Groceries" not in names and "Food at home" in names


def test_database_rejects_second_open_category_version(db):
    groceries = cat_id(db, "Groceries")
    db.add(CategoryVersion(category_id=groceries, name="Dup"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_database_rejects_duplicate_current_name_ignoring_case(db):
    identity = CategoryId(is_system=False)
    db.add(identity)
    db.flush()
    db.add(CategoryVersion(category_id=identity.category_id, name="groceries"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_database_rejects_null_category_unless_pending(uploaded):
    tx = tx_by_desc(uploaded, GROCER)
    uploaded.execute(text("SAVEPOINT s"))
    with pytest.raises(IntegrityError):
        uploaded.execute(text("UPDATE transactions SET category_source = 'model' WHERE id = :i"), {"i": tx.id})
    uploaded.execute(text("ROLLBACK TO SAVEPOINT s"))


def test_database_rejects_second_open_history_version(uploaded):
    tx = tx_by_desc(uploaded, GROCER)
    uploaded.add(TransactionLabelHistory(transaction_id=tx.id, category_source="pending",
                                         merchant=tx.merchant, merchant_source="pending"))
    with pytest.raises(IntegrityError):
        uploaded.flush()


def test_database_rejects_duplicate_memory_key(db):
    db.add(LabelMemory(description_key="SAME KEY", merchant="A", merchant_source="user"))
    db.flush()
    db.add(LabelMemory(description_key="SAME KEY", merchant="B", merchant_source="user"))
    with pytest.raises(IntegrityError):
        db.flush()


# --- ingest and enrichment --------------------------------------------------

def test_upload_labels_sections_and_leaves_the_rest_pending(uploaded):
    interest = tx_by_desc(uploaded, "INTEREST CHARGE ON PURCHASES")
    assert interest.category_source == "section"
    assert interest.category_id == category_store.system_category_id(uploaded, "interest")
    assert tx_by_desc(uploaded, GROCER).category_source == "pending"
    assert_history_matches(uploaded)


def test_enrichment_uses_one_model_call_per_key_and_remembers(uploaded):
    calls = []
    answers = {"GROCER MART SPRINGFIELD": ModelLabel("Groceries", "Grocer Mart", None),
               "CORNER CAFE RIVERTON": ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    result = cz.enrich_pending(uploaded, labeller=labeller_from(answers, calls))

    assert len(calls) == 1 and set(calls[0]) == set(answers)       # #101 and #202 share a key
    for desc in (GROCER, GROCER_2):
        tx = tx_by_desc(uploaded, desc)
        assert (tx.category_source, tx.merchant, tx.merchant_source) == ("model", "Grocer Mart", "model")
        assert tx.category == "Groceries"                           # transitional text copy
    memory = uploaded.get(LabelMemory, "GROCER MART SPRINGFIELD")
    assert memory.category_source == "model" and memory.merchant == "Grocer Mart"
    assert result["pending"] == 0
    assert_history_matches(uploaded)


def test_none_fit_goes_to_other_with_review_and_is_not_remembered(uploaded):
    answers = {"GROCER MART SPRINGFIELD": ModelLabel(NONE_FIT, "Grocer Mart", "Bulk Food"),
               "CORNER CAFE RIVERTON": ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    cz.enrich_pending(uploaded, labeller=labeller_from(answers))
    tx = tx_by_desc(uploaded, GROCER)
    assert tx.category_source == "none_fit" and tx.category_id == category_store.system_category_id(uploaded, "other")
    assert tx.needs_review
    item = uploaded.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "none_fit"))
    assert item.detail == {"suggestion": "Bulk Food"}
    memory = uploaded.get(LabelMemory, "GROCER MART SPRINGFIELD")
    assert memory.category_id is None          # merchant remembered, none_fit category never
    assert memory.merchant == "Grocer Mart"


def test_model_failure_leaves_rows_pending_and_flagged(uploaded):
    result = cz.enrich_pending(uploaded, labeller=labeller_from({}))
    tx = tx_by_desc(uploaded, CAFE)
    assert (tx.category_id, tx.category_source, tx.needs_review) == (None, "pending", True)
    assert uploaded.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "pending"))
    assert result["pending"] == 3
    assert uploaded.get(LabelMemory, "CORNER CAFE RIVERTON") is None


def test_enrichment_never_overwrites_a_pin_set_while_the_model_ran(uploaded):
    cafe = tx_by_desc(uploaded, CAFE)
    cafe_id = cafe.id
    shopping = cat_id(uploaded, "Shopping")

    def labeller(descriptions, names):
        from app.sessions import new_session
        other = new_session()
        cz.apply(other, {"kind": "pin", "params": {"transaction_id": cafe_id, "field": "category",
                                                    "value": shopping}}, None)
        other.close()
        return {k: ModelLabel("Groceries", "X", None) for k in descriptions}

    cz.enrich_pending(uploaded, labeller=labeller)
    uploaded.expire_all()
    cafe = uploaded.get(Transaction, cafe_id)
    assert (cafe.category_id, cafe.category_source) == (shopping, "pin")


# --- rules, precedence, tokens --------------------------------------------

@pytest.fixture
def labelled(uploaded):
    answers = {"GROCER MART SPRINGFIELD": ModelLabel("Groceries", "Grocer Mart", None),
               "CORNER CAFE RIVERTON": ModelLabel("Dining & DoorDash", "Corner Cafe", None)}
    cz.enrich_pending(uploaded, labeller=labeller_from(answers))
    return uploaded


def test_rule_relabels_matching_rows_and_never_touches_section_or_pins(labelled):
    db = labelled
    shopping = cat_id(db, "Shopping")
    cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx_by_desc(db, GROCER_2).id,
                                            "field": "category", "value": cat_id(db, "Entertainment")}}, None)
    result = change(db, "create_rule", pattern="GROCER MART", kind="contains", priority=10, category_id=shopping)

    assert result["effect"]["rows_changed"] == 1
    assert tx_by_desc(db, GROCER).category_source == "rule"
    assert tx_by_desc(db, GROCER_2).category_source == "pin"
    assert tx_by_desc(db, "INTEREST CHARGE ON PURCHASES").category_source == "section"
    assert_history_matches(db)


def test_merchant_pin_does_not_freeze_category(labelled):
    db = labelled
    cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx_by_desc(db, GROCER).id,
                                            "field": "merchant", "value": "My Grocer"}}, None)
    change(db, "create_rule", pattern="GROCER", kind="contains", priority=1, category_id=cat_id(db, "Shopping"))
    tx = tx_by_desc(db, GROCER)
    assert (tx.merchant, tx.merchant_source) == ("My Grocer", "pin")
    assert tx.category_source == "rule"


def test_equal_priority_conflict_is_refused_at_creation(labelled):
    db = labelled
    change(db, "create_rule", pattern="GROCER", kind="contains", priority=5, category_id=cat_id(db, "Shopping"))
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "create_rule", "params": {"pattern": "SPRINGFIELD", "kind": "contains",
                                                         "priority": 5, "category_id": cat_id(db, "Entertainment")}})
    assert e.value.detail["total"] >= 1


def test_equal_priority_conflict_at_ingest_is_flagged_not_decided(labelled):
    db = labelled
    change(db, "create_rule", pattern="RIVERTON", kind="contains", priority=5, category_id=cat_id(db, "Shopping"))
    change(db, "create_rule", pattern="ZEBRA", kind="contains", priority=5, category_id=cat_id(db, "Entertainment"))
    # Test harness: sessions share one connection, so release this session's open
    # savepoint before another session commits, or a later rollback here would
    # discard that commit. Real sessions each have their own connection.
    db.commit()
    pipeline.persist_statement(statement([("ZEBRA SHOP RIVERTON", "purchases")], date(2026, 1, 3), date(2026, 2, 2)),
                               "second.pdf")
    cz.enrich_pending(db, labeller=labeller_from({}))
    tx = tx_by_desc(db, "ZEBRA SHOP RIVERTON")
    assert tx.category_source == "pending" and tx.needs_review
    assert db.scalar(select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.kind == "conflict"))


def test_preview_token_goes_stale_after_an_upload(labelled):
    db = labelled
    payload = {"kind": "create_rule", "params": {"pattern": "CAFE", "kind": "contains", "priority": 1,
                                                 "category_id": cat_id(db, "Shopping")}}
    token = cz.preview(db, payload)["token"]
    pipeline.persist_statement(statement([("NEW PLACE", "purchases")], date(2026, 1, 3), date(2026, 2, 2)), "x.pdf")
    with pytest.raises(cz.StalePreview):
        cz.apply(db, payload, token)


def test_token_is_bound_to_the_payload(labelled):
    db = labelled
    payload = {"kind": "create_rule", "params": {"pattern": "CAFE", "kind": "contains", "priority": 1,
                                                 "category_id": cat_id(db, "Shopping")}}
    token = cz.preview(db, payload)["token"]
    other = {"kind": "create_rule", "params": {**payload["params"], "pattern": "GROCER"}}
    with pytest.raises(cz.StalePreview):
        cz.apply(db, other, token)


def test_preview_changes_nothing(labelled):
    db = labelled
    before = [cz.labels_of(t) for t in db.scalars(select(Transaction).order_by(Transaction.id)).all()]
    preview = cz.preview(db, {"kind": "create_rule", "params": {"pattern": "GROCER", "kind": "contains",
                                                               "priority": 1, "category_id": cat_id(db, "Shopping")}})
    assert preview["effect"]["rows_changed"] == 2
    after = [cz.labels_of(t) for t in db.scalars(select(Transaction).order_by(Transaction.id)).all()]
    assert before == after
    assert db.scalar(select(ChangeLog.change_id)) is None


def test_deleting_a_rule_falls_back_to_memory_without_calling_the_model(labelled):
    db = labelled
    created = change(db, "create_rule", pattern="GROCER", kind="contains", priority=1,
                     category_id=cat_id(db, "Shopping"))
    rule_id = db.scalar(select(ChangeLog.after).where(ChangeLog.change_id == created["change_id"]))["rule_id"]
    result = change(db, "delete_rule", rule_id=rule_id)
    assert result["effect"]["to_memory"] == 2
    tx = tx_by_desc(db, GROCER)
    assert (tx.category_source, tx.category) == ("memory", "Groceries")


# --- undo, rename, merge ------------------------------------------------------

def test_undo_is_a_compensating_change_and_history_is_append_only(labelled):
    db = labelled
    versions_before = db.scalar(text("SELECT COUNT(*) FROM transaction_label_history"))
    created = change(db, "create_rule", pattern="GROCER", kind="contains", priority=1,
                     category_id=cat_id(db, "Shopping"))
    change(db, "undo", change_id=created["change_id"])
    tx = tx_by_desc(db, GROCER)
    assert tx.category == "Groceries"
    # Two relabels per matching row (rule, then undo), nothing rewritten.
    assert db.scalar(text("SELECT COUNT(*) FROM transaction_label_history")) == versions_before + 4
    assert_history_matches(db)


def test_undo_is_refused_while_a_later_change_touched_the_same_rows(labelled):
    db = labelled
    first = change(db, "create_rule", pattern="GROCER", kind="contains", priority=1,
                   category_id=cat_id(db, "Shopping"))
    second = change(db, "create_rule", pattern="GROCER", kind="contains", priority=2,
                    category_id=cat_id(db, "Entertainment"))
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "undo", "params": {"change_id": first["change_id"]}})
    assert e.value.detail["blocking_change_ids"] == [second["change_id"]]


def test_rename_changes_no_labels_and_keeps_old_name_in_time(labelled):
    db = labelled
    groceries = cat_id(db, "Groceries")
    renamed_at = db.scalar(text("SELECT now()"))
    history_before = db.scalar(text("SELECT COUNT(*) FROM transaction_label_history"))
    change(db, "rename_category", category_id=groceries, name="Food at home")
    assert db.scalar(text("SELECT COUNT(*) FROM transaction_label_history")) == history_before
    assert tx_by_desc(db, GROCER).category == "Food at home"     # transitional text follows
    old = db.scalar(select(CategoryVersion.valid_from).where(CategoryVersion.category_id == groceries)
                    .order_by(CategoryVersion.valid_from))
    assert category_store.name_at(db, groceries, old) == "Groceries"


def test_merge_repoints_and_archives_and_undo_restores(labelled):
    db = labelled
    dining, shopping = cat_id(db, "Dining & DoorDash"), cat_id(db, "Shopping")
    merged = change(db, "merge_category", source_id=dining, target_id=shopping)
    assert tx_by_desc(db, CAFE).category == "Shopping"
    archived = next(c for c in category_store.current_categories(db) if c.category_id == dining)
    assert archived.archived
    change(db, "undo", change_id=merged["change_id"])
    assert tx_by_desc(db, CAFE).category == "Dining & DoorDash"
    assert_history_matches(db)


def test_archived_name_can_be_reused(labelled):
    db = labelled
    dining = cat_id(db, "Dining & DoorDash")
    change(db, "delete_category", category_id=dining)
    change(db, "create_category", name="Dining & DoorDash")
    assert sum(1 for c in category_store.current_categories(db) if c.name == "Dining & DoorDash") == 2


@pytest.mark.parametrize("kind,params", [
    ("rename_category", {"name": "Charges"}),
    ("delete_category", {}),
])
def test_system_categories_are_protected(labelled, kind, params):
    db = labelled
    fees = category_store.system_category_id(db, "fees")
    with pytest.raises(cz.ChangeRejected):
        cz.preview(db, {"kind": kind, "params": {"category_id": fees, **params}})


def test_pin_on_a_section_row_is_refused(labelled):
    db = labelled
    interest = tx_by_desc(db, "INTEREST CHARGE ON PURCHASES")
    with pytest.raises(cz.ChangeRejected):
        cz.apply(db, {"kind": "pin", "params": {"transaction_id": interest.id, "field": "category",
                                                "value": cat_id(db, "Shopping")}}, None)


# --- embeddings ----------------------------------------------------------------

def test_refresh_embeds_stale_rows_and_rename_makes_them_stale_again(labelled):
    db = labelled
    embed = lambda text: [0.0] * 384
    assert cz.refresh_embeddings(db, embed) == 4
    assert cz.refresh_embeddings(db, embed) == 0
    change(db, "rename_category", category_id=cat_id(db, "Groceries"), name="Food at home")
    assert cz.refresh_embeddings(db, embed) == 2


# --- review round 2: undo depth, dependencies, untested kinds -------------------

def rule_id_of(db, result):
    return db.scalar(select(ChangeLog.after).where(ChangeLog.change_id == result["change_id"]))["rule_id"]


def test_undo_two_overlapping_changes_newest_first(labelled):
    db = labelled
    first = change(db, "create_rule", pattern="GROCER", kind="contains", priority=1,
                   category_id=cat_id(db, "Shopping"))
    second = change(db, "create_rule", pattern="GROCER", kind="contains", priority=2,
                    category_id=cat_id(db, "Entertainment"))
    change(db, "undo", change_id=second["change_id"])
    change(db, "undo", change_id=first["change_id"])          # no longer stuck behind the undone pair
    assert tx_by_desc(db, GROCER).category == "Groceries"
    assert_history_matches(db)


def test_undo_of_a_category_is_blocked_while_a_rule_targets_it(labelled):
    db = labelled
    created = change(db, "create_category", name="Bulk Food")
    rule = change(db, "create_rule", pattern="GROCER", kind="contains", priority=1,
                  category_id=cat_id(db, "Bulk Food"))
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "undo", "params": {"change_id": created["change_id"]}})
    assert e.value.detail["blocking_change_ids"] == [rule["change_id"]]


def test_undoing_a_rule_delete_is_blocked_after_its_category_was_merged(labelled):
    db = labelled
    rule = change(db, "create_rule", pattern="CAFE", kind="contains", priority=1,
                  category_id=cat_id(db, "Dining & DoorDash"))
    deleted = change(db, "delete_rule", rule_id=rule_id_of(db, rule))
    merged = change(db, "merge_category", source_id=cat_id(db, "Dining & DoorDash"),
                    target_id=cat_id(db, "Shopping"))
    with pytest.raises(cz.ChangeRejected) as e:
        cz.preview(db, {"kind": "undo", "params": {"change_id": deleted["change_id"]}})
    assert merged["change_id"] in e.value.detail["blocking_change_ids"]


def test_unpin_falls_back_and_undo_restores_the_pin(labelled):
    db = labelled
    tx = tx_by_desc(db, GROCER)
    pinned = cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx.id, "field": "category",
                                                     "value": cat_id(db, "Shopping")}}, None)
    unpinned = cz.apply(db, {"kind": "unpin", "params": {"transaction_id": tx.id, "field": "category"}}, None)
    tx = tx_by_desc(db, GROCER)
    assert (tx.category, tx.category_source) == ("Groceries", "memory")
    change(db, "undo", change_id=unpinned["change_id"])
    tx = tx_by_desc(db, GROCER)
    assert (tx.category, tx.category_source) == ("Shopping", "pin")
    assert pinned["change_id"] < unpinned["change_id"]
    assert_history_matches(db)


def test_set_memory_normalises_the_key_and_relabels_and_undo_restores(labelled):
    db = labelled
    result = change(db, "set_memory", description_key="grocer mart #555 springfield",
                    category_id=cat_id(db, "Shopping"))
    assert db.get(LabelMemory, "GROCER MART SPRINGFIELD").category_source == "user"
    assert tx_by_desc(db, GROCER).category == "Shopping"
    change(db, "undo", change_id=result["change_id"])
    assert db.get(LabelMemory, "GROCER MART SPRINGFIELD").category_source == "model"
    assert tx_by_desc(db, GROCER).category == "Groceries"


def test_edit_rule_changes_its_target_and_undo_restores(labelled):
    db = labelled
    rule = change(db, "create_rule", pattern="CAFE", kind="contains", priority=1,
                  category_id=cat_id(db, "Shopping"))
    edited = change(db, "edit_rule", rule_id=rule_id_of(db, rule), category_id=cat_id(db, "Entertainment"))
    assert tx_by_desc(db, CAFE).category == "Entertainment"
    change(db, "undo", change_id=edited["change_id"])
    assert tx_by_desc(db, CAFE).category == "Shopping"


def test_undo_of_category_creation_archives_it(labelled):
    db = labelled
    created = change(db, "create_category", name="Bulk Food")
    change(db, "undo", change_id=created["change_id"])
    bulk = next(c for c in category_store.current_categories(db) if c.name == "Bulk Food")
    assert bulk.archived


def test_preview_leaves_no_trace_anywhere(labelled):
    db = labelled
    tables = ("transaction_label_history", "review_items", "category_rules", "category_versions",
              "label_memory", "change_log")
    counts = lambda: {t: db.scalar(text(f"SELECT COUNT(*) FROM {t}")) for t in tables}
    flags = lambda: [t.needs_review for t in db.scalars(select(Transaction).order_by(Transaction.id))]
    before, before_flags, version = counts(), flags(), cz.current_version(db)
    for payload in (
        {"kind": "create_rule", "params": {"pattern": "GROCER", "kind": "contains", "priority": 1,
                                           "category_id": cat_id(db, "Shopping")}},
        {"kind": "merge_category", "params": {"source_id": cat_id(db, "Groceries"),
                                              "target_id": cat_id(db, "Shopping")}},
        {"kind": "set_memory", "params": {"description_key": "CORNER CAFE RIVERTON", "category_id": None}},
    ):
        cz.preview(db, payload)
    assert counts() == before and flags() == before_flags and cz.current_version(db) == version


def test_delete_into_fees_is_refused(labelled):
    db = labelled
    with pytest.raises(cz.ChangeRejected):
        cz.preview(db, {"kind": "delete_category", "params": {
            "category_id": cat_id(db, "Shopping"), "into": category_store.system_category_id(db, "fees")}})


def test_merchant_pin_on_a_section_row_is_refused(labelled):
    db = labelled
    interest = tx_by_desc(db, "INTEREST CHARGE ON PURCHASES")
    with pytest.raises(cz.ChangeRejected):
        cz.apply(db, {"kind": "pin", "params": {"transaction_id": interest.id, "field": "merchant",
                                                "value": "Bank"}}, None)


def test_enrichment_with_nothing_to_do_does_not_invalidate_previews(labelled):
    db = labelled
    version = cz.current_version(db)
    cz.enrich_pending(db, labeller=labeller_from({}))
    assert cz.current_version(db) == version


def test_rule_delete_resets_merchant_to_the_printed_description(labelled):
    db = labelled
    db.execute(text("UPDATE label_memory SET merchant = NULL, merchant_source = NULL"))
    rule = change(db, "create_rule", pattern="CAFE", kind="contains", priority=1, merchant="Rule Cafe")
    assert tx_by_desc(db, CAFE).merchant == "Rule Cafe"
    change(db, "delete_rule", rule_id=rule_id_of(db, rule))
    tx = tx_by_desc(db, CAFE)
    assert (tx.merchant, tx.merchant_source) == (CAFE, "pending")


# --- delete without a target: labels withdrawn, rows re-decided (user decision A) ---

def test_delete_withdraws_the_category_and_the_model_re_decides(labelled):
    db = labelled
    dining = cat_id(db, "Dining & DoorDash")
    only_category = change(db, "create_rule", pattern="CORNER", kind="contains", priority=1, category_id=dining)
    with_merchant = change(db, "create_rule", pattern="CAFE", kind="contains", priority=2, category_id=dining,
                           merchant="Corner Cafe")
    result = change(db, "delete_category", category_id=dining)

    assert result["effect"]["to_pending"] == 1
    cafe = tx_by_desc(db, CAFE)
    assert (cafe.category_id, cafe.category_source) == (None, "pending")
    assert cafe.merchant == "Corner Cafe"                                    # merchant part kept
    assert db.get(cz.CategoryRule, rule_id_of(db, only_category)).deleted
    kept = db.get(cz.CategoryRule, rule_id_of(db, with_merchant))
    assert (kept.deleted, kept.sets_category_id) == (False, None)
    memory = db.get(LabelMemory, "CORNER CAFE RIVERTON")
    assert memory is None or memory.category_id is None

    # The background pass re-decides from the remaining categories; nothing is silently Other.
    offered = []
    def labeller(descriptions, names):
        offered.extend(names)
        return {k: ModelLabel("Shopping", "Corner Cafe", None) for k in descriptions}
    cz.enrich_pending(db, labeller=labeller)
    assert "Dining & DoorDash" not in offered
    assert tx_by_desc(db, CAFE).category == "Shopping"
    assert_history_matches(db)


def test_delete_with_into_is_a_merge(labelled):
    db = labelled
    change(db, "delete_category", category_id=cat_id(db, "Dining & DoorDash"), into=cat_id(db, "Groceries"))
    assert tx_by_desc(db, CAFE).category == "Groceries"


def test_delete_into_other_explicitly_is_refused(labelled):
    db = labelled
    with pytest.raises(cz.ChangeRejected):
        cz.preview(db, {"kind": "delete_category", "params": {
            "category_id": cat_id(db, "Shopping"), "into": category_store.system_category_id(db, "other")}})


def test_undo_of_delete_restores_rules_memory_and_labels(labelled):
    db = labelled
    dining = cat_id(db, "Dining & DoorDash")
    rule = change(db, "create_rule", pattern="CORNER", kind="contains", priority=1, category_id=dining)
    deleted = change(db, "delete_category", category_id=dining)
    change(db, "undo", change_id=deleted["change_id"])
    assert not db.get(cz.CategoryRule, rule_id_of(db, rule)).deleted
    assert tx_by_desc(db, CAFE).category == "Dining & DoorDash"
    assert_history_matches(db)


# --- final review of #21 --------------------------------------------------------

def test_undo_of_a_category_the_model_learned_heals_memory(labelled):
    db = labelled
    created = change(db, "create_category", name="Bulk Food")
    bulk = cat_id(db, "Bulk Food")
    # Simulate a later model pass filing GROCER rows into the new category (an
    # unlogged enrichment write), then undo the creation.
    db.execute(text("UPDATE label_memory SET category_id = :c WHERE description_key = 'GROCER MART SPRINGFIELD'"),
               {"c": bulk})
    db.execute(text("UPDATE transactions SET category_id = :c WHERE raw_description IN (:a, :b)"),
               {"c": bulk, "a": GROCER, "b": GROCER_2})
    db.execute(text("UPDATE transaction_label_history SET category_id = :c WHERE valid_to IS NULL AND transaction_id IN "
                    "(SELECT id FROM transactions WHERE raw_description IN (:a, :b))"),
               {"c": bulk, "a": GROCER, "b": GROCER_2})
    db.commit()     # harness: raw edits must be committed before code that rolls back its read
    change(db, "undo", change_id=created["change_id"])        # nothing logged blocks it
    assert tx_by_desc(db, GROCER).category_source == "pending"

    memory = db.get(LabelMemory, "GROCER MART SPRINGFIELD")
    assert memory.category_id is None                          # forgotten, not pointing at the archived id
    calls = []
    cz.enrich_pending(db, labeller=labeller_from(
        {"GROCER MART SPRINGFIELD": ModelLabel("Groceries", "Grocer Mart", None)}, calls))
    assert db.get(LabelMemory, "GROCER MART SPRINGFIELD").category_id == cat_id(db, "Groceries")
    cz.enrich_pending(db, labeller=labeller_from({}, calls))
    assert len(calls) == 1                                     # stabilised: not re-asked


def test_remember_replaces_an_archived_category(labelled):
    db = labelled
    groceries = cat_id(db, "Groceries")
    change(db, "delete_category", category_id=cat_id(db, "Shopping"))
    shopping_archived = next(c.category_id for c in category_store.current_categories(db)
                             if c.name == "Shopping" and c.archived)
    db.execute(text("UPDATE label_memory SET category_id = :c, category_source = 'model' "
                    "WHERE description_key = 'CORNER CAFE RIVERTON'"), {"c": shopping_archived})
    db.execute(text("UPDATE transactions SET category_id = NULL, category_source = 'pending' "
                    "WHERE raw_description = :d"), {"d": CAFE})
    db.commit()     # harness: raw edits must be committed before code that rolls back its read
    cz.enrich_pending(db, labeller=labeller_from({"CORNER CAFE RIVERTON": ModelLabel("Groceries", "Cafe", None)}))
    assert db.get(LabelMemory, "CORNER CAFE RIVERTON").category_id == groceries


def test_merge_effect_counts_pinned_rows(labelled):
    db = labelled
    dining = cat_id(db, "Dining & DoorDash")
    cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx_by_desc(db, GROCER).id,
                                            "field": "category", "value": dining}}, None)
    preview = cz.preview(db, {"kind": "merge_category", "params": {"source_id": dining,
                                                                  "target_id": cat_id(db, "Shopping")}})
    assert preview["effect"]["rows_changed"] == 2              # the pinned GROCER row and the CAFE row


def test_pin_and_unpin_report_their_row(labelled):
    db = labelled
    tx = tx_by_desc(db, GROCER)
    pinned = cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx.id, "field": "category",
                                                     "value": cat_id(db, "Shopping")}}, None)
    assert pinned["effect"]["rows_changed"] == 1


def test_delete_then_enrichment_then_undo_restores_the_category(labelled):
    db = labelled
    dining = cat_id(db, "Dining & DoorDash")
    cz.apply(db, {"kind": "pin", "params": {"transaction_id": tx_by_desc(db, GROCER_2).id,
                                            "field": "category", "value": dining}}, None)
    deleted = change(db, "delete_category", category_id=dining)
    # CAFE has nothing else to decide it: pending. The pinned GROCER_2 row loses
    # its pin but memory knows its store: back to Groceries.
    assert (deleted["effect"]["to_pending"], deleted["effect"]["to_memory"]) == (1, 1)
    cz.enrich_pending(db, labeller=labeller_from({"CORNER CAFE RIVERTON": ModelLabel("Shopping", "Cafe", None),
                                                  "GROCER MART SPRINGFIELD": ModelLabel("Groceries", "GM", None)}))
    assert tx_by_desc(db, CAFE).category == "Shopping"
    change(db, "undo", change_id=deleted["change_id"])
    assert tx_by_desc(db, CAFE).category == "Dining & DoorDash"
    assert (tx_by_desc(db, GROCER_2).category, tx_by_desc(db, GROCER_2).category_source) == ("Dining & DoorDash", "pin")
    assert_history_matches(db)
