"""Pure logic of ADR 0005: normalisation, rule validation and choice,
per-field precedence, tokens, review kinds, reset refusals and model-output
validation. No database, no Ollama. Invented descriptors only."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app import categorization as cz
from app import category_store, llm
from app.categorization import Context, Labels


# --- normalisation ----------------------------------------------------------

def test_store_numbers_share_a_key():
    assert category_store.description_key("GROCER MART #101 SPRINGFIELD") == "GROCER MART SPRINGFIELD"
    assert category_store.description_key("grocer mart #202 springfield") == "GROCER MART SPRINGFIELD"


@pytest.mark.parametrize("raw", ["12345", "#### 99", "   "])
def test_key_is_never_empty(raw):
    assert category_store.description_key(raw)


def test_normalise_strips_punctuation_and_spaces():
    assert category_store.normalise("  Corner-Café,  4455 ") == "CORNER CAFÉ 4455"


# --- rule validation and matching -------------------------------------------

@pytest.mark.parametrize("pattern,kind", [
    ("", "contains"), ("   ", "contains"), ("!!!", "contains"),
    ("x" * 201, "contains"), ("(", "regex"), ("(a+)+", "regex"), (r"(\w*)*", "regex"),
    ("(x+){2,}", "regex"), ("GROCER", "glob"),
])
def test_bad_patterns_are_refused(pattern, kind):
    with pytest.raises(cz.ChangeRejected):
        cz.validate_pattern(pattern, kind)


@pytest.mark.parametrize("pattern,kind", [("GROCER MART", "contains"), (r"^GROCER\s+MART", "regex"), ("a+b", "regex")])
def test_good_patterns_pass(pattern, kind):
    cz.validate_pattern(pattern, kind)


def rule(rule_id, pattern, priority, category=None, merchant=None, kind="contains"):
    return SimpleNamespace(rule_id=rule_id, pattern=pattern, kind=kind, priority=priority,
                           sets_category_id=category, sets_merchant=merchant)


def test_contains_ignores_case_and_punctuation():
    assert cz.rule_matches(rule(1, "grocer-mart", 0, 5), "GROCER MART #101")


def test_regex_matches_raw_text_case_insensitively():
    assert cz.rule_matches(rule(1, r"mart\s+#\d+", 0, 5, kind="regex"), "GROCER MART #101")


def test_highest_priority_wins_per_field():
    rules = [rule(1, "GROCER", 1, category=10), rule(2, "MART", 5, category=20), rule(3, "GROCER", 9, merchant="Grocer")]
    chosen, conflict = cz.choose_rule(rules, "GROCER MART", "category")
    assert (chosen.rule_id, conflict) == (2, None)
    chosen, _ = cz.choose_rule(rules, "GROCER MART", "merchant")
    assert chosen.rule_id == 3


def test_equal_priority_different_targets_is_a_conflict():
    rules = [rule(1, "GROCER", 5, category=10), rule(2, "MART", 5, category=20)]
    assert cz.choose_rule(rules, "GROCER MART", "category") == (
        None, [{"rule_id": 1, "target": 10}, {"rule_id": 2, "target": 20}])


def test_equal_priority_same_target_is_not_a_conflict():
    rules = [rule(1, "GROCER", 5, category=10), rule(2, "MART", 5, category=10)]
    chosen, conflict = cz.choose_rule(rules, "GROCER MART", "category")
    assert (chosen.rule_id, conflict) == (1, None)


# --- precedence ------------------------------------------------------------------

def ctx(rules=(), memory=None, archived=()):
    return Context(list(rules), memory or {}, {"fees": 1, "interest": 2}, 3, frozenset(archived))


def tx(**kw):
    base = dict(raw_description="GROCER MART #101", section="purchases", category_id=None,
                category_source="pending", category_rule_id=None, merchant="GROCER MART #101",
                merchant_source="pending", merchant_rule_id=None)
    return SimpleNamespace(**{**base, **kw})


def mem(category_id=None, merchant=None):
    return SimpleNamespace(category_id=category_id, merchant=merchant)


def test_section_beats_everything():
    labels, _ = cz.resolve(tx(section="interest", category_source="pin", category_id=9),
                           ctx([rule(1, "GROCER", 99, category=10)]))
    assert (labels.category_id, labels.category_source) == (2, "section")


def test_pin_beats_rule_and_memory():
    labels, _ = cz.resolve(tx(category_id=9, category_source="pin"),
                           ctx([rule(1, "GROCER", 99, category=10)], {"GROCER MART": mem(11)}))
    assert (labels.category_id, labels.category_source) == (9, "pin")


def test_rule_beats_memory():
    labels, _ = cz.resolve(tx(), ctx([rule(1, "GROCER", 0, category=10)], {"GROCER MART": mem(11)}))
    assert (labels.category_id, labels.category_source, labels.category_rule_id) == (10, "rule", 1)


def test_memory_fills_a_pending_field():
    labels, _ = cz.resolve(tx(), ctx(memory={"GROCER MART": mem(11, "Grocer Mart")}))
    assert (labels.category_id, labels.category_source) == (11, "memory")
    assert (labels.merchant, labels.merchant_source) == ("Grocer Mart", "memory")


def test_model_label_matching_memory_keeps_model_source():
    labels, _ = cz.resolve(tx(category_id=11, category_source="model"), ctx(memory={"GROCER MART": mem(11)}))
    assert labels.category_source == "model"


def test_a_row_that_loses_its_rule_falls_to_pending_without_memory():
    labels, _ = cz.resolve(tx(category_id=10, category_source="rule", category_rule_id=1), ctx())
    assert (labels.category_id, labels.category_source) == (None, "pending")


def test_conflict_skips_rules_and_falls_through_to_memory():
    rules = [rule(1, "GROCER", 5, category=10), rule(2, "MART", 5, category=20)]
    labels, conflicts = cz.resolve(tx(), ctx(rules, {"GROCER MART": mem(11)}))
    assert (labels.category_id, labels.category_source) == (11, "memory")
    assert [c["rule_id"] for c in conflicts["category"]] == [1, 2]


def test_merchant_pin_leaves_category_to_rules():
    labels, _ = cz.resolve(tx(merchant="Mine", merchant_source="pin"), ctx([rule(1, "GROCER", 0, category=10)]))
    assert (labels.merchant_source, labels.category_source) == ("pin", "rule")


# --- tokens and review kinds -------------------------------------------------

def test_token_binds_payload_and_version():
    payload = {"kind": "create_rule", "params": {"pattern": "A"}}
    assert cz.make_token(payload, 3) == cz.make_token({"params": {"pattern": "A"}, "kind": "create_rule"}, 3)
    assert cz.make_token(payload, 3) != cz.make_token(payload, 4)
    assert cz.make_token(payload, 3) != cz.make_token({"kind": "create_rule", "params": {"pattern": "B"}}, 3)


def test_review_kinds():
    pending = Labels(None, "pending", None, "X", "model", None)
    none_fit = Labels(3, "none_fit", None, "X", "model", None)
    assert cz.review_kinds(pending, {}, after_model=True) == {"pending"}
    assert cz.review_kinds(pending, {}, after_model=False) == set()       # waiting, not stuck
    assert cz.review_kinds(none_fit, {"category": [{}]}, after_model=False) == {"none_fit", "conflict"}
    assert cz.review_kinds(Labels(4, "model", None, "X", "model", None), {}, after_model=True) == set()


def test_archived_category_is_not_kept_from_model_or_memory():
    labels, _ = cz.resolve(tx(category_id=11, category_source="model"), ctx(memory={"GROCER MART": mem(11)},
                                                                            archived={11}))
    assert (labels.category_id, labels.category_source) == (None, "pending")


def test_pending_merchant_is_the_printed_description_not_a_stale_rule_value():
    labels, _ = cz.resolve(tx(merchant="Old Rule Name", merchant_source="rule", merchant_rule_id=7), ctx())
    assert (labels.merchant, labels.merchant_source) == ("GROCER MART #101", "pending")


# --- reset script refusals ------------------------------------------------------

def reset_args(**kw):
    return SimpleNamespace(**{"yes": True, "confirm_name": "finrag", "env": "production",
                              "allow_production": False, **kw})


def test_reset_refusals():
    from scripts.reset_db import refusal
    url = SimpleNamespace(database="finrag")
    assert refusal(reset_args(yes=False), url, None, 0)
    assert refusal(reset_args(confirm_name="other"), url, None, 0)
    assert refusal(reset_args(), url, "production", 0)                  # marked production
    assert refusal(reset_args(), url, None, 3)                          # unmarked but holds data
    assert refusal(reset_args(env="test"), url, "production", 0)        # never re-mark
    assert refusal(reset_args(allow_production=True), url, None, 3) is None
    assert refusal(reset_args(), url, None, 0) is None                  # empty, unmarked


def test_reset_never_touches_the_import_time_engine(monkeypatch):
    from scripts import reset_db
    import app.models as models

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("reset_db used the import-time engine")

    monkeypatch.setattr(models, "engine", Boom())
    fake_engine = MagicMock()
    monkeypatch.setattr(reset_db, "create_engine", lambda url: fake_engine)
    monkeypatch.setattr(reset_db, "inspect_target", lambda engine: ("production", 2))
    assert reset_db.main(["--database-url", "postgresql://u:secret@h/finrag", "--env", "production",
                          "--confirm-name", "finrag", "--yes"]) == 1


def test_reset_requires_env(capsys):
    from scripts import reset_db
    with pytest.raises(SystemExit):
        reset_db.main(["--database-url", "postgresql://u@h/finrag", "--confirm-name", "finrag", "--yes"])


# --- model output validation ----------------------------------------------------

def ollama_returns(monkeypatch, body):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": body}
    monkeypatch.setattr(llm.httpx, "post", MagicMock(return_value=response))
    return llm.httpx.post


def test_model_output_is_validated(monkeypatch):
    body = ('{"labels": [{"id": 0, "category": "Groceries", "merchant": "Grocer Mart"},'
            ' {"id": 1, "category": "Made Up", "merchant": "X"},'
            ' {"id": 2, "category": "none_fit", "merchant": "Zed", "suggestion": "Pet supplies and more"}]}')
    post = ollama_returns(monkeypatch, body)
    out = llm.label_descriptions({"A": "A 1", "B": "B 2", "C": "C 3"}, ["Groceries"])
    assert out["A"] == llm.ModelLabel("Groceries", "Grocer Mart", None)
    assert out["B"] is None                                     # off-list category rejected
    assert out["C"] == llm.ModelLabel("none_fit", "Zed", "Pet supplies and")
    sent = post.call_args.kwargs["json"]
    assert sent["options"]["temperature"] == 0
    assert sent["format"]["properties"]["labels"]["items"]["properties"]["category"]["enum"] == ["Groceries", "none_fit"]


@pytest.mark.parametrize("failure", [httpx.ConnectError("down"), ValueError("bad json")])
def test_model_failure_returns_none_for_every_key(monkeypatch, failure):
    if isinstance(failure, httpx.HTTPError):
        monkeypatch.setattr(llm.httpx, "post", MagicMock(side_effect=failure))
    else:
        ollama_returns(monkeypatch, "not json")
    assert llm.label_descriptions({"A": "a", "B": "b"}, ["Groceries"]) == {"A": None, "B": None}


# --- the environment guard ---------------------------------------------------------

class FakeCursor:
    def __init__(self, has_table, marker):
        self.results = [(has_table,)] + ([(marker,)] if marker is not None else [None])
        self.closed = False

    def execute(self, sql):
        pass

    def fetchone(self):
        return self.results.pop(0)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, has_table=True, marker="test"):
        self._cursor = FakeCursor(has_table, marker)
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def rollback(self):
        self.rolled_back = True


def test_guard_accepts_a_matching_marker_and_leaves_the_connection_clean(monkeypatch):
    from app import db_guard
    monkeypatch.setenv("FINRAG_ENV", "test")
    connection = FakeConnection(marker="test")
    db_guard._check(connection, None)
    assert connection.rolled_back


@pytest.mark.parametrize("env,has_table,marker", [
    ("test", True, "production"),     # a test run reaching the real database
    ("production", True, "test"),     # the app pointed at the test database
    ("test", False, None),            # never set up
    ("test", True, None),             # table but no row
])
def test_guard_refuses(monkeypatch, env, has_table, marker):
    from app import db_guard
    monkeypatch.setenv("FINRAG_ENV", env)
    with pytest.raises(db_guard.EnvironmentMismatch):
        db_guard._check(FakeConnection(has_table, marker), None)


def test_guard_refuses_an_unknown_environment(monkeypatch):
    from app import db_guard
    monkeypatch.setenv("FINRAG_ENV", "staging")
    with pytest.raises(db_guard.EnvironmentMismatch):
        db_guard._check(FakeConnection(), None)


def test_the_app_engine_has_the_guard_installed():
    from sqlalchemy import event
    from app import db_guard, models
    assert event.contains(models.engine, "connect", db_guard._check)
