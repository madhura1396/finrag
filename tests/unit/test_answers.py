"""Formatting, the exact answer and the placeholder-rewrite check (ADR 0004, detail 3)."""

from datetime import date
from decimal import Decimal

import httpx
import pytest

from app import answers
from app import read_config as rs
from app.calculations import Result
from app.read_config import Form, Range, month_range

D = Decimal
NOV = month_range(2026, 11)


# --- formatting -------------------------------------------------------------------

@pytest.mark.parametrize("amount,text", [
    (D("1234.5"), "$1,234.50"), (D("-45.2"), "$45.20"), (D("0.005"), "$0.01"), (D("0.004"), "$0.00"),
])
def test_money(amount, text):
    assert answers.fmt_money(amount) == text


def test_dates_and_ranges():
    assert answers.fmt_date(date(2025, 12, 5)) == "5 Dec 2025"
    assert answers.fmt_range(NOV) == "Nov 2026"
    assert answers.fmt_range(Range(date(2026, 5, 5), date(2026, 7, 5), "x")) == "5 May – 5 Jul 2026"
    assert answers.fmt_range(Range(date(2025, 12, 3), date(2026, 1, 2), "x")) == "3 Dec 2025 – 2 Jan 2026"
    assert answers.fmt_range(Range(None, None, "all")) == "all your data"


# --- exact answers -------------------------------------------------------------------

def test_total_spent_facts_carry_their_verbs():
    form = Form(type="total_spent", category="Groceries", period=NOV)
    result = Result("total_spent", {"total": D("245.10"), "count": 14, "direction": "spent",
                                    "refunds": D("30.00"), "refund_count": 1})
    answer = answers.render(form, result, None)
    assert answer.facts[0] == "you spent $245.10 on Groceries in Nov 2026"
    assert answer.exact == ("You spent $245.10 on Groceries in Nov 2026, that came from 14 charges, "
                            "and you also got $30.00 back in refunds.")


def test_table_answers_have_no_facts_to_rewrite():
    form = Form(type="ranking", period=NOV)
    result = Result("ranking", {"group_by": "merchant", "order": "most", "groups": 1},
                    table=[{"name": "Grocer Mart", "amount": D("10"), "count": 1}])
    answer = answers.render(form, result, None)
    assert answer.facts == [] and answer.table[0]["amount"] == "$10.00"
    assert answers.rewrite(form, answer, call=lambda p: pytest.fail("tables must not be rewritten")) == \
        (answer.exact, "skipped")


# --- the rewrite check -------------------------------------------------------------------

def test_faithful_arrangement_is_accepted_only_with_tokens_in_order():
    assert answers.check_rewrite("Here is your answer: [F1], and [F2].", 2)
    assert answers.check_rewrite("Here is what I found: [F1], [F2], and [F3].", 3)
    assert not answers.check_rewrite("[F3], [F2], and [F1].", 3)


@pytest.mark.parametrize("text", [
    "[F1].",                              # missing token
    "[F1] [F1] and [F2].",                 # duplicated token
    "[F1], [F2] and [F3].",                # unknown token
    "In 2025, [F1] and [F2].",             # digit outside tokens
    "[F1] " + "and so on " * 40 + "[F2]",  # too long
    "[F1] and [F2] didn't change.",        # contraction negation
])
def test_broken_rewrites_are_rejected(text):
    assert not answers.check_rewrite(text, 2)


@pytest.mark.parametrize("word", sorted(rs.GLUE_BANNED_WORDS - {"n't"}))
def test_every_closed_list_word_is_rejected_outside_tokens(word):
    assert not answers.check_rewrite(f"Well, {word} [F1] and [F2].", 2)


def test_words_inside_tokens_are_fine():
    form = Form(type="count", period=NOV)
    answer = answers.Answer("You had 3 charges in Nov 2026.", ["you had 3 charges in Nov 2026"], None)
    text, status = answers.rewrite(form, answer, call=lambda p: "Here is what I found: [F1]!")
    assert (text, status) == ("Here is what I found: you had 3 charges in Nov 2026!", "accepted")


def test_rejected_or_failed_rewrite_falls_back_to_exact():
    form = Form(type="count", period=NOV)
    answer = answers.Answer("You had 3 charges.", ["you had 3 charges"], None)
    assert answers.rewrite(form, answer, call=lambda p: "You spent [F1].") == ("You had 3 charges.", "rejected")

    def timeout(prompt):
        raise httpx.ReadTimeout("slow")
    assert answers.rewrite(form, answer, call=timeout) == ("You had 3 charges.", "skipped")


def test_the_rewrite_prompt_names_the_type_and_never_contains_the_facts():
    prompt = answers.rewrite_prompt("total_spent", 2)
    assert "[F1]" in prompt and "$" not in prompt
    assert "comparison" in answers.rewrite_prompt("compare", 3)


def test_substitution_is_single_pass():
    form = Form(type="search", search_text="tacos [F2]", period=Range(None, None, "all"))
    answer = answers.Answer("x", ['I found 1 charge that looks like "tacos [F2]"', "it comes to $7.00"], None,
                            {"capped": False})
    text, status = answers.rewrite(form, answer, call=lambda p: "Here is what I found: [F1], and [F2].")
    assert status == "accepted"
    assert text.startswith('Here is what I found: I found 1 charge that looks like "tacos [F2]", and it comes')


def test_search_singular_grammar_and_cap_caveat():
    form = Form(type="search", search_text="parking", period=Range(None, None, "all"))
    one = answers.render(form, Result("search", {"count": 1, "total": D("3"), "direction": "spent",
                                                 "capped": False}), None)
    assert 'I found 1 charge that looks like "parking"' in one.exact and "it comes to $3.00" in one.exact
    capped = answers.render(form, Result("search", {"count": 50, "total": D("3"), "direction": "spent",
                                                    "capped": True}), None)
    assert f"Only the {rs.SEARCH_CAP} closest matches are counted" in capped.exact


def test_scope_states_every_applied_filter():
    form = Form(type="total_spent", category="Groceries", merchant_text="costco", merchants=("Costco",), period=NOV)
    result = Result("total_spent", {"total": D("25.00"), "count": 1, "direction": "spent"})
    assert answers.render(form, result, None).facts[0] == "you spent $25.00 on Groceries at Costco in Nov 2026"


@pytest.mark.parametrize("text", ["[F1] × [F2].", "[F1] and ½ [F2].", "[F1] and [F2] %."])
def test_glue_outside_plain_ascii_is_rejected(text):
    assert not answers.check_rewrite(text, 2)
