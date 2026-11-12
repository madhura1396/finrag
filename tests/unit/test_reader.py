"""The position-based reader on synthetic statements (plan PR2). No DB, no model."""

from dataclasses import replace
from datetime import date
from decimal import Decimal

import fitz
import pytest

from app.profile import load_single
from app.reader import read_statement
from app.statement import StatementRejected, validate
from tests.fixtures.make_wells_fargo import Spec, Txn, make_statement, section_total, standard_spec

D = Decimal
PROFILE = load_single()


def read(tmp_path, spec, name="s.pdf"):
    path = tmp_path / name
    truth = make_statement(path, spec)
    return read_statement(path, PROFILE), truth


def checks(data):
    return {(m.check, m.section) for m in validate(data)}


def rows_of(data):
    return [(r.trans_date, r.description, r.amount, r.section) for r in data.rows]


# --- a valid statement --------------------------------------------------------------

def test_valid_statement_matches_ground_truth_and_reconciles(tmp_path):
    data, truth = read(tmp_path, standard_spec())
    assert rows_of(data) == truth
    assert validate(data) == []


def test_continuation_pages_neither_duplicate_nor_drop_rows(tmp_path):
    data, truth = read(tmp_path, standard_spec(rows_per_page=3))
    assert rows_of(data) == truth and validate(data) == []


def test_layout_features(tmp_path):
    data, _ = read(tmp_path, standard_spec())
    by_desc = {r.description: r for r in data.rows}
    assert "FOODDASH*TACO TOWN SPRINGFIELD ORDER 7781" in by_desc                  # wrapped description
    assert by_desc["MEGASTORE RETURN"].amount == D("19.99")                       # Credits: positive
    assert by_desc["CORNER CAFE 4455 RIVERTON"].amount == D("-8.40")              # Charges: negative
    assert by_desc["ONLINE PAYMENT THANK YOU"].section == "payments"               # blank card cell
    interest = [r for r in data.rows if r.section == "interest"]
    assert {r.trans_date for r in interest} == {data.period_end}                  # undated -> closing date
    assert D("0.00") in {r.amount for r in interest}                              # 0.00 row kept
    assert data.section_totals["fees"] == D("0.00")                               # total only, no rows
    assert data.section_totals["interest"] == D("12.34")                          # not the YTD box's 99.99
    assert all("IN 2026" not in r.description and "Year-to-Date" not in r.description for r in data.rows)
    assert data.summary["purchases"].value == section_total(standard_spec(), "purchases")   # wrapped label
    assert data.summary["previous_balance"].value == D("100.00")                  # right column not mixed in


def test_december_to_january_years(tmp_path):
    spec = Spec(date(2025, 12, 3), date(2026, 1, 2), [
        Txn("purchases", date(2025, 12, 28), "GROCER MART", D("-10.00")),
        Txn("purchases", date(2026, 1, 1), "CORNER CAFE", D("-5.00")),
    ])
    data, truth = read(tmp_path, spec)
    assert rows_of(data) == truth and validate(data) == []


def test_february_29_in_a_leap_year(tmp_path):
    spec = Spec(date(2028, 2, 3), date(2028, 3, 2), [Txn("purchases", date(2028, 2, 29), "GROCER MART", D("-10.00"))])
    data, truth = read(tmp_path, spec)
    assert rows_of(data) == truth and validate(data) == []


# --- broken statements are rejected with the right check -------------------------------

def test_dropped_row(tmp_path):
    spec = standard_spec()
    spec = replace(spec, totals={"purchases": section_total(spec, "purchases") + D("10.00")},
                   summary={"purchases": section_total(spec, "purchases") + D("10.00")})
    assert ("section_total", "purchases") in checks(read(tmp_path, spec)[0])


def test_total_off_by_one_cent(tmp_path):
    spec = standard_spec()
    spec = replace(spec, totals={"payments": D("250.01")})
    assert ("section_total", "payments") in checks(read(tmp_path, spec)[0])


def test_missing_total_line(tmp_path):
    assert ("section_total", "purchases") in checks(read(tmp_path, standard_spec(totals={"purchases": None}))[0])


def test_missing_summary(tmp_path):
    found = checks(read(tmp_path, standard_spec(include_summary=False))[0])
    assert ("summary_not_found", None) in found and ("no_printed_totals", None) in found


def test_missing_summary_anchor(tmp_path):
    assert ("summary_not_found", None) in checks(read(tmp_path, standard_spec(include_anchor=False))[0])


def test_summary_equation_off_by_one_cent(tmp_path):
    spec = standard_spec()
    from tests.fixtures.make_wells_fargo import summary_values
    spec = replace(spec, summary={"new_balance": summary_values(spec)["new_balance"] + D("0.01")})
    assert checks(read(tmp_path, spec)[0]) == {("summary_equation", None)}


def test_date_eight_days_before_the_period(tmp_path):
    spec = standard_spec()
    spec.rows.append(Txn("purchases", date(2026, 8, 26), "EARLY SHOP", D("-1.00")))
    assert ("date_window", "purchases") in checks(read(tmp_path, spec)[0])


def test_zero_rows(tmp_path):
    spec = Spec(date(2026, 9, 3), date(2026, 10, 2), [])
    assert ("no_rows", None) in checks(read(tmp_path, spec)[0])


def test_unparseable_amount(tmp_path):
    spec = standard_spec()
    spec.rows[4] = replace(spec.rows[4], raw_amount="8.4O")
    assert ("row_invalid", "purchases") in checks(read(tmp_path, spec)[0])


def test_unreadable_interest_date_is_invalid_not_closing_date(tmp_path):
    spec = standard_spec()
    spec.rows[7] = replace(spec.rows[7], raw_date="O9/3O")
    data, _ = read(tmp_path, spec)
    assert ("row_invalid", "interest") in checks(data)
    assert not any(r.description == "INTEREST CHARGE ON PURCHASES" for r in data.rows)


def test_unknown_summary_label(tmp_path):
    assert ("unknown_summary_label", None) in checks(read(tmp_path, standard_spec(extra_summary_label="Bonus Points"))[0])


def test_period_differs_on_page_two(tmp_path):
    spec = standard_spec(page2_period=(date(2026, 9, 4), date(2026, 10, 2)))
    assert ("period_mismatch", None) in checks(read(tmp_path, spec)[0])


def test_nonzero_cash_advances(tmp_path):
    spec = standard_spec(cash_advances=D("40.00"))
    assert ("summary_section", "cash_advances") in checks(read(tmp_path, spec)[0])


def test_no_text_layer_is_rejected(tmp_path):
    path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    with pytest.raises(StatementRejected) as e:
        read_statement(path, PROFILE)
    assert e.value.mismatches[0].check == "no_text_layer"


def test_mismatches_never_echo_cell_text(tmp_path):
    spec = standard_spec()
    spec.rows[4] = replace(spec.rows[4], raw_amount="8.4O")
    data, _ = read(tmp_path, spec)
    text = " ".join(f"{m.expected} {m.actual}" for m in data.reader_mismatches)
    assert "8.4O" not in text and "CORNER" not in text


# --- review round: dates and descriptions --------------------------------------------

def test_shifted_columns_reject_instead_of_storing_the_post_date(tmp_path):
    # Data cells sit left of their header labels: the trans date lands under "Card
    # Ending in" and the post date under "Trans Date". Rejected, never stored.
    spec = standard_spec(data_offset={"card": -30, "trans": -52, "post": -46})
    data, _ = read(tmp_path, spec)
    assert ("row_invalid", "purchases") in checks(data)
    assert not any(r.section == "purchases" for r in data.rows)


def test_a_post_date_alone_is_not_treated_as_undated(tmp_path):
    spec = standard_spec()
    spec.rows[7] = replace(spec.rows[7], raw_date="")      # trans date empty...
    path = tmp_path / "s.pdf"
    make_statement(path, spec)
    doc = fitz.open(str(path))
    # ...but a post date printed for the interest row
    for page in doc:
        for inst in page.search_for("INTEREST CHARGE ON PURCHASES"):
            page.insert_text((140, inst.y1 - 1.5), "10/02", fontsize=8, fontname="helv")
    doc.save(str(tmp_path / "s2.pdf"))
    data = read_statement(tmp_path / "s2.pdf", PROFILE)
    assert ("row_invalid", "interest") in checks(data)


def test_page_footer_is_never_glued_onto_a_description(tmp_path):
    path = tmp_path / "s.pdf"
    make_statement(path, standard_spec(rows_per_page=4))
    doc = fitz.open(str(path))
    for page in doc:
        page.insert_text((290, 760), "Continued on next page footer", fontsize=8, fontname="helv")
    doc.save(str(tmp_path / "f.pdf"))
    data = read_statement(tmp_path / "f.pdf", PROFILE)
    assert all("footer" not in r.description.lower() for r in data.rows)


def test_the_profile_date_format_reaches_the_reader(tmp_path):
    from dataclasses import replace as dc_replace
    path = tmp_path / "s.pdf"
    make_statement(path, standard_spec())                  # printed 09/03/2026 to 10/02/2026
    with pytest.raises(StatementRejected) as e:            # read day-first, the period reverses
        read_statement(path, dc_replace(PROFILE, date_format="%d/%m/%Y"))
    assert e.value.mismatches[0].check == "period_not_found"


def test_undated_fee_rows_get_the_closing_date(tmp_path):
    spec = standard_spec()
    spec.rows.append(Txn("fees", None, "LATE FEE", D("-29.00")))
    data, truth = read(tmp_path, spec)
    assert rows_of(data) == truth and validate(data) == []


def test_summary_label_wrapped_before_its_amount(tmp_path):
    # The older layout: label continues onto the line that carries the amount.
    data, truth = read(tmp_path, standard_spec(purchases_amount_first_line=False))
    assert rows_of(data) == truth and validate(data) == []
    assert data.summary["purchases"].value == section_total(standard_spec(), "purchases")


# --- review round: stacked header and summary wrap -----------------------------------

def test_a_one_line_header_still_reads(tmp_path):
    # The profile's continuation lines are optional: absent, nothing is skipped.
    data, truth = read(tmp_path, standard_spec(stacked_header=False))
    assert rows_of(data) == truth and validate(data) == []


def test_header_continuation_is_skipped_only_directly_under_the_header(tmp_path):
    # The same words further down are not skipped: here they sit inside the
    # purchases section and are rejected as a row without an amount.
    path = tmp_path / "s.pdf"
    make_statement(path, standard_spec())
    doc = fitz.open(str(path))
    page = doc[0]
    inst = page.search_for("GROCER MART")[0]
    page.insert_text((92, inst.y1 + 5), "Date", fontsize=8, fontname="helv")
    page.insert_text((140, inst.y1 + 5), "Date", fontsize=8, fontname="helv")
    doc.save(str(tmp_path / "s2.pdf"))
    assert ("row_invalid", "purchases") in checks(read_statement(tmp_path / "s2.pdf", PROFILE))


def _summary_lines(*rows):
    """Lines for reader._summary: each row is (label words, amount or None)."""
    from app.reader import Line, Word
    lines = [Line(0, [Word(46, 0, 80, "Previous"), Word(82, 0, 110, "Balance"), Word(230, 0, 260, "$100.00"),
                      Word(330, 0, 350, "Total"), Word(352, 0, 370, "Credit"), Word(372, 0, 390, "Limit")])]
    for i, (label, amount) in enumerate(rows, start=1):
        words, x = [], 46.0
        for w in label.split():
            words.append(Word(x, i * 12, x + 20, w)); x += 22
        if amount:
            words.append(Word(230, i * 12, 260, amount))
        lines.append(Line(0, words))
    return [lines]


def test_summary_attach_needs_the_completing_words_on_the_very_next_line():
    from app.reader import _summary
    values, mismatches = _summary(_summary_lines(
        ("Purchases, Balance Transfers &", "$5.00"), ("see note", None), ("Other Charges", None),
        ("New Balance", "$105.00")), PROFILE)
    assert "purchases" not in values
    assert "unknown_summary_label" in {m.check for m in mismatches}


def test_summary_unknown_label_with_a_non_completing_line_is_rejected():
    from app.reader import _summary
    values, mismatches = _summary(_summary_lines(
        ("Bonus Points", "$1.00"), ("Other Charges", None), ("New Balance", "$101.00")), PROFILE)
    assert "unknown_summary_label" in {m.check for m in mismatches}
