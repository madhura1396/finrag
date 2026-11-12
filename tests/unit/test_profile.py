"""The strict profile loader (plan PR2). Built from the real profile file, then broken."""

import copy
from pathlib import Path

import pytest
import yaml

from app import profile as prof

REAL = prof.PROFILES_DIR / "wells_fargo.credit_card.v1.yaml"


def base() -> dict:
    return yaml.safe_load(REAL.read_text())


def test_the_real_profile_loads():
    p = prof.load(REAL)
    assert {s.key for s in p.sections} == {"payments", "other_credits", "purchases", "fees", "interest"}
    assert p.undated_rows == {"interest", "fees"}


@pytest.mark.parametrize("break_it", [
    lambda d: d.update(surprise=1),                                     # unknown top-level key
    lambda d: d["summary"].pop("right_column_anchor"),                  # missing key
    lambda d: d["table"].__setitem__("header", "Card Ending in"),       # string where a list is expected
    lambda d: d["sections"][0].__setitem__("key", "groceries"),         # not a canonical section
    lambda d: d["summary"]["lines"]["payments"].__setitem__("sign", True),
    lambda d: d["period"].__setitem__("pattern", "Statement (\\d+)"),   # needs two groups
    lambda d: d["table"]["columns"].__setitem__("credit", "Money In"),  # not in the header
    lambda d: d["undated_rows"].append("cash_advances"),                # not a defined section
    lambda d: d["summary"]["lines"].pop("new_balance"),
])
def test_malformed_profiles_are_rejected(break_it):
    data = base()
    break_it(data)
    with pytest.raises(prof.ProfileError):
        prof.parse(data)


def test_unquoted_no_in_yaml_is_a_type_error(tmp_path):
    text = REAL.read_text().replace("heading: 'Payments'", "heading: no")
    path = tmp_path / "p.yaml"
    path.write_text(text)
    with pytest.raises(prof.ProfileError):
        prof.load(path)


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_profile_is_required(tmp_path, count):
    for i in range(count):
        (tmp_path / f"p{i}.yaml").write_text(REAL.read_text())
    with pytest.raises(prof.ProfileError):
        prof.load_single(tmp_path)


def test_app_code_contains_no_profile_strings():
    data = base()
    strings = [s["heading"] for s in data["sections"]] + [l["label"] for l in data["summary"]["lines"].values()]
    strings += [data["summary"]["right_column_anchor"], "Statement Period", "FOR THIS PERIOD"]
    app = Path(prof.__file__).parent
    for module in ("statement.py", "profile.py", "reader.py"):
        source = (app / module).read_text()
        for s in strings:
            assert s not in source, f"{module} contains profile string {s!r}"
