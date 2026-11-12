"""Load a bank layout profile (ADR 0002, decision 3; ADR 0003, 5d).

All bank-specific strings live in profiles/*.yaml; app/ holds none. Loading is
strict: unknown keys, missing keys and wrong types are rejected here, at load
time, rather than surfacing later as a silent misparse (the lesson of #6 and
#8). That also neutralises YAML's implicit typing: an unquoted `no` becomes
False, which is not a string, so it is refused.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.statement import SECTIONS

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
COLUMN_KEYS = ("trans_date", "post_date", "description", "credit", "debit")


class ProfileError(ValueError):
    """The profile file is malformed. Fix the file; never guess."""


@dataclass(frozen=True)
class Section:
    key: str
    heading: str
    total: re.Pattern


@dataclass(frozen=True)
class SummaryLine:
    key: str
    label: str
    sign: int
    section: str | None


@dataclass(frozen=True)
class Profile:
    name: str
    version: int
    period_pattern: re.Pattern
    date_format: str
    header: tuple[str, ...]
    columns: dict[str, str]          # column key -> header label
    y_tolerance: float
    sections: tuple[Section, ...]
    undated_rows: frozenset[str]
    summary_start: str
    summary_end: str
    summary_anchor: str
    summary_lines: tuple[SummaryLine, ...]
    header_continuation: tuple[str, ...] = ()   # exact lines under a stacked header


def _expect(mapping, keys: set[str], where: str, optional: set[str] = frozenset()) -> None:
    if not isinstance(mapping, dict):
        raise ProfileError(f"{where} must be a mapping")
    unknown = set(mapping) - keys - optional
    missing = keys - set(mapping)
    if unknown:
        raise ProfileError(f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise ProfileError(f"{where}: missing keys {sorted(missing)}")


def _str(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{where} must be a non-empty string, got {type(value).__name__}")
    return value


def _regex(value, where: str) -> re.Pattern:
    try:
        return re.compile(_str(value, where), re.IGNORECASE)
    except re.error as e:
        raise ProfileError(f"{where} is not a valid regex: {e}") from e


def parse(data) -> Profile:
    _expect(data, {"profile", "version", "period", "table", "sections", "undated_rows", "summary"}, "profile")
    if not isinstance(data["version"], int) or isinstance(data["version"], bool):
        raise ProfileError("version must be an integer")

    _expect(data["period"], {"pattern", "date_format"}, "period")
    period = _regex(data["period"]["pattern"], "period.pattern")
    if period.groups != 2:
        raise ProfileError("period.pattern must have exactly two groups (start, end)")
    date_format = _str(data["period"]["date_format"], "period.date_format")

    table = data["table"]
    _expect(table, {"header", "columns", "y_tolerance"}, "table", optional={"header_continuation"})
    if not isinstance(table["header"], list) or not table["header"]:
        raise ProfileError("table.header must be a list of labels")
    header = tuple(_str(h, "table.header[]") for h in table["header"])
    _expect(table["columns"], set(COLUMN_KEYS), "table.columns")
    columns = {k: _str(v, f"table.columns.{k}") for k, v in table["columns"].items()}
    for key, label in columns.items():
        if label not in header:
            raise ProfileError(f"table.columns.{key} '{label}' is not in table.header")
    continuation = table.get("header_continuation", [])
    if not isinstance(continuation, list):
        raise ProfileError("table.header_continuation must be a list of lines")
    continuation = tuple(_str(c, "table.header_continuation[]") for c in continuation)
    if isinstance(table["y_tolerance"], bool) or not isinstance(table["y_tolerance"], (int, float)) \
            or not 0 < table["y_tolerance"] < 20:
        raise ProfileError("table.y_tolerance must be a number of points between 0 and 20")

    if not isinstance(data["sections"], list) or not data["sections"]:
        raise ProfileError("sections must be a list")
    sections = []
    for i, s in enumerate(data["sections"]):
        _expect(s, {"key", "heading", "total"}, f"sections[{i}]")
        if s["key"] not in SECTIONS:
            raise ProfileError(f"sections[{i}].key {s['key']!r} is not one of {sorted(SECTIONS)}")
        sections.append(Section(s["key"], _str(s["heading"], f"sections[{i}].heading"),
                                _regex(s["total"], f"sections[{i}].total")))
    if len({s.key for s in sections}) != len(sections):
        raise ProfileError("sections: duplicate key")

    if not isinstance(data["undated_rows"], list):
        raise ProfileError("undated_rows must be a list")
    undated = frozenset(_str(k, "undated_rows[]") for k in data["undated_rows"])
    if not undated <= {s.key for s in sections}:
        raise ProfileError("undated_rows names a section the profile does not define")

    summary = data["summary"]
    _expect(summary, {"start_label", "end_label", "right_column_anchor", "lines"}, "summary")
    if not isinstance(summary["lines"], dict) or "new_balance" not in summary["lines"]:
        raise ProfileError("summary.lines must be a mapping that includes new_balance")
    lines = []
    for key, line in summary["lines"].items():
        _expect(line, {"label", "sign"}, f"summary.lines.{key}", optional={"section"})
        sign = line["sign"]
        allowed = (0,) if key == "new_balance" else (1, -1)
        if isinstance(sign, bool) or sign not in allowed:
            raise ProfileError(f"summary.lines.{key}.sign must be one of {allowed}")
        section = line.get("section")
        if section is not None and section not in SECTIONS:
            raise ProfileError(f"summary.lines.{key}.section {section!r} is not a known section")
        lines.append(SummaryLine(key, _str(line["label"], f"summary.lines.{key}.label"), sign, section))

    return Profile(
        name=_str(data["profile"], "profile"), version=data["version"], period_pattern=period,
        date_format=date_format,
        header=header, columns=columns, y_tolerance=float(table["y_tolerance"]), sections=tuple(sections),
        undated_rows=undated, summary_start=_str(summary["start_label"], "summary.start_label"),
        summary_end=_str(summary["end_label"], "summary.end_label"),
        summary_anchor=_str(summary["right_column_anchor"], "summary.right_column_anchor"),
        summary_lines=tuple(lines), header_continuation=continuation,
    )


def load(path: Path) -> Profile:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ProfileError(f"{Path(path).name} is not valid YAML") from e
    return parse(data)


def load_single(directory: Path = PROFILES_DIR) -> Profile:
    """The one profile in profiles/ (ADR 0002, decision 3: no selection logic yet).
    Zero or several is an error, raised at app startup."""
    files = sorted(Path(directory).glob("*.yaml"))
    if len(files) != 1:
        raise ProfileError(f"expected exactly one profile in {directory}, found {len(files)}")
    return load(files[0])
