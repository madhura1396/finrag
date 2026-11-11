"""Plain data types shared by the read path (ADR 0004). No I/O imports, so the
pure calculations can use them without pulling in HTTP or the database."""

import calendar
from dataclasses import dataclass, field
from datetime import date

from app import read_settings as rs


@dataclass(frozen=True)
class Range:
    start: date | None          # None: all data
    end: date | None
    label: str
    said: bool = True           # False when a default chose it

    @property
    def is_all(self) -> bool:
        return self.start is None

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


ALL = Range(None, None, "all data", said=False)


@dataclass(frozen=True)
class Trip:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class Form:
    type: str
    direction: str = "spent"
    group_by: str = "merchant"
    order: str = "most"
    limit: int = rs.DEFAULT_LIMIT
    category: str | None = None
    merchant_text: str | None = None
    merchants: tuple[str, ...] = ()
    trip: str | None = None
    search_text: str | None = None
    period: Range = ALL
    period_2: Range | None = None


@dataclass(frozen=True)
class Refusal:
    kind: str                   # 'unsupported', 'not_understood' or 'data'
    reason: str
    detail: dict = field(default_factory=dict)


def month_range(year: int, month: int, said: bool = True) -> Range:
    last = calendar.monthrange(year, month)[1]
    return Range(date(year, month, 1), date(year, month, last), f"{calendar.month_abbr[month]} {year}", said)
