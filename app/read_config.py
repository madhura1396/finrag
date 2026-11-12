"""Read-path settings and plain data types (ADR 0004).

Every tunable number on the read path lives here (ADR 0004 Consequences). All
values are unverified starting points, to be tuned once real history is
uploaded. Nothing else on the read path hard-codes a threshold.

The data types (Range, Form, Refusal) sit beside them. No I/O or
database imports, so the pure calculations can use this module without
pulling in HTTP or the database.
"""

import calendar
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

FORM_TIMEOUT_SECONDS = 30        # model unreachable or slower -> 503
REWRITE_TIMEOUT_SECONDS = 15     # slower -> the exact answer is shown

DEFAULT_LIMIT = 5
MAX_LIMIT = 50
LIST_CAP = 50                    # rows returned to the client
SEARCH_CUTOFF = 0.75             # cosine distance; unverified at cap 50
SEARCH_CAP = 50

MERCHANT_SIMILARITY = 0.90       # difflib ratio against the whole name or any word
MERCHANT_MIN_CONTAINS = 3        # typed text shorter than this never matches by containment
CATEGORY_NEAR = 0.75             # an unknown category this close to a category name is not tried as a merchant
CLOSEST_NAMES = 3

ANOMALY_RATIO = Decimal("1.5")           # spend above ratio x median of previous full months
ANOMALY_MIN_EXCESS = Decimal("50.00")    # ...and at least this far above it
ANOMALY_MIN_HISTORY_MONTHS = 3
RECURRING_MIN_MONTHS = 3                 # same merchant in at least 3...
RECURRING_WINDOW_MONTHS = 4              # ...of the last 4 full months

PENDING_BUCKET = "Uncategorised (pending)"

REWRITE_MAX_CHARS = 300

# Words the placeholder rewrite may not use outside a token (ADR 0004, detail 3).
# Closed: a word is allowed unless it is listed. Growing it is a reviewed change.
GLUE_BANNED_WORDS = frozenset({
    # comparison and ranking
    "more", "less", "fewer", "up", "down", "higher", "lower", "most", "least", "biggest", "smallest",
    "largest", "top", "bottom", "increase", "increased", "decrease", "decreased", "rise", "rose", "fell",
    "drop", "dropped", "than", "better", "worse",
    # negation
    "not", "no", "never", "none", "without", "nothing", "n't",
    # quantities and numbers
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "dozen", "hundred", "thousand", "million", "twice", "double", "triple", "half", "third", "quarter",
    "percent", "single", "once",
    # approximators
    "about", "around", "almost", "nearly", "roughly", "approximately", "over", "under", "above", "below",
    "just", "only",
    # exclusions
    "except", "excluding", "besides", "other",
    # direction verbs
    "spent", "spend", "spending", "paid", "pay", "received", "receive", "earned", "earn", "saved", "save",
    "refunded", "refund", "charged", "charge", "owe", "owed", "cost", "costs",
    # rates and frequency (added after review: "[F1] every day" changes the fact)
    "per", "each", "every", "daily", "weekly", "monthly", "yearly", "annually", "average", "averaged",
    "typically", "usually", "always", "often", "rarely",
    # more negation and doubt
    "cannot", "neither", "nor", "hardly", "barely", "scarcely", "unlikely", "doubt", "false", "wrong",
    # conditionals and hedges
    "if", "unless", "maybe", "perhaps", "might", "would", "could", "should", "possibly", "probably",
    "estimated", "estimate",
    # judgement about a figure
    "good", "bad", "great", "nice", "luckily", "unfortunately", "sadly", "well",
    # alternatives, arithmetic and limits (added in the final review of the read-path PR)
    "or", "instead", "minus", "plus", "until", "apart", "lacking", "supposedly", "excluded", "included",
})


def refused_log_path() -> Path:
    """Where refused questions are appended. Outside the repo, never committed,
    and never read by Claude (.claude/CLAUDE.md, Data safety)."""
    return Path(os.path.expanduser(os.getenv("FINRAG_REFUSED_LOG", "~/.finrag/refused_questions.log")))


# ---- Plain read-path data types ----

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
class Form:
    type: str
    direction: str = "spent"
    group_by: str = "merchant"
    order: str = "most"
    limit: int = DEFAULT_LIMIT
    category: str | None = None
    merchant_text: str | None = None
    merchants: tuple[str, ...] = ()
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
