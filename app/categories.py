"""Seed data for the categories dimension (ADR 0005). Nothing reads this at
runtime except app.db_setup.seed, which runs only on an empty database.

System categories are identified by a stable `system_key`, not a fixed numeric
ID, so seeding never fights the ID sequence.
"""

# (system_key, name). These cannot be renamed, merged, deleted, targeted by
# rules or pins, or offered to the model.
SYSTEM_CATEGORIES = [
    ("fees",     "Fees"),       # fixed by statement section (ADR 0003, 2b)
    ("interest", "Interest"),   # fixed by statement section (ADR 0003, 2b)
    ("other",    "Other"),      # the model's none_fit fallback
]

# User-owned defaults. After seeding they belong to the user: renamed, merged
# or archived defaults are never re-created.
DEFAULT_CATEGORIES = [
    "Groceries",
    "Dining & DoorDash",
    "Travel & Transport",
    "Shopping",
    "Telephone & Internet",
    "Entertainment",
    "Health & Fitness",
]

# Statement sections whose rows get a fixed system category.
SECTION_CATEGORY_KEYS = {"fees": "fees", "interest": "interest"}

# Category names that would collide with label states.
RESERVED_NAMES = {"none_fit", "pending"}
