"""User-owned categories and merchant labels (ADR 0005). The write side.

Every label comes from exactly one source, decided per field (category and
merchant independently), strongest first:

    section > pin > rule (by priority) > memory > model

Every change (rules, categories, memory, pins, undo) goes through one
pipeline: mutate the entities, then re-resolve every transaction's labels,
writing append-only SCD2 history for each row that changed. preview() runs
that pipeline inside a savepoint and rolls it back; apply() runs it for real
under a lock on data_version, after checking the preview token.

Functions take the session they act on. Callers get sessions from
app.sessions.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app import category_store
from app.categories import RESERVED_NAMES, SECTION_CATEGORY_KEYS
from app.embedding_text import build_embedded_text, is_stale
from app.llm import NONE_FIT, ModelLabel, label_descriptions
from app.models import (
    CategoryId, CategoryRule, CategoryVersion, ChangeLog, DataVersion, LabelMemory,
    ReviewItem, Transaction, TransactionLabelHistory,
)

MAX_PATTERN_LENGTH = 200
MAX_NAME_LENGTH = 60
MAX_MERCHANT_LENGTH = 80
PREVIEW_SAMPLES = 5
EMBED_BATCH_SIZE = 32
NO_PREVIEW_KINDS = {"pin", "unpin"}      # single-row changes: no token needed

# A group that contains a quantifier and is itself quantified: (a+)+, (\w*)*, (x+){2,}.
# The classic catastrophic-backtracking shape. Not exhaustive (ADR 0005, 2b).
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)\s*[+*{]")


class ChangeRejected(Exception):
    """The change is invalid as asked. Maps to 422."""

    def __init__(self, reason: str, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class StalePreview(Exception):
    """Data changed since the preview, or the payload differs. Maps to 409."""


def _now() -> datetime:
    # The timestamp columns are `without time zone`: store naive UTC so values
    # do not depend on the session time zone.
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------
# data_version and preview tokens
# --------------------------------------------------------------------------

def current_version(session: Session) -> int:
    return session.scalar(select(DataVersion.version).where(DataVersion.id == 1))


def lock_version(session: Session) -> int:
    """Serialise writers: every apply, upload and enrichment write takes this."""
    return session.scalar(select(DataVersion.version).where(DataVersion.id == 1).with_for_update())


def bump_version(session: Session) -> None:
    session.execute(update(DataVersion).where(DataVersion.id == 1).values(version=DataVersion.version + 1))


def make_token(payload: dict, version: int) -> str:
    """Binds the exact payload to the data version at preview time."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{version}|{canonical}".encode()).hexdigest()[:32]
    return f"{version}:{digest}"


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def validate_pattern(pattern: str, kind: str) -> None:
    if kind not in ("contains", "regex"):
        raise ChangeRejected("rule kind must be 'contains' or 'regex'")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ChangeRejected("rule pattern is empty")
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ChangeRejected(f"rule pattern is longer than {MAX_PATTERN_LENGTH} characters")
    if kind == "contains" and not category_store.normalise(pattern):
        raise ChangeRejected("rule pattern has no letters or digits")
    if kind == "regex":
        if _NESTED_QUANTIFIER.search(pattern):
            raise ChangeRejected("regex has nested repetition, which can hang matching; simplify it")
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ChangeRejected(f"invalid regex: {e}") from e


def rule_matches(rule: CategoryRule, raw_description: str) -> bool:
    if rule.kind == "contains":
        return category_store.normalise(rule.pattern) in category_store.normalise(raw_description)
    return re.search(rule.pattern, raw_description, re.IGNORECASE) is not None


def _target(rule: CategoryRule, field: str):
    return rule.sets_category_id if field == "category" else rule.sets_merchant


def choose_rule(rules: list[CategoryRule], raw_description: str, field: str):
    """Highest priority wins for this field. Equal-priority rules with different
    targets are a conflict: none applies. Returns (rule, conflict_rule_ids)."""
    matching = [r for r in rules if _target(r, field) is not None and rule_matches(r, raw_description)]
    if not matching:
        return None, None
    top = max(r.priority for r in matching)
    tops = [r for r in matching if r.priority == top]
    if len({_target(r, field) for r in tops}) > 1:
        return None, [{"rule_id": r.rule_id, "target": _target(r, field)} for r in sorted(tops, key=lambda r: r.rule_id)]
    return min(tops, key=lambda r: r.rule_id), None


# --------------------------------------------------------------------------
# Labels and resolution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Labels:
    category_id: int | None
    category_source: str
    category_rule_id: int | None
    merchant: str
    merchant_source: str
    merchant_rule_id: int | None


def labels_of(tx: Transaction) -> Labels:
    return Labels(tx.category_id, tx.category_source, tx.category_rule_id,
                  tx.merchant, tx.merchant_source, tx.merchant_rule_id)


@dataclass(frozen=True)
class MemorySnapshot:
    category_id: int | None
    merchant: str | None


@dataclass
class Context:
    rules: list[CategoryRule]
    # Snapshots, not ORM objects: memory written during an enrichment pass must
    # not change how later rows of the same pass resolve (decision C).
    memory: dict[str, MemorySnapshot]
    section_ids: dict[str, int]       # section -> system category id
    other_id: int
    archived: frozenset = frozenset()


def load_context(session: Session) -> Context:
    rules = session.scalars(
        select(CategoryRule).where(CategoryRule.deleted.is_(False)).order_by(CategoryRule.rule_id)
    ).all()
    memory = {m.description_key: MemorySnapshot(m.category_id, m.merchant)
              for m in session.scalars(select(LabelMemory)).all()}
    section_ids = {section: category_store.system_category_id(session, key)
                   for section, key in SECTION_CATEGORY_KEYS.items()}
    archived = frozenset(c.category_id for c in category_store.current_categories(session) if c.archived)
    return Context(list(rules), memory, section_ids, category_store.system_category_id(session, "other"), archived)


def resolve(tx: Transaction, ctx: Context) -> tuple[Labels, dict[str, list[dict]]]:
    """The labels this row should have now, without calling the model.
    A field nothing decides is 'pending'. Returns (labels, conflicts by field)."""
    key = category_store.description_key(tx.raw_description)
    memory = ctx.memory.get(key)
    conflicts: dict[str, list[dict]] = {}

    # Category
    if tx.section in ctx.section_ids:
        category = (ctx.section_ids[tx.section], "section", None)
    elif tx.category_source == "pin":
        category = (tx.category_id, "pin", None)
    else:
        rule, conflict = choose_rule(ctx.rules, tx.raw_description, "category")
        if conflict:
            conflicts["category"] = conflict
        if rule is not None:
            category = (rule.sets_category_id, "rule", rule.rule_id)
        elif memory is not None and memory.category_id is not None and memory.category_id not in ctx.archived:
            keep_model = tx.category_source == "model" and tx.category_id == memory.category_id
            category = (memory.category_id, "model" if keep_model else "memory", None)
        elif tx.category_source in ("model", "none_fit") and tx.category_id not in ctx.archived:
            category = (tx.category_id, tx.category_source, None)
        else:
            # Nothing decides it, or its category was archived: the model relabels it.
            category = (None, "pending", None)

    # Merchant
    if tx.section in ctx.section_ids:
        merchant = (tx.raw_description, "section", None)
    elif tx.merchant_source == "pin":
        merchant = (tx.merchant, "pin", None)
    else:
        rule, conflict = choose_rule(ctx.rules, tx.raw_description, "merchant")
        if conflict:
            conflicts["merchant"] = conflict
        if rule is not None:
            merchant = (rule.sets_merchant, "rule", rule.rule_id)
        elif memory is not None and memory.merchant is not None:
            keep_model = tx.merchant_source == "model" and tx.merchant == memory.merchant
            merchant = (memory.merchant, "model" if keep_model else "memory", None)
        elif tx.merchant_source == "model":
            merchant = (tx.merchant, "model", None)
        else:
            # Until labelled, the merchant is the printed description, never a
            # leftover rule or pin value (decision C).
            merchant = (tx.raw_description, "pending", None)

    return Labels(*category, *merchant), conflicts


def open_history(session: Session, tx: Transaction, change_id: int | None, at: datetime | None = None) -> None:
    labels = labels_of(tx)
    session.add(TransactionLabelHistory(transaction_id=tx.id, change_id=change_id,
                                        valid_from=at or _now(), **asdict(labels)))


def set_labels(session: Session, tx: Transaction, new: Labels, change_id: int | None,
               names: dict[int, str]) -> bool:
    """Write labels if they changed: close the open history version, open a new
    one, and update the current columns. Append-only."""
    if labels_of(tx) == new:
        return False
    at = _now()     # one instant: the closed version ends where the new one starts
    session.execute(
        update(TransactionLabelHistory)
        .where(TransactionLabelHistory.transaction_id == tx.id, TransactionLabelHistory.valid_to.is_(None))
        .values(valid_to=at)
    )
    for field, value in asdict(new).items():
        setattr(tx, field, value)
    session.flush()
    open_history(session, tx, change_id, at)
    return True


def review_kinds(labels: Labels, conflicts: dict, *, after_model: bool) -> set[str]:
    """`pending` is a review kind only once the model has tried and failed:
    a row that has merely fallen through is waiting, not stuck."""
    kinds = set()
    if after_model and (labels.category_source == "pending" or labels.merchant_source == "pending"):
        kinds.add("pending")
    if labels.category_source == "none_fit":
        kinds.add("none_fit")
    if conflicts:
        kinds.add("conflict")
    return kinds


def sync_review(session: Session, tx: Transaction, kinds: set[str], details: dict[str, dict],
                change_id: int | None, *, keep_pending: bool = False) -> None:
    """Open items for kinds that now apply, resolve those that no longer do.
    keep_pending leaves an existing `pending` item open while the row is still
    pending (a relabel that did not decide it is not a resolution)."""
    open_items = session.scalars(
        select(ReviewItem).where(ReviewItem.transaction_id == tx.id, ReviewItem.resolved_at.is_(None))
    ).all()
    if keep_pending and "pending" in (tx.category_source, tx.merchant_source):
        kinds = kinds | ({"pending"} & {item.kind for item in open_items})
    existing = {item.kind for item in open_items}
    for item in open_items:
        if item.kind not in kinds:
            item.resolved_at = _now()
            item.resolved_by_change_id = change_id
        elif item.kind in details and item.detail != details[item.kind]:
            item.detail = details[item.kind]       # e.g. the conflicting rules changed
    for kind in kinds - existing:
        session.add(ReviewItem(transaction_id=tx.id, kind=kind, detail=details.get(kind, {})))
    tx.needs_review = bool(kinds)


def relabel_all(session: Session, change_id: int | None) -> list[tuple[Transaction, Labels, Labels]]:
    """Re-resolve every transaction after an entity change. Never calls the model:
    a row that loses its rule or pin falls to memory or to 'pending'."""
    ctx = load_context(session)
    names = category_store.names_by_id(session)
    changed = []
    for tx in session.scalars(select(Transaction).order_by(Transaction.id)).all():
        before = labels_of(tx)
        after, conflicts = resolve(tx, ctx)
        if set_labels(session, tx, after, change_id, names):
            changed.append((tx, before, after))
        details = {"conflict": conflicts} if conflicts else {}
        sync_review(session, tx, review_kinds(after, conflicts, after_model=False), details, change_id,
                    keep_pending=True)
    return changed


# --------------------------------------------------------------------------
# Category helpers
# --------------------------------------------------------------------------

def _current_version(session: Session, category_id: int) -> CategoryVersion | None:
    return session.scalar(
        select(CategoryVersion).where(CategoryVersion.category_id == category_id,
                                      CategoryVersion.valid_to.is_(None))
    )


def _user_category(session: Session, category_id, *, allow_archived=False, allow_system=False,
                   allowed_system_keys: tuple = ()):
    if not isinstance(category_id, int):
        raise ChangeRejected("category_id must be an integer")
    identity = session.get(CategoryId, category_id)
    version = _current_version(session, category_id) if identity else None
    if identity is None or version is None:
        raise ChangeRejected(f"no category {category_id}")
    if identity.is_system and not (allow_system or identity.system_key in allowed_system_keys):
        raise ChangeRejected(f"'{version.name}' is a system category and cannot be used here")
    if version.archived and not allow_archived:
        raise ChangeRejected(f"'{version.name}' is archived")
    return identity, version


def _check_name(session: Session, name, *, ignore_category_id=None) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ChangeRejected("category name is empty")
    name = " ".join(name.split())
    if len(name) > MAX_NAME_LENGTH:
        raise ChangeRejected(f"category name is longer than {MAX_NAME_LENGTH} characters")
    if name.lower() in RESERVED_NAMES:
        raise ChangeRejected(f"'{name}' is reserved")
    for c in category_store.current_categories(session):
        if not c.archived and c.name.lower() == name.lower() and c.category_id != ignore_category_id:
            raise ChangeRejected(f"a category named '{c.name}' already exists")
    return name


def _new_version(session: Session, category_id: int, name: str, archived: bool, change_id) -> None:
    at = _now()
    current = _current_version(session, category_id)
    if current is not None:
        current.valid_to = at
        session.flush()
    session.add(CategoryVersion(category_id=category_id, name=name, archived=archived, change_id=change_id,
                                valid_from=at))
    session.flush()


def _rule_dict(rule: CategoryRule) -> dict:
    return {"rule_id": rule.rule_id, "pattern": rule.pattern, "kind": rule.kind, "priority": rule.priority,
            "sets_category_id": rule.sets_category_id, "sets_merchant": rule.sets_merchant,
            "deleted": rule.deleted}


def _memory_dict(m: LabelMemory | None) -> dict | None:
    if m is None:
        return None
    return {"description_key": m.description_key, "category_id": m.category_id,
            "category_source": m.category_source, "merchant": m.merchant,
            "merchant_source": m.merchant_source}


# --------------------------------------------------------------------------
# Mutations: one per change kind. Each returns (before, after, touched).
# --------------------------------------------------------------------------

def _touched(**kwargs) -> dict:
    return {k: sorted(set(v)) for k, v in kwargs.items() if v}


def _cats(*category_ids) -> list[int]:
    """Categories a change depends on count as touched, so undo cannot archive
    a category something still points at, or restore a pointer to one archived since."""
    return [c for c in category_ids if c is not None]


def _mut_create_category(session, p, change_id):
    name = _check_name(session, p.get("name"))
    identity = CategoryId(is_system=False)
    session.add(identity)
    session.flush()
    _new_version(session, identity.category_id, name, False, change_id)
    return {}, {"category_id": identity.category_id, "name": name}, _touched(categories=[identity.category_id])


def _mut_rename_category(session, p, change_id):
    identity, version = _user_category(session, p.get("category_id"))
    name = _check_name(session, p.get("name"), ignore_category_id=identity.category_id)
    before = {"category_id": identity.category_id, "name": version.name}
    _new_version(session, identity.category_id, name, False, change_id)
    return before, {"category_id": identity.category_id, "name": name}, _touched(categories=[identity.category_id])


def _merge(session, source_id, target_id, change_id):
    source, source_version = _user_category(session, source_id)
    # The target is always a user category: Fees and Interest come only from the
    # statement section, and Other only from a flagged none_fit (ADR 0005, D).
    target, _ = _user_category(session, target_id)
    if source.category_id == target.category_id:
        raise ChangeRejected("a category cannot be merged into itself")

    rules = session.scalars(select(CategoryRule).where(CategoryRule.sets_category_id == source.category_id,
                                                       CategoryRule.deleted.is_(False))).all()
    memories = session.scalars(select(LabelMemory).where(LabelMemory.category_id == source.category_id)).all()
    pins = session.scalars(select(Transaction).where(Transaction.category_id == source.category_id,
                                                     Transaction.category_source == "pin")).all()
    before = {"source_id": source.category_id, "source_name": source_version.name,
              "target_id": target.category_id, "rule_ids": [r.rule_id for r in rules],
              "memory_keys": [m.description_key for m in memories], "pin_tx_ids": [t.id for t in pins]}

    for rule in rules:
        rule.sets_category_id = target.category_id
        rule.updated_at = _now()
    for m in memories:
        m.category_id = target.category_id
        m.updated_at = _now()
    names = category_store.names_by_id(session)
    for tx in pins:
        # A category pin means "this row is in that category": it follows the merge.
        set_labels(session, tx, replace(labels_of(tx), category_id=target.category_id), change_id, names)
    _new_version(session, source.category_id, source_version.name, True, change_id)
    touched = _touched(categories=[source.category_id, target.category_id], rules=before["rule_ids"],
                       memory=before["memory_keys"], transactions=before["pin_tx_ids"])
    return before, {"archived": source.category_id, "into": target.category_id}, touched


def _mut_merge_category(session, p, change_id):
    return _merge(session, p.get("source_id"), p.get("target_id"), change_id)


def _mut_delete_category(session, p, change_id):
    """With `into`: a merge into that user category. Without: the category's
    labels are withdrawn and its rows re-decided (user decision, 2026-11-11):
    rules that only set it are retired (rules that also set a merchant keep
    that part), memory forgets it, pins on it are dropped, and its rows fall to
    `pending` for the model, which may only answer from the remaining
    categories. Nothing is moved into `Other` silently."""
    if p.get("into") is not None:
        before, after, touched = _merge(session, p.get("category_id"), p["into"], change_id)
        return {**before, "mode": "merge"}, after, touched

    identity, version = _user_category(session, p.get("category_id"))
    category_id = identity.category_id
    rules = session.scalars(select(CategoryRule).where(CategoryRule.sets_category_id == category_id,
                                                       CategoryRule.deleted.is_(False))).all()
    memories = session.scalars(select(LabelMemory).where(LabelMemory.category_id == category_id)).all()
    pins = session.scalars(select(Transaction).where(Transaction.category_id == category_id,
                                                     Transaction.category_source == "pin")).all()
    before = {"mode": "unassign", "category_id": category_id, "name": version.name,
              "rules": [_rule_dict(r) for r in rules], "memory": [_memory_dict(m) for m in memories],
              "pins": [{"transaction_id": t.id, "labels": asdict(labels_of(t))} for t in pins]}

    for rule in rules:
        if rule.sets_merchant:
            rule.sets_category_id = None            # keep the merchant part
        else:
            rule.deleted = True
        rule.updated_at = _now()
    for m in memories:
        m.category_id, m.category_source, m.updated_at = None, None, _now()
        if m.merchant is None:
            session.delete(m)
    names = category_store.names_by_id(session)
    for tx in pins:
        set_labels(session, tx, replace(labels_of(tx), category_id=None, category_source="pending",
                                        category_rule_id=None), change_id, names)
    _new_version(session, category_id, version.name, True, change_id)
    touched = _touched(categories=[category_id], rules=[r["rule_id"] for r in before["rules"]],
                       memory=[m["description_key"] for m in before["memory"]],
                       transactions=[p_["transaction_id"] for p_ in before["pins"]])
    return before, {"archived": category_id, "mode": "unassign"}, touched


def _rule_fields(session, p, existing: CategoryRule | None):
    pattern = p.get("pattern", existing.pattern if existing else None)
    kind = p.get("kind", existing.kind if existing else "contains")
    priority = p.get("priority", existing.priority if existing else 0)
    category_id = p.get("category_id", existing.sets_category_id if existing else None)
    merchant = p.get("merchant", existing.sets_merchant if existing else None)
    validate_pattern(pattern, kind)
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ChangeRejected("priority must be an integer")
    if category_id is None and not merchant:
        raise ChangeRejected("a rule must set a category, a merchant, or both")
    if category_id is not None:
        _user_category(session, category_id)
    if merchant is not None:
        if not isinstance(merchant, str) or not merchant.strip() or len(merchant) > MAX_MERCHANT_LENGTH:
            raise ChangeRejected(f"merchant must be 1 to {MAX_MERCHANT_LENGTH} characters")
        merchant = merchant.strip()
    return pattern, kind, priority, category_id, merchant


def _check_rule_conflicts(session, candidate: CategoryRule) -> None:
    """Refuse an equal-priority rule that would give an existing row a different
    target for the same field (ADR 0005, 2c)."""
    others = session.scalars(select(CategoryRule).where(
        CategoryRule.deleted.is_(False), CategoryRule.priority == candidate.priority,
        CategoryRule.rule_id != (candidate.rule_id or -1))).all()
    if not others:
        return
    conflicts = []
    for tx in session.scalars(select(Transaction)).all():
        if not rule_matches(candidate, tx.raw_description):
            continue
        for other in others:
            if not rule_matches(other, tx.raw_description):
                continue
            for field in ("category", "merchant"):
                mine, theirs = _target(candidate, field), _target(other, field)
                if mine is not None and theirs is not None and mine != theirs:
                    conflicts.append({"rule_id": other.rule_id, "field": field, "transaction_id": tx.id})
    if conflicts:
        raise ChangeRejected("an equal-priority rule gives some of the same rows a different label; "
                             "change the priority or the pattern",
                             {"conflicts": conflicts[:20], "total": len(conflicts)})


def _mut_create_rule(session, p, change_id):
    pattern, kind, priority, category_id, merchant = _rule_fields(session, p, None)
    rule = CategoryRule(pattern=pattern, kind=kind, priority=priority,
                        sets_category_id=category_id, sets_merchant=merchant)
    _check_rule_conflicts(session, rule)
    session.add(rule)
    session.flush()
    return {}, _rule_dict(rule), _touched(rules=[rule.rule_id], categories=_cats(rule.sets_category_id))


def _active_rule(session, rule_id) -> CategoryRule:
    rule = session.get(CategoryRule, rule_id) if isinstance(rule_id, int) else None
    if rule is None or rule.deleted:
        raise ChangeRejected(f"no active rule {rule_id}")
    return rule


def _mut_edit_rule(session, p, change_id):
    rule = _active_rule(session, p.get("rule_id"))
    before = _rule_dict(rule)
    pattern, kind, priority, category_id, merchant = _rule_fields(session, p, rule)
    rule.pattern, rule.kind, rule.priority = pattern, kind, priority
    rule.sets_category_id, rule.sets_merchant, rule.updated_at = category_id, merchant, _now()
    _check_rule_conflicts(session, rule)
    session.flush()
    return before, _rule_dict(rule), _touched(rules=[rule.rule_id],
                                              categories=_cats(before["sets_category_id"], category_id))


def _mut_delete_rule(session, p, change_id):
    rule = _active_rule(session, p.get("rule_id"))
    before = _rule_dict(rule)
    rule.deleted, rule.updated_at = True, _now()     # soft: history keeps the rule_id
    session.flush()
    return before, _rule_dict(rule), _touched(rules=[rule.rule_id], categories=_cats(rule.sets_category_id))


def _mut_set_memory(session, p, change_id):
    key = p.get("description_key")
    if not isinstance(key, str) or not key.strip():
        raise ChangeRejected("description_key is required")
    # Accept a raw description or a key; store the normalised key so it can match.
    key = category_store.description_key(key)
    if "category_id" not in p and "merchant" not in p:
        raise ChangeRejected("set category_id, merchant, or both")
    entry = session.get(LabelMemory, key)
    before = _memory_dict(entry)
    if entry is None:
        entry = LabelMemory(description_key=key)
        session.add(entry)
    if "category_id" in p:
        if p["category_id"] is None:
            entry.category_id, entry.category_source = None, None
        else:
            _user_category(session, p["category_id"])
            entry.category_id, entry.category_source = p["category_id"], "user"
    if "merchant" in p:
        merchant = p["merchant"]
        if merchant is None:
            entry.merchant, entry.merchant_source = None, None
        elif not isinstance(merchant, str) or not merchant.strip() or len(merchant) > MAX_MERCHANT_LENGTH:
            raise ChangeRejected(f"merchant must be 1 to {MAX_MERCHANT_LENGTH} characters")
        else:
            entry.merchant, entry.merchant_source = merchant.strip(), "user"
    entry.updated_at = _now()
    session.flush()
    if entry.category_id is None and entry.merchant is None:
        session.delete(entry)
        session.flush()
        entry = None
    after = _memory_dict(entry)
    return before, after, _touched(memory=[key], categories=_cats((before or {}).get("category_id"),
                                                                 (after or {}).get("category_id")))


def _pin_target(session, p) -> tuple[Transaction, str]:
    tx = session.get(Transaction, p.get("transaction_id")) if isinstance(p.get("transaction_id"), int) else None
    if tx is None:
        raise ChangeRejected(f"no transaction {p.get('transaction_id')}")
    field = p.get("field")
    if field not in ("category", "merchant"):
        raise ChangeRejected("field must be 'category' or 'merchant'")
    if tx.category_source == "section":
        raise ChangeRejected("fee and interest rows are labelled by their statement section and cannot be pinned")
    return tx, field


def _mut_pin(session, p, change_id):
    tx, field = _pin_target(session, p)
    before = {"transaction_id": tx.id, "field": field, "labels": asdict(labels_of(tx))}
    names = category_store.names_by_id(session)
    if field == "category":
        identity, _ = _user_category(session, p.get("value"))
        new = replace(labels_of(tx), category_id=identity.category_id, category_source="pin", category_rule_id=None)
    else:
        value = p.get("value")
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_MERCHANT_LENGTH:
            raise ChangeRejected(f"merchant must be 1 to {MAX_MERCHANT_LENGTH} characters")
        new = replace(labels_of(tx), merchant=value.strip(), merchant_source="pin", merchant_rule_id=None)
    set_labels(session, tx, new, change_id, names)
    cats = _cats(before["labels"]["category_id"], new.category_id) if field == "category" else []
    return (before, {"transaction_id": tx.id, "field": field, "value": p.get("value")},
            _touched(transactions=[tx.id], categories=cats))


def _mut_unpin(session, p, change_id):
    tx, field = _pin_target(session, p)
    source = tx.category_source if field == "category" else tx.merchant_source
    if source != "pin":
        raise ChangeRejected(f"the {field} of transaction {tx.id} is not pinned")
    before = {"transaction_id": tx.id, "field": field, "labels": asdict(labels_of(tx))}
    names = category_store.names_by_id(session)
    # Drop the pin; relabel_all then re-resolves the field from rules or memory,
    # or leaves it pending for the background model pass.
    if field == "category":
        new = replace(labels_of(tx), category_id=None, category_source="pending", category_rule_id=None)
    else:
        new = replace(labels_of(tx), merchant=tx.raw_description, merchant_source="pending", merchant_rule_id=None)
    set_labels(session, tx, new, change_id, names)
    cats = _cats(before["labels"]["category_id"]) if field == "category" else []
    return before, {"transaction_id": tx.id, "field": field}, _touched(transactions=[tx.id], categories=cats)


def _mut_undo(session, p, change_id):
    target_id = p.get("change_id")
    target = session.get(ChangeLog, target_id) if isinstance(target_id, int) else None
    if target is None:
        raise ChangeRejected(f"no change {target_id}")
    if target.kind == "undo":
        raise ChangeRejected("an undo cannot itself be undone; make the change again instead")
    if session.scalar(select(ChangeLog.change_id).where(ChangeLog.undoes == target.change_id,
                                                        ChangeLog.change_id != (change_id or -1))):
        raise ChangeRejected(f"change {target.change_id} is already undone")

    later = session.scalars(select(ChangeLog).where(ChangeLog.change_id > target.change_id,
                                                    ChangeLog.change_id != (change_id or -1))).all()
    # A change that was undone, and the undo that cancelled it, form a pair that
    # no longer has any effect, so neither blocks an older undo.
    undone = {c.undoes for c in later if c.undoes is not None}
    blockers = [c.change_id for c in later
                if c.kind != "undo" and c.change_id not in undone and _overlaps(c.touched, target.touched)]
    if blockers:
        raise ChangeRejected("later changes touched the same things; undo them first (newest first)",
                             {"blocking_change_ids": sorted(blockers, reverse=True)})

    restore = _UNDO[target.kind]
    restore(session, target.before, target.after, change_id)
    return {"undoes": target.change_id}, {"undoes": target.change_id}, target.touched


def _overlaps(a: dict, b: dict) -> bool:
    return any(set(a.get(k, [])) & set(b.get(k, [])) for k in ("categories", "rules", "memory", "transactions"))


# --- undo restorers: compensating changes, never history rewrites --------

def _undo_create_category(session, before, after, change_id):
    _, version = _user_category(session, after["category_id"], allow_archived=True)
    _new_version(session, after["category_id"], version.name, True, change_id)
    # The model may have learned this category for some descriptions since it was
    # created (enrichment is not a logged change, so it cannot block the undo).
    # Forget those so the next pass can relearn; user-set memory would have
    # blocked the undo instead.
    for entry in session.scalars(select(LabelMemory).where(LabelMemory.category_id == after["category_id"],
                                                           LabelMemory.category_source == "model")).all():
        entry.category_id, entry.category_source, entry.updated_at = None, None, _now()
        if entry.merchant is None:
            session.delete(entry)
    session.flush()


def _undo_rename_category(session, before, after, change_id):
    name = _check_name(session, before["name"], ignore_category_id=before["category_id"])
    _new_version(session, before["category_id"], name, False, change_id)


def _undo_merge(session, before, after, change_id):
    name = _check_name(session, before["source_name"], ignore_category_id=before["source_id"])
    _new_version(session, before["source_id"], name, False, change_id)
    for rule_id in before["rule_ids"]:
        session.get(CategoryRule, rule_id).sets_category_id = before["source_id"]
    for key in before["memory_keys"]:
        entry = session.get(LabelMemory, key)
        if entry is not None:
            entry.category_id = before["source_id"]
    names = category_store.names_by_id(session)
    for tx_id in before["pin_tx_ids"]:
        tx = session.get(Transaction, tx_id)
        set_labels(session, tx, replace(labels_of(tx), category_id=before["source_id"]), change_id, names)
    session.flush()


def _undo_delete_category(session, before, after, change_id):
    if before.get("mode") != "unassign":
        return _undo_merge(session, before, after, change_id)
    name = _check_name(session, before["name"], ignore_category_id=before["category_id"])
    _new_version(session, before["category_id"], name, False, change_id)
    for rule_before in before["rules"]:
        rule = session.get(CategoryRule, rule_before["rule_id"])
        rule.sets_category_id, rule.deleted, rule.updated_at = before["category_id"], False, _now()
    for memory_before in before["memory"]:
        entry = session.get(LabelMemory, memory_before["description_key"])
        if entry is None:
            entry = LabelMemory(description_key=memory_before["description_key"])
            session.add(entry)
        for field in ("category_id", "category_source", "merchant", "merchant_source"):
            setattr(entry, field, memory_before[field])
    names = category_store.names_by_id(session)
    for pin in before["pins"]:
        tx = session.get(Transaction, pin["transaction_id"])
        set_labels(session, tx, replace(labels_of(tx), category_id=before["category_id"], category_source="pin",
                                        category_rule_id=None), change_id, names)
    session.flush()


def _undo_create_rule(session, before, after, change_id):
    session.get(CategoryRule, after["rule_id"]).deleted = True
    session.flush()


def _usable_target(session, category_id) -> None:
    """Undo must never restore a pointer to a category that is archived now."""
    if category_id is not None:
        _user_category(session, category_id)


def _restore_rule(session, before, after, change_id):
    if not before["deleted"]:
        _usable_target(session, before["sets_category_id"])
    rule = session.get(CategoryRule, before["rule_id"])
    for field in ("pattern", "kind", "priority", "sets_category_id", "sets_merchant", "deleted"):
        setattr(rule, field, before[field])
    rule.updated_at = _now()
    if not rule.deleted:
        _check_rule_conflicts(session, rule)
    session.flush()


def _undo_set_memory(session, before, after, change_id):
    key = (before or after)["description_key"]
    entry = session.get(LabelMemory, key)
    if before is None:
        if entry is not None:
            session.delete(entry)
    else:
        _usable_target(session, before["category_id"])
        if entry is None:
            entry = LabelMemory(description_key=key)
            session.add(entry)
        for field in ("category_id", "category_source", "merchant", "merchant_source"):
            setattr(entry, field, before[field])
    session.flush()


def _undo_pin(session, before, after, change_id):
    tx = session.get(Transaction, before["transaction_id"])
    old = Labels(**before["labels"])
    names = category_store.names_by_id(session)
    if before["field"] == "category":
        if old.category_source == "pin":
            _usable_target(session, old.category_id)
            new = replace(labels_of(tx), category_id=old.category_id, category_source="pin", category_rule_id=None)
        else:                                  # re-resolved by relabel_all
            new = replace(labels_of(tx), category_id=None, category_source="pending", category_rule_id=None)
    else:
        if old.merchant_source == "pin":
            new = replace(labels_of(tx), merchant=old.merchant, merchant_source="pin", merchant_rule_id=None)
        else:
            new = replace(labels_of(tx), merchant=tx.raw_description, merchant_source="pending",
                          merchant_rule_id=None)
    set_labels(session, tx, new, change_id, names)


_UNDO = {
    "create_category": _undo_create_category,
    "rename_category": _undo_rename_category,
    "merge_category":  _undo_merge,
    "delete_category": _undo_delete_category,
    "create_rule":     _undo_create_rule,
    "edit_rule":       _restore_rule,
    "delete_rule":     _restore_rule,
    "set_memory":      _undo_set_memory,
    "pin":             _undo_pin,
    "unpin":           _undo_pin,
}

_MUTATIONS = {
    "create_category": _mut_create_category,
    "rename_category": _mut_rename_category,
    "merge_category":  _mut_merge_category,
    "delete_category": _mut_delete_category,
    "create_rule":     _mut_create_rule,
    "edit_rule":       _mut_edit_rule,
    "delete_rule":     _mut_delete_rule,
    "set_memory":      _mut_set_memory,
    "pin":             _mut_pin,
    "unpin":           _mut_unpin,
    "undo":            _mut_undo,
}
CHANGE_KINDS = tuple(_MUTATIONS)


# --------------------------------------------------------------------------
# Preview and apply
# --------------------------------------------------------------------------

def _effect(changed, names: dict[int, str]) -> dict:
    def describe(labels: Labels) -> dict:
        return {"category": names.get(labels.category_id), "category_source": labels.category_source,
                "merchant": labels.merchant, "merchant_source": labels.merchant_source}
    return {
        "rows_changed": len(changed),
        "to_pending": sum(1 for _, b, a in changed if a.category_source == "pending" and b.category_source != "pending"),
        "to_memory": sum(1 for _, b, a in changed if a.category_source == "memory" and b.category_source != "memory"),
        "samples": [{"transaction_id": tx.id, "description": tx.raw_description,
                     "before": describe(b), "after": describe(a)} for tx, b, a in changed[:PREVIEW_SAMPLES]],
    }


def _run(session: Session, payload: dict, change: ChangeLog | None):
    kind = payload.get("kind")
    if kind not in _MUTATIONS:
        raise ChangeRejected(f"unknown change kind {kind!r}", {"kinds": list(CHANGE_KINDS)})
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ChangeRejected("params must be an object")
    change_id = change.change_id if change is not None else None
    # Snapshot every row's labels first, so the effect also counts rows the
    # mutation itself wrote (pins re-pointed by a merge, dropped by a delete).
    snapshot = {tx.id: labels_of(tx) for tx in session.scalars(select(Transaction)).all()}
    before, after, touched = _MUTATIONS[kind](session, params, change_id)
    session.flush()
    relabel_all(session, change_id)
    changed = [(tx, snapshot[tx.id], labels_of(tx))
               for tx in session.scalars(select(Transaction).order_by(Transaction.id)).all()
               if tx.id in snapshot and labels_of(tx) != snapshot[tx.id]]
    touched = dict(touched)
    touched["transactions"] = sorted(set(touched.get("transactions", [])) | {tx.id for tx, _, _ in changed})
    return before, after, touched, changed


def preview(session: Session, payload: dict) -> dict:
    """What the change would do, computed for real inside a savepoint and rolled back."""
    version = current_version(session)
    savepoint = session.begin_nested()
    try:
        _, _, _, changed = _run(session, payload, None)
        effect = _effect(changed, category_store.names_by_id(session))
    finally:
        savepoint.rollback()
        session.expire_all()
    return {"token": make_token(payload, version), "data_version": version, "effect": effect}


def apply(session: Session, payload: dict, token: str | None) -> dict:
    """Apply a previewed change atomically under the data_version lock."""
    version = lock_version(session)
    if payload.get("kind") not in NO_PREVIEW_KINDS and token != make_token(payload, version):
        session.rollback()
        raise StalePreview("data changed since the preview, or the change differs from what was previewed; preview again")
    undoes = None
    if payload.get("kind") == "undo":
        undoes = (payload.get("params") or {}).get("change_id")
        if not isinstance(undoes, int) or session.get(ChangeLog, undoes) is None:
            session.rollback()
            raise ChangeRejected(f"no change {undoes}")
    try:
        change = ChangeLog(kind=payload.get("kind"), payload=payload, before={}, after={}, touched={},
                           undoes=undoes)
        session.add(change)
        session.flush()
        before, after, touched, changed = _run(session, payload, change)
        change.before, change.after, change.touched = before, after, touched
        effect = _effect(changed, category_store.names_by_id(session))
        bump_version(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return {"change_id": change.change_id, "effect": effect}


# --------------------------------------------------------------------------
# Enrichment: rows that rules, memory and pins leave pending
# --------------------------------------------------------------------------

def _pending_rows(session: Session, statement_id: int | None):
    query = select(Transaction).where(
        or_(Transaction.category_source == "pending", Transaction.merchant_source == "pending"))
    if statement_id is not None:
        query = query.where(Transaction.statement_id == statement_id)
    # populate_existing: never trust an already-loaded object's labels; another
    # writer (an apply, a pin) may have changed them since it was loaded.
    return session.scalars(query.order_by(Transaction.id).execution_options(populate_existing=True)).all()


def enrich_pending(session: Session, statement_id: int | None = None, labeller=label_descriptions) -> dict:
    """Label pending fields: rules and memory first, then the model for what is
    left. The model is called outside the lock; writes happen under it, after
    re-checking that each row is still pending (ADR 0005, decision 3)."""
    rows = _pending_rows(session, statement_id)
    ctx = load_context(session)
    needed: dict[str, str] = {}
    for tx in rows:
        labels, _ = resolve(tx, ctx)
        if "pending" in (labels.category_source, labels.merchant_source):
            needed.setdefault(category_store.description_key(tx.raw_description), tx.raw_description)
    offered = category_store.enrichment_categories(session)
    session.rollback()   # end the read before the slow model call

    model: dict[str, ModelLabel | None] = labeller(needed, [c.name for c in offered]) if needed else {}

    lock_version(session)
    try:
        ctx = load_context(session)
        name_to_id = {c.name: c.category_id for c in category_store.enrichment_categories(session)}
        names = category_store.names_by_id(session)
        # Memory learned in this pass is kept apart from ctx.memory, so rows that
        # share a key are all recorded as labelled by this model call, not by memory.
        learned: dict[str, LabelMemory] = {}
        labelled = 0
        wrote = False
        for tx in _pending_rows(session, statement_id):
            labels, conflicts = resolve(tx, ctx)
            key = category_store.description_key(tx.raw_description)
            result = model.get(key)
            details: dict[str, dict] = {"conflict": conflicts} if conflicts else {}

            if labels.category_source == "pending" and result is not None:
                if result.category == NONE_FIT:
                    labels = replace(labels, category_id=ctx.other_id, category_source="none_fit")
                    details["none_fit"] = {"suggestion": result.suggestion}
                elif result.category in name_to_id:
                    labels = replace(labels, category_id=name_to_id[result.category], category_source="model")
                    _remember(session, ctx, learned, key, category_id=name_to_id[result.category])
            if labels.merchant_source == "pending" and result is not None:
                labels = replace(labels, merchant=result.merchant[:MAX_MERCHANT_LENGTH], merchant_source="model")
                _remember(session, ctx, learned, key, merchant=labels.merchant)

            if set_labels(session, tx, labels, None, names):
                wrote = True
                if "pending" not in (labels.category_source, labels.merchant_source):
                    labelled += 1
            asked = key in model      # rows that became pending mid-call were not asked yet
            sync_review(session, tx, review_kinds(labels, conflicts, after_model=asked), details, None,
                        keep_pending=not asked)

        if wrote or learned:
            # Only a pass that changed labels or memory invalidates open previews.
            bump_version(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return label_counts(session, statement_id) | {"enriched": labelled}


def _remember(session: Session, ctx: Context, learned: dict, key: str, *, category_id=None, merchant=None) -> None:
    # A remembered category that has since been archived is treated as empty,
    # so memory always heals toward a usable label.
    """Memory only ever holds real labels (never none_fit or failures). Under the
    data_version lock, so concurrent passes cannot both insert the same key."""
    entry = learned.get(key) or session.get(LabelMemory, key)
    if entry is None:
        entry = LabelMemory(description_key=key)
        session.add(entry)
    learned[key] = entry
    if category_id is not None and (entry.category_id is None or entry.category_id in ctx.archived):
        entry.category_id, entry.category_source = category_id, "model"
    if merchant is not None and entry.merchant is None:
        entry.merchant, entry.merchant_source = merchant, "model"
    entry.updated_at = _now()
    session.flush()


def label_counts(session: Session, statement_id: int | None = None) -> dict:
    def count(*conditions) -> int:
        query = select(func.count()).select_from(Transaction).where(*conditions)
        if statement_id is not None:
            query = query.where(Transaction.statement_id == statement_id)
        return session.scalar(query)
    return {
        "needs_review": count(Transaction.needs_review.is_(True)),
        "pending": count(or_(Transaction.category_source == "pending", Transaction.merchant_source == "pending")),
    }


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def refresh_embeddings(session: Session, embed, batch_size: int = EMBED_BATCH_SIZE) -> int:
    """Re-embed every stale row in batches. Staleness is derived, so this is
    also the recovery path for any lost background task."""
    names = category_store.names_by_id(session)
    stale = [tx for tx in session.scalars(select(Transaction).order_by(Transaction.id)).all()
             if is_stale(tx, names.get(tx.category_id))]
    for start in range(0, len(stale), batch_size):
        for tx in stale[start:start + batch_size]:
            tx.embedded_text = build_embedded_text(tx, names.get(tx.category_id))
            tx.embedding = embed(tx.embedded_text)
        session.commit()
    return len(stale)
