-- ADR 0004 / ADR 0005: the read path no longer uses the transitional text copy
-- of the category name; category names come from category_versions.
-- Run by the user, after the read-path PR is merged:
--   psql "$DATABASE_URL" -f migrations/0002_drop_transitional_category.sql
-- A fresh database from scripts/reset_db.py never has this column.
--
-- Undo (only with the old code): ALTER TABLE transactions ADD COLUMN category TEXT;

ALTER TABLE transactions DROP COLUMN IF EXISTS category;
