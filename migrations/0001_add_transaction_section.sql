-- ADR 0003, decisions 1 and 5c. Run by the user, before PR1 code runs:
--   psql "$DATABASE_URL" -f migrations/0001_add_transaction_section.sql
--
-- Nullable until legacy rows are deleted and re-ingested (data step in
-- docs/plans/extraction-rebuild.md); 0003 then makes it NOT NULL.
-- IF NOT EXISTS: on a fresh database create_tables() already adds the column.
--
-- Undo (after reverting the code that uses the column):
--   ALTER TABLE transactions DROP COLUMN section;

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS section TEXT NULL;
