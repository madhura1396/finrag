---
paths:
  - "app/pipeline.py"
  - "app/statement.py"
  - "app/profile.py"
  - "app/reader.py"
  - "app/categorization.py"
  - "app/llm.py"
---

# Write path

- Extraction design is decided in `docs/adr/0002-profile-based-extraction.md`. Follow it; do not reopen it without asking me.
- Bank-specific strings and layout facts live only in `profiles/*.yaml`, loaded strictly by `profile.py`. Never add a bank string to `app/`.
- A statement is saved only if it reconciles with its own printed section totals and Account Summary; any mismatch rejects it with 422 and saves nothing.
- Labels: one source decides each field, strongest first: section > pin > rule > memory > model (ADR 0005). The model may pick only from the user's categories or answer `none_fit`. A row the model cannot label stays `pending`; it is never silently `Other`.
