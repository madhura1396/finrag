# profiles/

Bank layout profiles, one YAML file per bank and account type, versioned in the name (`<bank>.<account>.v<N>.yaml`).

- Extraction design is decided in `docs/adr/0002-profile-based-extraction.md`. Follow it; do not reopen it without asking me.
- This is the only home of bank-specific strings and layout facts: headings, total patterns, summary labels and signs, column anchors, tolerances. Never add a bank string to `app/`.
- `app/profile.py` loads profiles strictly: unknown keys, missing keys and wrong types are refused. Quote strings that YAML would otherwise turn into booleans or numbers.
- The app loads exactly one profile today (`load_single`). Adding a second profile forces a profile-selection decision; use the `write-adr` skill first.
- A new or changed profile needs a synthetic fixture (`tests/fixtures/`) with invented merchants. Check it against real statements only through `scripts/check_profiles.py`, which I run (see `scripts/CLAUDE.md`).
