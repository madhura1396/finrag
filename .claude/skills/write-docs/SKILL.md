---
name: write-docs
description: Write or update FinRAG documentation (README sections, docs/ pages, module guides) so it matches the code.
disable-model-invocation: true
argument-hint: <module, feature or file to document>
---

Document: $ARGUMENTS

Follow these steps in order.

1. Read the code for $ARGUMENTS first. The code is the source of truth.
2. List every claim the doc will make (what it does, inputs, outputs, failure behaviour, numbers). Verify each against the code and cite `file:line` in your working notes.
3. Check `README.md`, `decisions.md` and `docs/adr/` for statements on the same topic. List every contradiction with the code in your final message. Do not silently copy anything from `decisions.md`.
4. Structure design explanations as: what it does, why it was built this way, what the alternative is and what it costs, what breaks at scale.
5. Never invent numbers, benchmarks or accuracy figures. If a number is not produced by code or an eval in this repo, mark it "unverified".
6. Write only to `README.md` or files under `docs/`. Never edit files in `app/` or `tests/`. If a docstring is needed, list it as text in your final message for me to add.
7. If on `main`, create a branch `docs/<short-name>` first. Write the files, commit them, and report the branch, commit hash and a summary of what changed.
8. Never include real transaction data, account numbers or anything from `uploads/` or `.env`.
