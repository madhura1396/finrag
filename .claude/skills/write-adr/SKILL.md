---
name: write-adr
description: Evaluate an architectural decision with the user and record it as an ADR in docs/adr/. Use before any new dependency, schema change, module boundary change, cross-module contract, concurrency/transaction/error/security model change, or anything expensive to reverse.
argument-hint: <the decision to evaluate>
---

Decision: $ARGUMENTS

Before making the decision:

1. State the decision in one sentence and the requirement that forces it.
2. Give 2 to 4 real options. Include "keep what we have" when it applies.
3. For each option: how it works, pros, cons, cost (complexity, latency, money, privacy), and what breaks first at scale, on which axis, at roughly what number. If an option would send statement text, transactions or embeddings to an external API, that is its first con.
4. Recommend one, and name the condition that would change your recommendation.
5. Stop and wait for the user's choice. Write no code for it until they choose.
6. After they choose, write `docs/adr/NNNN-short-title.md` (next free number) with: Status, Date, Context, Options considered, Decision, Consequences.

Never put real transaction data or merchant names in the ADR.
