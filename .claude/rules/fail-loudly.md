# Fail loudly: a parse failure never becomes a wrong number

- A parse failure must never silently become a wrong number. Reject the statement and save nothing (ADR 0002).
- Reject rather than guess: when input is ambiguous or does not reconcile, refuse visibly instead of picking a plausible value.
