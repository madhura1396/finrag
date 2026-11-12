# Money is Decimal, never float

- Money is `Decimal` in Python and `Numeric` in the database. Never `float`, anywhere money is parsed, stored, compared or computed.
- Thresholds applied to money are `Decimal` too.
- Money leaves the API as strings (`str(Decimal)`).
