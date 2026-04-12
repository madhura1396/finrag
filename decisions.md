# Design Decisions

Every non-obvious choice made in this codebase — what we built, why we built it that way, and what we considered but rejected.

---

## models.py

### Why SQLAlchemy ORM instead of raw SQL
SQLAlchemy lets us define tables as Python classes. The alternative is writing raw SQL strings for every insert and query. ORM gives us type safety, relationship navigation, and protection against SQL injection at the cost of some abstraction overhead.

### Why `Numeric(10,2)` for amount instead of Float
Float uses binary floating point — it cannot represent 0.1 exactly in binary. This causes errors like `$19.999999` instead of `$20.00`. `Numeric(10,2)` is stored as an exact decimal by Postgres. Never use Float for money.

### Why `UniqueConstraint("period_start", "period_end")`
Two layers of duplicate upload protection. The Python layer checks before inserting. The database constraint is the hard stop — even if there is a bug in the Python code, Postgres will reject a second insert of the same period. Defense in depth.

### Why `is_credit` stored as String `"true"/"false"` instead of Boolean
SQLAlchemy Boolean columns behave inconsistently across databases when filtering. Storing as string makes the filter condition explicit: `is_credit = 'false'` in SQL, no ambiguity.

### Why `needs_review` flag
LLM batch calls occasionally miss a transaction or return malformed JSON for one row. Instead of failing the entire upload, we mark those rows with `needs_review = True` and surface them in a review queue. Silent failures are worse than flagged failures.

### Why `embedded_text` stored alongside `embedding`
The `embedding` column is 384 floats — unreadable by humans. `embedded_text` stores the sentence we fed to the model: `"On 12/06, spent $45.23 at Trader Joe's in Groceries"`. This lets us debug why a semantic search returned a particular result.

### Why `Vector(384)`
`all-MiniLM-L6-v2` produces 384-dimensional vectors. The dimension must match the model exactly — pgvector enforces this at insert time.

### Why `SessionLocal` as a factory
`SessionLocal()` creates a new database session each time it is called. Sessions are not thread-safe and should not be shared across requests. Each request gets its own session, uses it, and closes it.

### Why `lambda: datetime.now(timezone.utc)` instead of `datetime.utcnow()`
`datetime.utcnow()` is deprecated in Python 3.12+. It returns a naive datetime with no timezone info. `datetime.now(timezone.utc)` returns a timezone-aware datetime. The lambda is required because SQLAlchemy evaluates the default at insert time, not at class definition time.

---

## extractor.py

### Why fitz (PyMuPDF) instead of pdfplumber or PyPDF2
PyMuPDF is significantly faster for text extraction and handles Wells Fargo's PDF structure more reliably. pdfplumber is better for table detection but slower. PyPDF2 is outdated.

### Why filter pages with `len(text.strip()) < 10`
Wells Fargo PDFs sometimes include pages that contain only barcodes or tracking metadata stored as a thin invisible text layer. These pages return near-empty strings. The threshold of 10 characters filters them without risking removal of real content pages.

### Why regex for statement period instead of LLM
The date format `Statement Period MM/DD/YYYY to MM/DD/YYYY` is completely consistent across all Wells Fargo statements. Regex is deterministic, instant, and cannot hallucinate. LLM is unnecessary here and would add latency and risk.

### Why `re.VERBOSE` flag
Allows whitespace and comments inside the regex pattern. Without it, the regex is a single unreadable string. Verbose mode makes it possible to read and maintain the pattern.

### Why send only first 2 pages to `detect_structure()`
The LLM only needs to identify section headers and page ranges — information that always appears in the first 1-2 pages of a bank statement. Sending all 5 pages doubles the token count and caused timeouts in testing. First 2 pages is sufficient.

### Why fallback defaults in `detect_structure()`
If Ollama is down, times out, or returns malformed JSON, the system should degrade gracefully rather than crash. Fallback values match the standard Wells Fargo format. The trade-off is that it scans all pages instead of only the transaction pages.

### Why a state machine for `parse_transactions_from_text()`
PDF text extraction returns a flat list of lines with no structure. The state machine tracks: are we inside the transaction section yet, which section are we in (credits or purchases), is this line a new transaction or a continuation of the previous one. Each line means something different depending on what came before it — that is the definition of state.

### Why `_parse_date()` tries `period_start.year` first
Most transaction dates fall within the statement period year. We default to that year and only try `period_end.year` if the date falls outside the window. This handles December-January statements where some transactions are in one year and some in the other.

### Why `Decimal` in `_parse_amount()` instead of float
Same reason as the models.py decision. `Decimal("45.23")` is exact. `float("45.23")` is `45.22999999999999...`.

### Why underscore prefix on `_parse_date` and `_parse_amount`
Python convention for private helpers. The underscore signals that these functions are implementation details of this module, not part of the public API. They should not be imported or called from outside extractor.py.

---

## llm.py

### Why batch all transactions in one LLM call
One call per transaction would mean 30-50 LLM calls per statement upload. At 2-3 seconds per call that is 60-150 seconds of serial waiting. A single batch call takes ~15-30 seconds regardless of transaction count.

### Why track `missed_ids` with set difference
The LLM occasionally skips a row or drops it from the output JSON. We cannot rely on the LLM returning every row we sent. Set difference `all_ids - returned_ids` catches skipped rows, rows with malformed JSON that failed to parse, and any other silent omissions.

### Why `call_llm_single()` returns `None` instead of raising
A failed retry is expected, not exceptional. The caller handles `None` by setting `needs_review = True`. Using an exception for an expected case would make the calling code more complex.

### Why strip markdown code blocks from LLM response
LLMs are trained on documentation and tutorials where JSON is always shown inside code blocks. They default to this formatting even when explicitly told to return plain JSON. Stripping backticks before `json.loads()` is a required defense.

### Why pass fixed category list in the prompt
Without a fixed list the LLM will invent categories: "Food & Beverage", "Supermarkets", "Meal Delivery". The prompt forces it to choose from our exact category strings. Consistency in category names is required for SQL aggregation to work correctly.

---

## embedder.py

### Why `all-MiniLM-L6-v2`
Produces 384-dimensional vectors, runs entirely locally with no API call, fast inference (~10ms per sentence), and performs well on sentence similarity tasks. The alternative `text-embedding-ada-002` from OpenAI is more accurate but requires an API call per transaction, adds cost, and creates an external dependency.

### Why a singleton model with `_model = None`
Loading the model downloads ~90MB and takes 2-3 seconds. Loading it on every function call would add 3 seconds to every embedding request. The singleton loads it once on first call and reuses the loaded model for all subsequent calls.

### Why `normalize_embeddings=True`
Normalization makes every vector unit length (magnitude = 1). When vectors are unit length, cosine similarity equals dot product. pgvector's cosine distance operator uses this. Normalization also makes similarity scores comparable across different queries.

### Why embed a constructed sentence instead of the raw description
`"TRADER JOE S #534 SAN FRANCISCO CA"` gives the model almost no useful semantic signal — it is a formatting artifact. `"On 12/06, spent $45.23 at Trader Joe's in Groceries"` gives the model merchant name, category, amount, and date — much richer context for similarity matching.

### Why `vector.tolist()`
The model returns a numpy array. pgvector accepts either numpy arrays or Python lists. We convert to list to avoid numpy as an import dependency in pipeline.py and to make the value JSON-serializable if needed.

---

## pipeline.py

### Why split into `extract_from_pdf()` and `enrich_transactions()`
The original design ran LLM enrichment inside the upload transaction. If Ollama timed out, the database rolled back and all extracted transactions were lost. Splitting into two functions means raw transactions are committed to the DB first. LLM enrichment is a separate step that can fail, be retried, or be run as a background job without losing data.

### Why `db.flush()` before `db.commit()`
`flush()` sends the INSERT to Postgres within the current transaction and gets back the auto-generated `id`, but does not finalize the transaction. We need the `statement.id` before creating transaction rows (for the foreign key), and we need each `transaction.id` before calling the LLM (to match responses back to rows). `commit()` at the end finalizes everything atomically.

### Why `db.rollback()` in the except block
If anything fails during extraction — PDF parse error, duplicate statement, DB error — we want to undo all partial inserts from this request. Without rollback, a partial upload leaves orphaned rows in the database.

### Why set `merchant = raw_description` initially
At the time of DB insert, LLM has not run yet. The `merchant` column is non-nullable so we need a value. Setting it to `raw_description` means the row is always readable even before enrichment runs.

---

## query_router.py

### Why LLM-generated SQL instead of hardcoded query functions
Hardcoded functions cannot handle the full range of natural language questions. "Where did I spend most last month", "compare dining vs groceries in Q4", "which merchant charged me the most" all require different SQL. An LLM that knows the schema can generate the right query for any of these.

### Why always validate SQL before executing
LLMs can generate syntactically valid but destructive SQL. `DROP TABLE transactions` is valid SQL. `_validate_sql()` ensures only SELECT queries reach the database. Never execute LLM-generated SQL without validation.

### Why pass the schema in the SQL generation prompt
Without schema context the LLM invents column names and table names. With exact column names, valid category values, and data types in the prompt, hallucination is constrained to query logic rather than schema structure.

### Why a separate `_classify()` step
SQL and semantic search are fundamentally different retrieval strategies. Aggregation questions need SQL. Pattern-finding questions need vector search. Choosing the wrong path either returns wrong numbers or no results. The classification step costs one LLM call but routes correctly.

### Why default to `"sql"` if classification fails
If Ollama is down during classification, SQL is the safer default. A failed SQL query returns an error. A failed semantic search silently returns random transactions. SQL failures are louder and easier to debug.

### Why LLM only formats the answer, never computes
LLMs hallucinate numbers. Given "total is $387.28 + $45.23", a LLM might return $432.41 or $433.51 depending on how it samples. All arithmetic happens in SQL. The LLM receives pre-computed results and only converts them to natural language.

### Why `_format_answer()` passes the SQL alongside the results
Including the SQL in the prompt gives the LLM context about what was actually queried — date ranges, filters, groupings. This helps it describe the answer accurately: "your grocery spending last month" instead of just "your grocery spending".

---

## Overall architecture decisions

### Why local Ollama instead of OpenAI API
Bank statements contain sensitive personal financial data. Sending them to a third-party API is a privacy risk. Running Ollama locally means all data stays on the machine. The trade-off is lower model quality compared to GPT-4 or Claude.

### Why PostgreSQL + pgvector instead of a dedicated vector database like Pinecone or Chroma
Transactions are structured data that benefit from SQL aggregations, joins, and filtering. A dedicated vector DB has no SQL capability. pgvector adds vector search to Postgres — one database handles both structured queries and semantic search. Fewer moving parts, simpler deployment.

### Why hybrid retrieval (SQL + semantic) instead of pure RAG
Pure RAG embeds everything and retrieves by similarity. "How much did I spend on groceries" over embeddings would return the most semantically similar transactions, not a sum. Sums, averages, and comparisons require SQL. Semantic search handles exploratory questions where exact numbers are not the goal. The two complement each other.

### Why raw transactions are stored before LLM enrichment
Separating storage from enrichment means the system is resilient to LLM failures. The transaction data is the ground truth — it comes from the PDF. Merchant names and categories are derived data that can always be re-generated. Losing the ground truth on a LLM timeout would be unrecoverable.
