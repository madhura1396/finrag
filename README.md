# FinRAG — Personal Finance RAG System

A fully local RAG system that parses Wells Fargo credit card statements and answers natural language questions about spending. No data leaves the machine.

---

## What it does

Upload a PDF statement → transactions are extracted, cleaned, and stored in PostgreSQL → ask questions in plain English → the system routes to SQL for exact calculations or vector search for exploratory queries → an LLM formats the answer using real numbers from the database.

```
$ python demo.py

Q: How much did I spend on groceries in December 2025?
  Answer : You spent $46.38 on groceries in December 2025.
  Route  : SQL

Q: Which merchant did I spend the most at?
  Answer : You spent the most at eBay with a total of $1,278.02.
  Route  : SQL

Q: Any charges that look like online shopping?
  Answer : Yes — $610.60 from eBay on 2025-12-25 and $32.66 from Shein.
  Route  : SEMANTIC
```

---

## Architecture

```
PDF upload
    │
    ▼
extractor.py       — PyMuPDF text extraction, state machine parser
    │
    ▼
llm.py             — Ollama (llama3.2) batch enrichment: clean merchant names, assign categories
    │
    ▼
embedder.py        — all-MiniLM-L6-v2 embeddings stored in pgvector
    │
    ▼
PostgreSQL + pgvector
    │
    ▼
query_router.py
    ├── _classify()        — single-token LLM call routes to sql or semantic
    ├── _generate_sql()    — chain-of-thought prompt → PostgreSQL SELECT
    └── _semantic_search() — cosine similarity, reranked so best chunk is closest to question
    │
    ▼
Answer formatted by LLM using exact numbers from DB
```

**Stack:** FastAPI · PostgreSQL · pgvector · Ollama (llama3.2) · sentence-transformers · SQLAlchemy

---

## Key design decisions

**Hybrid retrieval — SQL + semantic search**
Aggregation questions ("how much did I spend") need SQL. Pattern questions ("find charges that look like subscriptions") need vector search. Choosing the wrong path either returns wrong numbers or no results. A classifier LLM call routes each question before retrieval.

**Chain-of-thought SQL generation**
The SQL prompt forces the model to reason through metric → table → filters → aggregation before writing the query. Each intermediate step writes decisions into context, making patterns like `ILIKE` and date truncation more likely to appear correctly in the final query.

**Recency bias in semantic retrieval**
Retrieved chunks are reordered so the highest-relevance chunk sits immediately before the question in the prompt. LLMs pay more attention to context near the question — this is a one-line reorder with measurable accuracy impact.

**Local-only**
Bank statements contain sensitive personal financial data. All LLM inference runs via Ollama. Embeddings use a local sentence-transformers model. Nothing is sent to an external API.

Full architecture decisions with reasoning at [decisions.md](decisions.md).

---

## Experiments

The `experiments/` folder documents the model internals research that informed the FinRAG design decisions.

### Attention & residual stream
- **`attention_steps.py`** — step-by-step attention mechanism: Q·Kᵀ → scale → softmax → ·V, showing how attention weights distribute across tokens
- **`inspect_attention.py`** — attention sink visualization: token 0 accumulates disproportionate attention weight across all heads (known property of transformer decoders)
- **`residual_stream.py`** — how the residual stream accumulates meaning across 16 layers for a financial token like "grew"
- **`residual_demo.py`** — numpy demo showing why residual connections prevent vanishing gradients

### FFN layers
- **`ffn_steps.py`** / **`ffn_steps_layer15.py`** — feed-forward network step-by-step: gate · up projection → SiLU activation → down projection; comparing layer 0 (syntactic) vs layer 15 (semantic)

### Token geometry
- **`real_anisotropy.py`** — cosine similarity matrices across token pairs at 4 depth snapshots, showing how representations become more anisotropic (clustered) in deeper layers
- **`temperature_visualization.py`** — how temperature scaling sharpens or flattens the softmax distribution; shows entropy collapse at T=0.1 and near-uniform distribution at T=2.0

### Perplexity & domain adaptation
- **`perplexity_analysis.py`** — token-level loss comparison between general English (perplexity 9.23) and financial text (perplexity 12.77). The model finds financial acronyms like EBITDA and SG&A harder to predict, explaining why domain-specific RAG retrieval matters.

### Quantization gap
- **`quant_gap_eval.py`** — measures accuracy delta between bfloat16 HuggingFace weights and Q4_K_M Ollama weights on FinRAG classifier and financial continuation tasks. Classifier accuracy gap: 0 pp. Financial token continuation agreement: 25% — the quantized model substitutes generic tokens ("That", "It") for precise financial continuations ("quarter", "expenditures") on rare vocabulary.

---

## Running locally

**Prerequisites:** PostgreSQL with pgvector extension, Ollama with llama3.2 pulled

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# edit .env: set DATABASE_URL=postgresql://user:password@localhost:5432/finrag

# Start the server
uvicorn app.main:app --reload

# Upload a statement
curl -X POST http://localhost:8000/upload \
  -F "file=@your_statement.pdf"

# Run the demo
python demo.py
```

**API endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload a PDF statement |
| POST | `/query` | Ask a question in plain English |
| GET | `/statements` | List uploaded statements |
| GET | `/statements/{id}/transactions` | List transactions for a statement |
| POST | `/trips` | Define a named trip by date range |

---

## Eval results

Semantic search accuracy on 7 query types against real transaction data:

| Embedding format | Accuracy |
|---|---|
| Category only | 67% |
| Merchant + category | 80% |
| Merchant + raw description + category | 73%* |

*Drop from 80% is due to LLM merchant name cleaning removing terms like "Park Meters" that were useful for matching. Raw description restores this signal but introduces noise from store numbers and transaction codes.
