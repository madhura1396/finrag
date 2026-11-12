# evals/

- Form eval: `python -m evals.read_path.run [--rewrite]` (Ollama up). Questions in `read_path/questions.yaml` are synthetic; never add a real question or merchant.
- Few-shot examples live in `app/query_router.TEACH_EXAMPLES`; the eval refuses any overlap with its questions. Keep the teach/test split: never copy an eval question into `TEACH_EXAMPLES`.
- Search eval (`python -m app.eval_search`) reads cases from `~/.finrag/search_eval.yaml` (or `FINRAG_SEARCH_EVAL`), outside the repo. Never read that file. `search_eval.example.yaml` shows the format with invented names.
- Report measured numbers only. A figure not produced by an eval in this repo is "unverified".
