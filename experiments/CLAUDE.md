# experiments/

- Model-internals research, not part of the app. `app/` never imports it.
- Needs torch, transformers, matplotlib, seaborn and llama_cpp, which are not in `requirements.txt`.
- Exploratory code is fine here, but the hard constraints (privacy, data safety) still apply: no statement data in experiments.
