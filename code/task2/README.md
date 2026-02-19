# Task 2 (Rebuilt Pipeline)

This folder replaces the old monolithic Task 2 code with stage scripts matching the induction plan.

## Stage scripts

- `01_sample_dev.py`
  - Samples teacher strings, deduplicates, splits into `dev_train` / `dev_eval`.
  - Saves teacher sanity stats (`avg_logprob_teacher`, length histogram, unigram/bigram).
- `02_cluster_preterminals.py`
  - Builds word context features from teacher `P(next|prefix ending in w)`.
  - Clusters words into preterminals and estimates `PT -> word` emissions.
- `03_induce_parses.py`
  - Uses deletion-score span scoring with CKY-style binary parsing.
- `04_induce_rules.py`
  - Clusters internal spans into latent nonterminals and extracts `A -> B C` probabilities.
  - Reports teacher-vs-PCFG logprob correlation and proxy TVD on `dev_eval`.
- `05_compress_and_em.py`
  - Prunes/compresses binary rules toward a target count (EM hook currently no-op placeholder).
- `export_csv.py`
  - Exports final grammar JSON to `pcfg2.csv` submission format.

## Core primitives (Step 0)

Shared in `task2/core.py`:

- `sample(model, n, max_len, temperature, top_p)`
- `logprob(model, seq)`
- `prefix_next_dist(model, prefix)`
- `pcfg_inside_prob(grammar, seq)`
- `avg_logprob_teacher(model, samples)`

## Quick run

From repo root:

```bash
.venv/bin/python code/task2/01_sample_dev.py --checkpoint code/checkpoints/pcfg2.pt --out-dir code/task2/artifacts --num-samples 20000 --device cpu
.venv/bin/python code/task2/02_cluster_preterminals.py --checkpoint code/checkpoints/pcfg2.pt --dev-train code/task2/artifacts/dev_train.jsonl --out code/task2/artifacts/preterminals.json --device cpu
.venv/bin/python code/task2/03_induce_parses.py --checkpoint code/checkpoints/pcfg2.pt --dev-train code/task2/artifacts/dev_train.jsonl --preterminals code/task2/artifacts/preterminals.json --out code/task2/artifacts/parses.jsonl --device cpu
.venv/bin/python code/task2/04_induce_rules.py --checkpoint code/checkpoints/pcfg2.pt --parses code/task2/artifacts/parses.jsonl --preterminals code/task2/artifacts/preterminals.json --dev-eval code/task2/artifacts/dev_eval.jsonl --out code/task2/artifacts/grammar_raw.json --device cpu
.venv/bin/python code/task2/05_compress_and_em.py --grammar-in code/task2/artifacts/grammar_raw.json --grammar-out code/task2/artifacts/grammar_final.json --target-binary-rules 25
.venv/bin/python code/task2/export_csv.py --grammar code/task2/artifacts/grammar_final.json --out pcfg2.csv
```

## Notes

- This is a clean baseline rewrite focused on modularity and reproducibility.
- `05_compress_and_em.py` includes compression; full inside-outside EM is left as a future extension.
