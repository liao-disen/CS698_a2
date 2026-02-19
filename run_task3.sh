.venv/bin/python code/task2/01_sample_dev.py \
  --checkpoint code/checkpoints/pcfg3.pt \
  --out-dir code/task3/artifacts \
  --num-samples 30000 \
  --max-len 24 \
  --temperature 1.0 \
  --top-p 1.0 \
  --device cpu

.venv/bin/python code/task2/02_cluster_preterminals.py \
  --checkpoint code/checkpoints/pcfg3.pt \
  --dev-train code/task3/artifacts/dev_train.jsonl \
  --out code/task3/artifacts/preterminals.json \
  --k-values 26 \
  --top-m 64 \
  --device cpu

.venv/bin/python code/task2/03_induce_parses.py \
  --checkpoint code/checkpoints/pcfg3.pt \
  --dev-train code/task3/artifacts/dev_train.jsonl \
  --preterminals code/task3/artifacts/preterminals.json \
  --out code/task3/artifacts/parses.jsonl \
  --device cpu

.venv/bin/python code/task2/04_induce_rules.py \
  --checkpoint code/checkpoints/pcfg3.pt \
  --parses code/task3/artifacts/parses.jsonl \
  --preterminals code/task3/artifacts/preterminals.json \
  --dev-eval code/task3/artifacts/dev_eval.jsonl \
  --out code/task3/artifacts/grammar_raw.json \
  --metrics-out code/task3/artifacts/rule_metrics.json \
  --num-nts 52 \
  --device cpu

.venv/bin/python code/task2/05_compress_and_em.py \
  --grammar-in code/task3/artifacts/grammar_raw.json \
  --grammar-out code/task3/artifacts/grammar_final.json \
  --target-binary-rules 50

.venv/bin/python code/task2/export_csv.py \
  --grammar code/task3/artifacts/grammar_final.json \
  --out pcfg3.csv

.venv/bin/python code/evaluate.py \
  --pcfg pcfg3.csv \
  --checkpoint code/checkpoints/pcfg3.pt \
  --num-samples 1000 \
  --device cpu
