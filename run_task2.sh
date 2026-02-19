.venv/bin/python code/task2/01_sample_dev.py \
  --checkpoint code/checkpoints/pcfg2.pt \
  --out-dir code/task2/artifacts \
  --num-samples 20000 \
  --max-len 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --device cpu

.venv/bin/python code/task2/02_cluster_preterminals.py \
  --checkpoint code/checkpoints/pcfg2.pt \
  --dev-train code/task2/artifacts/dev_train.jsonl \
  --out code/task2/artifacts/preterminals.json \
  --k-values 15,20,25,30 \
  --top-m 120 \
  --device cpu

.venv/bin/python code/task2/03_induce_parses.py \
  --checkpoint code/checkpoints/pcfg2.pt \
  --dev-train code/task2/artifacts/dev_train.jsonl \
  --preterminals code/task2/artifacts/preterminals.json \
  --out code/task2/artifacts/parses.jsonl \
  --device cpu

.venv/bin/python code/task2/04_induce_rules.py \
  --checkpoint code/checkpoints/pcfg2.pt \
  --parses code/task2/artifacts/parses.jsonl \
  --preterminals code/task2/artifacts/preterminals.json \
  --dev-eval code/task2/artifacts/dev_eval.jsonl \
  --out code/task2/artifacts/grammar_raw.json \
  --metrics-out code/task2/artifacts/rule_metrics.json \
  --num-nts 24 \
  --device cpu

.venv/bin/python code/task2/05_compress_and_em.py \
  --grammar-in code/task2/artifacts/grammar_raw.json \
  --grammar-out code/task2/artifacts/grammar_final.json \
  --target-binary-rules 25

.venv/bin/python code/task2/export_csv.py \
  --grammar code/task2/artifacts/grammar_final.json \
  --out pcfg2.csv

.venv/bin/python code/evaluate.py \
  --pcfg pcfg2.csv \
  --checkpoint code/checkpoints/pcfg2.pt \
  --num-samples 1000 \
  --device cpu
