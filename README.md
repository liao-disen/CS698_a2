# CS698 Assignment 2: Grammar Cryptanalysis

This repo contains my solutions for reconstructing PCFGs from Transformer checkpoints.

## What I have done

### Task 1 (pcfg1)
- Implemented `code/solve_task1.py` for the known-structure setting (estimate rule probabilities from sampled sentences).
- Generated `pcfg1.csv` from `code/checkpoints/pcfg1.pt`.
- Used `code/evaluate.py` to check fit by average log-likelihood on sampled sentences.

Run:
- `python code/solve_task1.py --checkpoint code/checkpoints/pcfg1.pt`
- `python code/evaluate.py --pcfg pcfg1.csv --checkpoint code/checkpoints/pcfg1.pt`

### Task 2 (pcfg2)
- Replaced the old monolithic scripts with a modular pipeline under `code/task2/`:
  - `01_sample_dev.py`: sample + split dev data
  - `02_cluster_preterminals.py`: build preterminal clusters
  - `03_induce_parses.py`: induce parses / latent structures
  - `04_induce_rules.py`: induce grammar rules
  - `05_compress_and_em.py`: compress + EM refinement
  - `export_csv.py`: export final CSV grammar
- Added `run_task2.sh` to run the full Task 2 workflow.
- Produced `pcfg2.csv` (current Task 2 output) and kept `pcfg2_initial.csv` as the earlier baseline.
- Deleted old Task 2 scripts:
  - `code/solve_task2.py`
  - `code/task2_pipeline.py`

Run:
- `bash run_task2.sh`
- `python code/evaluate.py --pcfg pcfg2.csv --checkpoint code/checkpoints/pcfg2.pt`

## Repository notes
- Generated Task 2 artifacts are now ignored via `.gitignore`:
  - `code/task2/artifacts/`
- Local environment files are ignored (`.venv/`, `.DS_Store`).
