# CS698 Assignment 2: Grammar Cryptanalysis

This repository contains the solutions and methodologies for reconstructing Probabilistic Context-Free Grammars (PCFGs) from Transformer language models.

## Project Structure
- `instruction.md`: Contains the assignment instructions.
- `plan.md`: Outlines the overall project strategy.
- `code/`:
    - `sampling.py`: Utility to sample sentences from a trained checkpoint.
    - `evaluate.py`: Script to evaluate a PCFG against model samples using average log-likelihood.
    - `pcfg_utils.py`: Core PCFG implementation (loading, saving, Inside algorithm).
    - `solve_task1.py`: Solution for Task 1 (known grammar structure, infer probabilities).
    - `solve_task2.py`: Solution for Task 2 (infer grammar structure and probabilities from English text).
    - `english_syntax_reference.txt`: A reference for common English syntax and POS tags.
    - `requirements.txt`: Python dependencies.
    - `checkpoints/`: Contains the trained model checkpoints (`pcfg1.pt`, `pcfg2.pt`, `pcfg3.pt`).
- `pcfg1.csv`: Inferred PCFG for Task 1.
- `pcfg2_initial.csv`: Initial skeleton PCFG for Task 2.

## Performance Estimation

To estimate performance locally, we use the `evaluate.py` script. Since a hidden test set is not available, this script uses a proxy method:

1.  **Sample Generation:** It generates a set of sentences by sampling from the provided Transformer checkpoint.
2.  **Log-Likelihood Calculation:** It then calculates the average log-probability assigned to these sampled sentences by the inferred PCFG using the Inside algorithm.

A higher average log-probability (closer to zero) indicates that the inferred PCFG better captures the distribution of the language model.

## Tasks

### Task 1: Getting Started
- **Methodology:** Count occurrences of grammatical patterns in sampled sentences to infer rule probabilities. The grammar structure was provided.
- **Execution:** `python code/solve_task1.py --checkpoint code/checkpoints/pcfg1.pt`
- **Evaluation:** `python code/evaluate.py --pcfg pcfg1.csv --checkpoint code/checkpoints/pcfg1.pt`

### Task 2: Realistic PCFG
- **Methodology:**
    1.  **Word Clustering:** Used K-Means on Transformer embeddings to group words into potential POS tags (clusters).
    2.  **Tag Sequence Generation:** Sampled sentences and converted them into sequences of predicted tags.
    3.  **Bigram Analysis:** Analyzed frequent tag bigrams to identify likely syntactic relationships.
    4.  **Rule Induction:** Generated preterminal rules (Tag -> word) based on word frequencies within clusters. Created skeleton nonterminal rules based on common English syntax and observed tag bigrams (e.g., `S -> CAT_0 CAT_1`, `NP -> DT NNS`).
- **Execution:** `python code/solve_task2.py --checkpoint code/checkpoints/pcfg2.pt --samples 200` (for initial skeleton generation)
- **Evaluation:** `python code/evaluate.py --pcfg pcfg2_initial.csv --checkpoint code/checkpoints/pcfg2.pt` (This uses the skeleton grammar and will likely show lower performance until nonterminal rules are refined).

## Self-Improvement Plan

For Task 2, the next step involves refining the nonterminal rules in `solve_task2.py` based on the observed tag bigrams and syntax reference, to produce a more complete and accurate PCFG. This refined grammar will then be evaluated.