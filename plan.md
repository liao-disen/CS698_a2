# Assignment 2: Grammar Cryptanalysis Plan

## Objective
Recover Probabilistic Context-Free Grammars (PCFGs) from trained Transformer language models.

## Infrastructure & Tools
We need a robust set of tools to analyze the models and verify our results.

1.  **`PCFG` Class (Python)**
    *   **Loading/Saving:** Read and write `.csv` files in the specified submission format.
    *   **Inside Algorithm:** Compute $P(x | G)$ for a given string $x$ and grammar $G$. This is crucial for the TVD calculation.
    *   **Sampling:** Generate strings from the PCFG to inspect outputs visually.
    *   **Viterbi Parse:** (Optional) Find the most likely parse tree for a string to debug structure.

2.  **Transformer Interface**
    *   Wrappers around the provided `sampling.py` to:
        *   Generate large datasets (e.g., 10k strings) from `pcfg1.pt`, `pcfg2.pt`, `pcfg3.pt`.
        *   (Optional) Extract next-token probabilities if needed for finer-grained analysis.

3.  **Evaluation Script (`evaluate.py`)**
    *   **Goal:** Estimate performance without the hidden test set.
    *   **Metric:** Log-Likelihood of held-out Transformer samples under our inferred PCFG.
    *   **Proxy TVD:**
        *   Generate a "Ground Truth" distribution $P_{proxy}$ by sampling a massive amount of data from the Transformer.
        *   Calculate TVD between $P_{proxy}$ and our $Q_{PCFG}$ on a validation set.
        *   $TVD(P, Q) = 0.5 \sum_x |P(x) - Q(x)|$.

## Task Strategy

### Task 1: Getting Started (Known Structure)
*   **Goal:** Estimate probabilities for the 10 known rules.
*   **Approach:**
    1.  Hardcode the known structure (S -> NP VP, etc.).
    2.  Sample $N=1000$ strings from `pcfg1.pt`.
    3.  **Counting:** Since the grammar is small and likely unambiguous for these simple sentences, we can parse the samples (or just count distinct patterns) to count how often each rule is used.
    4.  **Estimation:** $P(A \to \alpha) = \frac{Count(A \to \alpha)}{\sum_\beta Count(A \to \beta)}$.
    5.  **Refinement:** Use the Inside-Outside algorithm (EM) initialized with uniform probabilities to maximize the likelihood of the observed samples.

### Task 2: "Realistic" PCFG (Unknown Structure, English Words)
*   **Goal:** Reconstruct a 220-rule PCFG (20 non-terminals, 200 pre-terminals).
*   **Step 1: Identify Pre-terminals (Categories)**
    *   The terminals are English words.
    *   Use the Transformer's output distribution or simple co-occurrence statistics from samples to cluster words into categories (e.g., Nouns, Verbs, Determiners).
    *   Map each word $w$ to a tag $T$.
*   **Step 2: Grammar Induction on Tags**
    *   Replace words in sampled strings with their tags.
    *   We now have strings of tags (e.g., `DT NNS V`).
    *   **Algorithm:**
        *   Initialize a generic CNF grammar over the tags.
        *   Run Inside-Outside (EM) to learn rule probabilities and prune low-probability rules.
        *   Alternatively, use a heuristic approach: finding frequent substrings (constituents) to build the tree bottom-up.

### Task 3: Synthetic PCFG (Unknown Structure, Abstract)
*   **Goal:** Reconstruct PCFG for 26 pre-terminals ($A \to a$) and 50 expansion rules.
*   **Step 1: Mapping**
    *   Trivial mapping: $a \to A, b \to B, \dots$.
*   **Step 2: Structure Learning**
    *   Since we have strictly binary branching (CNF) and a limited number of non-terminals, this is a classic Grammar Induction problem.
    *   **Spectral Learning:** If applicable, spectral methods (using Hankel matrices of substring counts) can recover PCFG parameters.
    *   **EM Approach:** Initialize a fully connected CNF with a small number of non-terminals and run EM. If likelihood plateaus, add non-terminals (state-splitting) and retrain.

## Execution Plan
1.  **Setup:** Create `code/` directory, virtual environment, and install requirements.
2.  **Data Gen:** Generate 10k samples for each task using `sampling.py`.
3.  **Dev Loop:**
    *   Implement `pcfg_utils.py` (Class PCFG).
    *   Implement Task 1 solver (`solve_task1.py`). Verify with `evaluate.py`.
    *   Implement Task 2 solver (`solve_task2.py`) focusing on word clustering first.
    *   Implement Task 3 solver (`solve_task3.py`).
4.  **Submission:** Generate final `.csv` files and `README.md`.
