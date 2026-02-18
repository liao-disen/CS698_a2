import csv
import math
import random
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple


class Rule:
    def __init__(self, lhs: str, rhs: list[str], prob: float):
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.prob = float(prob)


class PCFG:
    """CNF-ish PCFG utilities for assignment tasks."""

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or []
        self._reindex()

    def _reindex(self):
        self.by_lhs = defaultdict(list)
        self.binary_rules = []  # A -> B C
        self.lex_rules = []  # A -> w
        self.lex_by_word = defaultdict(list)  # w -> list[(A, p)]
        self.binary_by_rhs = defaultdict(list)  # (B, C) -> list[(A, p)]
        self.nonterminals = set()

        for r in self.rules:
            self.by_lhs[r.lhs].append(r)
            self.nonterminals.add(r.lhs)
            if len(r.rhs) == 2:
                self.binary_rules.append(r)
                self.binary_by_rhs[(r.rhs[0], r.rhs[1])].append((r.lhs, r.prob))
            elif len(r.rhs) == 1:
                self.lex_rules.append(r)
                self.lex_by_word[r.rhs[0]].append((r.lhs, r.prob))

    def load_csv(self, path: str):
        self.rules = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lhs = row["LHS"].strip()
                rhs = row["RHS"].strip().split()
                p = float(row["Probability"])
                self.rules.append(Rule(lhs, rhs, p))
        self._reindex()

    # ---------- Probabilities ----------
    def sentence_prob(self, sent_tokens: list[str], start_symbol: str = "S") -> float:
        n = len(sent_tokens)
        if n == 0:
            return 0.0

        chart = defaultdict(float)  # (i, j, A) -> prob

        # lexical/preterminal
        for i, w in enumerate(sent_tokens):
            for lhs, p in self.lex_by_word.get(w, []):
                chart[(i, i + 1, lhs)] += p

        # inside
        for span in range(2, n + 1):
            for i in range(0, n - span + 1):
                j = i + span
                for k in range(i + 1, j):
                    # combine all possible B/C already present
                    left_syms = [A for A in self.nonterminals if chart[(i, k, A)] > 0.0]
                    right_syms = [A for A in self.nonterminals if chart[(k, j, A)] > 0.0]
                    for b in left_syms:
                        pb = chart[(i, k, b)]
                        for c in right_syms:
                            pc = chart[(k, j, c)]
                            for lhs, pr in self.binary_by_rhs.get((b, c), []):
                                chart[(i, j, lhs)] += pr * pb * pc

        return chart[(0, n, start_symbol)]

    def sentence_logprob(self, sent_tokens: list[str], start_symbol: str = "S") -> float:
        p = self.sentence_prob(sent_tokens, start_symbol=start_symbol)
        return math.log(p) if p > 0 else float("-inf")

    # ---------- Sampling ----------
    def sample_sentence(self, start_symbol: str = "S", max_steps: int = 200) -> list[str]:
        sent = [start_symbol]
        steps = 0

        while steps < max_steps:
            steps += 1
            nt_idx = next((i for i, tok in enumerate(sent) if tok in self.by_lhs), None)
            if nt_idx is None:
                break

            nt = sent[nt_idx]
            rules = self.by_lhs.get(nt, [])
            if not rules:
                break

            probs = [max(r.prob, 0.0) for r in rules]
            z = sum(probs)
            if z <= 0:
                probs = [1.0 / len(rules)] * len(rules)
            else:
                probs = [p / z for p in probs]

            chosen = random.choices(rules, weights=probs, k=1)[0]
            sent = sent[:nt_idx] + list(chosen.rhs) + sent[nt_idx + 1 :]

        # If unresolved nonterminals remain, drop them.
        return [t for t in sent if t not in self.by_lhs]

    # ---------- Diagnostics ----------
    def lhs_probability_sums(self) -> Dict[str, float]:
        out = {}
        for lhs, rules in self.by_lhs.items():
            out[lhs] = sum(r.prob for r in rules)
        return out

    def rule_count(self) -> int:
        return len(self.rules)

    def vocabulary(self) -> List[str]:
        return sorted({r.rhs[0] for r in self.lex_rules})

    def terminal_coverage_counts(self) -> Dict[str, int]:
        counts = Counter()
        for r in self.lex_rules:
            counts[r.rhs[0]] += 1
        return dict(counts)
