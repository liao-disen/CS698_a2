import csv
import math
from collections import defaultdict


class Rule:
    def __init__(self, lhs: str, rhs: list[str], prob: float):
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.prob = float(prob)


class PCFG:
    """Minimal CNF PCFG utilities for this assignment."""

    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules or []
        self._reindex()

    def _reindex(self):
        self.by_lhs = defaultdict(list)
        self.binary_rules = []  # A -> B C
        self.lex_rules = []     # A -> w
        for r in self.rules:
            self.by_lhs[r.lhs].append(r)
            if len(r.rhs) == 2:
                self.binary_rules.append(r)
            elif len(r.rhs) == 1:
                self.lex_rules.append(r)

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

    def sentence_logprob(self, sent_tokens: list[str], start_symbol: str = "S") -> float:
        """Inside algorithm for CNF grammars. Returns log P(sentence) or -inf."""
        n = len(sent_tokens)
        if n == 0:
            return float("-inf")

        chart = defaultdict(float)  # (i, j, A) -> prob

        # length-1 spans using lexical/preterminal rules
        for i, w in enumerate(sent_tokens):
            for r in self.lex_rules:
                if r.rhs[0] == w:
                    chart[(i, i + 1, r.lhs)] += r.prob

        # longer spans
        for span in range(2, n + 1):
            for i in range(0, n - span + 1):
                j = i + span
                for k in range(i + 1, j):
                    for r in self.binary_rules:
                        b = chart[(i, k, r.rhs[0])]
                        c = chart[(k, j, r.rhs[1])]
                        if b and c:
                            chart[(i, j, r.lhs)] += r.prob * b * c

        p = chart[(0, n, start_symbol)]
        return math.log(p) if p > 0 else float("-inf")
