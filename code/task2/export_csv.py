from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import csv

from task2.io_utils import read_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Export task2 grammar json to submission csv")
    ap.add_argument("--grammar", default="code/task2/artifacts/grammar_final.json")
    ap.add_argument("--out", default="pcfg2.csv")
    args = ap.parse_args()

    g = read_json(args.grammar)
    rows = []
    for r in g.get("binary_rules", []):
        rows.append(
            {
                "LHS": r["lhs"],
                "LHS Type": "nonterminal",
                "RHS": " ".join(r["rhs"]),
                "Probability": float(r["prob"]),
            }
        )
    for r in g.get("lex_rules", []):
        rows.append(
            {
                "LHS": r["lhs"],
                "LHS Type": "preterminal",
                "RHS": r["rhs"][0],
                "Probability": float(r["prob"]),
            }
        )

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "LHS", "LHS Type", "RHS", "Probability"])
        for i, r in enumerate(rows):
            w.writerow([i, r["LHS"], r["LHS Type"], r["RHS"], f"{r['Probability']:.6f}"])

    print(f"wrote {args.out} ({len(rows)} rules)")


if __name__ == "__main__":
    main()
