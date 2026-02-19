from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from collections import Counter, defaultdict

from task2.io_utils import read_json, write_json


def _renorm_binary(binary_rules):
    by_lhs = defaultdict(list)
    for r in binary_rules:
        by_lhs[r["lhs"]].append(r)

    out = []
    for lhs, rules in by_lhs.items():
        z = sum(max(float(r["prob"]), 0.0) for r in rules)
        if z <= 0:
            p = 1.0 / len(rules)
            for r in rules:
                out.append({"lhs": lhs, "rhs": r["rhs"], "prob": p})
        else:
            for r in rules:
                out.append({"lhs": lhs, "rhs": r["rhs"], "prob": float(r["prob"]) / z})
    return out


def _prune_to_target(binary_rules, target_rules: int):
    if target_rules <= 0 or len(binary_rules) <= target_rules:
        return binary_rules

    by_lhs = defaultdict(list)
    for r in binary_rules:
        by_lhs[r["lhs"]].append(r)

    keep = []
    used = set()
    for lhs, rules in by_lhs.items():
        best = max(rules, key=lambda x: float(x["prob"]))
        keep.append(best)
        used.add((lhs, tuple(best["rhs"])))

    rest = [r for r in binary_rules if (r["lhs"], tuple(r["rhs"])) not in used]
    rest.sort(key=lambda x: float(x["prob"]), reverse=True)

    budget = max(0, target_rules - len(keep))
    keep.extend(rest[:budget])
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: compress grammar size (optional EM placeholder)")
    ap.add_argument("--grammar-in", default="code/task2/artifacts/grammar_raw.json")
    ap.add_argument("--grammar-out", default="code/task2/artifacts/grammar_final.json")
    ap.add_argument("--target-binary-rules", type=int, default=25)
    ap.add_argument("--min-binary-prob", type=float, default=1e-4)
    ap.add_argument("--em-iters", type=int, default=0, help="Placeholder in this version")
    args = ap.parse_args()

    g = read_json(args.grammar_in)
    binary_rules = list(g["binary_rules"])

    binary_rules = [r for r in binary_rules if float(r["prob"]) >= args.min_binary_prob]
    binary_rules = _prune_to_target(binary_rules, args.target_binary_rules)
    binary_rules = _renorm_binary(binary_rules)

    # EM hook: intentionally left as no-op in this clean baseline.
    if args.em_iters > 0:
        print("Warning: EM is not implemented in this clean baseline; skipped.")

    g["binary_rules"] = binary_rules
    g["meta"] = g.get("meta", {})
    g["meta"].update(
        {
            "num_binary_rules": len(binary_rules),
            "target_binary_rules": args.target_binary_rules,
            "em_iters": args.em_iters,
        }
    )
    write_json(args.grammar_out, g)

    lhs_counts = Counter(r["lhs"] for r in binary_rules)
    print(f"saved {args.grammar_out}")
    print(f"num_binary_rules={len(binary_rules)}, lhs={len(lhs_counts)}")


if __name__ == "__main__":
    main()
