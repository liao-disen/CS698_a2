from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import math
from functools import lru_cache
from tqdm.auto import tqdm

from task2.core import load_teacher, logprob
from task2.io_utils import read_json, read_jsonl, write_json, write_jsonl


def _build_tree(i: int, j: int, split):
    if j - i == 1:
        return {"i": i, "j": j, "leaf": True}
    k = split[i][j]
    return {
        "i": i,
        "j": j,
        "leaf": False,
        "k": k,
        "left": _build_tree(i, k, split),
        "right": _build_tree(k, j, split),
    }


def _cky_best_parse(tokens, span_scores):
    n = len(tokens)
    dp = [[-1e18 for _ in range(n + 1)] for _ in range(n + 1)]
    split = [[-1 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(n):
        dp[i][i + 1] = 0.0

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            best = -1e18
            best_k = -1
            for k in range(i + 1, j):
                val = dp[i][k] + dp[k][j] + span_scores[(i, j)]
                if val > best:
                    best = val
                    best_k = k
            dp[i][j] = best
            split[i][j] = best_k

    return _build_tree(0, n, split), float(dp[0][n])


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: induce binary parses by deletion-score CKY")
    ap.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    ap.add_argument("--dev-train", default="code/task2/artifacts/dev_train.jsonl")
    ap.add_argument("--preterminals", default="code/task2/artifacts/preterminals.json")
    ap.add_argument("--out", default="code/task2/artifacts/parses.jsonl")
    ap.add_argument("--stats-out", default="code/task2/artifacts/parse_stats.json")
    ap.add_argument("--max-sents", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--progress", dest="progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args()

    teacher = load_teacher(args.checkpoint, device=args.device)
    rows = read_jsonl(args.dev_train)[: args.max_sents]
    pt = read_json(args.preterminals)["word_to_pt"]

    outputs = []
    parse_scores = []

    for row in tqdm(rows, desc="Induce parses", disable=not args.progress):
        toks = row["tokens"]
        if not toks or len(toks) > args.max_len:
            continue
        pts = [pt[w] for w in toks]

        @lru_cache(maxsize=None)
        def lp_for_deleted(i: int, j: int) -> float:
            x_del = toks[:i] + toks[j:]
            return logprob(teacher, x_del)

        lp_full = logprob(teacher, toks)
        span_scores = {}
        n = len(toks)
        for i in range(n):
            for j in range(i + 1, n + 1):
                l = max(1, j - i)
                lp_del = lp_for_deleted(i, j)
                span_scores[(i, j)] = (lp_full - lp_del) / l

        tree, score = _cky_best_parse(toks, span_scores)
        parse_scores.append(score)
        outputs.append(
            {
                "tokens": toks,
                "pts": pts,
                "count": int(row.get("count", 1)),
                "score": score,
                "tree": tree,
            }
        )

    write_jsonl(args.out, outputs)
    write_json(
        args.stats_out,
        {
            "num_parsed": len(outputs),
            "avg_parse_score": float(sum(parse_scores) / max(1, len(parse_scores))),
        },
    )

    print(f"parsed={len(outputs)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
